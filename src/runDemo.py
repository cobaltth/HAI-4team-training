import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from model import EventChartTransformer
from mp3toSpec import build_spectrogram, sanitize_filename
from parser import ChartEvent, events_to_frame_labels, events_to_numpy
from toOsu import events_txt_to_osu


# =========================================================
# 0. 디바이스
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[GENERATE] device: {device}")

MAX_DELTA    = EventChartTransformer.MAX_DELTA
MAX_DURATION = EventChartTransformer.MAX_DURATION
EOS_TYPE     = EventChartTransformer.EOS_TYPE


# =========================================================
# 1. 단일 window 이벤트 생성
# =========================================================
def _sample(logits: torch.Tensor, temperature: float, top_k: int = 0) -> int:
    """Argmax when temperature==0, multinomial sampling otherwise.

    top_k > 0 이면 상위 k개 logit만 남기고 나머지는 -inf로 마스킹한다.
    """
    if temperature <= 0:
        return logits.argmax(-1).item()
    if top_k > 0:
        k = min(top_k, logits.shape[-1])
        thresh = torch.topk(logits, k).values[-1]
        logits = logits.masked_fill(logits < thresh, float('-inf'))
    return torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1).item()


def generate_window_events(
    model: EventChartTransformer,
    spec_chunk: torch.Tensor,
    max_events: int = 256,
    temperature: float = 1.0,
    bpm: Optional[torch.Tensor] = None,
    prime_event: Optional[List[int]] = None,
    eos_threshold: float = 0.8,
    active_holds: Optional[dict] = None,
    hold_bias: float = 1.3,
    top_k_delta: int = 0,
    min_hold_dur: int = 3,
    min_gap_per_lane: int = 3,
    max_active_holds: int = 2,
    hold_end_cooldown: int = 2,
    snap_bpm: Optional[float] = None,
    max_chord_size: int = 2,
    lane_bias_window: int = 8,
    lane_bias_max_ratio: float = 0.6,
    cross_lane_gap: int = 2,
) -> List[List[int]]:
    """Generate delta-encoded events for one spec window.

    Args:
        model:          Trained EventChartTransformer.
        spec_chunk:     (T, n_mels) spectrogram window (already padded to window size).
        max_events:     Maximum number of events to generate.
        prime_event:    [delta, lane, note_type, dur] from previous window's last event.
                        Injected after BOS to carry cross-window context.
        eos_threshold:  EOS is suppressed until current_frame >= T * eos_threshold.

    Returns:
        List of [delta_frames, lane, note_type, duration_frames].
        Stops at EOS or when cumulative frame exceeds T.
    """
    model.eval()

    audio = spec_chunk.unsqueeze(0).to(device)  # (1, T, n_mels)
    T = audio.shape[1]

    with torch.no_grad():
        bpm_in = bpm.reshape(1) if bpm is not None else None
        memory = model.encode_spec(audio, bpm_in)       # (1, T, d_model)
        tgt    = model.bos.expand(1, -1, -1).clone()    # (1, 1, d_model)

        # Inject prime event from previous window (delta=0: "just before this window")
        if prime_event is not None:
            ev_t = torch.tensor([[[0, prime_event[1], prime_event[2], prime_event[3]]]],
                                dtype=torch.long, device=device)  # (1, 1, 4)
            tgt = torch.cat([tgt, model.embed_events(ev_t)], dim=1)

        generated = []
        current_frame = 0
        consecutive_delta0 = 0   # 현재 프레임에서 연속 delta=0 카운터

        # 레인별 상태 (제약조건용)
        last_note_frame:   dict = {0: -999, 1: -999, 2: -999, 3: -999}
        hold_end_frame:    dict = {0: -999, 1: -999, 2: -999, 3: -999}
        window_active_holds: dict = {}   # {lane: end_frame_in_window}

        for _ in range(max_events):
            N = tgt.shape[1]
            causal_mask = torch.triu(
                torch.ones(N, N, device=device), diagonal=1
            ).bool()

            out  = model.decoder(tgt, memory, tgt_mask=causal_mask)  # (1, N, d_model)
            last = out[:, -1, :]                                       # (1, d_model)

            delta_logits = model.fc_delta(last)[0].clone()
            lane_logits  = model.fc_lane(last)[0]
            type_logits  = model.fc_type(last)[0].clone()
            dur_logits   = model.fc_dur(last)[0]

            # 4-key 게임에서 한 프레임에 최대 4개 노트 → delta=0은 최대 3번 연속
            # 그 이상이면 delta=0 차단해서 runaway loop 방지
            if consecutive_delta0 >= max(max_chord_size - 1, 0):
                delta_logits[0] = float('-inf')

            # 전체 레인 통합 최소 간격: delta=0(코드)은 허용, 1~cross_lane_gap-1 차단
            if cross_lane_gap > 1:
                delta_logits[1:cross_lane_gap] = float('-inf')

            # BPM 스냅: beat 단위 배수 delta만 허용 (snap_bpm 지정 시)
            if snap_bpm is not None and snap_bpm > 0:
                fpb = 60000.0 / snap_bpm / 50.0
                valid_snaps = {0}
                for divisor in (4.0, 2.0, 1.0, 0.5):
                    v = round(fpb / divisor)
                    for tol in (-1, 0, 1):
                        d = v + tol
                        if 0 < d <= MAX_DELTA:
                            valid_snaps.add(d)
                for d in range(MAX_DELTA + 1):
                    if d not in valid_snaps:
                        delta_logits[d] = float('-inf')

            # Suppress EOS until threshold fraction of window is covered
            if current_frame < T * eos_threshold:
                type_logits[EOS_TYPE] = float('-inf')

            delta = _sample(delta_logits, temperature, top_k=top_k_delta)

            # 같은 레인 cooldown: 최소 간격 및 hold 종료 후 쉬는 시간 적용
            lane_logits = lane_logits.clone()
            blocked = [
                (current_frame - last_note_frame[li] < min_gap_per_lane or
                 current_frame - hold_end_frame[li] < hold_end_cooldown)
                for li in range(4)
            ]
            if not all(blocked):
                for li, b in enumerate(blocked):
                    if b:
                        lane_logits[li] = float('-inf')

            if len(generated) >= lane_bias_window:
                recent_lanes = [e[1] for e in generated[-lane_bias_window:]]
                for li in range(4):
                    if recent_lanes.count(li) / lane_bias_window >= lane_bias_max_ratio:
                        lane_logits[li] = float('-inf')

            lane  = _sample(lane_logits, temperature)

            # hold logit 억제: 훈련/추론 분포 차이 보정
            type_logits[1] -= hold_bias

            # 이전 window에서 이 레인에 hold가 진행 중이면 중복 hold 완전 차단
            if active_holds and lane in active_holds:
                type_logits[1] = float('-inf')

            # 동시 active hold 수 제한
            n_cur_holds = sum(1 for ef in window_active_holds.values() if ef > current_frame)
            if n_cur_holds >= max_active_holds:
                type_logits[1] = float('-inf')

            note_type = _sample(type_logits, temperature)
            dur = _sample(dur_logits, temperature)
            if note_type == 1 and min_hold_dur > 0 and dur < min_hold_dur:
                note_type = 0
                dur = 0

            if note_type == EOS_TYPE:
                break

            current_frame += delta
            if current_frame >= T:
                break

            # delta=0 연속 카운터 갱신
            if delta == 0:
                consecutive_delta0 += 1
            else:
                consecutive_delta0 = 0

            # 레인별 상태 업데이트 (제약조건용)
            last_note_frame[lane] = current_frame
            if note_type == 1:
                hold_end_frame[lane] = current_frame + dur
                window_active_holds[lane] = current_frame + dur

            # 모델 피드백은 샘플링된 원본 dur 사용, 저장 시에만 tap dur=0
            save_dur = 0 if note_type != 1 else dur
            generated.append([delta, lane, note_type, save_dur])

            # Embed this event and extend tgt
            ev  = torch.tensor([[[delta, lane, note_type, dur]]],
                               dtype=torch.long, device=device)  # (1, 1, 4)
            tgt = torch.cat([tgt, model.embed_events(ev)], dim=1)

    return generated


# =========================================================
# 2. 전체 곡 이벤트 생성 (overlapping windows 지원)
# =========================================================
def generate_full_chart(
    model: EventChartTransformer,
    spec: torch.Tensor,
    window: int = 512,
    stride: Optional[int] = None,
    max_events_per_window: int = 256,
    temperature: float = 1.0,
    bpm: Optional[torch.Tensor] = None,
    eos_threshold: float = 0.8,
    hold_bias: float = 1.3,
    top_k_delta: int = 0,
    min_hold_dur: int = 3,
    min_gap_per_lane: int = 3,
    max_active_holds: int = 2,
    hold_end_cooldown: int = 2,
    snap_bpm: Optional[float] = None,
    max_chord_size: int = 2,
    lane_bias_window: int = 8,
    lane_bias_max_ratio: float = 0.6,
    cross_lane_gap: int = 2,
) -> List[ChartEvent]:
    """Generate ChartEvent list for the full spectrogram.

    stride < window 이면 overlapping window 모드:
      - 각 window는 spec[start : start+window] 를 인코더 입력으로 사용
      - 이벤트는 [start, start+stride) 범위만 수집하고 나머지는 버림
      - EOS가 stride 이후에 발생해도 해당 구간은 다음 window가 커버
      stride=None 이면 stride=window (기존 non-overlapping 동작)

    Returns:
        List of ChartEvent with absolute time_frame, sorted by time.
    """
    _stride = stride if stride is not None else window

    T = spec.shape[0]
    all_events: List[ChartEvent] = []
    prime_event: Optional[List[int]] = None   # last collected event of previous window
    active_holds: dict = {}                   # {lane: end_abs_frame} — 진행 중인 hold

    for start in range(0, T, _stride):
        end         = min(start + window, T)   # 인코더 입력 범위
        # 첫 윈도우는 prime_event가 없어 모델이 큰 delta를 생성하는 경향이 있어
        # stride 대신 window 전체를 수집해 0~stride 구간 공백을 방지한다
        collect_end = min(start + (window if start == 0 else _stride), T)

        spec_chunk = spec[start:end]

        # Pad to full window size so the model sees a consistent input shape
        if spec_chunk.shape[0] < window:
            pad = window - spec_chunk.shape[0]
            spec_chunk = torch.cat([spec_chunk, torch.zeros(pad, spec_chunk.shape[1])])

        delta_events = generate_window_events(
            model, spec_chunk, max_events_per_window, temperature, bpm,
            prime_event=prime_event,
            eos_threshold=eos_threshold,
            active_holds=active_holds,
            hold_bias=hold_bias,
            top_k_delta=top_k_delta,
            min_hold_dur=min_hold_dur,
            min_gap_per_lane=min_gap_per_lane,
            max_active_holds=max_active_holds,
            hold_end_cooldown=hold_end_cooldown,
            snap_bpm=snap_bpm,
            max_chord_size=max_chord_size,
            lane_bias_window=lane_bias_window,
            lane_bias_max_ratio=lane_bias_max_ratio,
            cross_lane_gap=cross_lane_gap,
        )

        # Convert delta → absolute, collect only up to collect_end
        abs_frame = start
        last_collected: Optional[List[int]] = None
        for delta, lane, note_type, dur in delta_events:
            abs_frame += delta
            if abs_frame >= collect_end:
                break
            all_events.append(ChartEvent(abs_frame, lane, note_type, dur))
            last_collected = [delta, lane, note_type, dur]
            if note_type == 1:
                active_holds[lane] = abs_frame + dur   # hold 종료 프레임 기록

        # collect_end를 넘어서 끝난 hold만 유지
        active_holds = {lane: ef for lane, ef in active_holds.items() if ef > collect_end}

        # 다음 window의 prime으로 마지막 수집 이벤트 전달
        if last_collected is not None:
            prime_event = last_collected

    return all_events


# =========================================================
# 2-1. Hold 겹침 후처리
# =========================================================
def fix_hold_overlaps(events: List[ChartEvent]) -> List[ChartEvent]:
    """같은 레인에서 겹치는 hold 노트를 하나로 병합한다.

    레인별로 시간순 정렬 후, 앞 hold와 겹치는 뒤 hold는
    앞 hold의 end_frame을 max(prev_end, new_end)로 늘려 병합한다.
    tap은 그대로 통과.
    """
    # 레인별로 분리해서 hold 병합
    lane_holds: dict = {0: [], 1: [], 2: [], 3: []}
    taps: List[ChartEvent] = []

    for ev in sorted(events, key=lambda e: (e.time_frame, e.lane)):
        if ev.note_type == 1:
            lane_holds[ev.lane].append(ev)
        else:
            taps.append(ev)

    merged: List[ChartEvent] = list(taps)

    for lane, holds in lane_holds.items():
        if not holds:
            continue
        cur_start = holds[0].time_frame
        cur_end   = holds[0].time_frame + holds[0].duration_frames

        for ev in holds[1:]:
            ev_end = ev.time_frame + ev.duration_frames
            if ev.time_frame < cur_end:  # 겹침 → 병합
                cur_end = max(cur_end, ev_end)
            else:                         # 겹치지 않음 → 이전 구간 확정
                merged.append(ChartEvent(cur_start, lane, 1, cur_end - cur_start))
                cur_start = ev.time_frame
                cur_end   = ev_end

        merged.append(ChartEvent(cur_start, lane, 1, cur_end - cur_start))

    return sorted(merged, key=lambda e: (e.time_frame, e.lane))


# =========================================================
# 2-2. Hold 구간 내 tap 제거
# =========================================================
def fix_tap_in_hold(events: List[ChartEvent]) -> List[ChartEvent]:
    """hold 구간 내에 위치한 tap 이벤트를 제거한다.

    fix_hold_overlaps() 이후에 호출 — hold 구간이 이미 병합된 상태를 전제로 함.
    """
    hold_intervals: dict = {0: [], 1: [], 2: [], 3: []}
    for ev in events:
        if ev.note_type == 1:
            hold_intervals[ev.lane].append(
                (ev.time_frame, ev.time_frame + ev.duration_frames)
            )

    result = []
    for ev in events:
        if ev.note_type == 0:
            in_hold = any(s <= ev.time_frame <= e for s, e in hold_intervals[ev.lane])
            if in_hold:
                continue
        result.append(ev)
    return result


# =========================================================
# 2-3. 같은 레인 최소 간격 강제
# =========================================================
def enforce_min_gap(events: List[ChartEvent], min_gap: int = 3) -> List[ChartEvent]:
    """같은 레인에서 min_gap 미만 간격의 노트를 제거한다."""
    last_frame: dict = {0: -999, 1: -999, 2: -999, 3: -999}
    result = []
    for ev in sorted(events, key=lambda e: (e.time_frame, e.lane)):
        if ev.time_frame - last_frame[ev.lane] >= min_gap:
            result.append(ev)
            last_frame[ev.lane] = ev.time_frame
    return result


# =========================================================
# 2-4. 같은 (frame, lane) 중복 제거
# =========================================================
def dedup_events(events: List[ChartEvent]) -> List[ChartEvent]:
    """같은 (time_frame, lane) 위치에 중복된 이벤트를 하나로 줄인다.

    delta=0 collapse로 인해 동일 프레임에 같은 레인 노트가 수십 개 쌓이는
    현상을 후처리로 제거한다. 중복 중 hold가 있으면 hold를 우선 유지하고,
    없으면 첫 번째 tap만 남긴다.
    """
    seen: dict = {}   # (time_frame, lane) → ChartEvent
    for ev in sorted(events, key=lambda e: (e.time_frame, e.lane, -e.note_type)):
        key = (ev.time_frame, ev.lane)
        if key not in seen:
            seen[key] = ev
        elif ev.note_type == 1 and seen[key].note_type != 1:
            seen[key] = ev   # hold가 있으면 tap 대신 hold로 교체
    return sorted(seen.values(), key=lambda e: (e.time_frame, e.lane))


# =========================================================
# 3. 실행
# =========================================================
def _unique_stem(base: str, directory: Path) -> str:
    """Return base if both output files are free, else base_1, base_2, ..."""
    if not (directory / f"{base}-generated.txt").exists() and \
       not (directory / f"{base}-generated_events.txt").exists():
        return base
    i = 1
    while True:
        candidate = f"{base}_{i}"
        if not (directory / f"{candidate}-generated.txt").exists() and \
           not (directory / f"{candidate}-generated_events.txt").exists():
            return candidate
        i += 1


def _save_chart(
    chart_events: List[ChartEvent],
    song_name: str,
    output_dir: Path,
    num_frames: int,
) -> None:
    stem = _unique_stem(song_name, output_dir)
    generated_txt_path    = output_dir / f"{stem}-generated.txt"
    generated_events_path = output_dir / f"{stem}-generated_events.txt"

    frame_labels = events_to_frame_labels(chart_events, num_frames=num_frames)
    np.savetxt(generated_txt_path, frame_labels, fmt="%d")

    events_np = events_to_numpy(chart_events)
    _type_name = {0: "tap", 1: "hold"}
    with open(generated_events_path, "w", encoding="utf-8") as f:
        f.write("time_frame\tlane\ttype\tduration_frames\n")
        for row in events_np:
            t, lane, ntype, dur = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            f.write(f"{t}\t{lane}\t{_type_name.get(ntype, str(ntype))}\t{dur}\n")

    print(f"  Saved {generated_txt_path}  ({frame_labels.shape})")
    print(f"  Saved {generated_events_path}  ({events_np.shape})")

    osu_path = events_txt_to_osu(generated_events_path)
    print(f"  Saved {osu_path}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent

    # ── input 폴더 준비 ───────────────────────────────────
    input_dir  = root / "input"
    output_dir = root / "output"
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    mp3_files = sorted(input_dir.glob("*.mp3"))
    if not mp3_files:
        print(f"input/ 폴더에 MP3 파일이 없습니다: {input_dir}")
        raise SystemExit(0)

    print(f"Found {len(mp3_files)} MP3 file(s) in {input_dir}")

    # ── 모델 로드 (한 번만) ───────────────────────────────
    model = EventChartTransformer().to(device)
    model.load_state_dict(
        torch.load(root / "models" / "best.pt", map_location=device)
    )
    print("Model loaded.\n")

    ok, skip = 0, 0

    for mp3_path in mp3_files:
        print(f"[{mp3_path.name}]")
        try:
            # ── 스펙트로그램 변환 ─────────────────────────
            spec = build_spectrogram(mp3_path)
            print(f"  Spec shape: {spec.shape}")

            # ── BPM: 사이드카 JSON에서 읽기, 없으면 120 ──
            bpm_val = 120.0
            meta_path = mp3_path.with_suffix(".json")
            if meta_path.exists():
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                bpm_val = float(meta.get("BPM", 120) or 120)
            bpm_norm = torch.tensor(
                [(60000.0 / max(bpm_val, 60.0) / 50.0) / 20.0], dtype=torch.float
            ).to(device)
            print(f"  BPM: {bpm_val:.1f}  (normalized: {bpm_norm.item():.3f})")

            # ── 이벤트 생성 ───────────────────────────────
            chart_events = generate_full_chart(
                model, spec, stride=256, bpm=bpm_norm,
                eos_threshold=0.8, hold_bias=1.3, top_k_delta=20,
                snap_bpm=bpm_val,
                cross_lane_gap=2,
            )
            print(f"  Generated {len(chart_events)} events")

            chart_events = dedup_events(chart_events)
            print(f"  After dedup: {len(chart_events)} events")

            chart_events = fix_hold_overlaps(chart_events)
            print(f"  After overlap fix: {len(chart_events)} events")

            chart_events = fix_tap_in_hold(chart_events)
            print(f"  After tap-in-hold fix: {len(chart_events)} events")

            chart_events = enforce_min_gap(chart_events, min_gap=3)
            print(f"  After min-gap enforce: {len(chart_events)} events")

            # ── 저장 ─────────────────────────────────────
            song_name = sanitize_filename(mp3_path.stem)
            _save_chart(chart_events, song_name, output_dir, num_frames=spec.shape[0])

            ok += 1

        except Exception as e:
            print(f"  [SKIP] {e}")
            skip += 1

        print()

    print(f"완료 — 성공: {ok}, 실패: {skip}")

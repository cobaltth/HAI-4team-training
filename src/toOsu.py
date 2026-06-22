import json
import sys
from pathlib import Path

from parser import sanitize_filename


HOP_MS = 50

def _unique_path(directory: Path, stem: str, suffix: str) -> Path:
    candidate = directory / f"{stem}{suffix}"
    if not candidate.exists():
        return candidate
    i = 1
    while True:
        candidate = directory / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def events_txt_to_osu(events_txt_path: Path) -> Path:
    root = events_txt_path.resolve().parent.parent  # project root

    # ── 곡 이름 추출 ──────────────────────────────────────────
    # 파일명 패턴: 곡이름-generated_events.txt
    fname = events_txt_path.stem  # e.g. "Celldweller - One Good Reason-generated_events"
    if "-generated_events" in fname:
        song_name = fname[: fname.rfind("-generated_events")]
    else:
        song_name = fname

    # ── metadata.json 탐색 ────────────────────────────────────
    title         = song_name
    artist        = ""
    bpm           = 120.0
    audio_filename = "audio.mp3"
    hop_ms        = HOP_MS

    meta_path = root / "source" / song_name / "metadata.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        title          = meta.get("title") or song_name
        artist         = meta.get("artist") or ""
        bpm            = float(meta.get("BPM") or 120)
        audio_filename = meta.get("audio_filename") or "audio.mp3"
        hop_ms         = int(meta.get("frame_ms") or HOP_MS)

    beat_length = 60000.0 / max(bpm, 1.0)

    # ── events_txt 파싱 ───────────────────────────────────────
    hit_objects = []
    with open(events_txt_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4 or parts[0] == "time_frame":
                continue
            time_frame, lane, note_type, dur_frames = (
                int(parts[0]), int(parts[1]), parts[2], int(parts[3])
            )
            start_ms = time_frame * hop_ms
            x = lane * 128 + 64

            if note_type == "tap":
                hit_objects.append(f"{x},192,{start_ms},1,0,0:0:0:0:")
            else:  # hold
                end_ms = (time_frame + dur_frames) * hop_ms
                hit_objects.append(f"{x},192,{start_ms},128,0,{end_ms}:0:0:0:0:")

    # ── .osu 파일 조립 ────────────────────────────────────────
    osu_content = f"""osu file format v14

[General]
AudioFilename: {audio_filename}
AudioLeadIn: 0
Mode: 3

[Metadata]
Title:{title}
Artist:{artist}
Version:AI Generated
Creator:AI

[Difficulty]
HPDrainRate:5
CircleSize:4
OverallDifficulty:5
ApproachRate:5
SliderMultiplier:1.4
SliderTickRate:1

[TimingPoints]
0,{beat_length:.6f},4,2,1,60,1,0

[HitObjects]
"""
    osu_content += "\n".join(hit_objects) + "\n"

    # ── 출력 경로 결정 ────────────────────────────────────────
    out_dir = root / "OSUoutput"
    out_dir.mkdir(exist_ok=True)

    out_path = _unique_path(out_dir, f"{sanitize_filename(song_name)}-generated", ".osu")
    out_path.write_text(osu_content, encoding="utf-8")

    return out_path


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    output_dir = root / "output"

    targets = sorted(output_dir.glob("*-generated_events.txt"))
    if not targets:
        print("output/ 폴더에 *-generated_events.txt 파일이 없습니다.")
        sys.exit(0)

    for txt_path in targets:
        try:
            result = events_txt_to_osu(txt_path)
            print(f"Saved: {result}")
        except Exception as e:
            print(f"건너뜀: {txt_path.name} → {e}")

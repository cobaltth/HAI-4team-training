# Code Review — EventChartTransformer

> 작성일: 2026-04-30  
> 대상 파일: `src/model.py`, `src/traindemo.py`, `src/runDemo.py`

---

## 목차

1. [전체 아키텍처 개요](#1-전체-아키텍처-개요)
2. [src/model.py](#2-srcmodelpy)
3. [src/traindemo.py](#3-srctraindempopy)
4. [src/runDemo.py](#4-srcrundemospy)
5. [데이터 흐름 요약](#5-데이터-흐름-요약)
6. [설계 결정 및 트레이드오프](#6-설계-결정-및-트레이드오프)

---

## 1. 전체 아키텍처 개요

본 프로젝트는 오디오 스펙트로그램으로부터 4레인 리듬 게임의 차트(노트 시퀀스)를 자동 생성하는 **Encoder-Decoder Transformer** 시스템이다.

```
┌─────────────────────────────────────────────────────────┐
│  오디오 (mel-spectrogram)                                │
│        ↓  Encoder                                        │
│  audio context (memory)                                  │
│        ↓  Decoder (autoregressive)                       │
│  이벤트 시퀀스: [delta, lane, note_type, duration]        │
└─────────────────────────────────────────────────────────┘
```

### 파일별 역할

| 파일 | 역할 |
|------|------|
| `model.py` | 모델 아키텍처 정의 (Transformer) |
| `traindemo.py` | 데이터셋 구축 + 학습 루프 |
| `runDemo.py` | 추론 + 후처리 + 파일 저장 |

### 이벤트 토큰 구조

각 이벤트는 4개의 정수 필드로 표현된다:

| 인덱스 | 필드명 | 범위 | 의미 |
|--------|--------|------|------|
| 0 | `delta_frames` | 0 ~ 256 | 이전 이벤트로부터의 시간 간격 (프레임 단위, 1프레임=50ms) |
| 1 | `lane` | 0 ~ 3 | 노트가 위치하는 레인 번호 |
| 2 | `note_type` | 0, 1, 2 | 0=tap, 1=hold, 2=EOS(시퀀스 종료) |
| 3 | `duration_frames` | 0 ~ 128 | hold 노트의 지속 시간, tap/EOS는 항상 0 |

- `MAX_DELTA=256` → 최대 inter-event 간격: 256 × 50ms = **12.8초**
- `MAX_DURATION=128` → 최대 hold 길이: 128 × 50ms = **6.4초**

---

## 2. src/model.py

### 파일 역할

모델의 전체 아키텍처를 정의한다. 학습(`traindemo.py`)과 추론(`runDemo.py`) 양쪽에서 임포트해 사용한다.

---

### `sinusoidal_pe(T, d_model, device)` — 위치 인코딩

```
인자
  T        : 시퀀스 길이 (프레임 수)
  d_model  : 모델 임베딩 차원
  device   : 연산 디바이스

반환
  pe : (T, d_model) — 각 위치에 대한 sin/cos 혼합 벡터
```

**동작 원리**:
- 짝수 차원: `sin(pos / 10000^(2i/d_model))`
- 홀수 차원: `cos(pos / 10000^(2i/d_model))`
- 학습 파라미터 없음 — 수식으로 결정론적 계산
- 길이 제한 없이 임의의 `T`에 적용 가능 (학습 시 보지 못한 길이의 spec도 처리 가능)

---

### `class EventChartTransformer(nn.Module)` — 메인 모델

#### 클래스 상수

```python
MAX_DELTA    = 256   # inter-event 최대 간격 (프레임)
MAX_DURATION = 128   # hold 최대 지속 시간 (프레임)
EOS_TYPE     = 2     # EOS를 나타내는 note_type 값
```

#### `__init__(n_mels=128, d_model=256)` — 레이어 정의

**Encoder 관련**:

| 레이어 | 입력 | 출력 | 역할 |
|--------|------|------|------|
| `audio_proj` | `(B, T, n_mels)` | `(B, T, d_model)` | 멜 스펙트로그램 차원 변환 |
| `encoder` | `(B, T, d_model)` | `(B, T, d_model)` | 4층 TransformerEncoder, 오디오 컨텍스트 추출 |

TransformerEncoderLayer 설정: `nhead=8`, `dim_feedforward=1024`, `dropout=0.1`

**Event Embedding 관련**:

| 레이어 | 임베딩 크기 | 차원 | 역할 |
|--------|------------|------|------|
| `delta_embed` | `MAX_DELTA+1 = 257` | `d_model//4 = 64` | delta 필드 임베딩 |
| `lane_embed` | `4` | `d_model//4 = 64` | lane 필드 임베딩 |
| `type_embed` | `EOS_TYPE+1 = 3` | `d_model//4 = 64` | note_type 필드 임베딩 |
| `dur_embed` | `MAX_DURATION+1 = 129` | `d_model//4 = 64` | duration 필드 임베딩 |
| `event_proj` | — | `d_model → d_model` | 4개 임베딩 concat(256) → d_model(256) 투영 |
| `bpm_proj` | — | `1 → d_model` | BPM 스칼라 → 조건화 벡터 |

**BOS 토큰**: `nn.Parameter(torch.zeros(1, 1, d_model))` — 학습 가능한 시작 토큰

**Decoder 관련**:

| 레이어 | 역할 |
|--------|------|
| `decoder` | 4층 TransformerDecoder, 이벤트 시퀀스 자기회귀 생성 |
| `fc_delta` | `d_model → MAX_DELTA+1` (257 클래스 분류) |
| `fc_lane` | `d_model → 4` (4 클래스 분류) |
| `fc_type` | `d_model → EOS_TYPE+1` (3 클래스 분류) |
| `fc_dur` | `d_model → MAX_DURATION+1` (129 클래스 분류) |

TransformerDecoderLayer 설정: `nhead=8`, `dim_feedforward=1024`, `dropout=0.1`

---

#### `encode_spec(audio, bpm=None)` — 오디오 인코딩

```
인자
  audio : (B, T, n_mels) — 멜 스펙트로그램 배치
  bpm   : (B,) — 정규화된 BPM 값 (optional)

반환
  memory : (B, T, d_model) — Encoder 컨텍스트
```

**처리 흐름**:
```
audio (B,T,n_mels)
  → audio_proj        → (B,T,d_model)
  + sinusoidal_pe     → 위치 정보 추가
  → encoder           → memory (B,T,d_model)
  + bpm_proj(bpm)     → BPM 조건화 (bpm이 있을 때만)
```

BPM 조건화: `bpm_proj`로 만든 `(B, d_model)` 벡터를 `(B, 1, d_model)`로 reshape 후 memory 전체에 broadcast 덧셈. 동일 곡 내 모든 위치에 동일한 BPM 신호 주입.

---

#### `embed_events(events)` — 이벤트 임베딩

```
인자
  events : (B, N, 4) — 이벤트 배치 (정수 텐서)

반환
  (B, N, d_model) — 임베딩 벡터
```

**처리 흐름**:
```
events[:,:,0] clamp(0, MAX_DELTA)    → delta_embed → (B,N,64)
events[:,:,1] clamp(0, 3)            → lane_embed  → (B,N,64)
events[:,:,2] clamp(0, EOS_TYPE)     → type_embed  → (B,N,64)
events[:,:,3] clamp(0, MAX_DURATION) → dur_embed   → (B,N,64)
                                        cat → (B,N,256)
                                        event_proj  → (B,N,d_model)
```

clamp를 먼저 적용해 범위 밖의 값이 들어와도 Embedding 인덱스 오류가 발생하지 않음.

---

#### `forward(audio, events, bpm=None)` — Teacher-Forcing 학습용 Forward

```
인자
  audio  : (B, T, n_mels) — 멜 스펙트로그램
  events : (B, N, 4)      — 정답 이벤트 시퀀스 (delta 인코딩)
  bpm    : (B,)           — 정규화 BPM (optional)

반환 (tuple)
  delta_logits : (B, N, MAX_DELTA+1)
  lane_logits  : (B, N, 4)
  type_logits  : (B, N, EOS_TYPE+1)
  dur_logits   : (B, N, MAX_DURATION+1)
```

**처리 흐름**:
```
1. encode_spec(audio, bpm) → memory (B,T,d_model)

2. Decoder 입력 구성 (Teacher Forcing):
   bos       = self.bos.expand(B,-1,-1)      → (B,1,d_model)
   event_emb = embed_events(events)           → (B,N,d_model)
   tgt       = cat([bos, event_emb[:,:-1]])   → (B,N,d_model)
   (위치 i의 출력이 events[i]를 예측하도록 1칸 shift)

3. Causal mask 생성 (상삼각 bool 행렬, diagonal=1):
   미래 토큰을 볼 수 없도록 마스킹

4. decoder(tgt, memory, tgt_mask) → out (B,N,d_model)

5. Output heads:
   fc_delta(out)          → delta_logits
   fc_lane(out)           → lane_logits
   fc_type(out)           → type_logits
   fc_dur(out.detach())   → dur_logits  ← gradient 차단
```

**`out.detach()` 설계 의도**:
- `fc_dur`의 gradient가 decoder까지 역전파되지 않도록 차단
- tap이 전체의 약 85%를 차지하는 데이터 분포에서, decoder는 `loss_delta + loss_lane + loss_type`만으로 학습되어 자연스럽게 tap 편향을 형성
- duration head는 decoder 표현에 의존하지만, decoder 학습에 hold 편향을 주입하지 않음

---

## 3. src/traindemo.py

### 파일 역할

원본 데이터(`source/`)를 학습용 데이터셋(`dataset/`)으로 변환하고, 모델을 학습시켜 `models/` 에 체크포인트를 저장한다.

---

### `build_dataset(root)` — 데이터셋 구축

```
읽는 것:
  source/<song>/spec.pt               (T, n_mels) 멜 스펙트로그램
  source/<song>/<ver>/labels/events.npy  (N, 4) 절대 시간 이벤트
  source/<song>/metadata.json         BPM 정보

쓰는 것:
  dataset/<idx>/spec.pt   (T, n_mels)
  dataset/<idx>/events.pt (N, 4)  절대 time_frame 기준
  dataset/<idx>/bpm.pt    scalar float
```

**처리 상세**:
1. `source/` 하위 모든 곡 디렉터리 순회
2. 곡당 `spec.pt` 필수 확인, 없으면 skip
3. `metadata.json`에서 BPM 읽기 (없으면 120.0 기본값)
4. `events.npy`는 `rglob`으로 하위 디렉터리까지 탐색 (하나의 곡에 여러 버전/난이도 가능)
5. 이벤트 유효성 검사:
   - `ndim != 2` 또는 `shape[1] != 4` 이면 skip
   - `spec` 길이 초과 이벤트 제거 (`events[:, 0] < T`)
   - 5개 미만이면 skip (너무 짧은 샘플 제외)
6. 곡 × 난이도 조합마다 별도 인덱스(`idx`)로 저장

---

### `class EventRhythmDataset(Dataset)` — 학습 데이터셋

```python
__init__(dataset_root, spec_window=512, max_events=256)
```

- `spec_window`: 한 번에 모델에 입력할 스펙트로그램 프레임 수 (512 × 50ms = **25.6초**)
- `max_events`: 한 샘플의 최대 이벤트 수 (패딩 포함)

#### `__getitem__(idx)` — 샘플 반환

```
반환 (tuple)
  spec_chunk     : (spec_window, n_mels)   — 정규화되지 않은 멜 스펙트로그램
  win_events     : (max_events, 4)         — delta 인코딩, EOS+패딩 포함
  valid_mask     : (max_events,) bool      — 실제 이벤트 위치 True
  frames_per_beat: scalar float            — 정규화된 BPM
```

**처리 흐름**:

```
1. spec, events, bpm 로드

2. 랜덤 윈도우 추출:
   start_frame = randint(0, T - spec_window)
   spec_chunk  = spec[start:end]  (부족하면 zero-pad)

3. 이벤트 필터링:
   mask = (events[:,0] >= start_frame) & (events[:,0] < end_frame)
   win_events[:, 0] -= start_frame  (윈도우 시작 기준 상대 시간)

4. delta 변환: _to_delta(win_events)
   절대 시간 → inter-event 간격

5. 필드 clamp:
   delta   : clamp(0, MAX_DELTA)
   lane    : clamp(0, 3)
   note_type: clamp(0, 1)  ← EOS는 여기서 추가하므로 1까지만
   duration: clamp(0, MAX_DURATION)

6. EOS 토큰 append: [0, 0, 2, 0]

7. max_events로 truncate 또는 EOS 패딩

8. valid_mask: [:real_len] = True, [real_len:] = False

9. BPM 정규화: (60000 / bpm / 50) / 20
   → BPM 60  ≈ 1.0, BPM 120 ≈ 0.5, BPM 300 ≈ 0.2
```

---

### `_to_delta(events)` — 절대 시간 → Delta 변환

```
인자  events : (N, 4) 절대 time_frame 기준 이벤트
반환  out    : (N, 4) delta 인코딩 (복사본)
```

```python
out[i, 0] = events[i, 0] - events[i-1, 0]   # i >= 1
out[0, 0] = events[0, 0]                      # 윈도우 시작 → 첫 이벤트 거리
```

---

### `compute_loss(pred, events, valid_mask)` — 손실 함수

```
인자
  pred       : (delta_logits, lane_logits, type_logits, dur_logits)
  events     : (B, N, 4) 정답
  valid_mask : (B, N) bool

반환
  scalar loss
```

**동작**:
1. `valid_mask`로 padding 위치를 제외한 1D 인덱스 생성
2. 각 필드별 Cross-Entropy 손실 계산
3. `loss_dur`는 **hold 이벤트(`note_type == 1`)에서만** 계산
   - hold가 없는 배치면 `loss_dur = 0.0` (dummy tensor로 gradient 단절 방지)

**최종 손실**:
```
loss = loss_delta + loss_lane + loss_type + 0.5 × loss_dur
```

duration 가중치를 0.5로 낮게 설정한 이유: hold가 드문 상황에서 delta/lane/type 학습이 지배적이어야 함.

---

### `train(root, epochs=300, batch_size=32)` — 학습 루프

**DataLoader 설정**:
```python
num_workers=8        # 병렬 데이터 로딩
pin_memory=True      # CPU→GPU 전송 비동기 가속
persistent_workers=True  # 워커 프로세스 재사용
```

**학습 설정**:
- Optimizer: Adam, `lr=1e-4`
- Mixed Precision: `torch.amp.autocast + GradScaler` (CUDA 전용)

**체크포인트 저장 전략**:
```
매 에폭:
  latest.pt.tmp → latest.pt  (원자적 교체 — 중간 손상 방지)
  model state + optimizer state + epoch 저장

손실 개선 시:
  best.pt.tmp → best.pt
  model state만 저장 (추론용)
```

**TensorBoard**: `runs/` 디렉터리에 `Loss/train` 스칼라 기록

---

## 4. src/runDemo.py

### 파일 역할

학습된 `best.pt` 모델을 로드해, 전체 곡 스펙트로그램으로부터 이벤트 시퀀스를 자기회귀적으로 생성한다. 생성 후 후처리(중복 제거, hold 병합)를 거쳐 결과 파일로 저장한다.

---

### `_sample(logits, temperature, top_k=0)` — 토큰 샘플링

```
인자
  logits      : (num_classes,) 원시 logit 벡터
  temperature : 샘플링 온도
  top_k       : 상위 k개만 유지 (0이면 비활성)

반환
  int — 샘플링된 클래스 인덱스
```

| temperature | 동작 |
|-------------|------|
| `<= 0` | argmax (완전 결정론적) |
| `> 0` | softmax + multinomial 샘플링 |

`top_k > 0`이면 상위 k개 미만의 logit을 `-inf`로 마스킹 후 샘플링 → 희귀 토큰 샘플링 방지.

---

### `generate_window_events(model, spec_chunk, ...)` — 단일 윈도우 생성

```
인자
  model           : 학습된 EventChartTransformer
  spec_chunk      : (T, n_mels) 단일 윈도우 스펙트로그램
  max_events      : 최대 생성 이벤트 수 (기본 256)
  temperature     : 샘플링 온도 (기본 1.0)
  bpm             : (1,) 정규화 BPM (optional)
  prime_event     : 이전 윈도우의 마지막 이벤트 [delta, lane, type, dur]
  eos_threshold   : EOS 억제 윈도우 비율 (기본 0.8)
  active_holds    : {lane: end_abs_frame} 진행 중인 hold 상태
  hold_bias       : hold logit 감쇠 값 (기본 2.0)
  top_k_delta     : delta 샘플링 시 top-k (기본 0)
  min_hold_dur    : hold 최소 지속 시간 (기본 5)

반환
  List[List[int]] — [[delta, lane, note_type, duration], ...]
```

**자기회귀 생성 루프**:
```
초기:
  memory = encode_spec(audio, bpm)         # 한 번만 계산
  tgt    = [BOS]                           # (1, 1, d_model)
  prime_event가 있으면 tgt에 추가

매 스텝:
  1. causal_mask 생성 (tgt 현재 길이 기준)
  2. decoder(tgt, memory) → out
  3. out의 마지막 위치에서 4개 head 호출 → logits
  4. delta 샘플링 (consecutive_delta0 >= 3이면 delta=0 차단)
  5. lane 샘플링
  6. type_logits[1] -= hold_bias  (hold 억제)
     type_logits[EOS_TYPE] = -inf (current_frame < T*eos_threshold이면)
     active_holds에 해당 레인이 있으면 type_logits[1] = -inf
  7. note_type 샘플링
  8. note_type==1(hold)이면 dur_logits[:min_hold_dur] = -inf 후 샘플링
  9. note_type == EOS_TYPE → break
  10. current_frame += delta
      current_frame >= T → break
  11. 생성된 이벤트를 tgt에 concat
```

**제어 로직 상세**:

| 제어 | 조건 | 동작 | 목적 |
|------|------|------|------|
| `consecutive_delta0` | `>= 3` | `delta_logits[0] = -inf` | 4-key 기준 한 프레임에 최대 4개 노트 제한 |
| `eos_threshold` | `current_frame < T * 0.8` | `type_logits[EOS] = -inf` | 윈도우 80% 이전 조기 종료 방지 |
| `hold_bias` | 항상 | `type_logits[1] -= 2.0` | 훈련/추론 분포 차이로 인한 hold 과생성 보정 |
| `active_holds` | 해당 레인 진행 중 | `type_logits[1] = -inf` | 윈도우 경계에서 중복 hold 방지 |
| `min_hold_dur` | `note_type == 1` | `dur_logits[:5] = -inf` | 너무 짧은 hold 생성 방지 |

**저장 시 `save_dur` 처리**:
- 모델 피드백(`tgt` 확장)에는 샘플링된 원본 `dur` 사용 (모델이 자신의 hold 길이를 기억)
- 저장 시에는 tap/EOS의 `dur = 0` 강제 (데이터 일관성)

---

### `generate_full_chart(model, spec, window, stride, ...)` — 전체 곡 생성

```
인자
  model    : 학습된 EventChartTransformer
  spec     : (T, n_mels) 전체 곡 스펙트로그램
  window   : 인코더 입력 윈도우 크기 (기본 512)
  stride   : 윈도우 이동 간격 (None이면 window와 동일)
  ...      : generate_window_events와 동일한 생성 파라미터

반환
  List[ChartEvent] — 절대 time_frame 기준, 시간순 정렬
```

**Overlapping Window 전략**:

```
stride < window 일 때:

  window=512, stride=256 예시:
  ┌──────── window ────────┐
  [0 ........... 511]         → [0, 256) 구간만 수집
          [256 ........ 767]   → [256, 512) 구간만 수집
                  [512 ...] → ...

  각 윈도우는 512프레임 전체를 인코더로 보지만
  앞쪽 stride 프레임의 이벤트만 수집 → 윈도우 경계 부근 품질 향상
```

**크로스 윈도우 상태 전달**:
- `prime_event`: 이전 윈도우의 마지막 수집 이벤트를 다음 윈도우 BOS 직후에 삽입 → 시간 연속성 유지
- `active_holds`: 현재 진행 중인 hold `{lane: end_abs_frame}` 딕셔너리를 윈도우 간 전달

---

### `fix_hold_overlaps(events)` — Hold 중복 병합

```
인자  events : List[ChartEvent]
반환  List[ChartEvent] — 겹치는 hold가 병합된 이벤트 리스트
```

**알고리즘**:
1. tap과 hold를 분리
2. 레인별로 hold를 시간순 정렬
3. 겹치는 hold (`ev.time_frame < cur_end`) → `cur_end = max(cur_end, ev_end)` 확장
4. 겹치지 않으면 이전 구간 확정 후 새 구간 시작
5. tap + 병합된 hold를 시간순 재정렬

---

### `dedup_events(events)` — 중복 이벤트 제거

```
인자  events : List[ChartEvent]
반환  List[ChartEvent] — (time_frame, lane) 기준 중복 제거
```

**알고리즘**:
- 동일 `(time_frame, lane)`에 여러 이벤트가 있을 경우:
  - hold가 있으면 hold 우선 유지
  - 모두 tap이면 첫 번째만 유지
- `delta=0` collapse로 같은 프레임에 동일 레인 노트가 수십 개 쌓이는 현상 방지

---

### `__main__` — 실행 엔트리포인트

```
실행 순서:
1. best.pt 로드
2. source/ 첫 번째 spec.pt + metadata.json(BPM) 로드
3. generate_full_chart()  → 원본 이벤트 생성
4. dedup_events()         → 중복 제거
5. fix_hold_overlaps()    → hold 병합
6. output/<song>-generated.txt        저장 (프레임 행렬, numpy savetxt)
7. output/<song>-generated_events.txt 저장 (human-readable TSV)
```

**출력 파일**:
- `-generated.txt`: `events_to_frame_labels()`로 변환된 `(T, 4)` 행렬 → 시각화/재생용
- `-generated_events.txt`: TSV 형식 `time_frame / lane / type / duration_frames` → 사람이 직접 확인 가능

---

## 5. 데이터 흐름 요약

```
[원본 데이터]
  source/<song>/spec.pt          (T, n_mels)
  source/<song>/**/events.npy    (N, 4)  절대 time_frame
  source/<song>/metadata.json    BPM
        ↓
  build_dataset()
        ↓
[정제된 데이터셋]
  dataset/<idx>/spec.pt
  dataset/<idx>/events.pt
  dataset/<idx>/bpm.pt
        ↓
  EventRhythmDataset.__getitem__()
    - 랜덤 윈도우 추출
    - delta 변환
    - EOS 추가 + 패딩
        ↓
  train() — Teacher Forcing
    model.forward(spec, events, bpm)
    compute_loss(pred, events, valid_mask)
    Adam optimizer + Mixed Precision
        ↓
[모델 체크포인트]
  models/best.pt
  models/latest.pt
        ↓
  generate_full_chart()
    - Overlapping window 슬라이딩
    - generate_window_events() 자기회귀 생성
        ↓
  dedup_events() → fix_hold_overlaps()
        ↓
[출력 결과]
  output/<song>-generated.txt
  output/<song>-generated_events.txt
```

---

## 6. 설계 결정 및 트레이드오프

### `fc_dur`에 `out.detach()` 적용

- **결정**: duration head에 전달하는 decoder 출력에 `.detach()` 적용
- **이유**: tap(~85%)이 압도적으로 많은 데이터에서 decoder가 `loss_dur` 신호를 받으면 hold 방향으로 편향될 수 있음. decoder는 delta/lane/type loss만으로 학습되어 데이터 자연 분포(tap 편향)를 그대로 반영하도록 설계.
- **트레이드오프**: duration 예측 정확도가 decoder 표현 품질에 단방향으로만 의존. duration 예측이 나빠도 decoder는 영향받지 않음.

### Overlapping Window (`stride < window`)

- **결정**: 추론 시 stride=256, window=512로 50% 중첩
- **이유**: 윈도우 경계 근처의 이벤트는 인코더가 양쪽 컨텍스트를 충분히 보지 못해 품질이 낮아질 수 있음. 각 윈도우가 앞쪽 절반만 수집하게 함으로써 항상 윈도우 중앙부의 이벤트만 실제로 사용.
- **트레이드오프**: 추론 연산량 2배 증가.

### `hold_bias` (추론 시 hold logit 감쇠)

- **결정**: 추론 시 `type_logits[1] -= 2.0` 하드코딩
- **이유**: 학습 데이터 분포와 추론 시의 auto-regressive 분포 불일치(exposure bias)로 hold가 과생성되는 경향 보정.
- **트레이드오프**: 하이퍼파라미터를 직접 조정해야 하며, 곡이나 모델 버전에 따라 적절한 값이 달라질 수 있음.

### `consecutive_delta0` 제한

- **결정**: 같은 프레임에 delta=0 이벤트가 3회 이상 연속되면 차단
- **이유**: 4-key 게임에서 한 프레임에 최대 4개(레인 0~3)의 노트만 존재 가능. 그 이상은 물리적으로 불가능하므로 runaway loop 방지.
- **트레이드오프**: 이론상 4개를 정확히 생성한 직후 다음 위치는 막히지 않음 (카운터가 reset되므로). 논리적으로 완벽하진 않지만 실용적으로 충분.

### 원자적 체크포인트 저장

- **결정**: `*.pt.tmp`로 먼저 쓰고 `os.replace()`로 교체
- **이유**: 학습 도중 프로세스 중단 시 체크포인트 파일이 절반만 기록되어 손상되는 상황 방지. `os.replace()`는 POSIX 시스템에서 atomic 연산.
- **트레이드오프**: Windows에서도 동작하나, 크로스 드라이브 이동은 atomic 보장 안 됨 (같은 드라이브 내에서는 보장).

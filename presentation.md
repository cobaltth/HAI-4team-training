# AI 기반 리듬 게임 채보 자동 생성
## Transformer를 이용한 Event-based Chart Generation

**HAI 2026-1 Team 4**

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [배경 및 동기](#2-배경-및-동기)
3. [데이터 파이프라인](#3-데이터-파이프라인)
4. [모델 아키텍처](#4-모델-아키텍처)
5. [학습 방법](#5-학습-방법)
6. [채보 생성 (추론)](#6-채보-생성-추론)
7. [결과 및 한계](#7-결과-및-한계)
8. [결론](#8-결론)

---

## 1. 프로젝트 개요

### 무엇을 만들었나?

**MP3 음악 파일을 입력받아 4-key 리듬 게임의 채보를 자동으로 생성하는 AI 시스템**

```
[MP3 파일] ──► [AI 모델] ──► [.osu 채보 파일]
```

- 입력: 음악 파일 (MP3)
- 출력: osu!mania 형식의 4-key 채보

### 핵심 특징

| 특징 | 설명 |
|---|---|
| 이벤트 기반 표현 | 프레임별 레이블이 아닌 노트 이벤트 시퀀스로 표현 |
| Encoder-Decoder | 음악 → 컨텍스트 인코딩 후 자기회귀적 채보 생성 |
| 다중 노트 타입 | Tap / Hold / EOS 지원 |
| BPM 조건화 | BPM 정보를 모델에 직접 주입해 박자 인식 향상 |

---

## 2. 배경 및 동기

### 리듬 게임과 채보

리듬 게임(osu!mania, DJMAX, 비트매니아)의 채보는 수작업으로 제작됨

- 숙련된 채보 제작자가 곡 1개당 수 시간 소요
- 인기 없는 곡은 채보가 존재하지 않음
- 제작자의 주관이 강하게 반영되어 품질 편차가 큼

→ **AI가 음악을 듣고 자동으로 채보를 생성**할 수 있다면?

### 기존 방법의 한계

| 방법 | 한계 |
|---|---|
| 규칙 기반 (BPM 분할) | 음악의 뉘앙스를 반영하지 못함 |
| Frame-level 분류 | 노트 간 관계(패턴)를 모델링하지 못함 |
| CNN 기반 온셋 검출 | 레인 배치, hold 길이 등 채보 구조 생성 불가 |

→ **시퀀스-투-시퀀스 모델**로 채보를 하나의 언어처럼 생성

---

## 3. 데이터 파이프라인

### 3-1. 전체 흐름

```
rawsource/          source/              dataset/
(osu 비트맵) ──►  (spec.pt)    ──►   (spec.pt + events.pt)
(MP3, .osu)       (멜 스펙)          (윈도우 분할 완료)
```

### 3-2. 멜 스펙트로그램 변환

음악의 주파수 특성을 시간×주파수 행렬로 변환

```
MP3  ──►  파형  ──►  STFT  ──►  멜 필터뱅크  ──►  log 변환  ──►  정규화
                    (n_fft=1024)  (n_mels=128)
```

**주요 파라미터:**

| 파라미터 | 값 | 의미 |
|---|---|---|
| `hop_length` | sample_rate × 0.05 | 프레임 간격 = **50ms** |
| `n_mels` | 128 | 멜 주파수 빈 수 |
| `n_fft` | 1024 | FFT 윈도우 크기 |
| `f_min` | 27.5 Hz | 피아노 최저음 A0 (서브베이스 제외) |
| 출력 | (T, 128) | T = 곡 길이 / 50ms |

**고주파 강조 (High-Frequency Emphasis):**

멜 스케일은 저주파에 더 많은 bin을 할당해 고주파(하이햇·스네어)가 상대적으로 약하게 표현됨.
모델 입력 직전에 선형 가중치를 곱해 보정:

```python
freq_weight = torch.linspace(1.0, 2.0, n_mels)  # bin 0 → 1.0x, bin 127 → 2.0x
audio = audio * freq_weight
```

### 3-3. 이벤트 포맷

채보를 **4개 필드의 이벤트 시퀀스**로 표현:

```
[delta_frames, lane, note_type, duration_frames]
    ↑              ↑       ↑           ↑
  이전 노트와    레인    tap/hold/   hold 지속
  의 시간 간격   (0~3)    EOS         시간
  (0~256 프레임)         (0/1/2)   (0~128 프레임)
```

**예시 이벤트 시퀀스:**

```
[8, 0, tap, 0]   → 8프레임(400ms) 후 레인0에 탭
[0, 2, tap, 0]   → 같은 프레임 레인2에 탭 (코드)
[4, 1, hold, 12] → 4프레임 후 레인1에 12프레임짜리 홀드
[8, 3, tap, 0]   → 8프레임 후 레인3에 탭
[0, 0, EOS, 0]   → 시퀀스 종료
```

**Delta 인코딩의 장점:**
- 절대 시간 대신 상대 간격 → 음악 패턴의 반복 구조 학습에 유리
- 12.8초(256×50ms) 이상 간격은 클램핑 처리

---

## 4. 모델 아키텍처

### 4-1. 전체 구조

```
                    ┌─────────────────────────────────────────┐
                    │         EventChartTransformer            │
                    │                                          │
  (B, T, 128)       │  ┌──────────┐      ┌──────────────────┐ │
  멜 스펙트로그램 ──►│  │  Encoder  │─────►│     Decoder      │ │
                    │  │ (6 layer) │      │    (6 layer)     │ │
                    │  └──────────┘      └────────┬─────────┘ │
                    │                             │           │
                    │                    ┌────────▼─────────┐ │
                    │                    │   FC Output Heads │ │
                    │                    │  fc_delta (257)   │ │
                    │                    │  fc_lane  (4)     │ │
                    │                    │  fc_type  (3)     │ │
                    │                    │  fc_dur   (129)   │ │
                    │                    └───────────────────┘ │
                    └─────────────────────────────────────────┘
```

**모델 규모:**
- d_model = 512
- nhead = 8
- num_layers = 6 (인코더·디코더 각각)
- dim_feedforward = 2048
- 파라미터 수 ≈ 약 **86M**

### 4-2. 인코더 (음악 → 컨텍스트)

```python
audio_proj  = Linear(128, 512)    # 멜 빈 → d_model 투영
encoder     = TransformerEncoder  # 6-layer, self-attention
bpm_proj    = Linear(1, 512)      # BPM 조건화 벡터
```

1. 고주파 강조 가중치 적용
2. `audio_proj`로 128 → 512 차원 투영
3. Sinusoidal Positional Encoding 추가
4. 6-layer Transformer Encoder
5. BPM 임베딩을 모든 시간 스텝에 더해 박자 정보 주입

### 4-3. 이벤트 임베딩

각 이벤트 필드를 독립적으로 임베딩 후 연결:

```
delta_embed  (257, 128)  ─┐
lane_embed   (4,   128)  ─┤ concat → (512,) → event_proj → (512,)
type_embed   (3,   128)  ─┤
dur_embed    (129, 128)  ─┘
```

### 4-4. 디코더 (자기회귀 채보 생성)

```
[BOS] [event₀] [event₁] ... [eventₙ₋₁]
  ↓      ↓        ↓              ↓
 pred₀  pred₁   pred₂    ...  predₙ   ← Teacher Forcing (학습 시)
```

- Causal Mask로 미래 토큰 참조 차단
- Cross-Attention으로 음악 컨텍스트(Memory) 참조
- `fc_dur`는 디코더 출력을 **detach**해 사용
  → hold 지속시간 gradient가 인코더/디코더 역전파에 영향을 주지 않도록 차단

---

## 5. 학습 방법

### 5-1. Teacher Forcing

학습 시 모델의 이전 예측이 아닌 **정답 이벤트를 디코더 입력으로 사용**:

```
학습: [BOS, 정답₀, 정답₁, ...] → 각 위치에서 다음 이벤트 예측
추론: [BOS, 예측₀, 예측₁, ...] → 자기 예측을 다음 입력으로 사용
```

### 5-2. 손실 함수

4개 필드를 각각 Cross-Entropy로 학습:

```
L = L_delta + L_lane + L_type + 0.5×L_dur + 0.5×L_lane_reg
```

| 손실 항 | 역할 |
|---|---|
| `L_delta` | 노트 간격 예측 정확도 |
| `L_lane` | 레인 선택 정확도 |
| `L_type` | Tap/Hold/EOS 분류 정확도 |
| `L_dur` | Hold 지속시간 정확도 (Hold 이벤트에만 적용) |
| `L_lane_reg` | 레인 편향 방지 — 4개 레인이 균등하게 사용되도록 정규화 |

**Lane Regularization:**

```python
# 평균 레인 분포가 uniform [0.25, 0.25, 0.25, 0.25]에 가까워지도록
loss_lane_reg = KL(mean_lane_probs ‖ Uniform(4))
```

### 5-3. 학습 설정

| 항목 | 값 |
|---|---|
| Optimizer | Adam (lr = 1e-4) |
| Batch Size | 64 |
| Epochs | 200 |
| Mixed Precision | FP16 (GradScaler) |
| DataLoader | num_workers=8, pin_memory=True |
| 체크포인트 | `latest.pt` (매 에폭), `best.pt` (최저 loss) |

**Mixed Precision 학습의 효과:**
- GPU 메모리 사용량 약 절반
- 행렬 연산 속도 2~3배 향상 (Tensor Core 활용)

### 5-4. 데이터 증강

매 에폭마다 곡의 무작위 512프레임(25.6초) 구간을 샘플링:

```
곡 전체(T 프레임) ──► 무작위 start ──► spec[start:start+512]
                                     events[start:start+512]
```

→ 동일 데이터셋으로 다양한 패턴 학습 (사실상 무한 증강)

---

## 6. 채보 생성 (추론)

### 6-1. Sliding Window 생성

긴 곡을 512프레임 단위 윈도우로 처리:

```
         window (512 프레임 = 25.6초)
         ┌─────────────────────────┐
         │  인코더 입력 범위        │
    ─────┤                         ├─────
         │◄─── stride=256 ────►│
              수집 범위 (겹침)
```

- 윈도우가 겹쳐서 경계 부분 노트 누락 방지
- 이전 윈도우의 마지막 이벤트를 다음 윈도우에 `prime_event`로 전달

### 6-2. 자기회귀 생성 루프

```
memory = encode_spec(spec_chunk)

[BOS] → delta₀, lane₀, type₀, dur₀
      → [BOS, ev₀] → delta₁, lane₁, type₁, dur₁
      → [BOS, ev₀, ev₁] → ...
```

**샘플링 방식:**
- `temperature > 0`: Softmax 확률 분포에서 샘플링 (다양성)
- `top_k_delta = 20`: 상위 20개 delta 후보만 허용 (품질 유지)
- `temperature = 0`: Argmax (결정론적)

### 6-3. 제약 조건 (Logit Masking)

생성 중 물리적으로 불가능하거나 바람직하지 않은 이벤트를 확률 0으로 차단:

```
┌─────────────────────────────────────────────────────────────┐
│ Delta 제약                                                    │
│  - delta=0 연속 3회 이상 → delta=0 차단 (4-key 최대 코드 수) │
│  - cross_lane_gap=2 → delta=1 차단 (레인 간 최소 간격)       │
│  - snap_bpm → BPM 격자 외 delta 차단 (박자 정렬)            │
├─────────────────────────────────────────────────────────────┤
│ Lane 제약                                                     │
│  - min_gap_per_lane=3 → 같은 레인 150ms 이내 재사용 차단    │
│  - hold_end_cooldown=2 → Hold 종료 직후 쿨다운              │
│  - lane_bias → 특정 레인 60% 이상 사용 시 해당 레인 억제    │
├─────────────────────────────────────────────────────────────┤
│ Type 제약                                                     │
│  - hold_bias=1.3 → Hold 확률 하향 조정 (데이터 불균형 보정) │
│  - max_active_holds=2 → 동시 Hold 2개 이하로 제한           │
│  - EOS 억제 → 윈도우 80% 이전에는 EOS 생성 차단             │
└─────────────────────────────────────────────────────────────┘
```

**Hold Bias의 수학적 근거:**

학습 데이터에서 Tap : Hold ≈ 85% : 15% → 모델이 tap에 강하게 편향됨.
추론 시 Hold logit에서 1.3을 빼면:

```
exp(-1.3) / exp(0) ≈ 0.27  →  Hold 확률을 약 4배 하향
```

이를 통해 과도한 Hold 생성 없이 자연스러운 비율 달성.

### 6-4. 후처리 파이프라인

```
생성된 이벤트
    │
    ▼ dedup_events()        — 동일 (프레임, 레인) 중복 제거
    │
    ▼ fix_hold_overlaps()   — 같은 레인 겹치는 Hold 병합
    │
    ▼ fix_tap_in_hold()     — Hold 구간 내 Tap 제거
    │
    ▼ enforce_min_gap()     — 같은 레인 150ms 미만 노트 제거
    │
    ▼ events_txt_to_osu()   — osu!mania .osu 파일 변환
```

---

## 7. 결과 및 한계

### 7-1. 학습 결과

- 데이터셋: 약 15,000 샘플 (곡별 다수 버전 포함)
- 에폭당 윈도우: 512프레임 × 랜덤 샘플링 → 사실상 무한 증강
- Loss 수렴: 초기 ~6.0 → 수렴 시 ~2.0 수준

### 7-2. 생성 품질 분석

**잘 되는 부분:**
- 전반적인 박자 구조 포착 (4/4박자 패턴)
- BPM에 맞는 노트 간격
- 4개 레인의 균등한 사용 (lane regularization 효과)

**개선이 필요한 부분:**
- 고밀도 구간에서 리듬 이탈 발생
- Hold 노트 비율이 여전히 낮은 경향
- 복잡한 트릴·롤 패턴 표현 한계

### 7-3. 아키텍처 설계 시 주요 결정

| 결정 사항 | 선택 | 이유 |
|---|---|---|
| 표현 방식 | Event (delta 인코딩) | 패턴 반복 학습, 시퀀스 길이 대폭 단축 |
| 모델 구조 | Encoder-Decoder Transformer | 음악(인코더)과 채보(디코더) 역할 분리 |
| fc_dur detach | 적용 | Hold 길이 gradient가 타이밍 학습 방해 방지 |
| Lane Regularization | KL Divergence | Softmax CE만으로는 레인 편향 해소 불충분 |
| BPM 조건화 | Memory에 더하기 | 전체 컨텍스트에 박자 정보 일관되게 반영 |

---

## 8. 결론

### 요약

- Encoder-Decoder Transformer로 음악 → 채보 생성 파이프라인 구현
- 이벤트 기반 표현으로 채보의 순서·타이밍·패턴을 통합 모델링
- 다중 제약 조건(logit masking)으로 게임 규칙 준수 보장
- 고주파 강조, BPM 조건화, Lane Regularization 등 도메인 지식 적용

### 향후 발전 방향

1. **난이도 조건화**: Easy/Normal/Hard 난이도를 조건 벡터로 주입
2. **리듬 트랜지언트 검출**: 온셋 검출기를 전처리로 추가해 박자 정확도 향상
3. **인간 평가**: 실제 플레이어가 채보를 직접 플레이해 품질 측정
4. **더 큰 모델**: d_model=1024, num_layers=12로 스케일업

---

## 부록: 시스템 실행 방법

### 학습

```bash
# 1. rawsource/ 에 osu 비트맵 폴더 배치
# 2. source/ 에 멜 스펙트로그램 생성
python src/mp3toSpec.py

# 3. 학습 시작 (dataset/ 없으면 자동 빌드)
python src/traindemo.py
```

### 채보 생성

```bash
# 1. input/ 에 MP3 파일 배치
# 2. 옵션: input/곡이름.json 에 {"BPM": 170} 형식으로 BPM 지정
# 3. 생성 실행
python src/runDemo.py
# output/ 폴더에 .osu 파일 생성
```

### TensorBoard 모니터링

```bash
tensorboard --logdir runs/
```

# 4-Key 리듬게임 채보 생성 모델 — 분석 보고서 (최신)

> 분석 대상 파일: `mp3toSpec.py`, `traindemo.py`, `runDemo.py`, `parser.py`, `model.py`
> 이전 분석: `ANALYSIS_OLD.md`

---

## 현재 구조 요약

```
mp3/ogg/wav  → [mp3toSpec.py]  → source/<Artist - Title>/spec.pt (T, 128)
.osu         → [parser.py]     → source/<Artist - Title>/<Ver>/labels/events.npy (N, 4)
source/      → [traindemo.py]  → dataset/ → models/best.pt
spec.pt + best.pt → [runDemo.py] → output/<곡이름>-generated_events.txt
                                           <곡이름>-generated.txt
output/*-generated_events.txt → [toOsu.py] → OSUoutput/<곡이름>-generated.osu
```

### 채보 표현 방식 (현재 — event-based)
- 각 이벤트: `(delta_frames, lane, note_type, duration_frames)`
  - `delta_frames` : 이전 이벤트로부터의 프레임 간격 (0~256)
  - `lane` : 0~3
  - `note_type` : 0=tap, 1=hold, 2=EOS
  - `duration_frames` : hold 길이 (0~128), tap은 0
- 특수 토큰: BOS(learnable parameter), EOS(note_type=2)

### Mel-Spectrogram 파라미터 (mp3toSpec.py)
- `hop_length = int(sample_rate * 0.05)` → **프레임 1개 = 50ms**
- `n_mels = 128`, `n_fft = 1024`
- per-sample z-score 정규화 (log_mel 기준)
- 출력 shape: `(T, 128)`

### 모델 구조 (model.py — EventChartTransformer)
- **Encoder**: `Linear(128→256)` + Sinusoidal PE + `TransformerEncoder(4층, nhead=8)`
- **BPM conditioning**: `Linear(1→256)` → memory에 broadcast 합산
- **Event embedding**: delta/lane/type/dur 각각 `d_model//4` 임베딩 → concat → `Linear(256→256)`
- **Decoder**: `TransformerDecoder(4층, nhead=8)` + causal mask, teacher forcing
- **Output heads**: `fc_delta`, `fc_lane`, `fc_type`, `fc_dur` (각각 Linear)

---

## ANALYSIS_OLD.md 대비 해결된 문제

| 구 문제 | 해결 여부 | 비고 |
|---|---|---|
| 문제 1: Spec ↔ 채보 시간축 불일치 | ✅ 해결 | event-based 전환, time_frame = ms/50 |
| 문제 2: `note_emb.sum(dim=2)` 레인 소실 | ✅ 해결 | 4필드 별도 임베딩 후 concat |
| 문제 3: 롱노트 보정 로직 오류 | ✅ 해결 | event-based로 구조적 소멸, fix_hold_overlaps로 후처리 |
| 문제 4: pos_embed 2048 한계 | ✅ 해결 | Sinusoidal PE 교체 (`sinusoidal_pe()`) |
| 문제 5: Encoder + causal mask 모순 | ✅ 해결 | Encoder-Decoder 구조로 전환 |
| 문제 7: 극심한 데이터 불균형 | ✅ 해결 | event-based 전환으로 "전부 0" 문제 소멸 |

---

## 현재 남아있는 문제

### 🔴 치명

---

#### 문제 A: Hold 과다 생성 (학습 손실 비대칭)

**파일:** `traindemo.py` — `compute_loss()`

**원인:**
```python
return loss_delta + loss_lane + loss_type + 0.5 * loss_dur
```
- hold 예측 시 → `loss_type` + `0.5 * loss_dur` (1.5배 gradient)
- tap 예측 시 → `loss_type` 만 (1배 gradient)
- `loss_type`에 클래스 가중치 없음 → 학습 데이터가 tap ~85%임에도 미반영

**현재 임시 대응:** `runDemo.py`의 `fix_hold_overlaps()`로 겹친 hold 병합 (근본 해결 아님)

**근본 해결방안:**
```python
# 1) Type loss 클래스 가중치
type_weight = torch.tensor([0.3, 1.7, 0.5], device=type_logits.device)
loss_type = F.cross_entropy(..., weight=type_weight)

# 2) Duration loss 가중치 축소
return loss_delta + loss_lane + loss_type + 0.1 * loss_dur
```

---

### 🟠 중요

---

#### 문제 B: 생성 시 O(N²) 연산 — KV-cache 미구현

**파일:** `runDemo.py` — `generate_window_events()`

**원인:**
```python
for _ in range(max_events):
    out = model.decoder(tgt, memory, tgt_mask=causal_mask)  # tgt가 매 스텝 1씩 증가
```
- max_events=256 기준 약 32,768번의 attention 연산
- event-based 전환으로 frame-based보다 크게 개선됐으나 여전히 비효율

**해결 방향:** PyTorch KV-cache 구현 또는 `torch.compile()` 적용

---

#### 문제 C: CNN front-end 없음

**파일:** `model.py` — `encode_spec()`

**원인:**
```python
x = self.audio_proj(audio)  # Linear(128, 256) 한 층
```
- 음악의 지역적 패턴(onset, beat, transient)을 사전 추출 없이 Transformer에 직접 전달
- 시퀀스 길이 T가 커서 Encoder self-attention 비용 높음

**해결 방향:**
```python
self.cnn = nn.Sequential(
    nn.Conv1d(n_mels, 128, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv1d(128, d_model, kernel_size=3, padding=1), nn.ReLU(),
)
# forward: audio (B,T,n_mels) → (B,n_mels,T) → cnn → (B,T,d_model)
```

---

#### 문제 D: 난이도/스타일 컨트롤 없음

**파일:** `model.py`, `traindemo.py`

- BPM conditioning은 구현됨 (`bpm_proj`)
- 난이도 레벨 conditioning 없음 → 동일 모델이 Easy/Hard 모두 같은 밀도로 생성

**해결 방향:**
```python
self.difficulty_embed = nn.Embedding(10, d_model)
# forward: memory += difficulty_embed(diff).unsqueeze(1)
```

---

#### 문제 E: window 경계의 hold 단절 (부분 해결)

**파일:** `runDemo.py` — `generate_full_chart()`

- `prime_event`로 이전 window의 마지막 이벤트를 주입하는 구조는 있음
- 그러나 window 경계를 **걸쳐서 진행 중인 hold**(이전 window에서 시작해 현재 window로 넘어오는)의 상태는 전달 안 됨
- 경계에서 hold가 갑자기 끊기거나 중복 시작될 수 있음

---

### 🟡 관찰 / 개선 고려

---

#### 관찰 1: tap의 duration 피드백 분리

**파일:** `runDemo.py:99-105`

```python
save_dur = 0 if note_type != 1 else dur
generated.append([delta, lane, note_type, save_dur])
# 피드백에는 원본 dur 사용
ev = torch.tensor([[[delta, lane, note_type, dur]]], ...)
```
- tap 저장 시 `dur=0` 강제 → 훈련 데이터와 일치
- 피드백은 원본 `dur` 유지 → 모델 컨텍스트 왜곡 방지
- **의도적 설계로 올바른 처리**

---

#### 관찰 2: Sliding window EOS suppression

**파일:** `runDemo.py:82-84`

```python
if current_frame < T * eos_threshold:
    type_logits[EOS_TYPE] = float('-inf')
```
- window의 80%가 지나기 전에는 EOS 생성 억제
- 너무 이른 종료 방지에는 효과적이나, 마지막 20% 구간에서 EOS가 나오지 않으면 window 끝에서 강제 절단되므로 마지막 구간 노트 밀도가 낮아질 수 있음

---

## 수정 우선순위 로드맵

### Phase 1 — 즉시 수정 (학습 품질)
1. **문제 A-1**: `loss_type`에 클래스 가중치 추가 (`traindemo.py`)
2. **문제 A-2**: `loss_dur` 가중치 0.5 → 0.1 축소 (`traindemo.py`)
3. 재학습 후 hold/tap 비율 검증

### Phase 2 — 성능 개선
4. **문제 C**: CNN front-end 추가 (`model.py`)
5. **문제 B**: KV-cache 구현 (`runDemo.py`)

### Phase 3 — 기능 확장
6. **문제 D**: 난이도 conditioning 추가
7. **문제 E**: window 경계 hold 상태 전달 개선

---

## 참고: 파이프라인 전체 파일 목록

| 파일 | 역할 |
|---|---|
| `src/mp3toSpec.py` | 오디오 → spec.pt 변환 |
| `src/parser.py` | .osu 파싱, ChartEvent, events_to_frame_labels |
| `src/model.py` | EventChartTransformer 정의 |
| `src/traindemo.py` | 데이터셋 빌드, 학습 루프 |
| `src/runDemo.py` | 추론, hold 겹침 후처리, output 저장 |
| `src/toOsu.py` | generated_events.txt → .osu 변환 |

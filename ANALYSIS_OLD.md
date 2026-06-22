# 4-Key 리듬게임 채보 생성 모델 — 분석 보고서

> 분석 대상 파일: `mp3toSpec.py`, `traindemo.py`, `runDemo.py`, `parser.py`    
> 이 문서를 Claude Code에서 참조하여 수정 작업을 진행할 것.

---

## 현재 구조 요약

```
mp3 → [mp3toSpec.py] → spec.pt (T, 128)
.npy 채보 + spec.pt → [traindemo.py] → 트랜스포머 학습
spec.pt + best.pt → [runDemo.py] → generated.pt
```

### 채보 표현 방식 (현재)
- shape: `(T, 4)` — T개 프레임 × 4개 레인
- 각 셀 값: `0`=빈칸, `1`=일반노트, `2`=롱노트 시작, `3`=롱노트 유지, `4`=롱노트 끝
- 롱노트 상태전이 규칙: `2 → 3 → 3 → ... → 4`

### Mel-Spectrogram 파라미터 (mp3toSpec.py)
- `hop_length = int(sample_rate * 0.05)` → **프레임 1개 = 50ms**
- `n_mels = 128`
- `n_fft = 1024`
- per-sample z-score 정규화 적용됨 (log_mel 기준)
- 출력 shape: `(T, 128)`

---

## 전체 문제점 목록

### 🔴 치명 — 즉시 수정 필요

---

#### 문제 1: Spec ↔ 채보 시간축 불일치 (가장 치명적)

**파일:** `mp3toSpec.py`, `traindemo.py`

**원인:**
- Spec: `hop_length = sr * 0.05` → 프레임 간격 **50ms 고정**
- 채보(.npy): BPM 기반 행 구성 → BPM마다 행 간격이 다름
  - BPM 180 → 16분음표 단위 83ms
  - BPM 120 → 16분음표 단위 125ms
- `traindemo.py`에서 `T = min(spec.shape[0], labels.shape[0])`으로 행 수만 맞춤
- **결과: 음악과 노트가 체계적으로 어긋난 채 학습**

**해결방안 (선택지 2개):**

옵션 A — frame-based 유지 시:
```python
# 채보를 BPM + offset → 절대 시간(ms) → 50ms 그리드로 리샘플링
def chart_to_frames(chart_rows, bpm, offset_ms, hop_ms=50, total_frames=None):
    # 각 행의 절대 시간 = offset + row_idx * (60000 / bpm / subdivision)
    # 해당 시간 // hop_ms = 프레임 인덱스
    ...
```

옵션 B — 이벤트 방식으로 전환 시 (권장):
- `(time_ms, lane, type, duration_ms)` 형태로 변환
- Spec에서 해당 time_ms를 프레임 인덱스로 변환: `frame_idx = time_ms // 50`
- 시간축 불일치 문제 구조적으로 해결됨

---

#### 문제 2: `note_emb.sum(dim=2)` 레인 정보 완전 소실

**파일:** `traindemo.py` — `ChartTransformer.forward()`

**원인 코드:**
```python
note_emb = self.note_embed(notes).sum(dim=2)  # (B, T, 4, 256) → (B, T, 256)
```
- `notes` shape: `(B, T, 4)` — 4개 레인
- `self.note_embed(notes)` shape: `(B, T, 4, 256)`
- `.sum(dim=2)` → 4개 레인 임베딩을 단순 합산
- **레인 조합이 달라도 합이 같으면 동일한 벡터 → 레인 구분 불가**

**해결 코드:**
```python
# 방법 A: concat + linear
note_emb = self.note_embed(notes)           # (B, T, 4, d_model)
note_emb = note_emb.view(B, T, 4 * d_model) # (B, T, 4*d_model)
note_emb = self.note_proj(note_emb)         # Linear(4*d_model, d_model)

# 모델 __init__에 추가:
self.note_proj = nn.Linear(4 * d_model, d_model)
```

---

#### 문제 3: 롱노트 보정 로직 오류 (`prev==2` 누락)

**파일:** `runDemo.py` — `apply_longnote_mask()`

**원인 코드:**
```python
mask = (prev == 2)
logits[mask, lane, :] = -1e9
logits[mask, lane, 3] = 0   # ← 인덱스 3 = 롱노트 유지(값 3)
                             # ← 롱노트 끝(값 4) 허용 코드 없음!
```
- `prev==2`(롱노트 시작) 다음에 `3`(유지)과 `4`(끝) 모두 허용해야 함
- 현재 코드는 `3`(유지)만 허용하고 `4`(끝)를 허용하지 않음 (또는 반대)
- **결과: 롱노트가 항상 1프레임 만에 강제 종료**

**올바른 보정 로직:**
```python
def apply_longnote_mask(prev_notes, logits):
    logits = logits.clone()
    for lane in range(4):
        prev = prev_notes[:, lane]

        # 롱노트 시작(2) 다음: 유지(3) 또는 끝(4)만 허용
        mask = (prev == 2)
        logits[mask, lane, 0] = -1e9  # 빈칸 차단
        logits[mask, lane, 1] = -1e9  # 일반노트 차단
        logits[mask, lane, 2] = -1e9  # 롱노트 시작 차단
        # 3(유지), 4(끝)는 허용 → 건드리지 않음

        # 롱노트 유지(3) 다음: 유지(3) 또는 끝(4)만 허용
        mask = (prev == 3)
        logits[mask, lane, 0] = -1e9
        logits[mask, lane, 1] = -1e9
        logits[mask, lane, 2] = -1e9
        # 3(유지), 4(끝)는 허용

        # 롱노트 없는 상태: 롱노트 유지(3), 끝(4) 차단
        mask = (prev == 0) | (prev == 1) | (prev == 4)
        logits[mask, lane, 3] = -1e9
        logits[mask, lane, 4] = -1e9

    return logits
```

---

#### 문제 4: `pos_embed` 2048 한계 — 1분 42초 이상 즉시 오류

**파일:** `traindemo.py`, `runDemo.py`

**원인 코드:**
```python
self.pos_embed = nn.Embedding(2048, d_model)
```

**실제 충돌 시점 계산:**
- hop_length = sr * 0.05 → 44100Hz 기준 2205 samples
- 1프레임 = 50ms → 2048프레임 = **102.4초 = 1분 42초**
- 리듬게임 수록곡 평균 2~4분 → 생성 단계에서 대부분 IndexError 발생
- 학습 시 segment_length=512로 잘라서 현재는 우연히 안 터지는 상태

**해결 코드:**
```python
# 방법 A: Sinusoidal PE로 교체 (길이 제한 없음, 권장)
def sinusoidal_pe(T, d_model, device):
    pe = torch.zeros(T, d_model, device=device)
    pos = torch.arange(T, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe  # (T, d_model)

# forward()에서:
pos_emb = sinusoidal_pe(T, self.d_model, audio.device)

# 방법 B: Embedding 크기 확장
self.pos_embed = nn.Embedding(8192, d_model)
```

---

#### 문제 5: TransformerEncoder + causal mask 구조 모순

**파일:** `traindemo.py`, `runDemo.py`

**원인:**
```python
encoder_layer = nn.TransformerEncoderLayer(...)  # 양방향 구조
mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
x = self.transformer(x, mask=mask)  # causal mask로 단방향 강제
```
- `TransformerEncoder`는 양방향(bidirectional) 구조
- 여기에 causal mask를 씌우면 단방향처럼 동작하긴 하나 비효율적
- autoregressive 생성이 목적이면 `TransformerDecoder`가 올바른 구조

**해결 방향 (아키텍처 선택 필요):**
- 선택 A: `nn.TransformerDecoder` 사용 (Encoder-Decoder 구조)
  - Encoder: Spec을 처리
  - Decoder: 채보를 autoregressive하게 생성
- 선택 B: 이벤트 방식 + Encoder-Decoder (권장)
  - Encoder input: Spec 전체
  - Decoder input: 이벤트 시퀀스

---

#### 문제 6: 생성 시 O(T²) 연산

**파일:** `runDemo.py` — `generate_window()`

**원인 코드:**
```python
for t in range(T - 1):
    pred = model(spec_chunk[:, :t+1], notes[:, :t+1])  # 매번 전체 재계산
```
- T=512 기준: 512번의 forward pass, 각각 점점 길어지는 시퀀스
- 실질적으로 약 13만 번의 attention 연산

**해결 방향:**
- KV-cache 구현으로 각 스텝 O(1)로 단축
- 또는 이벤트 방식 전환으로 시퀀스 자체를 1/4 이하로 단축

---

### 🟠 중요 — 성능 및 품질 영향

---

#### 문제 7: 극심한 데이터 불균형 (frame-based의 구조적 문제)

**파일:** `traindemo.py`

**원인:**
- 전체 프레임의 약 90%가 `[0,0,0,0]`
- `weight = [0.1, 1, 1, 1, 1]`로 부분 보완하나 불충분
- **모델이 "전부 0" 예측으로 수렴하는 trivial solution에 빠지기 쉬움**

**해결방안:**
```python
# 방법 A: Focal Loss 적용
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, pred, target):
        ce = F.cross_entropy(pred, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

# 방법 B: 이벤트 방식 전환으로 근본 해결 (0이 없어짐)
```

---

#### 문제 8: Sliding window 경계에서 롱노트 단절

**파일:** `runDemo.py` — `generate_full_chart()`

**원인:**
```python
for start in range(0, T, stride):
    generated = generate_window(model, spec_chunk)  # 이전 상태 전달 없음
```
- 이전 window의 마지막 노트 상태(롱노트 진행 중 여부)가 다음 window에 전달 안 됨
- 롱노트가 window 경계를 넘으면 무조건 끊김

**해결 코드:**
```python
def generate_full_chart(model, spec, window=512, stride=256):
    T = spec.shape[0]
    final_notes = torch.zeros((T, 4), dtype=torch.long)
    prev_last_notes = torch.zeros(4, dtype=torch.long)  # 이전 window 마지막 상태

    for start in range(0, T, stride):
        end = min(start + window, T)
        spec_chunk = spec[start:end]
        generated = generate_window(model, spec_chunk, init_notes=prev_last_notes)
        prev_last_notes = generated[-1]  # 다음 window에 전달
        ...
```

---

#### 문제 9: Spec을 트랜스포머에 raw로 직접 입력

**파일:** `traindemo.py`

**원인:**
- Spec (T, 128)을 Linear 한 층만 거쳐 트랜스포머에 바로 입력
- 음악의 지역적 패턴(onset, beat)을 먼저 추출하지 않음
- 시퀀스 길이가 길어 attention 비용 폭발

**해결 방향:**
```python
# CNN front-end 추가
self.cnn = nn.Sequential(
    nn.Conv1d(n_mels, 128, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv1d(128, d_model, kernel_size=3, padding=1),
    nn.ReLU(),
)
# forward():
audio = audio.transpose(1, 2)          # (B, n_mels, T)
audio = self.cnn(audio).transpose(1, 2) # (B, T, d_model)
```

---

#### 문제 10: 난이도/스타일 컨트롤 불가

**파일:** `traindemo.py`

**해결 방향:**
```python
# 난이도 레벨을 조건 벡터로 주입
self.difficulty_embed = nn.Embedding(10, d_model)  # 난이도 0~9

# forward()에서:
diff_emb = self.difficulty_embed(difficulty)  # (B, d_model)
x = x + diff_emb.unsqueeze(1)  # 모든 타임스텝에 더함
```

---

### 🟡 핵심 설계 결정

---

#### 설계 선택: frame-based vs event-based 표현

| 항목 | frame-based (현재) | event-based (권장) |
|---|---|---|
| 표현 형식 | `(T, 4)` 행렬 | `(time_ms, lane, type, duration_ms)` |
| 시퀀스 길이 (3분 곡) | ~3,600 토큰 | ~400~900 토큰 |
| 데이터 불균형 | ~90%가 0 | 해당 없음 |
| 롱노트 구조 버그 | 별도 보정 필요 | 구조적으로 불필요 |
| 시간축 불일치 | 별도 정렬 필요 | time_ms로 자연 해결 |
| 전환 비용 | — | .npy 변환 전처리 필요 |

**event-based 토큰 설계 예시:**
```python
# 특수 토큰
PAD = 0
BOS = 1  # 시작
EOS = 2  # 끝

# 이벤트 토큰: (time_frame, lane, note_type, duration_frames)
# time_frame = time_ms // 50
# lane: 0~3
# note_type: 0=tap, 1=hold
# duration_frames: hold인 경우만 사용
```

---

## 수정 우선순위 로드맵

### Phase 1 — 즉시 수정 (코드 레벨 버그)
1. **문제 2**: `note_emb.sum(dim=2)` → concat + linear ///
2. **문제 4**: `pos_embed 2048` → Sinusoidal PE 교체///
3. **문제 3**: 롱노트 보정 로직 재작성///

### Phase 2 — 아키텍처 결정
4. **설계 선택**: frame-based 유지 vs event-based 전환///
5. **문제 1**: 시간축 정렬 (Phase 2 결정에 따라 방법이 달라짐)
6. **문제 5**: Encoder → Decoder 구조 전환

### Phase 3 — 성능 개선
7. **문제 6**: KV-cache 또는 시퀀스 단축
8. **문제 7**: Focal Loss 또는 오버샘플링
9. **문제 8**: Sliding window 컨텍스트 전달
10. **문제 9**: CNN front-end 추가

### Phase 4 — 기능 확장
11. **문제 10**: 난이도 조건 벡터 추가

---

## 참고: 관련 선행 연구
- **Beat Transformer** — 음악 beat tracking에 Transformer 적용
- **Roformer** — 상대적 위치 인코딩 (Sinusoidal PE보다 긴 시퀀스에 유리)
- **osu! ML 커뮤니티** — 이벤트 방식 채보 생성 선례 다수 존재

음악 파일을 입력받아 4-KEY 리듬 게임의 채보를 자동으로 생성하는 AI 모델

.osu파일에서 mp3, .osz파일을 추출하여 mp3는 mel-spectogram으로, .osz에서 나온 채보는 학습하기 쉬운 형태로 전처리했다.
encoder에 mel-spectogram을 입력값으로 넣고, decoder에 전처리된 채보(event)를 입력값으로 넣었다.

spcetogram shape: (T, d_mel) -> 시간 T에 따른 d_mel개의 음역대를 나타내는 음원 파일

event shape: (N, 4) -> N개의 event에 대해서 각각 4개의 label 지정 (time, lane, type, duration)

4개의 label을 기준으로 loss를 정의하고 학습을 진행했다.

<img width="296" height="263" alt="화면 캡처 2026-07-28 104737" src="https://github.com/user-attachments/assets/9274cc14-ea38-4714-98b5-6a7e760bd78e" />

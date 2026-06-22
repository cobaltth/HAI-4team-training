from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple

import torch
import torchaudio
#TODO:오디오파일외에 다른 효과음 파일을 잘못가져오는 경우 수정필요.
# --------------------------------------------------
# 1. 파일/폴더 이름 정리
# --------------------------------------------------
def sanitize_filename(name: str) -> str:
    if not name:
        return "UnknownTitle"

    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    name = name.strip().strip(".")

    return name or "UnknownTitle"


# --------------------------------------------------
# 2. .osu 파일에서 metadata 파싱
# --------------------------------------------------
def parse_osu_metadata(osu_path: Path) -> Tuple[str, str]:
    artist: Optional[str] = None
    title: Optional[str] = None

    in_metadata = False
    text = None

    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            text = osu_path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue

    if text is None:
        raise ValueError(f"인코딩 실패: {osu_path}")

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        if line.startswith("[") and line.endswith("]"):
            in_metadata = (line == "[Metadata]")
            continue

        if not in_metadata or ":" not in line:
            continue

        key, value = line.split(":", 1)
        key, value = key.strip(), value.strip()

        if key == "Artist":
            artist = value
        elif key == "Title":
            title = value

    return artist or "UnknownArtist", title or "UnknownTitle"


# --------------------------------------------------
# 3. metadata 찾기
# --------------------------------------------------
def find_metadata_from_any_osu(song_dir: Path) -> Tuple[str, str]:
    osu_files = sorted(song_dir.glob("*.osu"))
    if not osu_files:
        raise FileNotFoundError(f".osu 없음: {song_dir}")

    return parse_osu_metadata(osu_files[0])


# --------------------------------------------------
# 4. 오디오 파일 찾기
# --------------------------------------------------
def find_audio_in_folder(song_dir: Path) -> Path:
    for ext in ("*.mp3", "*.ogg", "*.wav"):
        files = sorted(song_dir.glob(ext))
        if files:
            return files[0]

    raise FileNotFoundError(f"audio 없음: {song_dir}")


# --------------------------------------------------
# 5. 멜 스펙트로그램 생성 (🔥 핵심 수정)
# --------------------------------------------------
def build_spectrogram(
    audio_path: Path,
    n_fft: int = 1024,
    n_mels: int = 128,
    f_min: float = 27.5,
    f_max: Optional[float] = None,
) -> torch.Tensor:

    waveform, sample_rate = torchaudio.load(str(audio_path))

    # 🔥 stereo → mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # 🔥 50ms hop_length
    hop_length = int(sample_rate * 0.05)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        power=2.0,
    )

    mel = mel_transform(waveform)  # (1, n_mels, T)

    # 🔥 (T, n_mels)로 변환
    mel = mel.squeeze(0).transpose(0, 1)

    # 🔥 log 변환 (안정화)
    mel = torch.clamp(mel, min=1e-10)
    log_mel = torch.log(mel)

    # 정규화 (optional but 추천)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)

    return log_mel  # (T, n_mels)


# --------------------------------------------------
# 6. 저장
# --------------------------------------------------
def save_spectrogram(
    output_root: Path,
    artist: str,
    title: str,
    spectrogram: torch.Tensor,
) -> Path:

    folder_name = f"{sanitize_filename(artist)} - {sanitize_filename(title)}"
    out_dir = output_root / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / "spec.pt"
    torch.save(spectrogram, path)

    return out_dir


# --------------------------------------------------
# 7. 전체 처리
# --------------------------------------------------
def process_rawsource(
    rawsource_root: Path,
    output_root: Path,
):
    all_dirs = sorted([p for p in rawsource_root.rglob("*") if p.is_dir()])

    ok, skip = 0, 0

    for song_dir in all_dirs:
        try:
            if not any(song_dir.glob("*.osu")):
                continue

            if not any(song_dir.glob(ext) for ext in ("*.mp3", "*.ogg", "*.wav")):
                continue

            artist, title = find_metadata_from_any_osu(song_dir)
            audio_path = find_audio_in_folder(song_dir)

            spec = build_spectrogram(audio_path)

            out_dir = save_spectrogram(output_root, artist, title, spec)

            ok += 1
            print(f"[OK] {song_dir} → {out_dir}")

        except Exception as e:
            skip += 1
            print(f"[SKIP] {song_dir} | {e}")

    print("\n완료")
    print(f"성공: {ok}")
    print(f"실패: {skip}")


# --------------------------------------------------
# 8. 실행
# --------------------------------------------------
if __name__ == "__main__":
    process_rawsource(
        rawsource_root=Path("rawsource"),
        output_root=Path("source"),
    )
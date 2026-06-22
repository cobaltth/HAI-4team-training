"""rawsource/ 내 모든 .osz 파일을 .zip으로 변환(이름 변경)한다.

.osz 파일은 ZIP 아카이브와 동일한 포맷이므로 확장자 변경만으로 변환된다.
원본 .osz 파일은 삭제된다.
"""

from pathlib import Path


def osz_to_zip(rawsource_root: Path) -> None:
    osz_files = sorted(rawsource_root.rglob("*.osz"))

    if not osz_files:
        print(f".osz 파일 없음: {rawsource_root}")
        return

    print(f"Found {len(osz_files)} .osz file(s)\n")

    ok, skip = 0, 0

    for osz_path in osz_files:
        zip_path = osz_path.with_suffix(".zip")
        try:
            if zip_path.exists():
                print(f"[SKIP] 이미 존재: {zip_path}")
                skip += 1
                continue

            osz_path.rename(zip_path)
            print(f"[OK] {osz_path.name} → {zip_path.name}")
            ok += 1

        except Exception as e:
            print(f"[ERROR] {osz_path.name} | {e}")
            skip += 1

    print(f"\n완료 — 성공: {ok}, 건너뜀: {skip}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    osz_to_zip(root / "rawsource")

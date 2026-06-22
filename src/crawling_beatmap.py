import requests
import browser_cookie3
from pathlib import Path
import zipfile
import time
import json
import queue
import threading

_CURSOR_STATE_PATH  = Path(__file__).resolve().parent.parent / "cursor_state.json"
_PENDING_PATH       = Path(__file__).resolve().parent.parent / "pending_downloads.json"
_pending_lock       = threading.Lock()

# ── cursor 상태 ────────────────────────────────────────────
def _load_cursor() -> "str | None":
    if _CURSOR_STATE_PATH.exists():
        try:
            return json.loads(_CURSOR_STATE_PATH.read_text(encoding="utf-8")).get("cursor")
        except Exception:
            return None
    return None

def _save_cursor(cursor: "str | None") -> None:
    if cursor:
        _CURSOR_STATE_PATH.write_text(json.dumps({"cursor": cursor}), encoding="utf-8")
    else:
        if _CURSOR_STATE_PATH.exists():
            _CURSOR_STATE_PATH.unlink()

# ── pending 큐 상태 ────────────────────────────────────────
def _load_pending() -> list:
    """파일에서 pending ID 목록을 읽는다. 락 내부에서만 호출."""
    if _PENDING_PATH.exists():
        try:
            return json.loads(_PENDING_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def _write_pending(ids: list) -> None:
    """pending ID 목록을 파일에 쓴다. 락 내부에서만 호출."""
    if ids:
        _PENDING_PATH.write_text(json.dumps(ids), encoding="utf-8")
    else:
        if _PENDING_PATH.exists():
            _PENDING_PATH.unlink()

def _add_pending(bmset_id: int) -> None:
    with _pending_lock:
        ids = _load_pending()
        if bmset_id not in ids:
            ids.append(bmset_id)
            _write_pending(ids)

def _remove_pending(bmset_id: int) -> None:
    with _pending_lock:
        ids = _load_pending()
        ids = [i for i in ids if i != bmset_id]
        _write_pending(ids)

# ──────────────────────────────────────────────────────────
osu_session_value = "eyJpdiI6Im8xUlNycW1remd5bjdscUoyK0lJSHc9PSIsInZhbHVlIjoiaTRLRlZVcWRJeFc2TzRQL0lPd3djMytKdmQ3MEd4emcvZ2FqZFR6Q3IrRXpuMnNTTU1aUTBKay9pMnYzS01UcDgxVENPd1dBcHFydVQ0WGJWM3NEN2xSRmRmT1VFSjErOVdjSU91dHFIV3RaSVdSSTVVMS9RdlRxZkQxQk5ZWXptc0k2cXBvNnIyaUFFN0ZVaTBmSWtRPT0iLCJtYWMiOiIxOTQyM2JlYjk3ZmNlNDY3YjhiMTc5OTJmMGVlZjFjYTg4YWY5ZWFkODQzYTYwNjNkNDBkNjg0MWVmY2YyOWY3IiwidGFnIjoiIn0%3D"

def get_download(beatmapID: int):
    download_url = f"https://osu.ppy.sh/beatmapsets/{beatmapID}/download"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/146.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,image/apng,*/*;q=0.8,"
            "application/signed-exchange;v=b3;q=0.7"
        ),
        "Referer": f"https://osu.ppy.sh/beatmapsets/{beatmapID}",
    }
    session1 = requests.Session()
    session1.cookies.set("osu_session", osu_session_value, domain="osu.ppy.sh", path="/")
    resp = session1.get(download_url, headers=headers, allow_redirects=False, timeout=20)

    if resp.status_code != 302:
        raise Exception("invalid session, try to reset osu_session_value")
    return resp.headers.get("Location"), beatmapID

def search(id_queue: queue.Queue, skip_ids: set) -> None:
    """beatmap 탐색. 발견한 ID를 queue와 pending 파일에 동시 기록."""
    root = Path(__file__).resolve().parent.parent / "rawsource"
    existing_folder_names = {p.name for p in root.iterdir() if p.is_dir()}

    session = requests.Session()

    search_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/134.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://osu.ppy.sh/beatmapsets",
        "X-Requested-With": "XMLHttpRequest",
    }

    base_url = (
        "https://osu.ppy.sh/beatmapsets/search"
        "?e=&c=&g=&l=&m=3&nsfw=&played=&q=key%3D4%20star%3E3%20star%3C5"
        "&r=&sort=&s=ranked"
    )

    try:
        cursor = _load_cursor()
        if cursor:
            print(f"[RESUME] cursor 복원: {cursor[:40]}...")
        for _ in range(1000):
            if cursor:
                search_url = f"{base_url}&cursor_string={cursor}"
            else:
                search_url = base_url
            resp = session.get(search_url, headers=search_headers, timeout=20)
            if resp.status_code != 302 and resp.status_code != 200:
                print("error: search request failed with status code", resp.status_code)
                break
            data = resp.json()

            beatmapsets = data.get("beatmapsets", [])

            for bmset in beatmapsets:
                bmset_id = bmset.get("id")

                has_target_diff = False
                for beatmap in bmset.get("beatmaps", []):
                    if beatmap.get("mode") == "mania" and beatmap.get("cs") == 4:
                        has_target_diff = True
                        break

                if not has_target_diff:
                    print(f"{bmset_id} skipped: no 4K mania diff")
                    continue

                if str(bmset_id) in existing_folder_names:
                    print(f"{bmset_id} already exists so skipped.")
                elif bmset_id in skip_ids:
                    print(f"{bmset_id} already pending so skipped.")
                else:
                    _add_pending(bmset_id)
                    id_queue.put(bmset_id)
                    print(bmset_id, bmset.get("title"))

            cursor = data.get("cursor_string")
            _save_cursor(cursor)
            if not cursor:
                break
    finally:
        id_queue.put(None)   # 탐색 완료 신호 (오류 시에도 반드시 전송)

def downloader(id_queue: queue.Queue) -> None:
    """queue에서 ID를 꺼내 다운로드. 성공 시 pending 파일에서 제거."""
    while True:
        bmset_id = id_queue.get()
        if bmset_id is None:
            break
        try:
            location, beatmapid = get_download(bmset_id)
            download_file(location, beatmapid)
            _remove_pending(bmset_id)
        except Exception as e:
            print(f"[ERROR] {bmset_id} 다운로드 실패: {e}")
            # pending에는 남겨둬서 다음 실행 시 재시도 가능

def download_file(location: str, beatmapID: int):
    base_dir = Path(__file__).resolve().parent.parent
    save_dir = base_dir / "rawsource"
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / f"{beatmapID}.zip"

    download_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/146.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,image/apng,*/*;q=0.8,"
            "application/signed-exchange;v=b3;q=0.7"
        ),
        "Referer": f"https://osu.ppy.sh/beatmapsets/{beatmapID}",
    }

    session = requests.Session()

    with session.get(location, headers=download_headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(file_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    extract_dir = file_path.parent / file_path.stem
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(file_path, "r") as zf:
        zf.extractall(extract_dir)

    file_path.unlink()  # 휴지통 없이 바로 삭제

    print("추출완료:", file_path)

def main():
    id_queue: queue.Queue = queue.Queue()

    # 이전 실행에서 탐색됐지만 다운로드되지 않은 ID 복원
    pending = _load_pending()
    skip_ids: set = set(pending)
    if pending:
        print(f"[RESUME] pending ID {len(pending)}개 복원: {pending}")
        for bmset_id in pending:
            id_queue.put(bmset_id)

    t_search   = threading.Thread(target=search,     args=(id_queue, skip_ids), daemon=True)
    t_download = threading.Thread(target=downloader, args=(id_queue,))

    t_search.start()
    t_download.start()

    t_search.join()
    t_download.join()

if __name__ == "__main__":
    main()
else:
    raise Exception("__name__ is not __main__")

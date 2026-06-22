from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Union
import math
import json
import re
from pathlib import Path

import numpy as np


# =========================================================
# Dataclasses
# =========================================================

@dataclass
class ManiaNote:
    lane: int
    start_time: int
    end_time: Optional[int]
    note_type: str


@dataclass
class ChartEvent:
    """Single note event in event-based representation.

    time_frame      : start_time // hop_ms  (absolute frame index)
    lane            : 0-3
    note_type       : 0 = tap, 1 = hold
    duration_frames : 0 for tap, >= 1 for hold
    """
    time_frame: int
    lane: int
    note_type: int
    duration_frames: int


# =========================================================
# Utilities
# =========================================================

def sanitize_filename(name: str) -> str:
    if not name:
        return "UnknownTitle"

    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    name = name.strip().strip(".")

    if not name:
        return "UnknownTitle"

    return name


# =========================================================
# .osu parser
# =========================================================

def parse_osu_mania_4k(osu_path: Union[str, Path]) -> Dict[str, Any]:
    section = None

    general = {}
    metadata = {}
    difficulty = {}
    hit_objects = []
    bpm = 0

    with open(osu_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line or line.startswith("//"):
                continue

            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1]
                continue

            if section in ("General", "Metadata", "Difficulty"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = value.strip()

                    if section == "General":
                        general[key] = value
                    elif section == "Metadata":
                        metadata[key] = value
                    elif section == "Difficulty":
                        difficulty[key] = value
            elif section == "HitObjects":
                hit_objects.append(line)
            elif section == "TimingPoints":
                time, beatLength, meter, sampleSet, sampleIndex, volume, uninherited, effects = line.split(",", 8)
                if uninherited == "1":
                    bpm = round(60000 / float(beatLength))

    mode = int(general.get("Mode", -1))
    circle_size = float(difficulty.get("CircleSize", -1))

    if mode != 3:
        raise ValueError(f"{osu_path} 는 osu!mania 파일이 아닙니다. Mode={mode}")

    if int(circle_size) != 4:
        raise ValueError(f"{osu_path} 는 4K 파일이 아닙니다. CircleSize={circle_size}")

    notes: List[ManiaNote] = []

    for obj in hit_objects:
        parts = obj.split(",")

        if len(parts) < 5:
            continue

        x = int(parts[0])
        time = int(parts[2])
        obj_type = int(parts[3])

        lane = math.floor(x * 4 / 512)
        lane = max(0, min(3, lane))

        is_hold = (obj_type & 128) > 0

        if is_hold:
            if len(parts) < 6:
                continue

            end_part = parts[5]
            end_time_str = end_part.split(":")[0]
            end_time = int(end_time_str)

            notes.append(ManiaNote(lane, time, end_time, "hold"))
        else:
            notes.append(ManiaNote(lane, time, None, "tap"))

    return {
        "audio_filename": general.get("AudioFilename"),
        "title": metadata.get("Title"),
        "artist": metadata.get("Artist"),
        "version": metadata.get("Version"),
        "id": metadata.get("BeatmapSetID"),
        "mode": mode,
        "circle_size": circle_size,
        "bpm": bpm,
        "notes": notes
    }


# =========================================================
# Event-based representation
# =========================================================

def notes_to_events(notes: List[ManiaNote], hop_ms: int = 50) -> List[ChartEvent]:
    """Convert ManiaNote list to ChartEvent list, sorted by (start_time, lane).

    Args:
        notes:  List of ManiaNote parsed from .osu file.
        hop_ms: Frame duration in ms (must match mp3toSpec hop_length). Default 50.

    Returns:
        List of ChartEvent sorted by start time.
    """
    events = []
    for note in sorted(notes, key=lambda n: (n.start_time, n.lane)):
        time_frame = round(note.start_time / hop_ms)
        if note.note_type == "tap":
            events.append(ChartEvent(time_frame, note.lane, 0, 0))
        else:
            dur = round((note.end_time - note.start_time) / hop_ms)
            events.append(ChartEvent(time_frame, note.lane, 1, max(1, dur)))
    return events


def events_to_numpy(events: List[ChartEvent]) -> np.ndarray:
    """Convert ChartEvent list to (N, 4) int32 array.

    Columns: [time_frame, lane, note_type, duration_frames]
    """
    if not events:
        return np.zeros((0, 4), dtype=np.int32)
    return np.array(
        [[e.time_frame, e.lane, e.note_type, e.duration_frames] for e in events],
        dtype=np.int32
    )


def events_from_numpy(arr: np.ndarray) -> List[ChartEvent]:
    """Reconstruct ChartEvent list from (N, 4) array."""
    return [ChartEvent(int(r[0]), int(r[1]), int(r[2]), int(r[3])) for r in arr]


def events_to_frame_labels(events: List[ChartEvent], num_frames: int) -> np.ndarray:
    """Convert ChartEvent list to frame-based (num_frames, 4) label array.

    Values: 0=empty, 1=tap, 2=hold_start, 3=hold_body, 4=hold_end

    Use this to convert model output back to a visualizable / playable format.
    """
    labels = np.zeros((num_frames, 4), dtype=np.uint8)
    for e in events:
        t = e.time_frame
        lane = e.lane
        if not (0 <= lane < 4):
            continue
        if e.note_type == 0:  # tap
            if 0 <= t < num_frames:
                labels[t, lane] = 1
        else:  # hold
            end_t = t + e.duration_frames
            if 0 <= t < num_frames:
                labels[t, lane] = 2
            for i in range(t + 1, min(end_t, num_frames)):
                labels[i, lane] = 3
            if 0 <= end_t < num_frames:
                labels[end_t, lane] = 4
    return labels


# =========================================================
# Save / Load
# =========================================================

def save_parsed_song(
    parsed: Dict[str, Any],
    events: np.ndarray,
    output_root: str = "source",
    frame_ms: int = 50,
    original_osu_path: Optional[str] = None
) -> Path:
    """Save parsed song data to disk (event-based).

    Creates:
        source/<Artist> - <Title>/metadata.json
        source/<Artist> - <Title>/<Version>/labels/events.npy
        source/<Artist> - <Title>/<Version>/labels/events.txt  (human-readable)

    Args:
        parsed:  Dict returned by parse_osu_mania_4k.
        events:  (N, 4) int32 array from events_to_numpy.
        output_root: Root directory for output.
        frame_ms: Frame duration used when building events.
        original_osu_path: Source .osu path for traceability.
    """
    artist = parsed.get("artist") or "UnknownArtist"
    title = parsed.get("title") or "UnknownTitle"
    version = parsed.get("version") or "UnknownVersion"

    folder_name = f"{sanitize_filename(artist)} - {sanitize_filename(title)}"

    root_dir = Path(output_root)
    song_dir = root_dir / folder_name
    version_dir = song_dir / sanitize_filename(version)
    labels_dir = version_dir / "labels"

    labels_dir.mkdir(parents=True, exist_ok=True)

    metadata_to_save = {
        "audio_filename": parsed.get("audio_filename"),
        "title": parsed.get("title"),
        "artist": parsed.get("artist"),
        "version": parsed.get("version"),
        "mode": parsed.get("mode"),
        "circle_size": parsed.get("circle_size"),
        "id": parsed.get("id"),
        "BPM": parsed.get("bpm"),
        "frame_ms": frame_ms,
        "num_notes": len(parsed.get("notes", [])),
        "num_events": int(events.shape[0]),
        "original_osu_path": str(original_osu_path) if original_osu_path else None,
        "notes": [asdict(n) for n in parsed.get("notes", [])]
    }

    with open(song_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_to_save, f, ensure_ascii=False, indent=2)

    np.save(labels_dir / "events.npy", events)

    _write_events_txt(labels_dir / "events.txt", events)

    return song_dir


def _write_events_txt(path: Path, events: np.ndarray) -> None:
    """Write events.npy as a human-readable text file for debugging.

    Format (tab-separated):
        time_frame  lane  type       duration_frames
        50          2     tap        0
        125         0     hold       8
    """
    type_name = {0: "tap", 1: "hold"}

    with open(path, "w", encoding="utf-8") as f:
        f.write("time_frame\tlane\ttype\tduration_frames\n")
        for row in events:
            t, lane, ntype, dur = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            f.write(f"{t}\t{lane}\t{type_name.get(ntype, str(ntype))}\t{dur}\n")


def process_osu_file_to_source(
    osu_path: Union[str, Path],
    output_root: str = "source",
    frame_ms: int = 50
) -> Path:
    parsed = parse_osu_mania_4k(osu_path)
    events = events_to_numpy(notes_to_events(parsed["notes"], hop_ms=frame_ms))
    saved_dir = save_parsed_song(
        parsed=parsed,
        events=events,
        output_root=output_root,
        frame_ms=frame_ms,
        original_osu_path=str(osu_path)
    )
    return saved_dir


def process_all_osu_files(osu_paths: List[Path], output_root: str, frame_ms: int) -> None:
    for osu_file in osu_paths:
        try:
            saved_dir = process_osu_file_to_source(
                osu_path=osu_file,
                output_root=output_root,
                frame_ms=frame_ms
            )
            print(f"저장 완료: {saved_dir}")
        except Exception as e:
            print(f"건너뜀: {osu_file} -> {e}")


if __name__ == "__main__":
    root = Path("rawsource")
    osu_paths = list(root.rglob("*.osu"))

    if osu_paths:
        process_all_osu_files(osu_paths, "source", 50)
    else:
        print("rawsource 안에서 .osu 파일을 찾지 못했습니다.")

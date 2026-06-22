import json
import os
import random
from pathlib import Path
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import EventChartTransformer


# =========================================================
# 0. 디바이스 설정
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

MAX_DELTA    = EventChartTransformer.MAX_DELTA
MAX_DURATION = EventChartTransformer.MAX_DURATION
EOS_TYPE     = EventChartTransformer.EOS_TYPE


# =========================================================
# 1. source → dataset 변환
# =========================================================
def build_dataset(root: Path) -> None:
    """Build event-based dataset from source directory.

    Reads:
        source/<song>/spec.pt          (T, n_mels)
        source/<song>/<ver>/labels/events.npy  (N, 4)

    Writes:
        dataset/<idx>/spec.pt          (T, n_mels)
        dataset/<idx>/events.pt        (N, 4)  absolute time_frame
    """
    source_root = root / "source"
    dataset_root = root / "dataset"
    dataset_root.mkdir(exist_ok=True)

    idx = 0

    for song_dir in source_root.iterdir():
        if not song_dir.is_dir():
            continue

        spec_path = song_dir / "spec.pt"
        if not spec_path.exists():
            continue

        spec = torch.load(spec_path)
        T = spec.shape[0]

        bpm_val = 120.0
        meta_path = song_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            bpm_val = float(meta.get("BPM", 120) or 120)
        bpm_tensor = torch.tensor(bpm_val, dtype=torch.float)

        for events_path in song_dir.rglob("events.npy"):
            try:
                events_np = np.load(events_path)

                if events_np.ndim != 2 or events_np.shape[1] != 4:
                    continue

                events_t = torch.from_numpy(events_np).long()

                # Filter events within spec length
                if events_t.shape[0] > 0:
                    mask = events_t[:, 0] < T
                    events_t = events_t[mask]

                if events_t.shape[0] < 5:
                    continue

                out_dir = dataset_root / f"{idx:05d}"
                out_dir.mkdir(exist_ok=True)

                torch.save(spec, out_dir / "spec.pt")
                torch.save(events_t, out_dir / "events.pt")
                torch.save(bpm_tensor, out_dir / "bpm.pt")

                idx += 1

            except Exception as e:
                print(f"[SKIP] {events_path} | {e}")

    print(f"Dataset built: {idx} samples")


# =========================================================
# 2. Dataset
# =========================================================
class EventRhythmDataset(Dataset):
    """Event-based dataset for teacher-forcing training.

    Each sample:
        spec   : (spec_window, n_mels)
        events : (max_events, 4) — delta-encoded, EOS-terminated, zero-padded

    Event columns: [delta_frames, lane, note_type, duration_frames]
        delta_frames : time since previous event (or window start for first event)
        note_type    : 0=tap, 1=hold, 2=EOS
    """

    def __init__(self, dataset_root, spec_window: int = 512, max_events: int = 256):
        self.root = Path(dataset_root)
        self.spec_window = spec_window
        self.max_events = max_events
        self.samples = [
            d for d in self.root.iterdir()
            if d.is_dir() and (d / "events.pt").exists()
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder = self.samples[idx]

        spec   = torch.load(folder / "spec.pt")      # (T, n_mels)
        events = torch.load(folder / "events.pt")    # (N, 4)
        bpm    = torch.load(folder / "bpm.pt")       # scalar (raw BPM)

        T = spec.shape[0]

        # Random spec window
        if T > self.spec_window:
            start_frame = random.randint(0, T - self.spec_window)
        else:
            start_frame = 0
        end_frame = min(start_frame + self.spec_window, T)

        spec_chunk = spec[start_frame:end_frame]
        if spec_chunk.shape[0] < self.spec_window:
            pad = self.spec_window - spec_chunk.shape[0]
            spec_chunk = torch.cat([spec_chunk, torch.zeros(pad, spec_chunk.shape[1])])

        # Filter events in window and make time_frame relative to window start
        if events.shape[0] > 0:
            mask = (events[:, 0] >= start_frame) & (events[:, 0] < end_frame)
            win_events = events[mask].clone()
            if win_events.shape[0] > 0:
                win_events[:, 0] -= start_frame
        else:
            win_events = torch.zeros((0, 4), dtype=torch.long)

        # Convert absolute time_frame → delta encoding
        win_events = _to_delta(win_events)

        # Clamp fields
        win_events[:, 0] = win_events[:, 0].clamp(0, MAX_DELTA)
        win_events[:, 1] = win_events[:, 1].clamp(0, 3)
        win_events[:, 2] = win_events[:, 2].clamp(0, 1)        # only tap/hold; EOS added below
        win_events[:, 3] = win_events[:, 3].clamp(0, MAX_DURATION)

        # Append EOS token
        eos = torch.tensor([[0, 0, EOS_TYPE, 0]], dtype=torch.long)
        win_events = torch.cat([win_events, eos], dim=0)

        # Truncate or pad to max_events (pad with EOS)
        N = win_events.shape[0]
        real_len = min(N, self.max_events)   # real events + first EOS (capped)
        if N >= self.max_events:
            win_events = win_events[:self.max_events]
        else:
            pad_token = torch.tensor([[0, 0, EOS_TYPE, 0]], dtype=torch.long).expand(
                self.max_events - N, -1
            )
            win_events = torch.cat([win_events, pad_token], dim=0)

        # valid_mask: True for real events + first EOS, False for padding
        valid_mask = torch.zeros(self.max_events, dtype=torch.bool)
        valid_mask[:real_len] = True

        # Normalize BPM: frames_per_beat / 20.0  (BPM 60→1.0, BPM 300→0.2)
        frames_per_beat = (60000.0 / bpm.clamp(min=60.0) / 50.0) / 20.0

        return spec_chunk, win_events, valid_mask, frames_per_beat


def _to_delta(events: torch.Tensor) -> torch.Tensor:
    """Convert absolute time_frame column to inter-event delta in-place (copy)."""
    if events.shape[0] == 0:
        return events
    out = events.clone()
    out[1:, 0] = events[1:, 0] - events[:-1, 0]
    # out[0, 0] stays as-is: delta from window start to first event
    return out


# =========================================================
# 3. Loss
# =========================================================
def compute_loss(pred, events: torch.Tensor, valid_mask: torch.Tensor,
                 lane_reg_weight: float = 0.5) -> torch.Tensor:
    """Compute cross-entropy loss over 4 event fields, ignoring padding positions.

    pred       : tuple (delta_logits, lane_logits, type_logits, dur_logits)
    events     : (B, N, 4) target events (delta-encoded)
    valid_mask : (B, N) bool — True for real events + first EOS, False for padding
    """
    delta_logits, lane_logits, type_logits, dur_logits = pred

    flat_mask = valid_mask.reshape(-1)   # (B*N,)

    delta_t = events[:, :, 0].clamp(0, MAX_DELTA).reshape(-1)[flat_mask]
    lane_t  = events[:, :, 1].clamp(0, 3).reshape(-1)[flat_mask]
    type_t  = events[:, :, 2].clamp(0, EOS_TYPE).reshape(-1)[flat_mask]
    dur_t   = events[:, :, 3].clamp(0, MAX_DURATION).reshape(-1)[flat_mask]

    loss_delta = F.cross_entropy(delta_logits.reshape(-1, MAX_DELTA + 1)[flat_mask], delta_t)

    valid_lane_logits = lane_logits.reshape(-1, 4)[flat_mask]
    loss_lane = F.cross_entropy(valid_lane_logits, lane_t)

    lane_probs      = F.softmax(valid_lane_logits, dim=-1)
    mean_lane_probs = lane_probs.mean(dim=0)
    uniform_target  = torch.full_like(mean_lane_probs, 0.25)
    loss_lane_reg   = -(uniform_target * torch.log(mean_lane_probs + 1e-8)).sum()

    loss_type  = F.cross_entropy(type_logits.reshape(-1, EOS_TYPE + 1)[flat_mask], type_t)

    hold_mask = (events[:, :, 2] == 1).reshape(-1)[flat_mask]
    if hold_mask.sum() > 0:
        loss_dur = F.cross_entropy(
            dur_logits.reshape(-1, MAX_DURATION + 1)[flat_mask][hold_mask],
            dur_t[hold_mask]
        )
    else:
        loss_dur = delta_logits.new_tensor(0.0)

    return loss_delta + loss_lane + loss_type + 0.5 * loss_dur + lane_reg_weight * loss_lane_reg


# =========================================================
# 4. 학습
# =========================================================
def train(root: Path, epochs: int = 200, batch_size: int =64) -> None:
    dataset = EventRhythmDataset(root / "dataset")
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,      # 병렬 데이터 로딩
        pin_memory=True,    # CPU→GPU 전송 가속
        persistent_workers=True,
    )

    model     = EventChartTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler    = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))  # mixed precision

    model_dir = root / "models"
    model_dir.mkdir(exist_ok=True)

    writer = SummaryWriter(log_dir=str(root / "runs"))

    best_loss = float("inf")

    for epoch in range(epochs):
        total_loss = 0.0
        model.train()

        for spec, events, valid_mask, bpm in loader:
            spec       = spec.to(device, non_blocking=True)
            events     = events.to(device, non_blocking=True)
            valid_mask = valid_mask.to(device, non_blocking=True)
            bpm        = bpm.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                pred = model(spec, events, bpm)
                loss = compute_loss(pred, events, valid_mask)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1}: {avg_loss:.4f}")

        writer.add_scalar("Loss/train", avg_loss, epoch + 1)

        _tmp = model_dir / "latest.pt.tmp"
        torch.save({
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch":     epoch,
        }, _tmp)
        os.replace(_tmp, model_dir / "latest.pt")

        if avg_loss < best_loss:
            best_loss = avg_loss
            _tmp = model_dir / "best.pt.tmp"
            torch.save(model.state_dict(), _tmp)
            os.replace(_tmp, model_dir / "best.pt")

    writer.close()


# =========================================================
# 5. 실행
# =========================================================
if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    dataset_root = root / "dataset"

    if not dataset_root.exists() or not any(dataset_root.iterdir()):
        build_dataset(root)
    else:
        print(f"Dataset folder exists ({sum(1 for _ in dataset_root.iterdir())} samples), skipping build.")

    train(root)

import math
import torch
import torch.nn as nn


# =========================================================
# Sinusoidal Positional Encoding (길이 제한 없음)
# =========================================================

def sinusoidal_pe(T: int, d_model: int, device: torch.device) -> torch.Tensor:
    """Returns (T, d_model) sinusoidal positional encoding."""
    pe = torch.zeros(T, d_model, device=device)
    pos = torch.arange(T, device=device).unsqueeze(1).float()
    div = torch.exp(
        torch.arange(0, d_model, 2, device=device).float()
        * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


# =========================================================
# Event-based Encoder-Decoder Transformer
# =========================================================

class EventChartTransformer(nn.Module):
    """Encoder-Decoder transformer for event-based chart generation.

    Encoder : spectrogram (T, n_mels) → context
    Decoder : event sequence autoregressively generated

    Each event token has 4 fields:
        delta_frames    : time since previous event, clamped to [0, MAX_DELTA]
        lane            : 0-3
        note_type       : 0=tap, 1=hold, 2=EOS
        duration_frames : hold length, clamped to [0, MAX_DURATION]; 0 for tap/EOS

    Training (forward):
        audio  : (B, T, n_mels)
        events : (B, N, 4)  — target events with delta encoding
        Returns 4 logit tensors each (B, N, num_classes).
        Internally shifts events right and prepends BOS for teacher forcing.

    Inference:
        Use encode_spec() once, then call decoder + fc heads incrementally.
        See generate_window_events() in runDemo.py for the loop pattern.
    """

    MAX_DELTA    = 256   # max inter-event gap in frames (256 * 50ms = 12.8s)
    MAX_DURATION = 128   # max hold duration in frames  (128 * 50ms = 6.4s)
    EOS_TYPE     = 2     # note_type value used as end-of-sequence signal

    def __init__(self, n_mels: int = 128, d_model: int = 512):
        super().__init__()
        self.d_model = d_model

        # ── Encoder ──────────────────────────────────────────
        self.audio_proj = nn.Linear(n_mels, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, batch_first=True,
            dim_feedforward=2048, dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=6)

        # ── Event field embeddings ────────────────────────────
        # delta: 0 … MAX_DELTA  (MAX_DELTA+1 values)
        self.delta_embed = nn.Embedding(self.MAX_DELTA + 1, d_model // 4)
        self.lane_embed  = nn.Embedding(4,                  d_model // 4)
        self.type_embed  = nn.Embedding(self.EOS_TYPE + 1,  d_model // 4)  # 0,1,2
        self.dur_embed   = nn.Embedding(self.MAX_DURATION + 1, d_model // 4)
        self.event_proj  = nn.Linear(d_model, d_model)
        self.bpm_proj    = nn.Linear(1, d_model)         # BPM conditioning

        # BOS token (learnable)
        self.bos = nn.Parameter(torch.zeros(1, 1, d_model))

        # ── Decoder ──────────────────────────────────────────
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=8, batch_first=True,
            dim_feedforward=2048, dropout=0.1
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=6)

        # ── Output heads ─────────────────────────────────────
        self.fc_delta = nn.Linear(d_model, self.MAX_DELTA + 1)
        self.fc_lane  = nn.Linear(d_model, 4)
        self.fc_type  = nn.Linear(d_model, self.EOS_TYPE + 1)
        self.fc_dur   = nn.Linear(d_model, self.MAX_DURATION + 1)

    # ─────────────────────────────────────────────────────────
    # Sub-routines (reused in training and inference)
    # ─────────────────────────────────────────────────────────

    def encode_spec(self, audio: torch.Tensor, bpm: torch.Tensor = None) -> torch.Tensor:
        """audio: (B, T, n_mels), bpm: (B,) normalized frames_per_beat → memory: (B, T, d_model)"""
        B, T, n_mels = audio.shape
        freq_weight = torch.linspace(1.0, 2.0, n_mels, device=audio.device)
        audio = audio * freq_weight
        x = self.audio_proj(audio)
        x = x + sinusoidal_pe(T, self.d_model, audio.device).unsqueeze(0)
        memory = self.encoder(x)
        if bpm is not None:
            bpm_emb = self.bpm_proj(bpm.unsqueeze(-1))  # (B,) → (B, d_model)
            memory = memory + bpm_emb.unsqueeze(1)       # broadcast → (B, T, d_model)
        return memory

    def embed_events(self, events: torch.Tensor) -> torch.Tensor:
        """events: (B, N, 4) → (B, N, d_model)

        Clamps each field before embedding so out-of-range values are safe.
        """
        delta = events[:, :, 0].clamp(0, self.MAX_DELTA)
        lane  = events[:, :, 1].clamp(0, 3)
        ntype = events[:, :, 2].clamp(0, self.EOS_TYPE)
        dur   = events[:, :, 3].clamp(0, self.MAX_DURATION)

        emb = torch.cat([
            self.delta_embed(delta),
            self.lane_embed(lane),
            self.type_embed(ntype),
            self.dur_embed(dur),
        ], dim=-1)                       # (B, N, d_model)
        return self.event_proj(emb)

    # ─────────────────────────────────────────────────────────
    # Training forward
    # ─────────────────────────────────────────────────────────

    def forward(self, audio: torch.Tensor, events: torch.Tensor, bpm: torch.Tensor = None):
        """Teacher-forcing forward pass.

        audio  : (B, T, n_mels)
        events : (B, N, 4)  — delta-encoded target events

        Internally builds decoder input as [BOS, embed(events[0..N-2])],
        so output position i predicts events[i] from BOS + events[0..i-1].

        Returns:
            delta_logits : (B, N, MAX_DELTA+1)
            lane_logits  : (B, N, 4)
            type_logits  : (B, N, EOS_TYPE+1)
            dur_logits   : (B, N, MAX_DURATION+1)
        """
        B, N, _ = events.shape

        memory = self.encode_spec(audio, bpm)                    # (B, T, d_model)

        bos       = self.bos.expand(B, -1, -1)                  # (B, 1, d_model)
        event_emb = self.embed_events(events)                    # (B, N, d_model)
        tgt = torch.cat([bos, event_emb[:, :-1]], dim=1)        # (B, N, d_model)

        causal_mask = torch.triu(
            torch.ones(N, N, device=audio.device), diagonal=1
        ).bool()

        out = self.decoder(tgt, memory, tgt_mask=causal_mask)   # (B, N, d_model)

        # fc_dur은 detach된 out을 사용:
        # loss_dur gradient가 fc_dur 너머 decoder까지 역전파되지 않도록 차단.
        # decoder는 loss_delta + loss_lane + loss_type 만으로 학습되므로
        # 데이터 빈도(tap 85%)가 자연스럽게 tap 편향을 만들어준다.
        return (
            self.fc_delta(out),
            self.fc_lane(out),
            self.fc_type(out),
            self.fc_dur(out.detach()),
        )

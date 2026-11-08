"""FIGR-enhanced Traffic-Diff model.

Core differences from the previous implementation:
1. One pooled contextual stream instead of fixed src/dst/time context channels.
2. Flow Interaction Graph Reasoning (FIGR) over mini-batch flow nodes.
3. Curriculum packet masking and block-wise wavelet masking.
4. Graph-edge masking + relation recovery as an additional pretraining objective.
5. Graph-conditioned reconstruction decoder and graph-aware classification.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def make_group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def masked_mean(x: torch.Tensor, valid: torch.Tensor, dim: int = 1) -> torch.Tensor:
    weights = valid.to(x.dtype).unsqueeze(-1)
    return (x * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)


def _as_long(x, device):
    if x is None:
        return None
    return torch.as_tensor(x, dtype=torch.long, device=device).view(-1)


# -----------------------------------------------------------------------------
# Curriculum Masking
# -----------------------------------------------------------------------------
class CurriculumMasking(nn.Module):
    """Mask valid packet tokens and contiguous temporal blocks in the scalogram."""
    def __init__(self, start_ratio=0.15, end_ratio=0.75):
        super().__init__()
        self.start_ratio = float(start_ratio)
        self.end_ratio = float(end_ratio)

    def ratio_at(self, progress: float) -> float:
        p = float(max(0.0, min(1.0, progress)))
        return self.start_ratio + p * (self.end_ratio - self.start_ratio)

    @staticmethod
    def _mask_valid_positions(valid, ratio):
        """Return keep mask (1=visible, 0=masked); padding remains visible but invalid."""
        bsz, length = valid.shape
        keep = torch.ones((bsz, length), dtype=torch.float32, device=valid.device)
        for b in range(bsz):
            ids = torch.nonzero(valid[b], as_tuple=False).flatten()
            n = int(ids.numel())
            if n <= 1:
                continue
            n_mask = max(1, min(n - 1, int(round(n * ratio))))
            perm = torch.randperm(n, device=valid.device)[:n_mask]
            keep[b, ids[perm]] = 0.0
        return keep

    @staticmethod
    def _block_mask(batch, length, ratio, device):
        """Create one or more contiguous masked blocks on the context time axis."""
        keep = torch.ones((batch, length), dtype=torch.float32, device=device)
        if length <= 1 or ratio <= 0:
            return keep
        block_len = max(1, min(length - 1, int(round(length * ratio))))
        for b in range(batch):
            start = int(torch.randint(0, max(1, length - block_len + 1), (1,), device=device).item())
            keep[b, start:start + block_len] = 0.0
            # A small random component prevents the mask from always having one fixed shape.
            random_mask = torch.rand(length, device=device) < (ratio * 0.08)
            keep[b, random_mask] = 0.0
            if keep[b].sum() < 1:
                keep[b, torch.randint(0, length, (1,), device=device)] = 1.0
        return keep

    def forward(self, raw_bytes, wavelet, progress=0.0):
        ratio = self.ratio_at(progress)
        batch, packets, _ = raw_bytes.shape
        valid_packets = raw_bytes.abs().sum(dim=-1) > 1e-8
        packet_keep = self._mask_valid_positions(valid_packets, ratio)
        raw_masked = raw_bytes * packet_keep.unsqueeze(-1)

        # wavelet shape: (B, 1, S, M). Block mask acts on the M/time axis.
        wave_keep = self._block_mask(batch, wavelet.size(-1), ratio, wavelet.device)
        wave_mask_expanded = wave_keep[:, None, None, :].expand_as(wavelet)
        wave_masked = wavelet * wave_mask_expanded
        return raw_masked, wave_masked, packet_keep, wave_keep, ratio


# -----------------------------------------------------------------------------
# Temporal encoder: byte embedding -> multi-kernel CNN -> packet Transformer
# -----------------------------------------------------------------------------
class TemporalEncoder(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=3, packet_bytes=64, dropout=0.1):
        super().__init__()
        self.packet_bytes = int(packet_bytes)
        byte_dim = max(32, d_model // 4)
        self.byte_embed = nn.Embedding(256, byte_dim)
        self.byte_pos = nn.Parameter(torch.randn(1, 1, packet_bytes, byte_dim) * 0.02)
        self.byte_proj = nn.Linear(byte_dim, d_model)

        branch_dim = max(32, d_model // 2)
        self.conv3 = nn.Sequential(
            nn.Conv1d(d_model, branch_dim, kernel_size=3, padding=1),
            make_group_norm(branch_dim), nn.GELU(),
            nn.Conv1d(branch_dim, branch_dim, kernel_size=3, padding=1),
            make_group_norm(branch_dim), nn.GELU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(d_model, branch_dim, kernel_size=5, padding=2),
            make_group_norm(branch_dim), nn.GELU(),
            nn.Conv1d(branch_dim, branch_dim, kernel_size=3, padding=1),
            make_group_norm(branch_dim), nn.GELU(),
        )
        self.conv7 = nn.Sequential(
            nn.Conv1d(d_model, branch_dim, kernel_size=7, padding=3),
            make_group_norm(branch_dim), nn.GELU(),
            nn.Conv1d(branch_dim, branch_dim, kernel_size=3, padding=1),
            make_group_norm(branch_dim), nn.GELU(),
        )
        self.conv_fuse = nn.Sequential(
            nn.Conv1d(branch_dim * 3, d_model, kernel_size=1),
            make_group_norm(d_model), nn.GELU(),
        )
        self.byte_attention = nn.Sequential(
            nn.Linear(d_model, max(16, d_model // 4)), nn.GELU(), nn.Linear(max(16, d_model // 4), 1)
        )
        # avg, max, attention, header, transport, payload
        self.packet_fuse = nn.Sequential(
            nn.Linear(d_model * 6, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model),
        )
        self.packet_position = nn.Parameter(torch.randn(1, cfg.MAX_PACKETS, d_model) * 0.01)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))
        self.flow_attention = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.Tanh(), nn.Linear(d_model // 2, 1))

    @staticmethod
    def _segment_mean(tokens, start, end):
        if end <= start:
            return torch.zeros(tokens.size(0), tokens.size(-1), device=tokens.device, dtype=tokens.dtype)
        return tokens[:, start:end].mean(dim=1)

    def forward(self, raw_bytes, packet_keep=None, valid_mask=None):
        bsz, n_packets, n_bytes = raw_bytes.shape
        byte_ids = raw_bytes.clamp(0.0, 1.0).mul(255.0).round().long()
        x = self.byte_embed(byte_ids) + self.byte_pos[:, :, :n_bytes, :]
        x = self.byte_proj(x).reshape(bsz * n_packets, n_bytes, -1).transpose(1, 2)

        x = self.conv_fuse(torch.cat([self.conv3(x), self.conv5(x), self.conv7(x)], dim=1))
        byte_tokens = x.transpose(1, 2)
        avg_pool = byte_tokens.mean(dim=1)
        max_pool = byte_tokens.max(dim=1).values
        byte_alpha = F.softmax(self.byte_attention(byte_tokens), dim=1)
        attn_pool = (byte_tokens * byte_alpha).sum(dim=1)
        header_pool = self._segment_mean(byte_tokens, 0, min(20, n_bytes))
        transport_pool = self._segment_mean(byte_tokens, min(20, n_bytes), min(40, n_bytes))
        payload_pool = self._segment_mean(byte_tokens, min(40, n_bytes), n_bytes)
        packet_repr = self.packet_fuse(torch.cat(
            [avg_pool, max_pool, attn_pool, header_pool, transport_pool, payload_pool], dim=-1
        )).reshape(bsz, n_packets, -1)

        if valid_mask is None:
            valid_mask = raw_bytes.abs().sum(dim=-1) > 1e-8
        valid_mask = valid_mask.bool()
        if packet_keep is not None:
            masked = packet_keep < 0.5
            packet_repr = torch.where(masked.unsqueeze(-1), self.mask_token.expand(bsz, n_packets, -1), packet_repr)

        packet_repr = packet_repr + self.packet_position[:, :n_packets, :]
        # Flow extraction guarantees >= MIN_PACKETS; safeguard malformed data anyway.
        padding_mask = ~valid_mask
        all_invalid = padding_mask.all(dim=1)
        if all_invalid.any():
            padding_mask = padding_mask.clone()
            padding_mask[all_invalid, 0] = False
        tokens = self.transformer(packet_repr, src_key_padding_mask=padding_mask)
        tokens = tokens.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        scores = self.flow_attention(tokens).squeeze(-1).masked_fill(~valid_mask, -1e4)
        alpha = F.softmax(scores, dim=1) * valid_mask.to(tokens.dtype)
        alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-8)
        flow = (tokens * alpha.unsqueeze(-1)).sum(dim=1)
        return tokens, flow


# -----------------------------------------------------------------------------
# Wavelet encoder: 2-D spectral CNN -> scale pooling -> temporal Transformer
# -----------------------------------------------------------------------------
class WaveletEncoder(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), make_group_norm(32), nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=5, padding=2), make_group_norm(64), nn.GELU(),
            nn.Conv2d(64, d_model, kernel_size=3, padding=1), make_group_norm(d_model), nn.GELU(),
        )
        self.scale_fuse = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.position = nn.Parameter(torch.randn(1, cfg.MAX_PACKETS, d_model) * 0.01)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=max(1, num_layers), norm=nn.LayerNorm(d_model))
        self.flow_attention = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.Tanh(), nn.Linear(d_model // 2, 1))

    def forward(self, wavelet, wave_keep=None, valid_mask=None):
        # Input: (B, 1, scales, packet/time points)
        bsz, _channels, _scales, n_time = wavelet.shape
        x = self.cnn(wavelet)
        x_avg = x.mean(dim=2).transpose(1, 2)     # (B, M, D)
        x_max = x.max(dim=2).values.transpose(1, 2)
        tokens = self.scale_fuse(torch.cat([x_avg, x_max], dim=-1))

        if valid_mask is None:
            valid_mask = wavelet.abs().sum(dim=(1, 2)) > 1e-8
        valid_mask = valid_mask.bool()
        if wave_keep is not None:
            mask = (wave_keep < 0.5) & valid_mask
            tokens = torch.where(mask.unsqueeze(-1), self.mask_token.expand(bsz, n_time, -1), tokens)

        tokens = tokens + self.position[:, :n_time, :]
        padding_mask = ~valid_mask
        all_invalid = padding_mask.all(dim=1)
        if all_invalid.any():
            padding_mask = padding_mask.clone()
            padding_mask[all_invalid, 0] = False
        tokens = self.transformer(tokens, src_key_padding_mask=padding_mask)
        tokens = tokens.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        scores = self.flow_attention(tokens).squeeze(-1).masked_fill(~valid_mask, -1e4)
        alpha = F.softmax(scores, dim=1) * valid_mask.to(tokens.dtype)
        alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-8)
        flow = (tokens * alpha.unsqueeze(-1)).sum(dim=1)
        return tokens, flow


# -----------------------------------------------------------------------------
# Flow Interaction Graph Reasoning (FIGR)
# -----------------------------------------------------------------------------
def build_graph_relations(graph_meta, batch_size, device, k_neighbors=2):
    """Build two label-independent relations inside a mini-batch.

    Relation 0: endpoint interaction (two flows share any endpoint IP).
    Relation 1: temporal-profile similarity (each node is linked to its nearest
                peers by log flow duration; adjacency is symmetrized).

    Capture IDs and labels are deliberately NOT used to construct edges. In this
    dataset convention, one PCAP may correspond to one class, so scoping graph edges
    by capture_id would make topology strongly label-correlated. Absolute packet
    timestamps are also not compared across independent captures; only each flow's
    duration (end_time - start_time) is used as a capture-clock-independent profile.

    graph_meta tuple: (src_id, dst_id, start_time, end_time, capture_id).
    """
    if graph_meta is None:
        src = torch.arange(batch_size, device=device)
        dst = src + batch_size + 1
        start = torch.arange(batch_size, device=device, dtype=torch.float32)
        end = start + 0.1
        capture = torch.zeros(batch_size, device=device, dtype=torch.long)
    else:
        src, dst, start, end, capture = graph_meta
        src = _as_long(src, device)
        dst = _as_long(dst, device)
        start = torch.as_tensor(start, dtype=torch.float32, device=device).view(-1)
        end = torch.as_tensor(end, dtype=torch.float32, device=device).view(-1)
        capture = _as_long(capture, device)  # retained for metadata compatibility only

    not_self = ~torch.eye(batch_size, dtype=torch.bool, device=device)
    shared_endpoint = (
        src[:, None].eq(src[None, :]) |
        src[:, None].eq(dst[None, :]) |
        dst[:, None].eq(src[None, :]) |
        dst[:, None].eq(dst[None, :])
    )
    endpoint_adj = not_self & shared_endpoint

    duration = (end - start).clamp_min(0.0)
    duration_profile = torch.log1p(duration)
    duration_gap = torch.abs(duration_profile[:, None] - duration_profile[None, :])
    temporal_adj = torch.zeros((batch_size, batch_size), dtype=torch.bool, device=device)

    # Sparse k-nearest duration-profile neighbors avoid an almost-complete graph when
    # all observed flows happen to have short durations.
    k = min(int(k_neighbors), max(0, batch_size - 1))
    if k > 0:
        distance = duration_gap.masked_fill(~not_self, float("inf"))
        nearest = torch.topk(distance, k=k, dim=1, largest=False).indices
        for row in range(batch_size):
            temporal_adj[row, nearest[row]] = True
    temporal_adj = temporal_adj | temporal_adj.transpose(0, 1)
    targets = torch.stack([endpoint_adj, temporal_adj], dim=-1).to(torch.float32)
    pair_mask = not_self
    parsed_meta = (src, dst, start, end, capture)
    return targets, pair_mask, parsed_meta

class RelationGraphLayer(nn.Module):
    """Relation-specific multi-head message passing with residual updates."""
    def __init__(self, d_model, nhead=4, num_relations=2, dropout=0.1):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"D_MODEL ({d_model}) must be divisible by graph heads ({nhead}).")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.num_relations = num_relations
        self.q = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_relations)])
        self.k = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_relations)])
        self.v = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_relations)])
        self.out = nn.Linear(d_model * num_relations, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adjacency):
        bsz = x.size(0)
        eye = torch.eye(bsz, dtype=torch.bool, device=x.device)
        messages = []
        for rel in range(self.num_relations):
            adj = adjacency[:, :, rel].bool() | eye
            q = self.q[rel](x).view(bsz, self.nhead, self.head_dim)
            k = self.k[rel](x).view(bsz, self.nhead, self.head_dim)
            v = self.v[rel](x).view(bsz, self.nhead, self.head_dim)
            scores = torch.einsum("ihd,jhd->hij", q, k) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(~adj.unsqueeze(0), -1e4)
            attn = F.softmax(scores, dim=-1)
            msg = torch.einsum("hij,jhd->ihd", attn, v).reshape(bsz, self.d_model)
            messages.append(msg)
        update = self.out(torch.cat(messages, dim=-1))
        x = self.norm1(x + self.dropout(update))
        x = self.norm2(x + self.ffn(x))
        return x


class FlowInteractionGraphReasoner(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            RelationGraphLayer(d_model, nhead=nhead, num_relations=2, dropout=dropout)
            for _ in range(num_layers)
        ])
        # Symmetric pair descriptor for relation recovery.
        pair_dim = d_model * 3
        self.relation_head = nn.Sequential(
            nn.Linear(pair_dim, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 2),
        )

    def forward(self, x, graph_meta=None, edge_mask_ratio=0.0):
        bsz = x.size(0)
        targets, pair_mask, parsed_meta = build_graph_relations(
            graph_meta, bsz, x.device,
            k_neighbors=cfg.GRAPH_K_NEIGHBORS,
        )
        src, dst, start, end, _capture = parsed_meta
        # Predict relations from pre-message-passing traffic representations only;
        # do not expose capture identifiers, labels, or absolute capture timestamps.
        xi = x[:, None, :].expand(bsz, bsz, -1)
        xj = x[None, :, :].expand(bsz, bsz, -1)
        pair_features = torch.cat([xi + xj, torch.abs(xi - xj), xi * xj], dim=-1)
        relation_logits = self.relation_head(pair_features)

        adjacency = targets.bool()
        ratio = float(max(0.0, min(0.95, edge_mask_ratio)))
        if self.training and ratio > 0.0:
            masked_relations = []
            for rel in range(targets.size(-1)):
                random_keep = torch.rand((bsz, bsz), device=x.device) >= ratio
                random_keep = torch.triu(random_keep, diagonal=1)
                random_keep = random_keep | random_keep.transpose(0, 1)
                masked_relations.append(adjacency[:, :, rel] & random_keep)
            adjacency = torch.stack(masked_relations, dim=-1)

        h = x
        for layer in self.layers:
            h = layer(h, adjacency)
        return h, relation_logits, targets, pair_mask


# -----------------------------------------------------------------------------
# Traffic-Diff
# -----------------------------------------------------------------------------
class TrafficDiff(nn.Module):
    def __init__(self, d_model=128, n_classes=10, packet_bytes=64):
        super().__init__()
        self.d_model = int(d_model)
        self.masking = CurriculumMasking(cfg.MASK_RATIO_START, cfg.MASK_RATIO_END)
        self.temporal_enc = TemporalEncoder(
            d_model=d_model, nhead=cfg.NHEAD, num_layers=cfg.NUM_LAYERS,
            packet_bytes=packet_bytes, dropout=cfg.DROPOUT,
        )
        self.wavelet_enc = WaveletEncoder(
            d_model=d_model, nhead=cfg.NHEAD,
            num_layers=max(1, cfg.NUM_LAYERS // 2), dropout=cfg.DROPOUT,
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads=cfg.NHEAD, batch_first=True, dropout=cfg.DROPOUT
        )
        self.cross_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Linear(d_model, d_model), nn.Sigmoid()
        )
        self.fusion_norm = nn.LayerNorm(d_model)
        self.fused_attention = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.Tanh(), nn.Linear(d_model // 2, 1))

        self.graph_input = nn.Sequential(nn.Linear(d_model * 3, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.figr = FlowInteractionGraphReasoner(
            d_model=d_model, nhead=max(1, min(4, cfg.NHEAD)), num_layers=2, dropout=cfg.DROPOUT
        )
        self.graph_conditioner = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.LayerNorm(d_model), nn.Dropout(cfg.DROPOUT)
        )

        # Dual-domain reconstruction heads.
        self.byte_recon = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, packet_bytes), nn.Sigmoid())
        self.wavelet_recon = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, cfg.SCALES), nn.Sigmoid())

        # Graph-aware six-way flow representation.
        feature_dim = d_model * 6
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, d_model * 2), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(d_model, n_classes),
        )
        self.aux_temporal = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, n_classes))
        self.aux_wavelet = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, n_classes))
        self.aux_graph = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, n_classes))

    def _cross_fuse(self, temporal_tokens, wavelet_tokens, temporal_valid, wavelet_valid):
        bsz = temporal_tokens.size(0)
        key_padding = ~wavelet_valid.bool()
        no_context = key_padding.all(dim=1)
        if no_context.any():
            key_padding = key_padding.clone()
            key_padding[no_context, 0] = False
        cross, _ = self.cross_attn(
            temporal_tokens, wavelet_tokens, wavelet_tokens,
            key_padding_mask=key_padding, need_weights=False,
        )
        has_context = wavelet_valid.any(dim=1).view(bsz, 1, 1).to(cross.dtype)
        cross = cross * has_context
        gate = self.cross_gate(torch.cat([temporal_tokens, cross], dim=-1))
        fused = self.fusion_norm(temporal_tokens + gate * cross)
        return fused.masked_fill(~temporal_valid.unsqueeze(-1), 0.0)

    def _pool(self, tokens, valid):
        scores = self.fused_attention(tokens).squeeze(-1).masked_fill(~valid, -1e4)
        alpha = F.softmax(scores, dim=1) * valid.to(tokens.dtype)
        alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return (tokens * alpha.unsqueeze(-1)).sum(dim=1)

    def _encode(self, raw_bytes, wavelet, graph_meta=None, progress=0.0, pretrain=False):
        temporal_valid = raw_bytes.abs().sum(dim=-1) > 1e-8
        if pretrain:
            raw_in, wave_in, packet_keep, wave_keep, current_ratio = self.masking(raw_bytes, wavelet, progress)
        else:
            raw_in, wave_in = raw_bytes, wavelet
            packet_keep, wave_keep, current_ratio = None, None, 0.0

        temporal_tokens, flow_t = self.temporal_enc(
            raw_in, packet_keep=packet_keep, valid_mask=temporal_valid
        )
        # Context validity derives from the uncorrupted target, not the masked view.
        wave_valid = wavelet[:, 0].abs().sum(dim=1) > 1e-8
        wave_tokens, flow_w = self.wavelet_enc(
            wave_in, wave_keep=wave_keep, valid_mask=wave_valid
        )
        fused_tokens = self._cross_fuse(temporal_tokens, wave_tokens, temporal_valid, wave_valid)
        flow_fused = self._pool(fused_tokens, temporal_valid)

        graph_base = self.graph_input(torch.cat([flow_t, flow_w, flow_fused], dim=-1))
        edge_ratio = 0.0
        if pretrain:
            p = float(max(0.0, min(1.0, progress)))
            edge_ratio = cfg.GRAPH_MASK_START + p * (cfg.GRAPH_MASK_END - cfg.GRAPH_MASK_START)
        flow_graph, relation_logits, relation_targets, relation_pair_mask = self.figr(
            graph_base, graph_meta=graph_meta, edge_mask_ratio=edge_ratio
        )

        # The graph embedding conditions each token before reconstructing either domain.
        graph_context = flow_graph.unsqueeze(1).expand(-1, fused_tokens.size(1), -1)
        denoise_tokens = self.graph_conditioner(torch.cat([fused_tokens, graph_context], dim=-1))
        denoise_tokens = denoise_tokens.masked_fill(~temporal_valid.unsqueeze(-1), 0.0)

        features = torch.cat([
            flow_t,
            flow_w,
            flow_fused,
            flow_graph,
            torch.abs(flow_t - flow_w),
            flow_t * flow_graph,
        ], dim=-1)
        return {
            "temporal_tokens": temporal_tokens,
            "wavelet_tokens": wave_tokens,
            "fused_tokens": fused_tokens,
            "denoise_tokens": denoise_tokens,
            "flow_t": flow_t,
            "flow_w": flow_w,
            "flow_fused": flow_fused,
            "flow_graph": flow_graph,
            "features": features,
            "packet_keep": packet_keep,
            "wave_keep": wave_keep,
            "current_mask_ratio": current_ratio,
            "relation_logits": relation_logits,
            "relation_targets": relation_targets,
            "relation_pair_mask": relation_pair_mask,
            "temporal_valid": temporal_valid,
        }

    def forward(self, raw_bytes, wavelet, mode="pretrain", graph_meta=None, progress=0.0):
        if wavelet.dim() == 3:
            wavelet = wavelet.unsqueeze(1)
        if mode == "pretrain":
            encoded = self._encode(raw_bytes, wavelet, graph_meta=graph_meta, progress=progress, pretrain=True)
            pred_bytes = self.byte_recon(encoded["denoise_tokens"])
            pred_wavelet = self.wavelet_recon(encoded["denoise_tokens"])
            return {
                "pred_bytes": pred_bytes,
                "pred_wavelet": pred_wavelet,
                "packet_keep": encoded["packet_keep"],
                "wave_keep": encoded["wave_keep"],
                "relation_logits": encoded["relation_logits"],
                "relation_targets": encoded["relation_targets"],
                "relation_pair_mask": encoded["relation_pair_mask"],
                "mask_ratio": encoded["current_mask_ratio"],
            }

        encoded = self._encode(raw_bytes, wavelet, graph_meta=graph_meta, progress=1.0, pretrain=False)
        if mode == "feature":
            return encoded["features"]
        features = self.feature_norm(encoded["features"])
        logits = self.classifier(features)
        aux_t = self.aux_temporal(encoded["flow_t"])
        aux_w = self.aux_wavelet(encoded["flow_w"])
        aux_g = self.aux_graph(encoded["flow_graph"])
        return logits, aux_t, aux_w, aux_g


def pretrain_loss(output, gt_bytes, gt_wavelet, cfg_obj=cfg):
    """Masked dual-domain reconstruction + relation recovery."""
    pred_bytes = output["pred_bytes"]
    pred_wave = output["pred_wavelet"]
    packet_keep = output["packet_keep"]
    wave_keep = output["wave_keep"]

    valid_packet = gt_bytes.abs().sum(dim=-1) > 1e-8
    packet_mask = (1.0 - packet_keep) * valid_packet.to(packet_keep.dtype)
    if packet_mask.sum() < 1:
        packet_mask = valid_packet.to(packet_keep.dtype)
    byte_err = (pred_bytes - gt_bytes).pow(2) * packet_mask.unsqueeze(-1)
    byte_loss = byte_err.sum() / (packet_mask.sum() * gt_bytes.size(-1)).clamp_min(1.0)

    # pred_wave: (B, M, S); target wave: (B, 1, S, M) -> (B, M, S)
    target_wave = gt_wavelet[:, 0].transpose(1, 2)
    wave_mask = (1.0 - wave_keep).unsqueeze(-1).expand_as(target_wave)
    if wave_mask.sum() < 1:
        wave_mask = torch.ones_like(target_wave)
    wave_err = (pred_wave - target_wave).pow(2) * wave_mask
    wave_loss = wave_err.sum() / wave_mask.sum().clamp_min(1.0)

    logits = output["relation_logits"]
    targets = output["relation_targets"]
    pair_mask = output["relation_pair_mask"]
    relation_losses = []
    for rel in range(targets.size(-1)):
        valid_logits = logits[:, :, rel][pair_mask]
        valid_targets = targets[:, :, rel][pair_mask]
        if valid_targets.numel() == 0:
            continue
        positives = valid_targets.sum()
        negatives = valid_targets.numel() - positives
        pos_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, 20.0)
        relation_losses.append(F.binary_cross_entropy_with_logits(
            valid_logits, valid_targets, pos_weight=pos_weight
        ))
    relation_loss = torch.stack(relation_losses).mean() if relation_losses else pred_bytes.new_zeros(())

    total = byte_loss + cfg_obj.WAVELET_LOSS_WEIGHT * wave_loss + cfg_obj.GRAPH_RELATION_LOSS_WEIGHT * relation_loss
    return total


def finetune_loss(logits, aux_t, aux_w, aux_g, labels, class_weights=None):
    ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.0)
    main = ce(logits, labels)
    return main + cfg.AUX_LOSS_WEIGHT * (ce(aux_t, labels) + ce(aux_w, labels) + ce(aux_g, labels))

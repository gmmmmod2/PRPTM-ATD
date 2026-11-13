import math

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


def sequence_mask(x, mask=None):
    if mask is None:
        mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
    mask = mask.to(device=x.device, dtype=torch.bool)
    if mask.shape != x.shape[:2]:
        raise ValueError('seq_mask must have shape [B,L]')
    if not mask.any(1).all() or ((~mask[:, :-1]) & mask[:, 1:]).any():
        raise ValueError('Each trajectory must be nonempty and right-padded')
    return mask


class PeriodicEmbedding(nn.Module):
    def __init__(self, T, embedding_dim, sigma=1.0):
        super().__init__()
        if T <= 0 or T % 2:
            raise ValueError('T must be a positive even integer')
        self.c = nn.Parameter(torch.randn(1, T // 2) * sigma)
        self.linear = nn.Linear(T, embedding_dim)

    def forward(self, x):
        phases = 2 * math.pi * self.c * x.to(self.c.dtype)
        return self.linear(torch.cat((phases.sin(), phases.cos()), dim=-1))


class POIBERTEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, padding_idx=0):
        super().__init__()
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)

    def _mean(self, ids):
        mask = ids.ne(self.padding_idx).unsqueeze(-1)
        return (self.embedding(ids) * mask).sum(-2) / mask.sum(-2).clamp_min(1)

    def forward(self, poi_ids, neighbor_ids):
        return self._mean(poi_ids), self._mean(neighbor_ids)


class RiskEmbedding(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, dropout=0.1,
                 is_double=False, use_spatial=True):
        super().__init__()
        self.is_double = is_double
        self.use_spatial = use_spatial
        self.text_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2), nn.ReLU(),
            nn.Linear(input_dim // 2, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.time_proj = PeriodicEmbedding(hidden_dim, hidden_dim)
        self.spatial_proj = nn.Linear(3, hidden_dim) if use_spatial else None
        self.fusion = nn.Sequential(
            nn.Linear((3 if use_spatial else 2) * hidden_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout))
        self.incident_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1))

    def forward(self, text_seq, time_seq, spatial=None):
        if text_seq.ndim == 4 and time_seq.ndim == 3:
            time_seq = time_seq.unsqueeze(2).expand(*text_seq.shape[:-1], 1)
        features = [self.text_proj(text_seq)]
        if self.use_spatial:
            if spatial is None or spatial.shape != (*text_seq.shape[:-1], 3):
                raise ValueError('Provide aligned spatial coordinates [...,3] for Eq.(4)')
            features.append(self.spatial_proj(spatial.to(text_seq.dtype)))
        features.append(self.time_proj(time_seq))
        return self.fusion(torch.cat(features, -1))

    def predict_incident(self, aggregated_risk):
        return self.incident_head(aggregated_risk) if self.is_double else None


class GCNApproxFusion(nn.Module):
    def __init__(self, in_dim=512, out_dim=512, use_neighbors=True):
        super().__init__()
        self.linear_self = nn.Linear(in_dim, out_dim)
        self.linear_neigh = nn.Linear(in_dim, out_dim)
        self.activation = nn.ReLU()
        self.use_neighbors = use_neighbors

    def forward(self, self_feat, neighbor_feat, neighbor_mask=None):
        if not self.use_neighbors:
            # 保持原消融语义：关闭邻域传播，直接使用初始风险。
            if self_feat.shape[-1] != self.linear_self.out_features:
                raise ValueError('Bypass requires equal input/output dimensions')
            return self_feat
        if neighbor_mask is None:
            neighbor_mask = torch.ones(neighbor_feat.shape[:-1],
                                       device=neighbor_feat.device, dtype=torch.bool)
        mask = neighbor_mask.unsqueeze(-1).to(neighbor_feat.dtype)
        count = mask.sum(-2)
        mean = neighbor_feat.masked_fill(~neighbor_mask.unsqueeze(-1), 0).sum(-2) / count.clamp_min(1)
        # 无真实邻居时，邻居分支连同 bias 一起移除。
        neighbor_term = self.linear_neigh(mean) * count.gt(0)
        return self.activation(self.linear_self(self_feat) + neighbor_term)


class RiskExposure(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=512, dropout=0.1):
        super().__init__()
        if hidden_dim % 2:
            raise ValueError('hidden_dim must be even for BiLSTM')
        self.time_proj = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False), nn.Dropout(dropout))
        self.fusion = nn.Sequential(nn.Linear(in_dim + hidden_dim, hidden_dim),
            nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.bilstm = nn.LSTM(hidden_dim, hidden_dim // 2, batch_first=True,
                              bidirectional=True)

    def forward(self, risk, staytime, seq_mask=None):
        mask = sequence_mask(risk, seq_mask)
        x = self.fusion(torch.cat((risk, self.time_proj(staytime)), -1))
        packed = pack_padded_sequence(x, mask.sum(1).cpu(), batch_first=True,
                                      enforce_sorted=False)
        packed, _ = self.bilstm(packed)
        return pad_packed_sequence(packed, batch_first=True, total_length=x.size(1))[0]


class TimePatchLSTM(nn.Module):
    def __init__(self, patch_size, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        if patch_size <= 0:
            raise ValueError('patch_size must be positive')
        self.patch_size = patch_size
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, seq_mask=None, return_mask=False):
        mask = sequence_mask(x, seq_mask)
        b, length, channels = x.shape
        p = self.patch_size
        pad = (-length) % p
        x = F.pad(x, (0, 0, 0, pad))
        mask = F.pad(mask, (0, pad), value=False)
        n = x.size(1) // p
        patches = x.reshape(b * n, p, channels)
        lengths = mask.reshape(b * n, p).sum(1)
        valid = lengths.gt(0)
        packed = pack_padded_sequence(patches[valid], lengths[valid].cpu(),
                                      batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        encoded = self.proj(h[-1])
        tokens = encoded.new_zeros(b * n, channels)
        tokens[valid] = encoded
        tokens = tokens.reshape(b, n, channels)
        return (tokens, valid.reshape(b, n)) if return_mask else tokens


class TransformerEncoder(nn.Module):
    def __init__(self, dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(dim, num_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)

    def forward(self, x, mask=None):
        return self.encoder(x, src_key_padding_mask=None if mask is None else ~mask)


class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x, mask=None):
        scores = self.attn(x)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(-1), -torch.inf)
        return (x * scores.softmax(1)).sum(1)


def risk_entropy(p_key):
    if not torch.isfinite(p_key).all() or ((p_key < 0) | (p_key > 1)).any():
        raise ValueError('p_key must be finite and in [0,1]')
    p = p_key.float()
    tiny = torch.finfo(p.dtype).tiny
    return -p * p.clamp_min(tiny).log() - (1-p) * (1-p).clamp_min(tiny).log()


class JointLoss(nn.Module):
    def __init__(self, alpha=0.6, gamma=2.0, lambda_entropy=0.5):
        super().__init__()
        if not 0 <= alpha <= 1 or gamma < 0 or lambda_entropy < 0:
            raise ValueError('Invalid loss hyperparameters')
        self.alpha, self.gamma, self.lambda_entropy = alpha, gamma, lambda_entropy

    def forward(self, outputs, labels, entropy_target=None, seq_mask=None):
        logp = outputs['logits'].log_softmax(-1)
        labels = labels.long().reshape(-1)
        logpt = logp.gather(1, labels[:, None]).squeeze(1)
        alpha = torch.where(labels.eq(1), self.alpha, 1-self.alpha)
        focal = (-alpha * (1-logpt.exp()).pow(self.gamma) * logpt).mean()
        mse = focal.new_zeros(())
        pred = outputs.get('incident_pred_center')
        if pred is not None and self.lambda_entropy > 0:
            if entropy_target is None:
                raise ValueError('Provide risk entropy target; legacy incident is not automatically entropy')
            if entropy_target.ndim == 2:
                entropy_target = entropy_target.unsqueeze(-1)
            if entropy_target.shape != pred.shape:
                raise ValueError('entropy_target shape must match [B,L,1]')
            if seq_mask is None:
                seq_mask = outputs.get('seq_mask')
            mask = sequence_mask(pred, seq_mask).unsqueeze(-1)
            target = entropy_target[mask]
            if not torch.isfinite(target).all() or ((target < 0) | (target > math.log(2)+1e-6)).any():
                raise ValueError('Entropy target must lie in [0,ln(2)]')
            mse = (pred[mask] - target).square().mean()
        return {'loss': focal + self.lambda_entropy * mse,
                'focal_loss': focal, 'entropy_loss': mse}

import torch
from torch import nn

from . import Componet as _components
JointLoss = _components.JointLoss
risk_entropy = _components.risk_entropy


class PRPTMATD(nn.Module):
    def __init__(self, *, vocab_size=700, input_dim=768, hidden_dim=512,
                 patch_sizes=(10, 20, 30), num_layers=1, num_heads=8, n_mlp=3,
                 num_layers_intra=2, num_layers_inter=2, dropout=0.1,
                 is_double=True, use_neighbors=True, use_spatial=True):
        super().__init__()
        if not patch_sizes or any(p <= 0 for p in patch_sizes):
            raise ValueError('Provide positive patch sizes')
        if hidden_dim % 2 or hidden_dim % num_heads:
            raise ValueError('hidden_dim must be even and divisible by num_heads')
        self.patch_sizes = tuple(patch_sizes)
        self.num_scales = len(patch_sizes)
        self.is_double = is_double
        self.use_neighbors = use_neighbors
        self.encoder = _components.POIBERTEncoder(vocab_size, input_dim)
        self.riskEmbedding = _components.RiskEmbedding(
            input_dim, hidden_dim, dropout, is_double, use_spatial)
        self.gcn = _components.GCNApproxFusion(hidden_dim, hidden_dim, use_neighbors)
        self.exp = _components.RiskExposure(hidden_dim, hidden_dim, dropout)
        self.patch_extractors = nn.ModuleList([
            _components.TimePatchLSTM(p, hidden_dim, hidden_dim, num_layers) for p in patch_sizes])
        self.intra_transformer = _components.TransformerEncoder(
            hidden_dim, num_heads, num_layers_intra, dropout)
        self.scale_emb = nn.Parameter(torch.randn(self.num_scales, hidden_dim))
        self.inter_transformer = _components.TransformerEncoder(
            hidden_dim, num_heads, num_layers_inter, dropout)
        self.pool = _components.AttentionPool(hidden_dim)
        self.mlp = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout)) for _ in range(n_mlp)])
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, data):
        poi, neighbors = self.encoder(data['poi_ids'], data['neighbor_ids'])
        hour = data['hour']
        stay = data['stay_duration']
        if hour.ndim == 2:
            hour = hour.unsqueeze(-1)
        if stay.ndim == 2:
            stay = stay.unsqueeze(-1)
        mask = _components.sequence_mask(poi, data.get('seq_mask', hour.squeeze(-1).ge(0)))
        hour = hour.masked_fill(~mask.unsqueeze(-1), 0)
        stay = stay.to(poi.dtype).masked_fill(~mask.unsqueeze(-1), 0)
        spatial = data.get('spatial')
        if spatial is not None:
            spatial = spatial.masked_fill(~mask.unsqueeze(-1), 0)
        point_risk = self.riskEmbedding(poi, hour, spatial)
        if self.use_neighbors:
            neighbor_mask = data.get('neighbor_mask', data['neighbor_ids'].ne(0).any(-1))
            neighbor_mask = neighbor_mask.bool() & mask.unsqueeze(-1)
            neighbor_spatial = data.get('neighbor_spatial')
            if neighbor_spatial is not None:
                neighbor_spatial = neighbor_spatial.masked_fill(~neighbor_mask.unsqueeze(-1), 0)
            neighbor_risk = self.riskEmbedding(neighbors, hour, neighbor_spatial)
            grid_risk = self.gcn(point_risk, neighbor_risk, neighbor_mask)
        else:
            grid_risk = self.gcn(point_risk, None)
        grid_risk = grid_risk.masked_fill(~mask.unsqueeze(-1), 0)
        incident = self.riskEmbedding.predict_incident(grid_risk)
        if incident is not None:
            incident = incident.masked_fill(~mask.unsqueeze(-1), 0)
        exposure = self.exp(grid_risk, stay, mask)
        all_tokens, all_masks = [], []
        for i, extractor in enumerate(self.patch_extractors):
            tokens, patch_mask = extractor(exposure, mask, return_mask=True)
            tokens = self.intra_transformer(tokens, patch_mask)
            all_tokens.append(tokens + self.scale_emb[i])
            all_masks.append(patch_mask)
        token_mask = torch.cat(all_masks, 1)
        fused = self.inter_transformer(torch.cat(all_tokens, 1), token_mask)
        pooled = self.pool(fused, token_mask)
        for layer in self.mlp:
            pooled = layer(pooled)
        return {'logits': self.classifier(pooled), 'incident_pred_center': incident,
                'incident_pred_neighbors': None, 'seq_mask': mask}

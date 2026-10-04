from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any, Dict, Optional

import torch
from torch import nn


MODALITIES = ("T", "A", "V")


class SinusoidalPosition(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        table = torch.zeros(max_len, d_model)
        table[:, 0::2] = torch.sin(position * div)
        table[:, 1::2] = torch.cos(position * div[: table[:, 1::2].shape[1]])
        self.register_buffer("table", table.unsqueeze(0), persistent=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.table[:, : value.shape[1]]


def projection(input_dim: int, d_model: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, d_model),
        nn.GELU(),
        nn.LayerNorm(d_model),
        nn.Dropout(dropout),
    )


class SharedCompensator(nn.Module):
    """One shared cross-attention layer for all target modalities/times."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        ffn_dim: int,
        dropout: float,
        max_len: int,
    ) -> None:
        super().__init__()
        self.modality_embedding = nn.Embedding(3, d_model)
        self.time_embedding = nn.Embedding(max_len, d_model)
        self.attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.null_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.null_token, std=0.02)

    def forward(
        self, projected: torch.Tensor, effective_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, modalities, dim = projected.shape
        device = projected.device
        time_ids = torch.arange(steps, device=device)
        modality_ids = torch.arange(modalities, device=device)
        context_embedding = (
            self.time_embedding(time_ids)[None, :, None, :]
            + self.modality_embedding(modality_ids)[None, None, :, :]
        )
        query = context_embedding.expand(batch, -1, -1, -1).reshape(
            batch, steps * modalities, dim
        )
        source = (projected + context_embedding).reshape(batch, steps * modalities, dim)
        source_valid = effective_mask.reshape(batch, steps * modalities)
        any_source = source_valid.any(dim=1)
        null = self.null_token.expand(batch, -1, -1)
        key_value = torch.cat([source, null], dim=1)
        # NULL is visible only when the sample has no legal K/V positions.
        null_padding = any_source.unsqueeze(1)
        key_padding_mask = torch.cat([~source_valid, null_padding], dim=1)
        attended, _ = self.attention(
            query,
            key_value,
            key_value,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        hidden = self.norm1(query + attended)
        reconstructed = self.norm2(hidden + self.ffn(hidden)).reshape(
            batch, steps, modalities, dim
        )
        return reconstructed, ~any_source


class MissingAwareGate(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.modality_embedding = nn.Embedding(3, d_model)
        self.network = nn.Sequential(
            nn.Linear(2 * d_model + 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        completed: torch.Tensor,
        effective_mask: torch.Tensor,
        applicable_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, modalities, _ = completed.shape
        modality_ids = torch.arange(modalities, device=completed.device)
        modality_features = self.modality_embedding(modality_ids)[None, None].expand(
            batch, steps, -1, -1
        )
        missing_features = effective_mask.to(completed.dtype).unsqueeze(2).expand(
            -1, -1, modalities, -1
        )
        features = torch.cat([completed, missing_features, modality_features], dim=-1)
        logits = self.network(features).squeeze(-1)
        has_candidate = applicable_mask.any(dim=-1, keepdim=True)
        safe_logits = logits.masked_fill(~applicable_mask, -torch.inf)
        safe_logits = torch.where(has_candidate, safe_logits, torch.zeros_like(safe_logits))
        weights = torch.softmax(safe_logits, dim=-1)
        weights = torch.where(has_candidate, weights, torch.zeros_like(weights))
        return logits, weights


class UnifiedSentimentModel(nn.Module):
    def __init__(
        self,
        pretrained_name: str = "bert-base-uncased",
        freeze_bert: bool = True,
        bert_unfreeze_last_n_layers: int = 0,
        local_files_only: bool = False,
        text_dim: int = 768,
        audio_dim: int = 74,
        vision_dim: int = 35,
        d_model: int = 128,
        nhead: int = 4,
        compensation_ffn: int = 256,
        transformer_layers: int = 2,
        transformer_ffn: int = 256,
        dropout: float = 0.1,
        num_classes: int = 3,
        max_len: int = 50,
        bounded_regression: bool = False,
        text_encoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        if text_encoder is None:
            from transformers import AutoModel

            text_encoder = AutoModel.from_pretrained(
                pretrained_name, local_files_only=local_files_only
            )
        self.text_encoder = text_encoder
        self.pretrained_name = pretrained_name
        self._frozen_text_modules: list[nn.Module] = []
        self.bert_unfreeze_last_n_layers = int(bert_unfreeze_last_n_layers)
        if self.bert_unfreeze_last_n_layers < 0:
            raise ValueError("bert_unfreeze_last_n_layers must be non-negative")
        self.freeze_bert = bool(freeze_bert)
        self.bounded_regression = bounded_regression
        if self.bert_unfreeze_last_n_layers > 0:
            for parameter in self.text_encoder.parameters():
                parameter.requires_grad_(False)
            encoder = getattr(self.text_encoder, "encoder", None)
            layers = getattr(encoder, "layer", None)
            if layers is None:
                raise ValueError(
                    "Partial BERT fine-tuning requires text_encoder.encoder.layer"
                )
            if self.bert_unfreeze_last_n_layers > len(layers):
                raise ValueError(
                    "bert_unfreeze_last_n_layers exceeds encoder layer count"
                )
            for layer in layers[-self.bert_unfreeze_last_n_layers :]:
                for parameter in layer.parameters():
                    parameter.requires_grad_(True)
            self._frozen_text_modules.extend(
                layers[: -self.bert_unfreeze_last_n_layers]
            )
            embeddings = getattr(self.text_encoder, "embeddings", None)
            if embeddings is not None:
                self._frozen_text_modules.append(embeddings)
            pooler = getattr(self.text_encoder, "pooler", None)
            if pooler is not None:
                self._frozen_text_modules.append(pooler)
            self.freeze_bert = False
        elif self.freeze_bert:
            for parameter in self.text_encoder.parameters():
                parameter.requires_grad_(False)
            self.text_encoder.eval()
        else:
            for parameter in self.text_encoder.parameters():
                parameter.requires_grad_(True)

        encoder_dim = int(getattr(getattr(text_encoder, "config", None), "hidden_size", text_dim))
        self.text_projection = projection(encoder_dim, d_model, dropout)
        self.audio_projection = projection(audio_dim, d_model, dropout)
        self.vision_projection = projection(vision_dim, d_model, dropout)
        self.compensator = SharedCompensator(
            d_model, nhead, compensation_ffn, dropout, max_len
        )
        self.gate = MissingAwareGate(d_model, dropout)
        self.fusion = projection(3 * d_model, d_model, dropout)
        self.position = SinusoidalPosition(d_model, max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=transformer_ffn,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.backbone = nn.TransformerEncoder(layer, transformer_layers)
        self.classification_head = nn.Linear(d_model, num_classes)
        self.regression_head = nn.Linear(d_model, 1)

    def train(self, mode: bool = True) -> "UnifiedSentimentModel":
        super().train(mode)
        if self.freeze_bert:
            self.text_encoder.eval()
        elif self.bert_unfreeze_last_n_layers > 0:
            for module in self._frozen_text_modules:
                module.eval()
        return self

    def freeze_text_encoder(self) -> None:
        """Freeze the full text encoder, including after partial fine-tuning."""
        for parameter in self.text_encoder.parameters():
            parameter.requires_grad_(False)
        self.freeze_bert = True
        self.bert_unfreeze_last_n_layers = 0
        self._frozen_text_modules = [self.text_encoder]
        self.text_encoder.eval()

    def _encode_text(self, text_bert: torch.Tensor) -> torch.Tensor:
        input_ids = text_bert[:, 0].long()
        attention_mask = text_bert[:, 1].long()
        token_type_ids = text_bert[:, 2].long()
        context = torch.no_grad() if self.freeze_bert else nullcontext()
        with context:
            output = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            return output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]

    def encode_projected(
        self, text_bert: torch.Tensor, audio: torch.Tensor, vision: torch.Tensor
    ) -> torch.Tensor:
        text_hidden = self._encode_text(text_bert)
        return torch.stack(
            [
                self.text_projection(text_hidden),
                self.audio_projection(audio),
                self.vision_projection(vision),
            ],
            dim=2,
        )

    def forward_from_projected(
        self,
        projected: torch.Tensor,
        valid_mask: torch.Tensor,
        observed_mask: torch.Tensor,
        artificial_mask: Optional[torch.Tensor] = None,
        enable_imputation: bool = True,
        valid_mask_by_modality: Optional[torch.Tensor] = None,
        enable_gate: bool = True,
        return_explain_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        del return_explain_features  # Contract is stable: features are always returned.
        valid_mask = valid_mask.bool()
        if valid_mask_by_modality is None:
            valid_mask_by_modality = valid_mask.unsqueeze(-1).expand(-1, -1, 3)
        applicable = valid_mask_by_modality.bool() & valid_mask.unsqueeze(-1)
        observed = observed_mask.bool() & applicable
        artificial = (
            torch.zeros_like(observed)
            if artificial_mask is None
            else artificial_mask.bool() & applicable
        )
        if torch.any(artificial & ~observed):
            raise ValueError("Artificial mask must satisfy A <= P_mod * O")
        effective = observed & ~artificial
        reconstructed, used_null = self.compensator(projected, effective)
        if enable_imputation:
            completed = torch.where(effective.unsqueeze(-1), projected, reconstructed)
        else:
            completed = torch.where(
                effective.unsqueeze(-1), projected, torch.zeros_like(projected)
            )
        completed = completed * applicable.unsqueeze(-1).to(completed.dtype)

        gate_logits, gate_weights = self.gate(completed, effective, applicable)
        if not enable_gate:
            candidates = applicable.to(completed.dtype)
            gate_weights = candidates / candidates.sum(dim=-1, keepdim=True).clamp_min(1.0)
            gate_logits = torch.zeros_like(gate_weights)
        weighted = completed * gate_weights.unsqueeze(-1)
        fused_seq = self.fusion(weighted.reshape(weighted.shape[0], weighted.shape[1], -1))
        fused_seq = self.position(fused_seq)
        backbone_seq = self.backbone(fused_seq, src_key_padding_mask=~valid_mask)
        weights = valid_mask.unsqueeze(-1).to(backbone_seq.dtype)
        pooled = (backbone_seq * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        cls_logits = self.classification_head(pooled)
        raw_reg = self.regression_head(pooled)
        reg_pred = 3.0 * torch.tanh(raw_reg) if self.bounded_regression else raw_reg
        return {
            "cls_logits": cls_logits,
            "reg_pred": reg_pred,
            "projected": projected,
            "completed": completed,
            "reconstructed": reconstructed,
            "gate_logits": gate_logits,
            "gate_weights": gate_weights,
            "fused_seq": fused_seq,
            "backbone_seq": backbone_seq,
            "pooled": pooled,
            "valid_mask": valid_mask,
            "valid_mask_by_modality": applicable,
            "observed_mask": observed,
            "effective_mask": effective,
            "artificial_mask": artificial,
            "used_null_prior": used_null,
        }

    def forward(
        self,
        text_bert: torch.Tensor,
        audio: torch.Tensor,
        vision: torch.Tensor,
        valid_mask: torch.Tensor,
        observed_mask: torch.Tensor,
        artificial_mask: Optional[torch.Tensor] = None,
        enable_imputation: bool = True,
        return_explain_features: bool = False,
        valid_mask_by_modality: Optional[torch.Tensor] = None,
        enable_gate: bool = True,
    ) -> Dict[str, torch.Tensor]:
        text_input = text_bert.clone()
        if artificial_mask is not None:
            text_missing = artificial_mask[..., 0].bool() & valid_mask.bool()
            # Text hiding occurs before contextualisation while attention remains valid.
            text_input[:, 0] = torch.where(
                text_missing,
                torch.full_like(text_input[:, 0], 103),
                text_input[:, 0],
            )
        projected = self.encode_projected(text_input, audio, vision)
        return self.forward_from_projected(
            projected=projected,
            valid_mask=valid_mask,
            observed_mask=observed_mask,
            artificial_mask=artificial_mask,
            enable_imputation=enable_imputation,
            valid_mask_by_modality=valid_mask_by_modality,
            enable_gate=enable_gate,
            return_explain_features=return_explain_features,
        )

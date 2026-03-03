from typing import Dict, Optional, Union, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .transformer import Transformer, AbsolutePositionalEmbedding
from .BinsTupleHead import EnhancedBinnedTupleTokenHeads
from .TempoComposerEmbedding import (
    ComposerEmbedding,
    TempoComposerFusion,
    EnhancedComposerConditionedTransformer
)
from .TempoPredictor import TempoPredictor, MultiScaleTempoPredictor
from .TupleTokenEmbedding import EnhancedTupleTokenEmbeddings


@dataclass
class EnhancedScorePerformerOutput:
    """Enhanced output structure for ScorePerformer model with tempo prediction."""
    logits: Dict[str, Tensor]
    predicted_tempo: Optional[Tensor] = None  # [batch, 2] - predicted tempo params
    loss: Optional[Tensor] = None
    losses: Optional[Dict[str, Tensor]] = None


# Token type definitions
# SCORE_TOKEN_TYPES = [
#     'pitch_str', 'position', 'duration', 'is_staccato', 'is_accent', 'part_id'
#     # Note: removed 'global_tempo_mean', 'global_tempo_std' as they will be predicted
# ]
SCORE_TOKEN_TYPES = [
    'note_idx', 'accidental_idx', 'octave_idx', 'position', 'duration', 'is_staccato', 'is_accent', 'part_id'
]

CONTINUOUS_TOKEN_TYPES = [
    'onset_deviation_in_seconds', 'duration_deviation_in_seconds', 'local_tempo',
    'duration', 'velocity'
    # Note: global tempo will be handled separately
]

PERFORMANCE_TOKEN_TYPES = [
    'pitch_int', 'duration', 'is_staccato', 'is_accent', 'part_id',
    'onset_deviation_in_seconds', 'duration_deviation_in_seconds',
    'local_tempo', 'velocity', 'sustain_level'
]

class ScorePerformer(nn.Module):
    def __init__(
            self,
            num_tokens: Dict[str, int],  # performance tokens
            num_score_tokens: Optional[Dict[str, int]] = None,  # score tokens
            dim: int = 512,
            max_seq_len: int = 2048,
            # Score encoder parameters
            score_encoder_depth: int = 4,
            score_encoder_heads: int = 8,
            # Performance decoder parameters
            perf_decoder_depth: int = 6,
            perf_decoder_heads: int = 8,
            # Shared parameters
            dim_head: int = 64,
            ff_mult: int = 4,
            dropout: float = 0.1,
            attn_dropout: float = 0.1,
            emb_dropout: float = 0.1,
            # Embedding parameters
            token_emb_mode: str = "cat",
            # Composer conditioning parameters
            num_composers: Optional[int] = None,
            composer_embedding_dim: int = 128,
            use_composer_conditioning: bool = True,
            # Musical position parameters
            use_musical_position: bool = True,
            score_position_key: str = "position",
            perf_position_key: str = "position",
            musical_pos_config: Optional[dict] = None,
            # Binned parameters
            binned_config: Optional[dict] = None,
            # regression parameters
            regression_config: Optional[dict] = None,
            # Tempo prediction parameters
            use_tempo_prediction: bool = True,
            tempo_predictor_type: str = "standard",  # "standard" or "multiscale"
            tempo_hidden_dim: int = 256,
            tempo_fusion_type: str = "concat",  # "concat", "add", "cross_attention"
            condition_dim: int = 128,
            # Training parameters
            tempo_loss_weight: float = 1.0,
            **kwargs
    ):
        super().__init__()

        # Handle score tokens (remove tempo-related tokens since we'll predict them)
        if num_score_tokens is None:
            num_score_tokens = {
                k: v for k, v in num_tokens.items()
                if k in SCORE_TOKEN_TYPES
            }

        self.use_composer_conditioning = use_composer_conditioning
        self.use_musical_position = use_musical_position
        self.use_tempo_prediction = use_tempo_prediction
        self.tempo_loss_weight = tempo_loss_weight

        self.num_tokens = num_tokens  # performance
        self.num_score_tokens = num_score_tokens  # score
        self.condition_dim = condition_dim

        self.bins_config = (binned_config or {})

        self.regression_config = (regression_config or {})

        # Musical position config
        default_pos_config = {
            "embedding_type": "hybrid",
            "max_position": 3000.0,
            "scale_factor": 0.1,
            "num_learned_positions": 1000
        }
        musical_pos_config = {**default_pos_config, **(musical_pos_config or {})}

        # Positional embeddings
        if use_musical_position:
            self.pos_emb = None
        else:
            self.pos_emb = AbsolutePositionalEmbedding(dim, max_seq_len)

        # Token embeddings
        self.score_token_emb = EnhancedTupleTokenEmbeddings(
            num_tokens=self.num_score_tokens,
            continuous_tokens=CONTINUOUS_TOKEN_TYPES,
            output_dim=dim,
            dropout=emb_dropout,
            mode=token_emb_mode,
            use_musical_position=use_musical_position,
            position_key=score_position_key,
            musical_pos_config=musical_pos_config
        )

        self.perf_token_emb = EnhancedTupleTokenEmbeddings(
            num_tokens=self.num_tokens,
            continuous_tokens=CONTINUOUS_TOKEN_TYPES,
            output_dim=dim,
            dropout=emb_dropout,
            mode=token_emb_mode,
            use_musical_position=use_musical_position,
            position_key=perf_position_key,
            musical_pos_config=musical_pos_config
        )

        # Tempo prediction
        if use_tempo_prediction:
            if tempo_predictor_type == "multiscale":
                self.tempo_predictor = MultiScaleTempoPredictor(
                    input_dim=dim,
                    hidden_dim=tempo_hidden_dim
                )
            else:
                self.tempo_predictor = TempoPredictor(
                    input_dim=dim,
                    hidden_dim=tempo_hidden_dim
                )

        # Composer embedding and conditioning
        if use_composer_conditioning and num_composers is not None:
            self.composer_embedding = ComposerEmbedding(
                num_composers=num_composers,
                embedding_dim=composer_embedding_dim,
                dropout=emb_dropout
            )

            # Tempo+Composer fusion
            self.condition_fusion = TempoComposerFusion(
                tempo_dim=2,  # mean, std
                composer_dim=composer_embedding_dim,
                output_dim=condition_dim,
                fusion_type=tempo_fusion_type
            )

            # Enhanced transformers with joint conditioning
            self.score_encoder = EnhancedComposerConditionedTransformer(
                dim=dim,
                depth=score_encoder_depth,
                heads=score_encoder_heads,
                dim_head=dim_head,
                ff_mult=ff_mult,
                dropout=dropout,
                attn_dropout=attn_dropout,
                causal=False,
                cross_attend=False,
                condition_dim=condition_dim
            )

            self.perf_decoder = EnhancedComposerConditionedTransformer(
                dim=dim,
                depth=perf_decoder_depth,
                heads=perf_decoder_heads,
                dim_head=dim_head,
                ff_mult=ff_mult,
                dropout=dropout,
                attn_dropout=attn_dropout,
                causal=True,
                cross_attend=True,
                condition_dim=condition_dim
            )
        else:
            self.composer_embedding = None
            self.condition_fusion = None

            # Standard transformers without conditioning
            self.score_encoder = Transformer(
                dim=dim, depth=score_encoder_depth, heads=score_encoder_heads,
                dim_head=dim_head, ff_mult=ff_mult, dropout=dropout,
                attn_dropout=attn_dropout, causal=False, cross_attend=False
            )

            self.perf_decoder = Transformer(
                dim=dim, depth=perf_decoder_depth, heads=perf_decoder_heads,
                dim_head=dim_head, ff_mult=ff_mult, dropout=dropout,
                attn_dropout=attn_dropout, causal=True, cross_attend=True
            )

        # Output head
        self.lm_head = EnhancedBinnedTupleTokenHeads(
            input_dim=dim,
            num_tokens=self.num_tokens,
            binned_tokens=self.bins_config,
            regression_tokens=self.regression_config

            
        )

    def predict_tempo(
            self,
            score_encoded: Tensor,
            score_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Predict tempo parameters from score encoding.

        Args:
            score_encoded: [batch, seq_len, dim] - encoded score
            score_mask: [batch, seq_len] - score mask

        Returns:
            tempo_params: [batch, 2] - (mean_tempo, std_tempo)
        """
        if not self.use_tempo_prediction:
            raise RuntimeError("Tempo prediction is disabled")

        return self.tempo_predictor(score_encoded, score_mask)

    def prepare_condition(
            self,
            tempo_params: Optional[Tensor] = None,
            composer_ids: Optional[Tensor] = None
    ) -> Optional[Tensor]:
        """
        Prepare joint tempo+composer conditioning vector.

        Args:
            tempo_params: [batch, 2] - (mean_tempo, std_tempo)
            composer_ids: [batch] - composer IDs

        Returns:
            condition: [batch, condition_dim] - joint conditioning vector
        """
        if not self.use_composer_conditioning:
            return None

        # Get composer embedding
        composer_emb = None
        if composer_ids is not None:
            composer_emb = self.composer_embedding(composer_ids)

        # Fuse tempo and composer information
        condition = self.condition_fusion(tempo_params, composer_emb)
        return condition

    def encode_score(
            self,
            score_tokens: Tensor,
            score_mask: Optional[Tensor] = None,
            condition: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Encode musical score with optional joint conditioning.

        Args:
            score_tokens: [batch, score_seq_len, num_score_token_types]
            score_mask: [batch, score_seq_len] - True for valid positions
            condition: [batch, condition_dim] - joint tempo+composer condition

        Returns:
            score_encoded: [batch, score_seq_len, dim] - encoded score
            score_mask: [batch, score_seq_len] - mask (potentially modified)
        """
        # Token embedding
        score_emb = self.score_token_emb(score_tokens)

        # Add positional embedding if not using musical position
        if not self.use_musical_position and self.pos_emb is not None:
            score_emb = score_emb + self.pos_emb(score_emb)

        # Apply joint conditioning if available
        if self.use_composer_conditioning and condition is not None:
            score_encoded = self.score_encoder(score_emb, condition, mask=score_mask)
        else:
            score_encoded = self.score_encoder(score_emb, mask=score_mask)

        return score_encoded, score_mask

    def decode_performance(
            self,
            perf_tokens: Tensor,
            score_encoded: Optional[Tensor] = None,
            score_mask: Optional[Tensor] = None,
            perf_mask: Optional[Tensor] = None,
            condition: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        Decode performance with optional joint conditioning.

        Args:
            perf_tokens: [batch, perf_seq_len, num_perf_token_types]
            score_encoded: [batch, score_seq_len, dim] - encoded score
            score_mask: [batch, score_seq_len] - score mask
            perf_mask: [batch, perf_seq_len] - performance mask
            condition: [batch, condition_dim] - joint tempo+composer condition

        Returns:
            logits: Dict[str, Tensor] - logits for each token type
        """
        # Token embedding
        perf_emb = self.perf_token_emb(perf_tokens)

        # Add positional embedding if not using musical position
        if not self.use_musical_position and self.pos_emb is not None:
            perf_emb = perf_emb + self.pos_emb(perf_emb)

        # Apply joint conditioning if available
        if self.use_composer_conditioning and condition is not None:
            perf_decoded = self.perf_decoder(
                perf_emb, condition,
                mask=perf_mask,
                context=score_encoded,
                context_mask=score_mask
            )
        else:
            perf_decoded = self.perf_decoder(
                perf_emb,
                mask=perf_mask,
                context=score_encoded,
                context_mask=score_mask
            )

        # Generate logits
        logits = self.lm_head(perf_decoded)
        return logits

    def compute_loss(
            self,
            logits: Dict[str, Tensor],
            labels: Tensor,
            predicted_tempo: Optional[Tensor] = None,
            target_tempo: Optional[Tensor] = None
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute total loss including both performance and tempo prediction losses.

        Args:
            logits: Performance prediction logits
            labels: Performance labels
            predicted_tempo: [batch, 2] - predicted tempo params
            target_tempo: [batch, 2] - target tempo params

        Returns:
            total_loss: Combined loss
            losses: Dictionary of individual losses
        """
        losses = {}

        # Performance prediction loss
        token_keys = PERFORMANCE_TOKEN_TYPES
        perf_losses = self.lm_head.compute_losses(logits, labels, token_keys)
        losses.update(perf_losses)

        perf_loss = sum(perf_losses.values()) / len(perf_losses) if perf_losses else torch.tensor(0.0)
        losses['performance_total'] = perf_loss

        # Tempo prediction loss
        tempo_loss = torch.tensor(0.0)
        if (self.use_tempo_prediction and
                predicted_tempo is not None and
                target_tempo is not None):
            tempo_loss = self.tempo_predictor.compute_loss(predicted_tempo, target_tempo)
            losses['tempo_prediction'] = tempo_loss

        # Combined loss
        total_loss = perf_loss + self.tempo_loss_weight * tempo_loss
        losses['total'] = total_loss

        return total_loss, losses

    def forward(
            self,
            perf_tokens: Tensor,
            score_tokens: Optional[Tensor] = None,
            perf_mask: Optional[Tensor] = None,
            score_mask: Optional[Tensor] = None,
            labels: Optional[Tensor] = None,
            composer_ids: Optional[Tensor] = None,
            target_tempo: Optional[Tensor] = None,  # [batch, 2] for training
            use_predicted_tempo: bool = False,  # Whether to use predicted tempo for conditioning
            teacher_force_tempo: bool = False  # Whether to use target tempo for conditioning
    ) -> EnhancedScorePerformerOutput:
        """
        Forward pass supporting multiple training modes.

        Args:
            perf_tokens: [batch, perf_seq_len, num_perf_token_types]
            score_tokens: [batch, score_seq_len, num_score_token_types]
            perf_mask: [batch, perf_seq_len] - performance mask
            score_mask: [batch, score_seq_len] - score mask
            labels: [batch, perf_seq_len, num_perf_token_types] - targets
            composer_ids: [batch] - composer IDs
            target_tempo: [batch, 2] - target tempo for training tempo predictor
            use_predicted_tempo: If True, use predicted tempo for conditioning
            teacher_force_tempo: If True, use target tempo for conditioning

        Returns:
            EnhancedScorePerformerOutput with logits, tempo prediction, and losses
        """
        # Handle CLM shifting
        if labels is not None:
            perf_tokens = perf_tokens[:, :-1]
            if perf_mask is not None:
                perf_mask = perf_mask[:, :-1]
            labels = labels[:, 1:]

        # Encode score (initially without tempo conditioning)
        score_encoded = None
        predicted_tempo = None

        if score_tokens is not None:
            # First pass: encode score without tempo for tempo prediction
            initial_condition = self.prepare_condition(
                tempo_params=None,
                composer_ids=composer_ids
            )
            score_encoded, score_mask = self.encode_score(
                score_tokens, score_mask, initial_condition
            )

            # Predict tempo from score encoding
            if self.use_tempo_prediction:
                predicted_tempo = self.predict_tempo(score_encoded, score_mask)

        # Determine which tempo to use for conditioning
        conditioning_tempo = None
        if teacher_force_tempo and target_tempo is not None:
            conditioning_tempo = target_tempo
        elif use_predicted_tempo and predicted_tempo is not None:
            conditioning_tempo = predicted_tempo

        # Prepare final conditioning with chosen tempo
        final_condition = self.prepare_condition(
            tempo_params=conditioning_tempo,
            composer_ids=composer_ids
        )

        # Re-encode score with final conditioning (if needed)
        if score_tokens is not None and conditioning_tempo is not None:
            score_encoded, score_mask = self.encode_score(
                score_tokens, score_mask, final_condition
            )

        # Decode performance
        logits = self.decode_performance(
            perf_tokens,
            score_encoded=score_encoded,
            score_mask=score_mask,
            perf_mask=perf_mask,
            condition=final_condition
        )

        # Compute loss if labels provided
        loss = None
        losses = None
        if labels is not None:
            loss, losses = self.compute_loss(
                logits, labels, predicted_tempo, target_tempo
            )

        return EnhancedScorePerformerOutput(
            logits=logits,
            predicted_tempo=predicted_tempo,
            loss=loss,
            losses=losses
        )

    def generate(
            self,
            score_tokens: Tensor,
            score_mask: Optional[Tensor] = None,
            composer_ids: Optional[Tensor] = None,
            max_length: int = 512,
            temperature: float = 1.0,
            use_predicted_tempo: bool = True
    ) -> Tuple[Dict[str, Tensor], Optional[Tensor]]:
        """
        Generate performance autoregressively.

        Args:
            score_tokens: [batch, score_seq_len, num_score_token_types]
            score_mask: [batch, score_seq_len]
            composer_ids: [batch]
            max_length: Maximum generation length
            temperature: Sampling temperature
            use_predicted_tempo: Whether to use predicted tempo

        Returns:
            generated_tokens: Dict of generated token sequences
            predicted_tempo: Predicted tempo parameters
        """
        self.eval()
        batch_size = score_tokens.size(0)
        device = score_tokens.device

        with torch.no_grad():
            # Encode score and predict tempo
            initial_condition = self.prepare_condition(
                tempo_params=None,
                composer_ids=composer_ids
            )
            score_encoded, _ = self.encode_score(score_tokens, score_mask, initial_condition)

            predicted_tempo = None
            if self.use_tempo_prediction:
                predicted_tempo = self.predict_tempo(score_encoded, score_mask)

            # Prepare final conditioning
            conditioning_tempo = predicted_tempo if use_predicted_tempo else None
            final_condition = self.prepare_condition(
                tempo_params=conditioning_tempo,
                composer_ids=composer_ids
            )

            # Re-encode with final conditioning
            if conditioning_tempo is not None:
                score_encoded, score_mask = self.encode_score(
                    score_tokens, score_mask, final_condition
                )

            # Initialize generation (this would need proper implementation)
            # For now, return placeholder
            generated_tokens = {token_type: torch.zeros(batch_size, max_length, dtype=torch.long, device=device)
                                for token_type in PERFORMANCE_TOKEN_TYPES}

            return generated_tokens, predicted_tempo
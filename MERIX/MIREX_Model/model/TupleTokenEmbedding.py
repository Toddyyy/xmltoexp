from typing import Dict, Optional, Union, List, Tuple, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from abc import ABC, abstractmethod
import math
from einops import rearrange


class BaseTokenEmbedding(ABC):
    """Base class for custom token embedding strategies."""

    @abstractmethod
    def create_embedding(self, num_tokens: int, emb_dim: int) -> nn.Module:
        """Create the embedding module."""
        pass

    @abstractmethod
    def forward(self, tokens: Tensor, embedding_module: nn.Module) -> Tensor:
        """Apply embedding to tokens."""
        pass


class DefaultEmbedding(BaseTokenEmbedding):
    """Default embedding strategy using nn.Embedding for discrete tokens."""

    def create_embedding(self, num_tokens: int, emb_dim: int) -> nn.Module:
        emb = nn.Embedding(num_tokens, emb_dim, padding_idx=0)
        nn.init.normal_(emb.weight, std=0.02)
        return emb

    # def forward(self, tokens: Tensor, embedding_module: nn.Module) -> Tensor:
    #     # Ensure tokens are integers for embedding lookup
    #     return embedding_module(tokens.long())
    def forward(self, tokens: Tensor, embedding_module: nn.Module) -> Tensor:
        # print(f"Max token value: {tokens.max().item()}, Embedding size: {embedding_module.weight.size(0)}")

        # Ensure that the index is in the effective range
        valid_tokens = torch.clamp(tokens.long(), 0, embedding_module.weight.size(0) - 1)

        # Record the number of truncated indexes
        num_clamped = (tokens.long() >= embedding_module.weight.size(0)).sum().item()
        # if num_clamped > 0:
        #     print(f"WARNING: {num_clamped} token indices were out of range and have been clamped")

        return embedding_module(valid_tokens)


class LearnedContinuousEmbedding(BaseTokenEmbedding):
    """
    Learned embedding for continuous values.
    This is better than SinusoidalEmbedding for features that need to be learned from data.
    """

    def __init__(self, value_range: Tuple[float, float] = (-1.0, 1.0), normalize: bool = True):
        self.value_range = value_range
        self.normalize = normalize

    def create_embedding(self, num_tokens: int, emb_dim: int) -> nn.Module:
        # Multi-layer projection for better representation
        return nn.Sequential(
            nn.Linear(1, emb_dim // 2),
            nn.LayerNorm(emb_dim // 2),
            nn.ReLU(),
            nn.Linear(emb_dim // 2, emb_dim),
            nn.LayerNorm(emb_dim)
        )

    def forward(self, tokens: Tensor, embedding_module: nn.Module) -> Tensor:
        # Normalize to [-1, 1] if specified
        if self.normalize:
            min_val, max_val = self.value_range
            tokens = 2 * (tokens - min_val) / (max_val - min_val) - 1
            tokens = torch.clamp(tokens, -1.0, 1.0)

        # Reshape for linear layer [..., 1]
        continuous_tokens = tokens.float().unsqueeze(-1)
        return embedding_module(continuous_tokens)


class MusicalPositionalEmbedding(nn.Module):
    """
    Position embedding specifically for musical position values.
    Handles continuous beat positions (e.g., 0.5, 1.0, 1.5, 2.75).
    """

    def __init__(
            self,
            dim: int,
            max_position: float = 1000.0,
            embedding_type: str = "hybrid",  # "sinusoidal", "learned", "hybrid"
            num_learned_positions: int = 2000,
            scale_factor: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.max_position = max_position
        self.embedding_type = embedding_type
        self.scale_factor = scale_factor
        self.scale = dim ** -0.5

        if embedding_type == "learned":
            # Pure learned embeddings with interpolation
            self.emb = nn.Embedding(num_learned_positions, dim)
            nn.init.normal_(self.emb.weight, std=0.02)
            self.position_scale = (num_learned_positions - 1) / max_position

        elif embedding_type == "hybrid":
            # Combination of learned and sinusoidal (recommended)
            self.learned_emb = nn.Embedding(num_learned_positions, dim // 2)
            nn.init.normal_(self.learned_emb.weight, std=0.02)
            self.position_scale = (num_learned_positions - 1) / max_position
            self.sin_dim = dim - dim // 2

    def _sinusoidal_encoding(self, positions: Tensor, dim: int) -> Tensor:
        """Generate sinusoidal position encodings for continuous positions."""
        batch_size, seq_len = positions.shape
        device = positions.device

        # Create div_term for different frequencies
        div_term = torch.exp(
            torch.arange(0, dim, 2, device=device) *
            -(math.log(10000.0) / dim)
        )

        # Scale positions for better discrimination
        scaled_positions = positions * self.scale_factor

        embeddings = torch.zeros(batch_size, seq_len, dim, device=device)

        # Apply sin to even indices
        embeddings[..., 0::2] = torch.sin(
            scaled_positions.unsqueeze(-1) * div_term
        )

        # Apply cos to odd indices
        if dim > 1:
            cos_dim = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
            embeddings[..., 1::2] = torch.cos(
                scaled_positions.unsqueeze(-1) * div_term[:cos_dim]
            )

        return embeddings

    def _learned_with_interpolation(self, positions: Tensor) -> Tensor:
        """Learned embeddings with linear interpolation for continuous positions."""
        # Scale positions to embedding indices
        scaled_pos = positions * self.position_scale

        # Get lower and upper indices
        lower_idx = torch.floor(scaled_pos).long()
        upper_idx = torch.ceil(scaled_pos).long()

        # Clamp to valid range
        lower_idx = torch.clamp(lower_idx, 0, self.emb.num_embeddings - 1)
        upper_idx = torch.clamp(upper_idx, 0, self.emb.num_embeddings - 1)

        # Get embeddings
        lower_emb = self.emb(lower_idx)
        upper_emb = self.emb(upper_idx)

        # Interpolation weights
        alpha = (scaled_pos - lower_idx.float()).unsqueeze(-1)

        # Linear interpolation
        return lower_emb * (1 - alpha) + upper_emb * alpha

    def forward(self, positions: Tensor) -> Tensor:
        """
        Args:
            positions: [batch, seq_len] - musical position values (e.g., 0.5, 1.0, 1.5, 2.75)
        Returns:
            embeddings: [batch, seq_len, dim]
        """
        # Clamp positions to valid range
        positions = torch.clamp(positions, 0, self.max_position)

        if self.embedding_type == "sinusoidal":
            return self._sinusoidal_encoding(positions, self.dim) * self.scale

        elif self.embedding_type == "learned":
            return self._learned_with_interpolation(positions) * self.scale

        elif self.embedding_type == "hybrid":
            # Learned component with interpolation
            scaled_pos = positions * self.position_scale
            lower_idx = torch.floor(scaled_pos).long()
            upper_idx = torch.ceil(scaled_pos).long()
            lower_idx = torch.clamp(lower_idx, 0, self.learned_emb.num_embeddings - 1)
            upper_idx = torch.clamp(upper_idx, 0, self.learned_emb.num_embeddings - 1)

            lower_emb = self.learned_emb(lower_idx)
            upper_emb = self.learned_emb(upper_idx)
            alpha = (scaled_pos - lower_idx.float()).unsqueeze(-1)
            learned_emb = lower_emb * (1 - alpha) + upper_emb * alpha

            # Sinusoidal component
            sin_emb = self._sinusoidal_encoding(positions, self.sin_dim)

            # Concatenate
            combined = torch.cat([learned_emb, sin_emb], dim=-1)
            return combined * self.scale

        else:
            raise ValueError(f"Unknown embedding_type: {self.embedding_type}")


class EnhancedTupleTokenEmbeddings(nn.Module):
    """
    Enhanced embedding module that handles both discrete and continuous tokens,
    with special treatment for musical position encoding.
    """

    def __init__(
            self,
            num_tokens: dict,
            continuous_tokens: List[str] = None,  # List of token types that are continuous
            continuous_value_ranges: Dict[str, Tuple[float, float]] = None,  # Value ranges for normalization
            emb_dims: dict = None,
            output_dim: int = 512,
            dropout: float = 0.0,
            mode: str = "cat",
            tie_keys: Optional[dict] = None,
            # Musical position parameters
            use_musical_position: bool = False,
            position_key: str = "position",
            musical_pos_config: Optional[dict] = None
    ):
        super().__init__()

        self.mode = mode
        self.tie_keys = tie_keys or {}
        self.use_musical_position = use_musical_position
        self.position_key = position_key
        self.continuous_tokens = continuous_tokens or []

        # Default value ranges for continuous features
        default_value_ranges = {
            'duration': (0.0, 8.0),
            'onset_deviation_in_beats': (-2.0, 2.0),
            'duration_deviation_in_beats': (-2.0, 2.0),
            'local_tempo': (30.0, 200.0),
            'velocity': (0.0, 127.0),
            'sustain_level': (0.0, 127.0)
        }
        self.value_ranges = {**default_value_ranges, **(continuous_value_ranges or {})}

        # Default embedding dimensions
        default_emb_dims = {
            'pitch_str': 64, 'pitch_int': 128, 'duration': 32,
            'velocity': 128, 'tempo': 32, 'position': 32,
            'is_staccato': 8, 'is_accent': 8, 'part_id': 16,
            'onset_deviation_in_beats': 32, 'duration_deviation_in_beats': 32,
            'local_tempo': 32, 'sustain_level': 32
        }
        emb_dims = emb_dims or default_emb_dims

        # Store token keys in order
        self.token_keys = list(num_tokens.keys())

        # Create embedding layers for each token type
        embeddings = {}
        embedding_strategies = {}

        for key, num in num_tokens.items():
            # Skip position if using musical position embedding
            if self.use_musical_position and key == self.position_key:
                continue

            emb_dim = emb_dims.get(key, 64)

            # Check if this key should share embeddings with another
            if key in self.tie_keys:
                tied_key = self.tie_keys[key]
                if tied_key in embeddings:
                    embeddings[key] = embeddings[tied_key]
                    embedding_strategies[key] = embedding_strategies[tied_key]
                continue

            # Determine embedding strategy
            if key in self.continuous_tokens:
                # Use learned continuous embedding for continuous features
                value_range = self.value_ranges.get(key, (-1.0, 1.0))
                strategy = LearnedContinuousEmbedding(value_range=value_range, normalize=True)
            else:
                # Use default embedding for discrete features
                strategy = DefaultEmbedding()

            embeddings[key] = strategy.create_embedding(num, emb_dim)
            embedding_strategies[key] = strategy

        self.embeddings = nn.ModuleDict(embeddings)
        self.embedding_strategies = embedding_strategies

        # Musical position embedding
        if use_musical_position:
            pos_config = musical_pos_config or {}
            self.musical_pos_emb = MusicalPositionalEmbedding(
                dim=output_dim,  # Use full output dim for position
                **pos_config
            )

        # Calculate total embedding dimension
        total_dim = 0
        for key in num_tokens.keys():
            if self.use_musical_position and key == self.position_key:
                continue  # Position is handled separately
            total_dim += emb_dims.get(key, 64)

        # Projection layer to output dimension
        if total_dim != output_dim:
            self.project = nn.Linear(total_dim, output_dim)
        else:
            self.project = nn.Identity()

        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Args:
            tokens: [batch, seq_len, num_token_types]
        Returns:
            embeddings: [batch, seq_len, output_dim]
        """
        token_embs = []
        position_values = None

        # Process each token type
        for i, key in enumerate(self.token_keys):
            token_slice = tokens[..., i]

            # Extract position values for later use
            if self.use_musical_position and key == self.position_key:
                position_values = token_slice.float()
                continue  # Skip embedding for position

            # Apply appropriate embedding strategy
            if key in self.embeddings:
                emb_layer = self.embeddings[key]
                strategy = self.embedding_strategies[key]
                token_emb = strategy.forward(token_slice, emb_layer)
                token_embs.append(token_emb)

        # Combine embeddings
        if len(token_embs) > 0:
            if self.mode == "cat":
                combined = torch.cat(token_embs, dim=-1)
            else:  # sum mode
                # Ensure all embeddings have the same dimension for sum mode
                combined = sum(token_embs)
        else:
            # If no token embeddings (shouldn't happen), create zero embedding
            batch_size, seq_len = tokens.shape[:2]
            combined = torch.zeros(batch_size, seq_len, self.project.in_features, device=tokens.device)

        # Project to output dimension
        output = self.project(combined)
        output = self.norm(output)
        output = self.dropout(output)

        # Add musical position embedding if enabled
        if self.use_musical_position and position_values is not None:
            musical_pos_emb = self.musical_pos_emb(position_values)
            output = output + musical_pos_emb

        return output
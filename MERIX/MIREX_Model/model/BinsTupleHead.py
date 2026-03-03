import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Optional, List, Tuple, Union
import numpy as np


class BinnedTokenHead(nn.Module):
    """
    token head support binning
    """

    def __init__(
            self,
            input_dim: int,
            num_bins: int,
            value_range: Tuple[float, float] = (0.0, 1.0),
            custom_boundaries: Optional[Union[List[float], np.ndarray, Tensor]] = None
    ):
        super().__init__()

        self.num_bins = num_bins
        self.value_range = value_range

        # bin boundary
        if custom_boundaries is not None:
            if isinstance(custom_boundaries, (list, np.ndarray)):
                boundaries = torch.tensor(custom_boundaries, dtype=torch.float32)
            else:
                boundaries = custom_boundaries.float()
            assert len(boundaries) == num_bins + 1, f"Need {num_bins + 1} boundaries，only have {len(boundaries)}"
            self.register_buffer('bin_boundaries', boundaries)
        else:
            min_val, max_val = value_range
            boundaries = torch.linspace(min_val, max_val, num_bins + 1)
            self.register_buffer('bin_boundaries', boundaries)

        # linear layer
        self.linear = nn.Linear(input_dim, num_bins)

    def values_to_bins(self, values: Tensor) -> Tensor:
        """
        Convert continuous values to bin indices

        Args:
            values: [batch, seq_len] continuous value
        Returns:
            bin_indices: [batch, seq_len] bin indexing (0 to num_bins-1)
        """
        bin_indices = torch.searchsorted(self.bin_boundaries[1:], values.float(), right=False)
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)
        return bin_indices.long()

    def bins_to_values(self, bin_indices: Tensor) -> Tensor:
        """
        Convert bin indices back to continuous values (take the centre of the bin)

        Args:
            bin_indices: [batch, seq_len] bin index
        Returns:
            values: [batch, seq_len] continuous value
        """
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)
        left_bounds = self.bin_boundaries[bin_indices]
        right_bounds = self.bin_boundaries[bin_indices + 1]
        center_values = (left_bounds + right_bounds) / 2.0
        return center_values

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim]
        Returns:
            logits: [batch, seq_len, num_bins]
        """
        return self.linear(x)

    def compute_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Calculating binned projected losses

        Args:
            logits: [batch, seq_len, num_bins] logits
            targets: [batch, seq_len] Target values (continuous values or token indices)
        Returns:
            loss:
        """
        # Convert target values to bin indices
        bin_targets = self.values_to_bins(targets)

        bin_targets = bin_targets.long()

        # padding (-100)
        mask = (targets != -100)
        bin_targets = bin_targets * mask.long() + (-100) * (~mask).long()

        # cross-entropy loss
        try:
            loss = F.cross_entropy(
                logits.transpose(1, 2),  # [batch, num_bins, seq_len]
                bin_targets,  # [batch, seq_len]
                ignore_index=-100
            )
            return loss
        except Exception as e:
            print(f"Error in cross_entropy: {str(e)}")
            print(f"logits.shape={logits.shape}, logits.dtype={logits.dtype}")
            print(f"bin_targets.shape={bin_targets.shape}, bin_targets.dtype={bin_targets.dtype}")
            raise


class StandardTokenHead(nn.Module):
    """Standard token prediction header for discrete classification"""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim]
        Returns:
            logits: [batch, seq_len, num_classes]
        """
        return self.linear(x)

    def compute_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            logits: [batch, seq_len, num_classes]
            targets: [batch, seq_len] Target Category Index
        Returns:
            loss:
        """
        targets = targets.long()

        try:
            loss = F.cross_entropy(
                logits.transpose(1, 2),  # [batch, num_classes, seq_len]
                targets,  # [batch, seq_len]
                ignore_index=-100
            )
            return loss
        except Exception as e:
            print(f"Error in cross_entropy: {str(e)}")
            print(f"logits.shape={logits.shape}, logits.dtype={logits.dtype}")
            print(f"targets.shape={targets.shape}, targets.dtype={targets.dtype}")
            raise


class RegressionTokenHead(nn.Module):
    """
    Regression head for continuous value prediction
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: Optional[int] = None,
            dropout: float = 0.1,
            activation: str = 'relu',
            value_range: Optional[Tuple[float, float]] = None,
            use_tanh_output: bool = False
    ):
        """
        Args:
            input_dim
            hidden_dim: Hidden layer dimension, if None then direct linear mapping to 1-dimension
            dropout: dropout
            activation: activation function('relu', 'gelu', 'swish')
            value_range:
            use_tanh_output:
        """
        super().__init__()

        self.value_range = value_range
        self.use_tanh_output = use_tanh_output

        if hidden_dim is not None:
            # two-layer network
            self.layers = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
        else:
            # 单层网络
            self.layers = nn.Linear(input_dim, 1)

        # 如果使用tanh输出，添加tanh层
        if use_tanh_output:
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = nn.Identity()

    def _get_activation(self, activation: str) -> nn.Module:
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'swish':
            return nn.SiLU()  # SiLU is Swish
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim]
        Returns:
            values: [batch, seq_len, 1]
        """
        output = self.layers(x)  # [batch, seq_len, 1]
        output = self.output_activation(output)

        # 如果指定了值域范围，进行缩放
        if self.value_range is not None and self.use_tanh_output:
            min_val, max_val = self.value_range
            # 将tanh的(-1,1)映射到(min_val, max_val)
            output = (output + 1) / 2 * (max_val - min_val) + min_val

        return output.squeeze(-1)  # [batch, seq_len]

    def compute_loss(
            self,
            predictions: Tensor,
            targets: Tensor,
            loss_type: str = 'mse',
            reduction: str = 'mean'
    ) -> Tensor:
        """
        Args:
            predictions: [batch, seq_len]
            targets: [batch, seq_len]
            loss_type: loss function type ('mse', 'mae', 'huber', 'smooth_l1')
            reduction: loss function aggregation ('mean', 'sum', 'none')
        Returns:
            loss:
        """
        mask = (targets != -100.0)

        if not mask.any():
            # If all values are padding, return 0 loss
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        # Calculate losses only for non-padding positions
        valid_predictions = predictions[mask]
        valid_targets = targets[mask]

        if loss_type == 'mse':
            loss = F.mse_loss(valid_predictions, valid_targets, reduction=reduction)
        elif loss_type == 'mae':
            loss = F.l1_loss(valid_predictions, valid_targets, reduction=reduction)
        elif loss_type == 'huber':
            loss = F.huber_loss(valid_predictions, valid_targets, reduction=reduction)
        elif loss_type == 'smooth_l1':
            loss = F.smooth_l1_loss(valid_predictions, valid_targets, reduction=reduction)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        return loss


class PitchDecompositionHead(nn.Module):
    """
    Pitch decomposition head for pitch_int tokens
    Includes original pitch head plus pitch class (pc) and octave (oct) auxiliary heads
    All three are used for auxiliary training losses
    """

    def __init__(self, input_dim: int, num_pitch_classes: int):
        super().__init__()
        # Original pitch head
        self.pitch_head = nn.Linear(input_dim, num_pitch_classes)
        # Pitch class head (0-11, 12 classes)
        self.pc_head = nn.Linear(input_dim, 12)
        # Octave head (assuming reasonable octave range, e.g., 0-10, 11 classes)
        self.oct_head = nn.Linear(input_dim, 11)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            x: [batch, seq_len, input_dim]
        Returns:
            logits: {'pitch': [batch, seq_len, num_pitch_classes],
                    'pc': [batch, seq_len, 12],
                    'oct': [batch, seq_len, 11]}
        """
        return {
            'pitch': self.pitch_head(x),
            'pc': self.pc_head(x),
            'oct': self.oct_head(x)
        }

    def compute_loss(self, logits: Dict[str, Tensor], pitch_targets: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            logits: {'pitch': [...], 'pc': [...], 'oct': [...]}
            pitch_targets: [batch, seq_len] original pitch values
        Returns:
            losses: {'pitch': loss_value, 'pc': loss_value, 'oct': loss_value}
        """
        losses = {}

        # Original pitch loss
        pitch_targets_long = pitch_targets.long()
        losses['pitch'] = F.cross_entropy(
            logits['pitch'].transpose(1, 2),  # [batch, num_pitch_classes, seq_len]
            pitch_targets_long,  # [batch, seq_len]
            ignore_index=-100
        )

        # Decompose pitch into pc and octave for auxiliary losses
        mask = (pitch_targets != -100)

        # For valid pitches, compute pc and octave
        valid_pitches = torch.where(mask, pitch_targets, torch.zeros_like(pitch_targets))
        pc_targets = valid_pitches % 12
        oct_targets = valid_pitches // 12

        # Apply padding mask
        pc_targets = torch.where(mask, pc_targets, torch.full_like(pc_targets, -100))
        oct_targets = torch.where(mask, oct_targets, torch.full_like(oct_targets, -100))

        # Ensure targets are long
        pc_targets = pc_targets.long()
        oct_targets = oct_targets.long()

        # Compute auxiliary losses
        try:
            losses['pc'] = F.cross_entropy(
                logits['pc'].transpose(1, 2),  # [batch, 12, seq_len]
                pc_targets,  # [batch, seq_len]
                ignore_index=-100
            )

            losses['oct'] = F.cross_entropy(
                logits['oct'].transpose(1, 2),  # [batch, 11, seq_len]
                oct_targets,  # [batch, seq_len]
                ignore_index=-100
            )
        except Exception as e:
            print(f"Error in PitchDecompositionHead loss computation: {str(e)}")
            print(f"pc_logits.shape={logits['pc'].shape}, pc_targets.shape={pc_targets.shape}")
            print(f"oct_logits.shape={logits['oct'].shape}, oct_targets.shape={oct_targets.shape}")
            raise

        return losses


class EnhancedBinnedTupleTokenHeads(nn.Module):
    """
    Prediction header for multiple token types
    Support for using different prediction strategies for different token types:
    - Binning (BinnedTokenHead)
    - Standard classification (StandardTokenHead)
    - Regression (RegressionTokenHead)
    - Special handling for pitch_int with pitch class and octave decomposition
    """

    def __init__(
            self,
            input_dim: int,
            num_tokens: Dict[str, int],
            binned_tokens: Optional[Dict[str, dict]] = None,
            regression_tokens: Optional[Dict[str, dict]] = None,
            pitch_decomposition: bool = True
    ):
        """
        Args:
            input_dim: Input feature dimensions
            num_tokens: {'token_type': vocab_size}
            binned_tokens: Types of tokens that need to be sub-binned and their configuration
            regression_tokens: Types of tokens to be regressed and their configuration
            pitch_decomposition: Whether or not to use pitch decomposition for pitch_int
        """
        super().__init__()

        self.binned_tokens = binned_tokens or {}
        self.regression_tokens = regression_tokens or {}
        self.pitch_decomposition = pitch_decomposition

        heads = {}
        for token_type, vocab_size in num_tokens.items():
            if token_type == "pitch_int" and self.pitch_decomposition:
                # special pitch 3-loss head
                heads[token_type] = PitchDecompositionHead(input_dim, vocab_size)
            elif token_type in self.regression_tokens:
                # Regression head
                config = self.regression_tokens[token_type]
                heads[token_type] = RegressionTokenHead(
                    input_dim=input_dim,
                    hidden_dim=config.get('hidden_dim', None),
                    dropout=config.get('dropout', 0.1),
                    activation=config.get('activation', 'relu'),
                    value_range=config.get('value_range', None),
                    use_tanh_output=config.get('use_tanh_output', False)
                )
            elif token_type in self.binned_tokens:
                # bins head
                config = self.binned_tokens[token_type]
                heads[token_type] = BinnedTokenHead(
                    input_dim=input_dim,
                    num_bins=config.get('num_bins', vocab_size),
                    value_range=config.get('value_range', (0.0, float(vocab_size))),
                    custom_boundaries=config.get('custom_boundaries', None)
                )
            else:
                # Standard Discrete Classification Header
                heads[token_type] = StandardTokenHead(
                    input_dim=input_dim,
                    num_classes=vocab_size
                )

        self.heads = nn.ModuleDict(heads)

    def forward(self, x: Tensor) -> Dict[str, Union[Tensor, Dict[str, Tensor]]]:
        """
        Args:
            x: [batch, seq_len, input_dim]
        Returns:
            results: {'token_type': [batch, seq_len, vocab_size/num_bins/1]} or
                    {'pitch_int': {'pc': [...], 'oct': [...]}} for pitch decomposition
        """
        results = {}
        for token_type, head in self.heads.items():
            results[token_type] = head(x)
        return results

    def compute_losses(
            self,
            logits: Dict[str, Union[Tensor, Dict[str, Tensor]]],
            labels: Tensor,
            token_keys: List[str],
            loss_weights: Optional[Dict[str, float]] = None,
            regression_loss_types: Optional[Dict[str, str]] = None
    ) -> Dict[str, Tensor]:
        """
        Args:
            logits: output of forward()
            labels: [batch, seq_len, num_token_types]
            token_keys: Sequential list of token types, corresponding to the last dimension of labels
            loss_weights: Loss weights for each token type
            regression_loss_types: Type of loss function for regression tokens
        Returns:
            losses: {'token_type': loss_value} or
                   {'pitch_int_pc': loss_value, 'pitch_int_oct': loss_value} for pitch
        """
        losses = {}
        loss_weights = loss_weights or {}
        regression_loss_types = regression_loss_types or {}

        for i, token_type in enumerate(token_keys):
            try:
                if token_type in self.heads:
                    token_labels = labels[..., i]  # [batch, seq_len]

                    if torch.any(token_labels != -100):
                        if token_type == "pitch_int" and self.pitch_decomposition:
                            pitch_losses = self.heads[token_type].compute_loss(
                                logits[token_type],
                                token_labels
                            )

                            # Add all three pitch losses and weights
                            pitch_weight = loss_weights.get(token_type, 1.0)
                            pc_weight = loss_weights.get(f'{token_type}_pc', 1.0)
                            oct_weight = loss_weights.get(f'{token_type}_oct', 1.0)

                            losses[token_type] = pitch_losses['pitch'] * pitch_weight
                            losses[f'{token_type}_pc'] = pitch_losses['pc'] * pc_weight
                            losses[f'{token_type}_oct'] = pitch_losses['oct'] * oct_weight

                        elif token_type in self.regression_tokens:
                            # regression loss
                            loss_type = regression_loss_types.get(token_type, 'mse')
                            loss = self.heads[token_type].compute_loss(
                                logits[token_type],
                                token_labels.float(),
                                loss_type=loss_type
                            )

                            # weights
                            weight = loss_weights.get(token_type, 1.0)
                            losses[token_type] = loss * weight

                        else:
                            # Standard Loss Calculations
                            if token_type not in self.binned_tokens:
                                token_labels = token_labels.long()

                            loss = self.heads[token_type].compute_loss(
                                logits[token_type],
                                token_labels
                            )

                            # weights
                            weight = loss_weights.get(token_type, 1.0)
                            losses[token_type] = loss * weight
                    else:
                        print(f"  Skipping {token_type} as all labels are padding")
                else:
                    print(f"  WARNING: {token_type} not found in heads")
            except Exception as e:
                print(f"ERROR processing token_type {token_type}: {str(e)}")
                raise

        return losses

    def predict_tokens(
            self,
            logits: Dict[str, Union[Tensor, Dict[str, Tensor]]],
            temperature: float = 1.0,
            sample: bool = True
    ) -> Dict[str, Tensor]:
        """
        Args:
            logits
            temperature
            sample
        Returns:
            predictions: {'token_type': [batch, seq_len]} Predicted token value
        """
        predictions = {}

        for token_type, logit in logits.items():
            if token_type == "pitch_int" and self.pitch_decomposition:
                # For pitch, only the original pitch logits are used for prediction
                pitch_logit = logit['pitch']

                if temperature != 1.0:
                    pitch_logit = pitch_logit / temperature

                if sample:
                    probs = F.softmax(pitch_logit, dim=-1)
                    predicted_indices = torch.multinomial(
                        probs.view(-1, probs.size(-1)),
                        num_samples=1
                    ).view(probs.shape[:-1])
                else:
                    predicted_indices = torch.argmax(pitch_logit, dim=-1)

                predictions[token_type] = predicted_indices

            elif token_type in self.regression_tokens:
                # Regression prediction
                predictions[token_type] = logit

            else:
                # Standard head
                if temperature != 1.0:
                    logit = logit / temperature

                if sample:
                    # sampling
                    probs = F.softmax(logit, dim=-1)
                    predicted_indices = torch.multinomial(
                        probs.view(-1, probs.size(-1)),
                        num_samples=1
                    ).view(probs.shape[:-1])
                else:
                    predicted_indices = torch.argmax(logit, dim=-1)

                # For sub-binned tokens, convert the bin index back to the original value
                if token_type in self.binned_tokens:
                    head = self.heads[token_type]
                    predicted_values = head.bins_to_values(predicted_indices)
                    predictions[token_type] = predicted_values
                else:
                    predictions[token_type] = predicted_indices

        return predictions
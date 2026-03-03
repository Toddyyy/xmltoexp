import os
import json
import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import argparse
import yaml
from pathlib import Path
import re

# Import your modules
from dataset import ScorePerformanceDataset, SCORE_FEATURE_NAMES, PERFORMANCE_FEATURE_NAMES, PitchDecomposer
from model.model import ScorePerformer, PERFORMANCE_TOKEN_TYPES, SCORE_TOKEN_TYPES
from model.custom_boundaries import generate_binned_config_with_boundaries


@dataclass(frozen=True)
class RenderingPerformanceNoteToken:
    """Output format for generated performance."""
    velocity: int  # MIDI velocity
    onset_deviation_in_seconds: float
    duration_deviation_in_seconds: float
    local_tempo: float
    sustain_level: int = 0


class ScorePerformanceGenerator:
    """Generator for score-to-performance conversion using guided overlapping windows with adaptive tempo."""

    def __init__(
            self,
            model_path: str,
            config_path: str = None,
            device: str = 'auto',
            sequence_length: int = 512,
            overlap_length: int = 0,
            temperature: float = 1.0,
            top_p: float = 0.9,
            use_predicted_tempo: bool = True
    ):
        """
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to model config file
            device: Device to use for inference
            sequence_length: Length of each sequence segment (default: 512)
            overlap_length: Length of overlap between segments (default: 0, no overlap)
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            use_predicted_tempo: Whether to use predicted tempo (True) or default tempo (False)
        """
        self.sequence_length = sequence_length
        self.overlap_length = overlap_length
        self.stride = sequence_length - overlap_length if overlap_length > 0 else sequence_length
        self.temperature = temperature
        self.top_p = top_p
        self.use_predicted_tempo = use_predicted_tempo

        # Fixed strategies
        self.overlap_strategy = 'cumulative'  # Always use cumulative strategy
        self.tempo_strategy = 'adaptive'      # Always use adaptive tempo

        # Validate overlap parameters
        if overlap_length >= sequence_length:
            raise ValueError(f"Overlap length ({overlap_length}) must be less than sequence length ({sequence_length})")
        if overlap_length < 0:
            raise ValueError("Overlap length must be non-negative")

        # Initialize PitchDecomposer
        self.pitch_decomposer = PitchDecomposer()
        self.use_decomposed_pitch = True

        # Set device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"Using device: {self.device}")
        print(f"Using decomposed pitch representation: {self.use_decomposed_pitch}")
        print(f"Sequence length: {self.sequence_length}, Overlap: {self.overlap_length}, Stride: {self.stride}")
        print(f"Overlap strategy: {self.overlap_strategy}")
        print(f"Tempo strategy: {self.tempo_strategy}")

        # Load model
        self.model = self._load_model(model_path, config_path)
        self.model.eval()

        # Performance token mapping
        self.performance_feature_names = [
            'pitch_int', 'duration', 'is_staccato', 'is_accent',
            'part_id', 'onset_deviation_in_seconds', 'duration_deviation_in_seconds',
            'local_tempo', 'velocity', 'sustain_level'
        ]

        # Cache for tempo predictions and previous segment outputs
        self.tempo_history = []  # Store tempo history for smooth transitions
        self.previous_segment_cache = None

    def _load_model(self, model_path: str, config_path: str = None) -> ScorePerformer:
        """Load trained model from checkpoint."""
        print(f"Loading model from {model_path}")

        # checkpoint = torch.load(model_path, map_location=self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Load config
        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            if 'config' in checkpoint:
                config = checkpoint['config']
            else:
                raise ValueError("No config found. Please provide config_path.")

        # Setup model configuration
        num_tokens, num_score_tokens = self._setup_model_config()
        binned_config = generate_binned_config_with_boundaries()
        regression_config = {
            'onset_deviation_in_seconds': {
                'hidden_dim': 256,
                'dropout': 0.2,
                'activation': 'gelu',
                'value_range': (-4.0, 4.0),
                'use_tanh_output': True
            },
            'duration_deviation_in_seconds': {
                'hidden_dim': 256,
                'dropout': 0.2,
                'activation': 'gelu',
                'value_range': (-3.0, 4.0),
                'use_tanh_output': True
            }
        }

        # Create model
        model = ScorePerformer(
            num_tokens=num_tokens,
            num_score_tokens=num_score_tokens,
            dim=config['model']['dim'],
            max_seq_len=config['model']['max_seq_len'],
            score_encoder_depth=config['model']['score_encoder_depth'],
            score_encoder_heads=config['model']['score_encoder_heads'],
            perf_decoder_depth=config['model']['perf_decoder_depth'],
            perf_decoder_heads=config['model']['perf_decoder_heads'],
            dim_head=config['model']['dim_head'],
            ff_mult=config['model']['ff_mult'],
            dropout=config['model']['dropout'],
            attn_dropout=config['model']['attn_dropout'],
            emb_dropout=config['model']['emb_dropout'],
            token_emb_mode=config['model']['token_emb_mode'],
            num_composers=config['model']['num_composers'],
            composer_embedding_dim=config['model']['composer_embedding_dim'],
            use_composer_conditioning=config['model']['use_composer_conditioning'],
            use_musical_position=config['model']['use_musical_position'],
            binned_config=binned_config,
            regression_config=regression_config,
            use_tempo_prediction=config['model'].get('use_tempo_prediction', True),
            tempo_predictor_type=config['model'].get('tempo_predictor_type', 'standard'),
            tempo_hidden_dim=config['model'].get('tempo_hidden_dim', 256),
            tempo_fusion_type=config['model'].get('tempo_fusion_type', 'concat'),
            condition_dim=config['model'].get('condition_dim', 128),
            tempo_loss_weight=config['training'].get('tempo_loss_weight', 1.0)
        )

        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device)

        print(f"Model loaded successfully")
        print(f"Tempo prediction enabled: {model.use_tempo_prediction}")
        return model

    def _setup_model_config(self):
        """Setup model configuration for decomposed pitch format."""
        num_score_tokens = {
            'note_idx': 7,
            'accidental_idx': 5,
            'octave_idx': 8,
            'position': 1,
            'duration': 1,
            'is_staccato': 2,
            'is_accent': 2,
            'part_id': 2,
        }

        num_tokens = {
            'pitch_int': 128,
            'duration': 1,
            'is_staccato': 2,
            'is_accent': 2,
            'part_id': 2,
            'onset_deviation_in_seconds': 1,
            'duration_deviation_in_seconds': 1,
            'local_tempo': 1,
            'velocity': 1,
            'sustain_level': 2
        }

        return num_tokens, num_score_tokens

    def _nucleus_sampling(self, logits: torch.Tensor, top_p: float = 0.9) -> torch.Tensor:
        """Apply nucleus (top-p) sampling."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))

        return logits

    def _sample_from_logits(
            self,
            logits: Dict[str, torch.Tensor],
            temperature: float = 1.0,
            top_p: float = 0.9
    ) -> Dict[str, torch.Tensor]:
        """Sample from logits with temperature and nucleus sampling."""
        predictions = {}
        regression_tokens = ['onset_deviation_in_seconds', 'duration_deviation_in_seconds']

        for token_type, logit in logits.items():
            if isinstance(logit, dict):
                if token_type == "pitch_int":
                    logit = logit['pitch']
                else:
                    continue

            if token_type in regression_tokens:
                predictions[token_type] = logit
                continue

            if temperature != 1.0:
                logit = logit / temperature

            if top_p < 1.0:
                logit = self._nucleus_sampling(logit, top_p)

            probs = torch.softmax(logit, dim=-1)
            if len(probs.shape) == 3:
                batch_size, seq_len, vocab_size = probs.shape
                probs_2d = probs.view(-1, vocab_size)
                sampled_indices = torch.multinomial(probs_2d, 1).view(batch_size, seq_len)
            else:
                sampled_indices = torch.multinomial(probs, 1).squeeze(-1)

            if hasattr(self.model.lm_head, 'binned_tokens') and token_type in self.model.lm_head.binned_tokens:
                head = self.model.lm_head.heads[token_type]
                predicted_values = head.bins_to_values(sampled_indices)
                predictions[token_type] = predicted_values
            else:
                predictions[token_type] = sampled_indices

        return predictions

    def _pitch_str_to_midi(self, pitch_str: str) -> int:
        """Convert pitch string to MIDI number using PitchDecomposer logic."""
        note_idx, accidental_idx, octave_idx = self.pitch_decomposer.parse_pitch(pitch_str)
        
        note_to_midi_offset = {0: 0, 1: 2, 2: 4, 3: 5, 4: 7, 5: 9, 6: 11}
        base_offset = note_to_midi_offset[note_idx]
        octave_num = octave_idx - self.pitch_decomposer.octave_offset
        accidental_offsets = {0: -2, 1: -1, 2: 0, 3: 1, 4: 2}
        accidental_offset = accidental_offsets[accidental_idx]
        
        midi_number = (octave_num + 1) * 12 + base_offset + accidental_offset
        midi_number = max(0, min(127, midi_number))
        
        return midi_number

    def _extract_score_features(self, score_tokens: List[Dict]) -> torch.Tensor:
        """Extract score features from tokens using decomposed pitch representation."""
        score_features = []
        
        for token in score_tokens:
            score_note = token['score_note_token']
            
            if 'note_idx' in score_note and 'accidental_idx' in score_note and 'octave_idx' in score_note:
                note_idx = score_note['note_idx']
                accidental_idx = score_note['accidental_idx']
                octave_idx = score_note['octave_idx']
            else:
                pitch_str = score_note.get('pitch', 'C4')
                note_idx, accidental_idx, octave_idx = self.pitch_decomposer.parse_pitch(pitch_str)
            
            position = round(score_note.get('position', 0.0), 2)
            duration = round(score_note.get('duration', 0.5), 2)
            is_staccato = 1 if score_note.get('is_staccato', False) else 0
            is_accent = 1 if score_note.get('is_accent', False) else 0
            
            part_id_str = score_note.get('part_id', 'P1-Staff1')
            part_id = 0 if 'Staff1' in part_id_str else 1
            
            score_features.append([
                note_idx, accidental_idx, octave_idx, position, 
                duration, is_staccato, is_accent, part_id
            ])
        
        score_tensor = torch.tensor(score_features, dtype=torch.float32)
        score_tensor[:, 4] = torch.clamp(score_tensor[:, 4], min=0.01, max=8.0)
        
        return score_tensor

    def _create_initial_performance_tokens(self, score_tokens: List[Dict], 
                                         previous_performance: Optional[torch.Tensor] = None,
                                         overlap_start: int = 0) -> torch.Tensor:
        """Create initial performance tokens, using previous segment outputs as ground truth in overlap region."""
        batch_size = 1
        seq_len = len(score_tokens)
        num_perf_features = len(self.performance_feature_names)

        perf_tokens = torch.zeros(batch_size, seq_len, num_perf_features, dtype=torch.float32)

        for i, score_token in enumerate(score_tokens):
            score_note = score_token['score_note_token']

            # Get pitch as MIDI number
            if 'pitch' in score_note:
                pitch_str = score_note['pitch']
                pitch_int = self._pitch_str_to_midi(pitch_str)
            else:
                pitch_int = 60

            # Score-derived features (always from ground truth)
            perf_tokens[0, i, 0] = pitch_int  # pitch_int
            perf_tokens[0, i, 1] = score_note.get('duration', 0.5)  # duration
            perf_tokens[0, i, 2] = 1 if score_note.get('is_staccato', False) else 0  # is_staccato
            perf_tokens[0, i, 3] = 1 if score_note.get('is_accent', False) else 0  # is_accent
            
            part_id_str = score_note.get('part_id', 'P1-Staff1')
            perf_tokens[0, i, 4] = 0 if 'Staff1' in part_id_str else 1  # part_id

            # Performance-specific features
            if (previous_performance is not None and 
                i < self.overlap_length):
                # Use previous segment's predictions as ground truth for overlap region
                prev_idx = overlap_start + i
                if prev_idx < previous_performance.size(1):
                    # Copy performance-specific features from previous segment (as ground truth)
                    perf_tokens[0, i, 5:] = previous_performance[0, prev_idx, 5:]
                else:
                    # Fallback to defaults if index out of bounds
                    perf_tokens[0, i, 5] = 0.0    # onset_deviation_in_seconds
                    perf_tokens[0, i, 6] = 0.0    # duration_deviation_in_seconds
                    perf_tokens[0, i, 7] = 120.0  # local_tempo
                    perf_tokens[0, i, 8] = 64     # velocity
                    perf_tokens[0, i, 9] = 0      # sustain_level
            else:
                # Default values for non-overlap positions (will be generated)
                perf_tokens[0, i, 5] = 0.0    # onset_deviation_in_seconds
                perf_tokens[0, i, 6] = 0.0    # duration_deviation_in_seconds
                perf_tokens[0, i, 7] = 120.0  # local_tempo
                perf_tokens[0, i, 8] = 64     # velocity
                perf_tokens[0, i, 9] = 0      # sustain_level

        return perf_tokens.to(self.device)

    def _predict_adaptive_tempo(
            self,
            score_encoded: torch.Tensor,
            score_mask: torch.Tensor,
            segment_idx: int,
            composer_ids: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """Predict tempo using adaptive strategy."""
        
        if not (self.model.use_tempo_prediction and self.use_predicted_tempo):
            return None
            
        # Always predict fresh tempo
        initial_condition = self.model.prepare_condition(
            tempo_params=None,
            composer_ids=composer_ids
        )
        score_encoded_temp, _ = self.model.encode_score(score_encoded.detach(), score_mask, initial_condition)
        raw_tempo = self.model.predict_tempo(score_encoded_temp, score_mask)
        
        if segment_idx == 0 or len(self.tempo_history) == 0:
            predicted_tempo = raw_tempo
            print(f"Initial tempo prediction: mean={predicted_tempo[0, 0]:.1f}, std={predicted_tempo[0, 1]:.1f}")
        else:
            # Apply exponential smoothing
            alpha = 0.3  # Smoothing factor (0 = no change, 1 = full change)
            prev_tempo = self.tempo_history[-1]
            
            smoothed_mean = alpha * raw_tempo[0, 0] + (1 - alpha) * prev_tempo[0, 0]
            smoothed_std = alpha * raw_tempo[0, 1] + (1 - alpha) * prev_tempo[0, 1]
            
            predicted_tempo = torch.tensor([[smoothed_mean, smoothed_std]], 
                                         device=raw_tempo.device, dtype=raw_tempo.dtype)
            
            print(f"Raw prediction: mean={raw_tempo[0, 0]:.1f}, std={raw_tempo[0, 1]:.1f}")
            print(f"Smoothed tempo: mean={predicted_tempo[0, 0]:.1f}, std={predicted_tempo[0, 1]:.1f}")
        
        # Store in history for future smoothing
        self.tempo_history.append(predicted_tempo.clone())
        
        return predicted_tempo

    def _update_performance_tokens_with_score_ground_truth(
            self,
            perf_tokens: torch.Tensor,
            score_tokens: List[Dict],
            predictions: Dict[str, torch.Tensor],
            pos: int
    ) -> torch.Tensor:
        """Update performance tokens for the next position."""
        if pos >= len(score_tokens) - 1:
            return perf_tokens

        score_note = score_tokens[pos + 1]['score_note_token']

        # Score-related features - ALWAYS use ground truth
        if 'pitch' in score_note:
            pitch_str = score_note['pitch']
            pitch_int = self._pitch_str_to_midi(pitch_str)
        else:
            pitch_int = 60

        perf_tokens[0, pos + 1, 0] = pitch_int  # pitch_int
        perf_tokens[0, pos + 1, 1] = score_note.get('duration', 0.5)  # duration
        perf_tokens[0, pos + 1, 2] = 1 if score_note.get('is_staccato', False) else 0  # is_staccato
        perf_tokens[0, pos + 1, 3] = 1 if score_note.get('is_accent', False) else 0  # is_accent
        
        part_id_str = score_note.get('part_id', 'P1-Staff1')
        perf_tokens[0, pos + 1, 4] = 0 if 'Staff1' in part_id_str else 1  # part_id

        # Performance-specific features - use model predictions
        performance_specific_features = [
            'onset_deviation_in_seconds',  # index 5
            'duration_deviation_in_seconds',  # index 6
            'local_tempo',  # index 7
            'velocity',  # index 8
            'sustain_level'  # index 9
        ]

        regression_features = ['onset_deviation_in_seconds', 'duration_deviation_in_seconds', 'velocity']

        for j, token_type in enumerate(performance_specific_features):
            feature_idx = j + 5
            if token_type in predictions:
                pred_tensor = predictions[token_type]
                
                if token_type in regression_features:
                    if pred_tensor.dim() == 2:
                        value = pred_tensor[0, 0]
                    elif pred_tensor.dim() == 1:
                        value = pred_tensor[0]
                    else:
                        value = pred_tensor.flatten()[0]
                else:
                    if pred_tensor.dim() == 1:
                        value = pred_tensor[0]
                    elif pred_tensor.dim() == 2:
                        value = pred_tensor[0, 0]
                    else:
                        value = pred_tensor.flatten()[0]
                
                if hasattr(value, 'item'):
                    value = value.item()
                
                perf_tokens[0, pos + 1, feature_idx] = value

        return perf_tokens

    def _safe_extract_value(self, tensor: torch.Tensor, position: int) -> float:
        """Safely extract a value from tensor at given position."""
        if tensor is None:
            return 0.0
            
        if tensor.dim() == 0:
            return tensor.item()
        elif tensor.dim() == 1:
            if position < tensor.size(0):
                return tensor[position].item()
            else:
                return tensor[0].item()
        elif tensor.dim() == 2:
            if position < tensor.size(1):
                return tensor[0, position].item()
            else:
                return tensor[0, 0].item()
        else:
            flattened = tensor.flatten()
            if len(flattened) > 0:
                return flattened[0].item()
            else:
                return 0.0

    @torch.no_grad()
    def generate_window(
            self,
            score_tokens: List[Dict],
            composer_id: int = 0,
            previous_performance: Optional[torch.Tensor] = None,
            overlap_start: int = 0,
            segment_idx: int = 0
    ) -> Tuple[torch.Tensor, List[RenderingPerformanceNoteToken], Optional[torch.Tensor]]:
        """Generate performance for a single window with cumulative overlap and adaptive tempo."""
        batch_size = 1
        seq_len = len(score_tokens)

        # Extract score features
        score_features = self._extract_score_features(score_tokens).unsqueeze(0).to(self.device)
        score_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=self.device)
        composer_ids = torch.tensor([composer_id], device=self.device) if composer_id is not None else None

        # Adaptive tempo prediction
        predicted_tempo = self._predict_adaptive_tempo(
            score_features, score_mask, segment_idx, composer_ids
        )

        # Prepare final condition
        final_condition = self.model.prepare_condition(
            tempo_params=predicted_tempo if self.use_predicted_tempo else None,
            composer_ids=composer_ids
        )

        # Encode score
        score_encoded, score_mask = self.model.encode_score(score_features, score_mask, final_condition)

        # Initialize performance tokens with previous segment as ground truth in overlap region
        perf_tokens = self._create_initial_performance_tokens(
            score_tokens, 
            previous_performance, 
            overlap_start
        )

        # Determine generation start point
        generation_start = 0
        if previous_performance is not None and self.overlap_length > 0:
            # Start generation after overlap region (previous segment performance is ground truth)
            generation_start = min(self.overlap_length, seq_len - 1)
            print(f"  Starting generation from position {generation_start} (after overlap)")

        # Autoregressive generation - start from after overlap region
        for pos in range(generation_start, seq_len):
            current_perf_tokens = perf_tokens[:, :pos + 1]
            perf_mask = torch.ones(batch_size, pos + 1, dtype=torch.bool, device=self.device)

            logits = self.model.decode_performance(
                current_perf_tokens,
                score_encoded=score_encoded,
                score_mask=score_mask,
                perf_mask=perf_mask,
                condition=final_condition
            )

            current_logits = {}
            for token_type, logit_tensor in logits.items():
                if isinstance(logit_tensor, dict):
                    current_logits[token_type] = {
                        k: v[:, pos:pos + 1] for k, v in logit_tensor.items()
                    }
                else:
                    current_logits[token_type] = logit_tensor[:, pos:pos + 1]

            predictions = self._sample_from_logits(
                current_logits,
                temperature=self.temperature,
                top_p=self.top_p
            )

            if pos < seq_len - 1:
                perf_tokens = self._update_performance_tokens_with_score_ground_truth(
                    perf_tokens, score_tokens, predictions, pos
                )

        # Convert to rendering tokens (no fusion needed - overlap region already has ground truth)
        rendering_tokens = []
        for i in range(seq_len):
            # Use values directly from perf_tokens (which contains ground truth for overlap)
            rendering_token = RenderingPerformanceNoteToken(
                velocity=int(perf_tokens[0, i, 8].item()),
                onset_deviation_in_seconds=float(perf_tokens[0, i, 5].item()),
                duration_deviation_in_seconds=float(perf_tokens[0, i, 6].item()),
                local_tempo=float(perf_tokens[0, i, 7].item()),
                sustain_level=int(perf_tokens[0, i, 9].item())
            )
            rendering_tokens.append(rendering_token)

        return perf_tokens, rendering_tokens, predicted_tempo

    def generate_full_sequence(
            self,
            all_score_tokens: List[Dict],
            composer_id: int = 0
    ) -> Tuple[List[Dict], List[Dict]]:
        """Generate performance for a full sequence using guided overlapping segments with adaptive tempo."""
        total_length = len(all_score_tokens)
        all_rendering_tokens = []
        segment_tempos = []

        overlap_str = f"overlap={self.overlap_length}" if self.overlap_length > 0 else "non-overlapping"
        print(f"Generating performance for {total_length} notes using {overlap_str} segments")
        print(f"Segment size: {self.sequence_length}, Stride: {self.stride}")
        print(f"Strategy: cumulative overlap with adaptive tempo")

        # Clear caches
        self.tempo_history = []
        self.previous_segment_cache = None

        segment_idx = 0
        current_position = 0

        while current_position < total_length:
            # Calculate segment boundaries
            segment_start = current_position
            segment_end = min(segment_start + self.sequence_length, total_length)
            segment_tokens = all_score_tokens[segment_start:segment_end]
            segment_size = len(segment_tokens)

            print(f"\nProcessing segment {segment_idx + 1}:")
            print(f"  Range: tokens {segment_start} to {segment_end - 1} (size: {segment_size})")

            # Calculate actual output range for this segment BEFORE generation
            output_start = 0
            output_end = segment_size

            # For overlapping segments, skip the overlap region in subsequent segments
            if segment_idx > 0 and self.overlap_length > 0:
                actual_overlap = min(self.overlap_length, segment_start)
                output_start = actual_overlap
                print(f"  Overlap: {actual_overlap} tokens with previous segment")
                print(f"  Skipping first {actual_overlap} tokens (overlap region)")

            # Calculate the actual note IDs this segment contributes to the final output
            actual_output_start = segment_start + output_start
            actual_output_end = min(segment_start + output_end, total_length)
            actual_output_size = actual_output_end - actual_output_start

            # Check if this segment would contribute any new tokens
            if actual_output_size <= 0:
                print(f"  Segment would contribute {actual_output_size} tokens, skipping...")
                break

            # Determine overlap parameters for this segment
            if segment_idx == 0:
                # First segment
                previous_performance = None
                overlap_start = 0
            else:
                # Subsequent segments
                previous_performance = self.previous_segment_cache
                overlap_start = max(0, segment_start - self.overlap_length)

            # Generate for this segment
            segment_performance, rendering_tokens, segment_tempo = self.generate_window(
                segment_tokens,
                composer_id=composer_id,
                previous_performance=previous_performance,
                overlap_start=overlap_start,
                segment_idx=segment_idx
            )

            # Cache this segment's performance for next iteration
            self.previous_segment_cache = segment_performance

            # Store tempo information with actual note ranges
            tempo_info = {
                'segment_idx': segment_idx,
                'segment_token_range': f"{segment_start}-{segment_end - 1}",
                'segment_size': segment_size,
                'output_note_range': f"{actual_output_start}-{actual_output_end - 1}",
                'output_note_count': actual_output_size,
                'overlap_with_previous': min(self.overlap_length, segment_start) if segment_idx > 0 else 0,
                'avg_tempo': float(segment_tempo[0, 0]) if segment_tempo is not None else None,
                'std_tempo': float(segment_tempo[0, 1]) if segment_tempo is not None else None
            }
            segment_tempos.append(tempo_info)

            if segment_tempo is not None:
                print(f"  Predicted tempo: mean={segment_tempo[0, 0]:.1f} BPM, std={segment_tempo[0, 1]:.1f}")

            # Convert rendering tokens to dict format
            tokens_to_add = rendering_tokens[output_start:output_end]
            
            # Ensure we don't exceed the total length
            tokens_added = 0
            for i, token in enumerate(tokens_to_add):
                global_pos = segment_start + output_start + i
                
                # Check if we've reached the end of the sequence
                if global_pos >= total_length:
                    break
                    
                original_score_token = all_score_tokens[global_pos]
                score_note = original_score_token['score_note_token']
                
                token_dict = {
                    'score': {
                        'pitch': score_note.get('pitch', 'C4'),
                        'position': score_note.get('position', 0.0),
                        'duration': score_note.get('duration', 0.5),
                        'is_staccato': score_note.get('is_staccato', False),
                        'is_accent': score_note.get('is_accent', False),
                        'part_id': score_note.get('part_id', 'P1-Staff1'),
                        **{k: v for k, v in score_note.items() 
                           if k not in ['pitch', 'position', 'duration', 'is_staccato', 'is_accent', 'part_id']}
                    },
                    'performance': {
                        'velocity': token.velocity,
                        'onset_deviation_in_seconds': token.onset_deviation_in_seconds,
                        'duration_deviation_in_seconds': token.duration_deviation_in_seconds,
                        'local_tempo': token.local_tempo,
                        'sustain_level': token.sustain_level
                    },
                    'segment_info': {
                        'segment_idx': segment_idx,
                        'segment_avg_tempo': float(segment_tempo[0, 0]) if segment_tempo is not None else None,
                        'segment_std_tempo': float(segment_tempo[0, 1]) if segment_tempo is not None else None,
                        'global_position': global_pos,
                        'local_position_in_segment': output_start + i
                    }
                }
                all_rendering_tokens.append(token_dict)
                tokens_added += 1

            print(f"  Added {tokens_added} tokens to final output")

            # Move to next segment
            current_position += self.stride
            segment_idx += 1

            # Enhanced stopping condition: stop if we've covered all tokens or next segment won't contribute
            if len(all_rendering_tokens) >= total_length:
                print(f"  Reached target length {total_length}, stopping generation")
                break

        # Verify output length
        if len(all_rendering_tokens) != total_length:
            print(f"⚠️ WARNING: Output has {len(all_rendering_tokens)} tokens, expected {total_length}")
            # Trim excess if any
            all_rendering_tokens = all_rendering_tokens[:total_length]
        else:
            print(f"\n✅ Generation complete. Generated {len(all_rendering_tokens)} tokens in {len(segment_tempos)} segments")

        return all_rendering_tokens, segment_tempos

    def generate_from_json(
            self,
            input_json_path: str,
            output_json_path: str,
            composer_id: int = 0
    ):
        """Generate performance from a JSON file and save results."""
        # Load input data
        with open(input_json_path, 'r') as f:
            input_data = json.load(f)

        score_tokens = input_data['full_tokens']
        print(f"Loaded {len(score_tokens)} score tokens from {input_json_path}")
        score_tokens = sorted(score_tokens, key=lambda x: x['score_note_token']['position'])
        print(f"  Sorted {len(score_tokens)} notes by position")

        # Generate performance
        performance_tokens, segment_tempos = self.generate_full_sequence(score_tokens, composer_id)

        # Create output data with enhanced metadata
        output_data = {
            'metadata': {
                'input_file': input_json_path,
                'total_notes': len(performance_tokens),
                'composer_id': composer_id,
                'generation_params': {
                    'sequence_length': self.sequence_length,
                    'overlap_length': self.overlap_length,
                    'stride': self.stride,
                    'overlap_strategy': self.overlap_strategy,
                    'tempo_strategy': self.tempo_strategy,
                    'temperature': self.temperature,
                    'top_p': self.top_p,
                    'use_predicted_tempo': self.use_predicted_tempo,
                    'use_decomposed_pitch': self.use_decomposed_pitch
                },
                'segment_details': segment_tempos
            },
            'note_by_note_results': performance_tokens
        }

        # Save output
        output_dir = os.path.dirname(output_json_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_json_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nSaved generated performance to {output_json_path}")
        print(f"Output includes {len(segment_tempos)} segment tempo predictions with note ranges")

        # Print segment summary
        print(f"\nSegment Summary:")
        for i, seg in enumerate(segment_tempos):
            print(f"  Segment {i}: Notes {seg['output_note_range']} ({seg['output_note_count']} notes) - Tempo: {seg['avg_tempo']:.1f} BPM")


def main():
    parser = argparse.ArgumentParser(description='Generate performance from score using cumulative overlap with adaptive tempo')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--input_json', type=str, required=True)
    parser.add_argument('--output_json', type=str, required=True)
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--composer_id', type=int, default=0)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--sequence_length', type=int, default=512)
    parser.add_argument('--overlap_length', type=int, default=0,
                        help='Length of overlap between segments (0 for non-overlapping)')
    parser.add_argument('--no_tempo_prediction', action='store_true')

    args = parser.parse_args()

    # Create generator with fixed strategies
    generator = ScorePerformanceGenerator(
        model_path=args.model_path,
        config_path=args.config_path,
        device=args.device,
        sequence_length=args.sequence_length,
        overlap_length=args.overlap_length,
        temperature=args.temperature,
        top_p=args.top_p,
        use_predicted_tempo=not args.no_tempo_prediction
    )

    # Generate performance
    generator.generate_from_json(
        input_json_path=args.input_json,
        output_json_path=args.output_json,
        composer_id=args.composer_id
    )

    print("Generation completed successfully!")


if __name__ == '__main__':
    main()

# 使用示例：
# python generate_overlap.py --model_path model.pt --input_json input.json --output_json output.json --sequence_length 512 --overlap_length 256 --composer_id 10 --config_path config.yaml

# 使用示例：
# python generate_overlap.py --model_path model.pt --input_json input.json --output_json output.json --sequence_length 512 --overlap_length 256 --composer_id 10 --config_path config.yaml

# 使用示例：
# python generate_overlap.py --model_path ./check/sustain_classify_20250815_073720_best.pt --input_json option_list/Vers_la_Flamme,_Op.72_(Scriabin,_Aleksandr)_sorted_score_tokens.json --output_json generate_results/Scriabin_Vers_la_Flamme,_Op.72.json --composer_id 14 --config_path config.yaml --sequence_length 512 --overlap_length 256


# python generate_overlap.py --model_path ./check/sustain_classify_20250815_073720_best.pt --input_json option_list/Prokofiev_Concerto_No.2_Op.16_Mvt._1_sorted_score_tokens.json --output_json generate_results/Prokofiev_Concerto_No.2_Op.16.json --composer_id 9 --config_path config.yaml --sequence_length 512 --overlap_length 256

# python generate_overlap.py --model_path ./check/sustain_classify_20250815_073720_best.pt --input_json option_list/Schubert_Moments_Musicaux_D._780_No._3_in_F_Minor-Piano_sorted_score_tokens.json --output_json generate_results/Schubert_Moments_Musicaux_D._780_No._3.json --composer_id 12 --config_path config.yaml --sequence_length 512 --overlap_length 256

# 'Bach': 0,
# 'Beethoven': 1,
# 'Brahms': 2,
# 'Chopin': 3,
# 'Debussy': 4,
# 'Glinka': 5,
# 'Haydn': 6,
# 'Liszt': 7,
# 'Mozart': 8,
# 'Prokofiev': 9,
# 'Rachmaninoff': 10,
# 'Ravel': 11,
# 'Schubert': 12,
# 'Schumann': 13,
# 'Scriabin': 14
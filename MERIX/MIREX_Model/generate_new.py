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
    """Generator for score-to-performance conversion using sliding windows with tempo prediction."""

    def __init__(
            self,
            model_path: str,
            config_path: str = None,
            device: str = 'auto',
            sequence_length: int = 512,
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
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            use_predicted_tempo: Whether to use predicted tempo (True) or default tempo (False)
        """
        self.sequence_length = sequence_length
        self.temperature = temperature
        self.top_p = top_p
        self.use_predicted_tempo = use_predicted_tempo

        # Initialize PitchDecomposer
        self.pitch_decomposer = PitchDecomposer()
        self.use_decomposed_pitch = True  # Flag to indicate we're using decomposed pitch

        # Set device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"Using device: {self.device}")
        print(f"Using decomposed pitch representation: {self.use_decomposed_pitch}")
        print(f"Using non-overlapping segments of size: {self.sequence_length}")

        # Load model
        self.model = self._load_model(model_path, config_path)
        self.model.eval()

        # Performance token mapping (from dataset.py)
        self.performance_feature_names = [
            'pitch_int', 'duration', 'is_staccato', 'is_accent',
            'part_id', 'onset_deviation_in_seconds', 'duration_deviation_in_seconds',
            'local_tempo', 'velocity', 'sustain_level'
        ]

        # Cache for tempo predictions (no longer needed for sliding windows)
        self.tempo_cache = None

    def _load_model(self, model_path: str, config_path: str = None) -> ScorePerformer:
        """Load trained model from checkpoint."""
        print(f"Loading model from {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)

        # Load config
        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            # Try to load config from checkpoint
            if 'config' in checkpoint:
                config = checkpoint['config']
            else:
                raise ValueError("No config found. Please provide config_path.")

        # Setup model configuration
        num_tokens, num_score_tokens = self._setup_model_config()
        binned_config = generate_binned_config_with_boundaries()
        regression_config = {
    'onset_deviation_in_seconds': {
        'hidden_dim': 256,           # 最大容量
        'dropout': 0.2,              # 高正则化
        'activation': 'gelu',        # 平滑激活
        'value_range': (-4.0, 4.0),  # 宽松范围
        'use_tanh_output': True
    },
    'duration_deviation_in_seconds': {
        'hidden_dim': 256,           # 最大容量
        'dropout': 0.2,              # 高正则化
        'activation': 'gelu',        # 平滑激活
        'value_range': (-3.0, 4.0),  # 宽松范围
        'use_tanh_output': True
    }
    # 'velocity': {
    #     'hidden_dim': 128,           # 中等容量
    #     'dropout': 0.1,              # 中等正则化
    #     'activation': 'relu',        # 非负特性
    #     'value_range': (5.0, 125.0), # 实用范围
    #     'use_tanh_output': True
    # }
}

        # Create model with new parameters
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
        # Score token configuration with decomposed pitch
        num_score_tokens = {
            'note_idx': 7,          # 7 note names (C, D, E, F, G, A, B)
            'accidental_idx': 5,    # 5 accidentals (double flat to double sharp)
            'octave_idx': 8,       # 11 octaves (from -1 to 9)
            'position': 1,
            'duration': 1,
            'is_staccato': 2,
            'is_accent': 2,
            'part_id': 2,
        }

        # Performance token configuration
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

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter sorted indices to original indices
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
        
        # Define regression tokens
        regression_tokens = ['onset_deviation_in_seconds', 'duration_deviation_in_seconds']
    
        for token_type, logit in logits.items():
            if isinstance(logit, dict):  # Handle pitch decomposition
                if token_type == "pitch_int":
                    logit = logit['pitch']
                else:
                    continue
    
            # Check if this is a regression token
            if token_type in regression_tokens:
                # For regression, logit is already the predicted value
                # No sampling needed, just return as is
                predictions[token_type] = logit
                continue
    
            # For classification/binned tokens, apply sampling
            if temperature != 1.0:
                logit = logit / temperature
    
            if top_p < 1.0:
                logit = self._nucleus_sampling(logit, top_p)
    
            # Sample
            probs = torch.softmax(logit, dim=-1)
            if len(probs.shape) == 3:  # [batch, seq, vocab]
                batch_size, seq_len, vocab_size = probs.shape
                probs_2d = probs.view(-1, vocab_size)
                sampled_indices = torch.multinomial(probs_2d, 1).view(batch_size, seq_len)
            else:
                sampled_indices = torch.multinomial(probs, 1).squeeze(-1)
    
            # Convert binned tokens back to values if needed
            if hasattr(self.model.lm_head, 'binned_tokens') and token_type in self.model.lm_head.binned_tokens:
                head = self.model.lm_head.heads[token_type]
                predicted_values = head.bins_to_values(sampled_indices)
                predictions[token_type] = predicted_values
            else:
                predictions[token_type] = sampled_indices
    
        return predictions

    def _pitch_str_to_midi(self, pitch_str: str) -> int:
        """Convert pitch string to MIDI number using PitchDecomposer logic."""
        # Parse pitch using PitchDecomposer
        note_idx, accidental_idx, octave_idx = self.pitch_decomposer.parse_pitch(pitch_str)
        
        # Convert back to MIDI number
        # Note offsets for MIDI calculation
        note_to_midi_offset = {0: 0, 1: 2, 2: 4, 3: 5, 4: 7, 5: 9, 6: 11}  # C, D, E, F, G, A, B
        
        # Get base note offset
        base_offset = note_to_midi_offset[note_idx]
        
        # Convert octave index back to octave number
        octave_num = octave_idx - self.pitch_decomposer.octave_offset
        
        # Convert accidental index to semitone offset
        accidental_offsets = {0: -2, 1: -1, 2: 0, 3: 1, 4: 2}  # double flat to double sharp
        accidental_offset = accidental_offsets[accidental_idx]
        
        # Calculate MIDI number
        midi_number = (octave_num + 1) * 12 + base_offset + accidental_offset
        
        # Clamp to valid MIDI range (0-127)
        midi_number = max(0, min(127, midi_number))
        
        return midi_number

    def _extract_score_features(self, score_tokens: List[Dict]) -> torch.Tensor:
        """
        Extract score features from tokens using decomposed pitch representation.
        Returns: [seq_len, 8] tensor without tempo fields
        """
        score_features = []
        
        for token in score_tokens:
            score_note = token['score_note_token']
            
            # Check if token already has decomposed format
            if 'note_idx' in score_note and 'accidental_idx' in score_note and 'octave_idx' in score_note:
                # Token already has decomposed indices
                note_idx = score_note['note_idx']
                accidental_idx = score_note['accidental_idx']
                octave_idx = score_note['octave_idx']
            else:
                # Need to decompose pitch string
                pitch_str = score_note.get('pitch', 'C4')
                note_idx, accidental_idx, octave_idx = self.pitch_decomposer.parse_pitch(pitch_str)
            
            position = round(score_note.get('position', 0.0), 2)
            duration = round(score_note.get('duration', 0.5), 2)
            is_staccato = 1 if score_note.get('is_staccato', False) else 0
            is_accent = 1 if score_note.get('is_accent', False) else 0
            
            # Convert part_id to index
            part_id_str = score_note.get('part_id', 'P1-Staff1')
            part_id = 0 if 'Staff1' in part_id_str else 1
            
            score_features.append([
                note_idx,         # 0: note name index
                accidental_idx,   # 1: accidental index
                octave_idx,       # 2: octave index
                position,         # 3: position in beats
                duration,         # 4: duration in beats
                is_staccato,      # 5: staccato flag
                is_accent,        # 6: accent flag
                part_id,          # 7: part id (0=right, 1=left)
            ])
        
        score_tensor = torch.tensor(score_features, dtype=torch.float32)
        
        # Apply clamping to duration
        score_tensor[:, 4] = torch.clamp(score_tensor[:, 4], min=0.01, max=8.0)
        
        return score_tensor

    def _create_initial_performance_tokens(self, score_tokens: List[Dict]) -> torch.Tensor:
        """Create initial performance tokens from score tokens."""
        batch_size = 1
        seq_len = len(score_tokens)
        num_perf_features = len(self.performance_feature_names)

        # Initialize performance tokens based on score tokens
        perf_tokens = torch.zeros(batch_size, seq_len, num_perf_features, dtype=torch.float32)

        for i, score_token in enumerate(score_tokens):
            score_note = score_token['score_note_token']

            # Get pitch as MIDI number
            if 'pitch' in score_note:
                pitch_str = score_note['pitch']
                pitch_int = self._pitch_str_to_midi(pitch_str)
            else:
                # If pitch is not provided, reconstruct from indices
                # This shouldn't normally happen in your use case
                pitch_int = 60  # Default to C4

            perf_tokens[0, i, 0] = pitch_int  # pitch_int
            perf_tokens[0, i, 1] = score_note.get('duration', 0.5)  # duration
            perf_tokens[0, i, 2] = 1 if score_note.get('is_staccato', False) else 0  # is_staccato
            perf_tokens[0, i, 3] = 1 if score_note.get('is_accent', False) else 0  # is_accent
            
            part_id_str = score_note.get('part_id', 'P1-Staff1')
            perf_tokens[0, i, 4] = 0 if 'Staff1' in part_id_str else 1  # part_id

            # Performance-specific features - will be generated
            perf_tokens[0, i, 5] = 0.0  # onset_deviation_in_seconds
            perf_tokens[0, i, 6] = 0.0  # duration_deviation_in_seconds
            perf_tokens[0, i, 7] = 120.0  # local_tempo (default)
            perf_tokens[0, i, 8] = 64  # velocity (default)
            perf_tokens[0, i, 9] = 0  # sustain_level (default)

        return perf_tokens.to(self.device)

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
                pitch_int = 60  # Default
        
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
        
            # Define which features are regression-based
            regression_features = ['onset_deviation_in_seconds', 'duration_deviation_in_seconds', 'velocity']
        
            for j, token_type in enumerate(performance_specific_features):
                feature_idx = j + 5  # Start from index 5
                if token_type in predictions:
                    pred_tensor = predictions[token_type]
                    
                    # Handle different prediction types
                    if token_type in regression_features:
                        # Regression predictions are [batch, seq] shape
                        if pred_tensor.dim() == 2:
                            value = pred_tensor[0, 0]  # [batch, seq]
                        elif pred_tensor.dim() == 1:
                            value = pred_tensor[0]  # [batch]
                        else:
                            value = pred_tensor.flatten()[0]
                    else:
                        # Binned or standard predictions
                        if pred_tensor.dim() == 1:
                            value = pred_tensor[0]
                        elif pred_tensor.dim() == 2:
                            value = pred_tensor[0, 0]
                        else:
                            value = pred_tensor.flatten()[0]
                    
                    # Ensure it's a scalar
                    if hasattr(value, 'item'):
                        value = value.item()
                    
                    perf_tokens[0, pos + 1, feature_idx] = value
        
            return perf_tokens

    def _safe_extract_value(self, tensor: torch.Tensor, position: int) -> float:
        """Safely extract a value from tensor at given position, handling different dimensions."""
        if tensor is None:
            return 0.0
            
        # Handle different tensor shapes
        if tensor.dim() == 0:  # scalar
            return tensor.item()
        elif tensor.dim() == 1:  # [seq] or [batch]
            if position < tensor.size(0):
                return tensor[position].item()
            else:
                return tensor[0].item()
        elif tensor.dim() == 2:  # [batch, seq]
            if position < tensor.size(1):
                return tensor[0, position].item()
            else:
                return tensor[0, 0].item()
        else:  # Higher dimensions, flatten and take first
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
            use_cache: bool = None,
            cache_start_idx: int = 0
    ) -> Tuple[torch.Tensor, List[RenderingPerformanceNoteToken], Optional[torch.Tensor]]:
        """
        Generate performance for a single window with tempo prediction.

        Returns:
            predictions_tensor: Full prediction tensor for this window
            rendering_tokens: List of RenderingPerformanceNoteToken for output
            predicted_tempo: Predicted tempo parameters [mean, std] or None
        """
        if use_cache is None:
            use_cache = False  # Default to False for non-overlapping segments

        batch_size = 1
        seq_len = len(score_tokens)

        # Extract score features using decomposed pitch
        score_features = self._extract_score_features(score_tokens).unsqueeze(0).to(self.device)

        # Create masks
        score_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=self.device)

        # Create composer_ids tensor
        composer_ids = torch.tensor([composer_id], device=self.device) if composer_id is not None else None

        # Tempo prediction and condition preparation
        predicted_tempo = None
        if self.model.use_tempo_prediction and self.use_predicted_tempo:
            # Use cached tempo if available
            if use_cache and self.tempo_cache is not None:
                predicted_tempo = self.tempo_cache
                print(f"Using cached tempo: mean={predicted_tempo[0, 0]:.1f}, std={predicted_tempo[0, 1]:.1f}")
            else:
                # Predict new tempo
                initial_condition = self.model.prepare_condition(
                    tempo_params=None,
                    composer_ids=composer_ids
                )
                score_encoded, _ = self.model.encode_score(score_features, score_mask, initial_condition)
                
                # Predict tempo
                predicted_tempo = self.model.predict_tempo(score_encoded, score_mask)
                print(f"Predicted tempo: mean={predicted_tempo[0, 0]:.1f}, std={predicted_tempo[0, 1]:.1f}")
                
                # Cache predicted tempo
                self.tempo_cache = predicted_tempo

        # Prepare final condition (with tempo and composer)
        final_condition = self.model.prepare_condition(
            tempo_params=predicted_tempo if self.use_predicted_tempo else None,
            composer_ids=composer_ids
        )

        # Encode score with final condition
        score_encoded, score_mask = self.model.encode_score(score_features, score_mask, final_condition)

        # Initialize performance tokens
        perf_tokens = self._create_initial_performance_tokens(score_tokens)

        # Autoregressive generation
        for pos in range(seq_len):
            # Current performance tokens up to position pos
            current_perf_tokens = perf_tokens[:, :pos + 1]
            perf_mask = torch.ones(batch_size, pos + 1, dtype=torch.bool, device=self.device)

            # Get logits using the model's decode_performance method
            logits = self.model.decode_performance(
                current_perf_tokens,
                score_encoded=score_encoded,
                score_mask=score_mask,
                perf_mask=perf_mask,
                condition=final_condition
            )

            # Extract logits for current position
            current_logits = {}
            for token_type, logit_tensor in logits.items():
                if isinstance(logit_tensor, dict):  # Handle pitch decomposition
                    current_logits[token_type] = {
                        k: v[:, pos:pos + 1] for k, v in logit_tensor.items()
                    }
                else:
                    current_logits[token_type] = logit_tensor[:, pos:pos + 1]

            # Sample predictions for performance-specific features only
            predictions = self._sample_from_logits(
                current_logits,
                temperature=self.temperature,
                top_p=self.top_p
            )

            # Update performance tokens for next iteration
            if pos < seq_len - 1:
                perf_tokens = self._update_performance_tokens_with_score_ground_truth(
                    perf_tokens, score_tokens, predictions, pos
                )

        # Final forward pass to get all predictions
        perf_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=self.device)
        logits = self.model.decode_performance(
            perf_tokens,
            score_encoded=score_encoded,
            score_mask=score_mask,
            perf_mask=perf_mask,
            condition=final_condition
        )

        # Get final predictions
        final_predictions = self._sample_from_logits(
            logits,
            temperature=self.temperature,
            top_p=self.top_p
        )

        # Convert to RenderingPerformanceNoteToken format
        rendering_tokens = []
        predictions_list = []

        for i in range(seq_len):
            # Extract values for this position - use safe extraction
            pred_dict = {}
            for token_type in self.performance_feature_names:
                if token_type in final_predictions:
                    # Use safe extraction method to handle different tensor dimensions
                    pred_dict[token_type] = self._safe_extract_value(final_predictions[token_type], i)
                else:
                    # Use default values for missing predictions
                    if token_type == 'pitch_int':
                        pred_dict[token_type] = 60
                    elif token_type == 'velocity':
                        pred_dict[token_type] = 64
                    elif token_type == 'local_tempo':
                        pred_dict[token_type] = 120.0
                    else:
                        pred_dict[token_type] = 0

            predictions_list.append(pred_dict)

            # Create rendering token
            rendering_token = RenderingPerformanceNoteToken(
                velocity=int(pred_dict['velocity']),
                onset_deviation_in_seconds=float(pred_dict['onset_deviation_in_seconds']),
                duration_deviation_in_seconds=float(pred_dict['duration_deviation_in_seconds']),
                local_tempo=float(pred_dict['local_tempo']),
                sustain_level=int(pred_dict['sustain_level'])
            )
            rendering_tokens.append(rendering_token)

        return final_predictions, rendering_tokens, predicted_tempo

    def generate_full_sequence(
            self,
            all_score_tokens: List[Dict],
            composer_id: int = 0
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Generate performance for a full sequence using non-overlapping segments.

        Returns:
            Tuple of:
            - List of performance tokens with segment tempo info AND score info
            - List of segment tempo predictions
        """
        total_length = len(all_score_tokens)
        all_rendering_tokens = []
        segment_tempos = []

        print(f"Generating performance for {total_length} notes using non-overlapping segments")
        print(f"Segment size: up to {self.sequence_length} tokens")
        print(f"Tempo prediction: {'Enabled' if self.use_predicted_tempo else 'Disabled'}")
        print(f"Using decomposed pitch representation")

        # Clear cache at start (no longer needed for non-overlapping)
        self.tempo_cache = None

        # Process non-overlapping segments
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

            # Generate for this segment (no cache needed for non-overlapping)
            _, rendering_tokens, segment_tempo = self.generate_window(
                segment_tokens,
                composer_id=composer_id,
                use_cache=False,  # No cache for non-overlapping segments
                cache_start_idx=0
            )

            # Store segment tempo information
            tempo_info = {
                'segment_idx': segment_idx,
                'start_token': segment_start,
                'end_token': segment_end - 1,
                'segment_size': segment_size,
                'avg_tempo': float(segment_tempo[0, 0]) if segment_tempo is not None else None,
                'std_tempo': float(segment_tempo[0, 1]) if segment_tempo is not None else None
            }
            segment_tempos.append(tempo_info)

            if segment_tempo is not None:
                print(f"  Predicted tempo: mean={segment_tempo[0, 0]:.1f} BPM, std={segment_tempo[0, 1]:.1f}")

            # Convert rendering tokens to dict format with segment tempo info AND score info
            for i, token in enumerate(rendering_tokens):
                # Get the corresponding score token (ensuring alignment)
                original_score_token = all_score_tokens[segment_start + i]
                score_note = original_score_token['score_note_token']
                
                # Create combined token with score info first, then performance info
                token_dict = {
                    # ===== SCORE INFORMATION (first) =====
                    'score': {
                        'pitch': score_note.get('pitch', 'C4'),
                        'position': score_note.get('position', 0.0),
                        'duration': score_note.get('duration', 0.5),
                        'is_staccato': score_note.get('is_staccato', False),
                        'is_accent': score_note.get('is_accent', False),
                        'part_id': score_note.get('part_id', 'P1-Staff1'),
                        # Add any other score fields that might exist
                        **{k: v for k, v in score_note.items() 
                           if k not in ['pitch', 'position', 'duration', 'is_staccato', 'is_accent', 'part_id']}
                    },
                    
                    # ===== PERFORMANCE INFORMATION (second) =====
                    'performance': {
                        'velocity': token.velocity,
                        'onset_deviation_in_seconds': token.onset_deviation_in_seconds,
                        'duration_deviation_in_seconds': token.duration_deviation_in_seconds,
                        'local_tempo': token.local_tempo,
                        'sustain_level': token.sustain_level
                    },
                    
                    # ===== METADATA =====
                    'segment_info': {
                        'segment_idx': segment_idx,
                        'segment_avg_tempo': float(segment_tempo[0, 0]) if segment_tempo is not None else None,
                        'segment_std_tempo': float(segment_tempo[0, 1]) if segment_tempo is not None else None,
                        'global_position': segment_start + i  # Global position in the full sequence
                    }
                }
                all_rendering_tokens.append(token_dict)

            print(f"  Added {len(rendering_tokens)} tokens from segment {segment_idx + 1}")

            # Move to next segment
            current_position = segment_end
            segment_idx += 1

        # Verify output length
        if len(all_rendering_tokens) != total_length:
            print(f"⚠️ WARNING: Output has {len(all_rendering_tokens)} tokens, expected {total_length}")
        else:
            print(f"\n✅ Generation complete. Generated {len(all_rendering_tokens)} tokens in {len(segment_tempos)} segments")

        return all_rendering_tokens, segment_tempos

    def generate_from_json(
            self,
            input_json_path: str,
            output_json_path: str,
            composer_id: int = 0
    ):
        """Generate performance from a JSON file and save results with segment tempo information AND score information."""
        # Load input data
        with open(input_json_path, 'r') as f:
            input_data = json.load(f)

        score_tokens = input_data['full_tokens']
        print(f"Loaded {len(score_tokens)} score tokens from {input_json_path}")
        score_tokens = sorted(score_tokens, key=lambda x: x['score_note_token']['position'])
        print(f"  Sorted {len(score_tokens)} notes by position")

        # Generate performance with segment tempos
        performance_tokens, segment_tempos = self.generate_full_sequence(score_tokens, composer_id)

        # Convert to serializable format with enhanced tempo information
        output_data = {
            'metadata': {
                'input_file': input_json_path,
                'total_notes': len(performance_tokens),
                'composer_id': composer_id,
                'generation_params': {
                    'sequence_length': self.sequence_length,
                    'temperature': self.temperature,
                    'top_p': self.top_p,
                    'use_predicted_tempo': self.use_predicted_tempo,
                    'use_decomposed_pitch': self.use_decomposed_pitch,
                    'non_overlapping_segments': True
                },
                # Segment-wise tempo predictions
                'segment_tempos': segment_tempos
            },
            # Performance tokens now include BOTH score and performance info, plus segment info
            'note_by_note_results': performance_tokens
        }

        # Save output
        output_dir = os.path.dirname(output_json_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_json_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nSaved generated performance to {output_json_path}")
        print(f"Output includes {len(segment_tempos)} segment tempo predictions")
        print(f"Each note contains: score info + performance predictions + segment metadata")


def main():
    parser = argparse.ArgumentParser(description='Generate performance from score using ScorePerformer with tempo prediction')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--input_json', type=str, required=True,
                        help='Path to input JSON file with score tokens')
    parser.add_argument('--output_json', type=str, required=True,
                        help='Path to save output JSON file')
    parser.add_argument('--config_path', type=str, default=None,
                        help='Path to model config file')
    parser.add_argument('--composer_id', type=int, default=0,
                        help='Composer ID for conditioning (0-14)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Nucleus sampling parameter')
    parser.add_argument('--sequence_length', type=int, default=512,
                        help='Sequence length for non-overlapping segments')
    parser.add_argument('--no_tempo_prediction', action='store_true',
                        help='Disable tempo prediction (use default tempo)')

    args = parser.parse_args()

    # Create generator (simplified without stride and cache options)
    generator = ScorePerformanceGenerator(
        model_path=args.model_path,
        config_path=args.config_path,
        device=args.device,
        sequence_length=args.sequence_length,
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
# python generate_new.py --model_path ./check/sustain_classify_20250815_073720/sustain_classify_20250815_073720_best.pt --input_json competition_data/12_Romances_Op.21__Sergei_Rachmaninoff_Zdes_khorosho_-_Arrangement_for_solo_piano_sorted_score_tokens.json --output_json generate_results/Rachmaninoff_12_Romances_Op_unsorted.json --composer_id 10 --config_path config.yaml --sequence_length 512

# python generate_new.py --model_path ./check/sustain_classify_20250815_073720/sustain_classify_20250815_073720_best.pt --input_json competition_data/32_Variations_in_C_minor_WoO_80_First_5_sorted_score_tokens.json --output_json generate_results/Beethoven_32_Variations_in_C_minor_WoO_80.json --composer_id 1 --config_path config.yaml --sequence_length 512

# python generate_new.py --model_path ./check/sustain_classify_20250815_073720/sustain_classify_20250815_073720_best.pt --input_json competition_data/CAPRICCIO_en_sol_mineur_HWV_483_-_Handel_sorted_score_tokens.json --output_json generate_results/Handel_but_Bach_CAPRICCIO_en_sol.json --composer_id 0 --config_path config.yaml --sequence_length 512

# python generate_new.py --model_path ./check/sustain_classify_20250815_073720/sustain_classify_20250815_073720_best.pt --input_json competition_data/With_Dog-teams_sorted_score_tokens.json --output_json generate_results/256Amy_but_Ranch_With_Dog.json --composer_id 10 --config_path config.yaml --sequence_length 512

# python generate_new.py --model_path ./check/sustain_classify_20250815_073720/sustain_classify_20250815_073720_best.pt --input_json competition_data/With_Dog-teams_sorted_score_tokens.json --output_json generate_results/Amy_but_brahms_With_Dog.json --composer_id 2 --config_path config.yaml --sequence_length 512

# 使用tempo预测：
# python generate.py --model_path ./check/regression_20250813_093907/regression_20250813_093907_best.pt --input_json test_bach.json --output_json generate_results/test_bach.json --composer_id 0 --config_path config.yaml --sequence_length 512
#
# 禁用tempo预测：

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
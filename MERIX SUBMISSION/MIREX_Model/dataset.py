import os
import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional, Union, Any
import glob
import numpy as np
import re


class PitchDecomposer:
    """
    Decompose pitch strings into note name, accidental, and octave components.
    """

    def __init__(self):
        # Note name mapping (C=0, D=1, E=2, F=3, G=4, A=5, B=6)
        self.note_to_idx = {
            'C': 0, 'D': 1, 'E': 2, 'F': 3, 'G': 4, 'A': 5, 'B': 6
        }

        # Accidental mapping (double flat to double sharp)
        self.accidental_to_idx = {
            '--': 0,  # double flat
            '-': 1,  # flat
            '': 2,  # natural (no accidental)
            '#': 3,  # sharp
            '##': 4  # double sharp
        }

        # Reverse mappings for debugging
        self.idx_to_note = {v: k for k, v in self.note_to_idx.items()}
        self.idx_to_accidental = {v: k for k, v in self.accidental_to_idx.items()}

        # Octave range: typically -1 to 9 in MIDI, but we'll be flexible
        self.min_octave = -1
        self.max_octave = 9
        self.octave_offset = abs(self.min_octave)  # to make indices positive

    def parse_pitch(self, pitch_str: str) -> Tuple[int, int, int]:
        """
        Parse pitch string into (note_idx, accidental_idx, octave_idx).

        Examples:
            'C4' -> (0, 2, 5)  # C, natural, octave 4
            'F#3' -> (3, 3, 4)  # F, sharp, octave 3
            'A--1' -> (5, 0, 2)  # A, double flat, octave 1
            'C##7' -> (0, 4, 8)  # C, double sharp, octave 7
        """
        # Regex to match note name, accidentals, and octave
        pattern = r'^([A-G])(#{1,2}|--?|)(-?\d+)$'
        match = re.match(pattern, pitch_str)

        if not match:
            # Fallback for unrecognized format
            print(f"Warning: Could not parse pitch '{pitch_str}', using C4 as fallback")
            return self.parse_pitch('C4')

        note_name, accidental, octave_str = match.groups()

        # Get indices
        note_idx = self.note_to_idx[note_name]
        accidental_idx = self.accidental_to_idx[accidental]
        octave_num = int(octave_str)
        octave_idx = octave_num + self.octave_offset

        # Clamp octave to valid range
        octave_idx = max(0, min(octave_idx, self.max_octave + self.octave_offset))

        return note_idx, accidental_idx, octave_idx

    def get_vocab_sizes(self) -> Dict[str, int]:
        """Get vocabulary sizes for each component."""
        return {
            'note_name': len(self.note_to_idx),  # 7
            'accidental': len(self.accidental_to_idx),  # 5
            'octave': self.max_octave - self.min_octave + 1  # 11 (from -1 to 9)
        }


class ScorePerformanceDataset(Dataset):
    """
    Dataset for paired score and performance data with sliding window extraction.
    Enhanced version with recalculated deviations and tempo statistics.
    """

    def __init__(self, data_dir: str, sequence_length: int = 512, stride: int = 256):
        """
        Args:
            data_dir: Directory containing JSON files
            sequence_length: Length of each sequence (default: 512)
            stride: Stride for sliding window (default: 256)
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.stride = stride
        self.use_decomposed_pitch = True

        # Composer mapping from paste.txt
        self.composer_to_id = {
            'Bach': 0,
            'Beethoven': 1,
            'Brahms': 2,
            'Chopin': 3,
            'Debussy': 4,
            'Glinka': 5,
            'Haydn': 6,
            'Liszt': 7,
            'Mozart': 8,
            'Prokofiev': 9,
            'Rachmaninoff': 10,
            'Ravel': 11,
            'Schubert': 12,
            'Schumann': 13,
            'Scriabin': 14,
            'Balakirev':15
        }

        if self.use_decomposed_pitch:
            self.pitch_decomposer = PitchDecomposer()
        else:
            # Pitch mapping from paste.txt (keeping for compatibility)
            self.pitch_str_to_idx = {
                'A##1': 0, 'A##2': 1, 'A##3': 2, 'A##4': 3, 'A##5': 4,
                'A#1': 5, 'A#2': 6, 'A#3': 7, 'A#4': 8, 'A#5': 9, 'A#6': 10,
                # ... (keeping the full mapping for backward compatibility)
                'C4': 84  # Default fallback
            }

        # Part ID mapping
        self.part_id_to_idx = {
            'P1-Staff1': 0,  # right
            'P1-Staff2': 1,  # left
            'P2-Staff1': 0,  # right
            'P2-Staff2': 1,  # left
            'P1-Staff3': 0,  # right
            'P1-Staff4': 1,  # left
        }

        # Load all JSON files and create sequences
        self.sequences = []
        self._load_all_sequences()
        self._print_sustain_stats_from_files()

        # Collect tempo statistics for the entire dataset
        self._collect_and_print_tempo_statistics()

        # Print dataset statistics
        print(f"\n📊 Dataset Statistics:")
        print(f"Total sequences created: {len(self.sequences)}")
        print(f"Sequence length: {self.sequence_length}")
        print(f"Stride: {self.stride}")


    def _extract_composer_from_path(self, score_path: str) -> str:
        """
        Extract composer name from score path.
        
        Examples:
            "datasets/asap_dataset/Bach/Fugue/bwv_846/xml_score.musicxml" -> "Bach"
            "datasets/vienna4x22/musicxml/Chopin_op38.musicxml" -> "Chopin"
            "datasets/vienna4x22/musicxml/Mozart_K331_1st-mov.musicxml" -> "Mozart"
            "datasets/vienna4x22/musicxml/Schubert_D783_no15.musicxml" -> "Schubert"
        
        Args:
            score_path: Path to the music score file
            
        Returns:
            Composer name extracted from the path, or 'Unknown' if not found
        """
        # Normalize path separators and split
        parts = score_path.replace('\\', '/').split('/')
        parts_lower = [p.lower() for p in parts]
        
        # Case 1: asap_dataset format - composer is the folder after 'asap_dataset'
        if 'asap_dataset' in parts_lower:
            idx = parts_lower.index('asap_dataset')
            if idx + 1 < len(parts):
                return parts[idx + 1].split()[0].strip().title()
        
        # Case 2: vienna4x22 format - composer is before the first underscore in filename
        elif 'vienna4x22' in parts_lower:
            filename = parts[-1].split('.')[0]
            if '_' in filename:
                return filename.split('_')[0].strip().title()

        print('no way')
        return 'Unknown'

    def _print_sustain_stats_from_files(self):
        """从每个 JSON 的 full_tokens 统计 sustain_level=0/1 的占比（去重、不受滑窗影响）"""
        json_files = glob.glob(os.path.join(self.data_dir, '*.json'))
        total_notes = 0
        sustain_one = 0

        for jf in json_files:
            try:
                with open(jf, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[warn] failed to read {jf}: {e}")
                continue

            for tok in data.get('full_tokens', []):
                # 可选过滤：只统计真正的音符 token
                if 'performance_note_token' not in tok or 'score_note_token' not in tok:
                    continue

                lvl = tok.get('sustain_level', 0)
                try:
                    lvl = int(lvl)
                except Exception:
                    lvl = 0

                # 你的下游把非零都当作 1
                if lvl != 0:
                    sustain_one += 1
                total_notes += 1

        sustain_zero = total_notes - sustain_one
        p0 = (sustain_zero / total_notes) if total_notes else 0.0
        p1 = (sustain_one / total_notes) if total_notes else 0.0

        print("\n🎹 Sustain-level distribution (unique tokens, no window double-count):")
        print(f"  level=0: {sustain_zero} ({p0:.2%})")
        print(f"  level=1: {sustain_one} ({p1:.2%})")

        # 需要的话保存到对象里，便于外部访问
        self.sustain_stats = {
            "total_notes": int(total_notes),
            "level0": int(sustain_zero),
            "level1": int(sustain_one),
            "p0": float(p0),
            "p1": float(p1),
        }

    def _load_all_sequences(self):
        """Load all JSON files and create sliding window sequences."""
        json_files = glob.glob(os.path.join(self.data_dir, '*.json'))

        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Extract composer from path
            composer = self._extract_composer_from_path(data['metadata']['score_path'])
            composer_id = self.composer_to_id.get(composer)
            if composer_id is None:
                # 打印未知作曲家及其来源文件，方便排查
                score_path = (data.get('metadata') or {}).get('score_path', 'N/A')
                print(f"[warn] Unknown composer '{composer}' from file: {json_file} | score_path: {score_path}. "
                      f"Defaulting composer_id=0.")
                composer_id = 0

            # Get full tokens
            full_tokens = data['full_tokens']
            num_tokens = len(full_tokens)

            # Create sliding window sequences
            start_indices = []

            # Regular sliding window
            for start_idx in range(0, num_tokens - self.sequence_length + 1, self.stride):
                start_indices.append(start_idx)

            # Handle the last sequence
            if num_tokens >= self.sequence_length:
                # Always include the last sequence_length tokens
                last_start = num_tokens - self.sequence_length
                if len(start_indices) == 0 or start_indices[-1] != last_start:
                    start_indices.append(last_start)

            # Create sequences
            for start_idx in start_indices:
                end_idx = start_idx + self.sequence_length
                sequence_tokens = full_tokens[start_idx:end_idx]

                self.sequences.append({
                    'tokens': sequence_tokens,
                    'composer_id': composer_id,
                    'file_path': json_file,
                    'start_idx': start_idx
                })

    def _collect_and_print_tempo_statistics(self):
        """Collect and print tempo statistics from all sequences."""
        print("\n🎵 Collecting tempo statistics from dataset...")

        all_avg_tempos = []
        all_std_tempos = []

        # Sample a subset of sequences for efficiency (or all if dataset is small)
        sample_size = min(1000, len(self.sequences))
        sample_indices = np.random.choice(len(self.sequences), sample_size, replace=False)

        for idx in sample_indices:
            tokens = self.sequences[idx]['tokens']
            avg_tempo, std_tempo = self.extract_global_tempo(tokens)
            all_avg_tempos.append(avg_tempo)
            all_std_tempos.append(std_tempo)

        # Convert to numpy arrays for statistics
        avg_tempos_array = np.array(all_avg_tempos)
        std_tempos_array = np.array(all_std_tempos)

        print(f"📈 Dataset Tempo Statistics (from {sample_size} sequences):")
        print(f"   Average Tempo:")
        print(f"     Range: [{avg_tempos_array.min():.1f}, {avg_tempos_array.max():.1f}] BPM")
        print(f"     Mean: {avg_tempos_array.mean():.1f} BPM")
        print(f"     Std: {avg_tempos_array.std():.1f} BPM")
        print(f"   Tempo Standard Deviation:")
        print(f"     Range: [{std_tempos_array.min():.1f}, {std_tempos_array.max():.1f}]")
        print(f"     Mean: {std_tempos_array.mean():.1f}")
        print(f"     Std: {std_tempos_array.std():.1f}")

        # Store statistics for potential use in model configuration
        self.tempo_stats = {
            'avg_tempo_range': (float(avg_tempos_array.min()), float(avg_tempos_array.max())),
            'avg_tempo_mean': float(avg_tempos_array.mean()),
            'avg_tempo_std': float(avg_tempos_array.std()),
            'std_tempo_range': (float(std_tempos_array.min()), float(std_tempos_array.max())),
            'std_tempo_mean': float(std_tempos_array.mean()),
            'std_tempo_std': float(std_tempos_array.std())
        }

    def extract_global_tempo(self, tokens: List[Dict], tempo_type: str = 'arithmetic') -> Tuple[float, float]:
        """
        提取全局tempo统计信息。
        支持三种不同的平均tempo计算方法。

        Args:
            tokens: List of token dictionaries
            tempo_type: 计算方法 -  'arithmetic', 'weighted'
                       - 'arithmetic': 简单算术平均
                       - 'weighted': 按duration加权平均

        Returns:
            Tuple of (global_avg_tempo, normalized_tempo_std)
        """
        # 获取所有local tempos用于计算std
        local_tempos = [token['local_tempo'] for token in tokens
                        if token.get('local_tempo') is not None and token['local_tempo'] > 0]

        if not local_tempos:
            return 120.0, 1.0

        local_tempos_array = np.array(local_tempos)

        if tempo_type == 'arithmetic':
            global_avg_tempo = float(np.mean(local_tempos_array))

        elif tempo_type == 'weighted':
            # weight with duration
            global_avg_tempo = self._calculate_weighted_tempo(tokens)

        else:
            raise ValueError(f"Unknown tempo_type: {tempo_type}. Use 'arithmetic', or 'weighted'")
        global_avg_tempo = float(np.mean(local_tempos_array))
        # 限制在合理范围内
        global_avg_tempo = np.clip(global_avg_tempo, 5, 500)

        # 计算tempo的标准差
        global_std_tempo = float(np.std(local_tempos_array))

        # 标准化std
        max_expected_std = 25.0
        global_std_tempo = 3.0 * np.log(1 + global_std_tempo) / np.log(1 + max_expected_std)

        return float(global_avg_tempo), float(global_std_tempo)

    def _calculate_weighted_tempo(self, tokens: List[Dict]) -> float:
        """
        计算加权平均tempo，权重为每个音符的duration。

        Returns:
            float: Weighted average tempo in BPM
        """
        total_weighted_tempo = 0.0
        total_duration = 0.0

        for token in tokens:
            local_tempo = token.get('local_tempo', 120.0)
            if local_tempo <= 0:
                local_tempo = 120.0

            duration = token['score_note_token']['duration']

            total_weighted_tempo += local_tempo * duration
            total_duration += duration

        if total_duration > 0:
            weighted_avg_tempo = total_weighted_tempo / total_duration
            return float(np.clip(weighted_avg_tempo, 20, 300))

        return 120.0

    def calculate_score_timing_with_avg_tempo(self, tokens: List[Dict], avg_tempo: float) -> Tuple[
        List[float], List[float]]:
        """
        使用平均tempo计算每个音符的理论演奏时间。
        使用统一的avg_tempo而不是local_tempo。

        Args:
            tokens: List of token dictionaries
            avg_tempo: Average tempo in BPM

        Returns:
            Tuple[List[float], List[float]]: (onset_seconds, duration_seconds)
        """
        if avg_tempo <= 0:
            avg_tempo = 120.0

        # 转换系数
        beats_to_seconds = 60.0 / avg_tempo

        # 计算onset和duration
        onset_seconds = []
        duration_seconds = []

        for token in tokens:
            score_token = token['score_note_token']

            # 使用avg_tempo转换
            onset_sec = score_token['position'] * beats_to_seconds
            duration_sec = score_token['duration'] * beats_to_seconds

            onset_seconds.append(onset_sec)
            duration_seconds.append(duration_sec)

        # # 计算全局时间偏移（对齐第一个音符）
        # if len(tokens) > 0:
        #     first_perf_onset = tokens[0]['performance_note_token']['onset_sec']
        #     first_score_onset = onset_seconds[0]
        #     global_offset = first_perf_onset - first_score_onset
        # 
        #     # 应用全局偏移
        #     onset_seconds = [t + global_offset for t in onset_seconds]

        return onset_seconds, duration_seconds

    def calculate_deviations_with_alignment(self, tokens: List[Dict], tempo_type: str = 'arithmetic') -> Tuple[
        List[float], List[float]]:
        """
        计算deviation，使用avg_tempo而不是local_tempo。

        Args:
            tokens: List of token dictionaries
            tempo_type: 使用哪种平均tempo计算方法

        Returns:
            Tuple[List[float], List[float]]: (onset_deviations_sec, duration_deviations_sec)
        """
        # 获取平均tempo
        avg_tempo, _ = self.extract_global_tempo(tokens, tempo_type=tempo_type)

        # 使用avg_tempo计算score timing
        score_onsets, score_durations = self.calculate_score_timing_with_avg_tempo(tokens, avg_tempo)

        # 计算原始deviations
        raw_onset_devs = []
        duration_devs = []

        for i, token in enumerate(tokens):
            perf_token = token['performance_note_token']

            raw_onset_dev = perf_token['onset_sec'] - score_onsets[i]
            raw_onset_devs.append(raw_onset_dev)

            duration_dev = perf_token['duration_sec'] - score_durations[i]
            duration_devs.append(duration_dev)

        # 对onset deviation进行对齐校正
        # 使用中位数作为基准（更稳健）
        median_offset = np.median(raw_onset_devs)

        # 校正onset deviations - 移除系统性偏移
        onset_devs_corrected = [dev - median_offset for dev in raw_onset_devs]

        return onset_devs_corrected, duration_devs

    def extract_performance_features(self, tokens: List[Dict], tempo_type: str = 'arithmetic') -> torch.Tensor:
        """
        提取performance特征，使用avg_tempo计算deviation。

        Args:
            tokens: List of token dictionaries
            tempo_type: 'arithmetic' or 'weighted'

        Returns:
            torch.Tensor: Performance features
        """
        perf_features = []

        # 计算deviations（使用avg_tempo方法）
        onset_deviations, duration_deviations = self.calculate_deviations_with_alignment(
            tokens, tempo_type=tempo_type
        )

        for i, token in enumerate(tokens):
            perf_token = token['performance_note_token']
            score_token = token['score_note_token']

            pitch_int = perf_token['pitch']
            duration = round(score_token['duration'], 2)
            is_staccato = 1 if score_token['is_staccato'] else 0
            is_accent = 1 if score_token['is_accent'] else 0
            part_id = self.part_id_to_idx.get(score_token['part_id'], 1)

            # 使用计算的deviations
            onset_deviation = round(onset_deviations[i], 3)
            duration_deviation = round(duration_deviations[i], 3)

            local_tempo = round(token['local_tempo'], 1)
            velocity = int(perf_token['velocity'])
            sustain_level = int(token.get('sustain_level', 0))
            sustain_level = 1 if sustain_level != 0 else 0

            perf_features.append([
                pitch_int,
                duration,
                is_staccato,
                is_accent,
                part_id,
                onset_deviation,
                duration_deviation,
                local_tempo,
                velocity,
                sustain_level
            ])

        return torch.tensor(perf_features, dtype=torch.float32)

    def extract_score_features(self, tokens: List[Dict], tempo_type: str = 'arithmetic') -> torch.Tensor:
        """
        提取score特征，包含全局tempo信息。

        Args:
            tokens: List of token dictionaries
            tempo_type: 'arithmetic' or 'weighted'

        Returns:
            torch.Tensor: Score features
        """
        score_features = []

        # 计算全局tempo（使用指定的方法）
        global_avg_tempo, global_std_tempo = self.extract_global_tempo(tokens, tempo_type=tempo_type)

        for token in tokens:
            score_token = token['score_note_token']

            if self.use_decomposed_pitch:
                # 使用分解的音高表示
                note_idx, accidental_idx, octave_idx = self.pitch_decomposer.parse_pitch(
                    score_token['pitch']
                )

                position = round(score_token['position'], 2)
                duration = round(score_token['duration'], 2)
                is_staccato = 1 if score_token['is_staccato'] else 0
                is_accent = 1 if score_token['is_accent'] else 0
                part_id = self.part_id_to_idx.get(score_token['part_id'], 1)

                score_features.append([
                    note_idx,
                    accidental_idx,
                    octave_idx,
                    position,
                    duration,
                    is_staccato,
                    is_accent,
                    part_id,
                    global_avg_tempo,  # 使用计算的avg_tempo
                    global_std_tempo,
                ])
            else:
                # 使用原始音高表示
                pitch_idx = self.pitch_str_to_idx.get(score_token['pitch'], 84)
                position = round(score_token['position'], 2)
                duration = round(score_token['duration'], 2)
                is_staccato = 1 if score_token['is_staccato'] else 0
                is_accent = 1 if score_token['is_accent'] else 0
                part_id = self.part_id_to_idx.get(score_token['part_id'], 1)

                score_features.append([
                    pitch_idx,
                    position,
                    duration,
                    is_staccato,
                    is_accent,
                    part_id,
                    global_avg_tempo,  # 使用计算的avg_tempo
                    global_std_tempo,
                ])

        score_tensor = torch.tensor(score_features, dtype=torch.float32)

        # 应用clamping
        clamped_score_features = clamp_score_features(score_tensor, SCORE_FEATURE_NAMES)

        return clamped_score_features

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary containing:
            - 'score_features': [seq_len, 10] tensor (includes tempo for extraction)
            - 'performance_features': [seq_len, 10] tensor
            - 'labels': [seq_len, 10] tensor (same as performance_features for AR training)
            - 'score_mask': [seq_len] tensor of ones
            - 'perf_mask': [seq_len] tensor of ones
            - 'composer_ids': scalar tensor
        """
        sequence_data = self.sequences[idx]
        tokens = sequence_data['tokens']

        # Extract features
        score_features = self.extract_score_features(tokens)
        performance_features = self.extract_performance_features(tokens)
        composer_id = torch.tensor(sequence_data['composer_id'], dtype=torch.long)

        # For autoregressive training, labels are the same as performance features
        # The model will handle the shifting internally
        labels = performance_features.clone()

        # Apply clamping to handle outliers in the dataset
        labels = clamp_performance_features(labels, PERFORMANCE_FEATURE_NAMES)

        # Since we have no padding, masks are all ones
        seq_len = len(tokens)
        score_mask = torch.ones(seq_len, dtype=torch.bool)
        perf_mask = torch.ones(seq_len, dtype=torch.bool)

        return {
            'score_features': score_features,
            'performance_features': performance_features,
            'labels': labels,
            'score_mask': score_mask,
            'perf_mask': perf_mask,
            'composer_ids': composer_id,  # Changed from 'composer_id' to 'composer_ids' for consistency
        }


def clamp_score_features(
        score_features: torch.Tensor,
        feature_names: List[str],
        clamp_config: Optional[Dict[str, Dict]] = None
) -> torch.Tensor:
    """
    Clamp score features to specified ranges to handle outliers.

    Args:
        score_features: [batch_size, seq_len, num_features] or [seq_len, num_features] tensor
        feature_names: List of feature names corresponding to the last dimension
        clamp_config: Dictionary with clamping configuration for specific features

    Returns:
        Clamped score features tensor (same shape as input)
    """
    # Default clamping configuration for score features
    default_clamp_config = {
        'duration': {
            'value_range': (0.01, 8.0),  # Minimum 0.01 beats, maximum 8 beats (double whole note)
        },
        'global_tempo_mean': {
            'value_range': (5.0, 500.0),  # Reasonable tempo range
        },
        'global_tempo_std': {
            'value_range': (0.1, 50.0),  # Reasonable std range
        }
    }

    # Merge with provided config
    clamp_config = {**default_clamp_config, **(clamp_config or {})}

    # Clone to avoid modifying original tensor
    clamped_features = score_features.clone()

    # Apply clamping for specified features
    for feature_name, config in clamp_config.items():
        if feature_name in feature_names:
            feature_idx = feature_names.index(feature_name)
            min_val, max_val = config['value_range']

            # Clamp the values
            if clamped_features.dim() == 3:  # [batch, seq_len, features]
                clamped_features[:, :, feature_idx] = torch.clamp(
                    clamped_features[:, :, feature_idx],
                    min=min_val,
                    max=max_val
                )
            else:  # [seq_len, features]
                clamped_features[:, feature_idx] = torch.clamp(
                    clamped_features[:, feature_idx],
                    min=min_val,
                    max=max_val
                )

    return clamped_features


def clamp_performance_features(
        performance_features: torch.Tensor,
        feature_names: List[str],
        clamp_config: Optional[Dict[str, Dict]] = None
) -> torch.Tensor:
    """
    Clamp performance features to specified ranges to handle outliers.

    Args:
        performance_features: [batch_size, seq_len, num_features] or [seq_len, num_features] tensor
        feature_names: List of feature names corresponding to the last dimension
        clamp_config: Dictionary with clamping configuration for specific features

    Returns:
        Clamped performance features tensor (same shape as input)
    """
    # Default clamping configuration
    default_clamp_config = {
        'onset_deviation_in_seconds': {
            'value_range': (-4.0, 4.0),
        },
        'duration_deviation_in_seconds': {
            'value_range': (-3.0, 4.0),
        },
        # 'local_tempo': {
        #     'value_range': (5.0, 500.0),
        # },
        'duration': {
            'value_range': (0.01, 8.0),  # Same as score duration (double whole note max)
        }
    }

    # Merge with provided config
    clamp_config = {**default_clamp_config, **(clamp_config or {})}

    # Clone to avoid modifying original tensor
    clamped_features = performance_features.clone()

    # Apply clamping for specified features
    for feature_name, config in clamp_config.items():
        if feature_name in feature_names:
            feature_idx = feature_names.index(feature_name)
            min_val, max_val = config['value_range']

            # Clamp the values
            if clamped_features.dim() == 3:  # [batch, seq_len, features]
                clamped_features[:, :, feature_idx] = torch.clamp(
                    clamped_features[:, :, feature_idx],
                    min=min_val,
                    max=max_val
                )
            else:  # [seq_len, features]
                clamped_features[:, feature_idx] = torch.clamp(
                    clamped_features[:, feature_idx],
                    min=min_val,
                    max=max_val
                )

    return clamped_features


# Updated feature names for reference
SCORE_FEATURE_NAMES = [
    'note_idx', 'accidental_idx', 'octave_idx', 'position', 'duration', 'is_staccato', 'is_accent', 'part_id',
    'global_tempo_mean', 'global_tempo_std'
]

PERFORMANCE_FEATURE_NAMES = [
    'pitch_int', 'duration', 'is_staccato', 'is_accent',
    'part_id', 'onset_deviation_in_seconds', 'duration_deviation_in_seconds',
    'local_tempo', 'velocity', 'sustain_level'
]


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to handle batching.

    Args:
        batch: List of dictionaries from __getitem__

    Returns:
        Dictionary containing:
        - 'score_features': [batch_size, seq_len, 10]
        - 'performance_features': [batch_size, seq_len, 10]
        - 'labels': [batch_size, seq_len, 10]
        - 'score_mask': [batch_size, seq_len]
        - 'perf_mask': [batch_size, seq_len]
        - 'composer_ids': [batch_size]
    """
    score_features = torch.stack([item['score_features'] for item in batch])
    performance_features = torch.stack([item['performance_features'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    score_mask = torch.stack([item['score_mask'] for item in batch])
    perf_mask = torch.stack([item['perf_mask'] for item in batch])
    composer_ids = torch.stack([item['composer_ids'] for item in batch])

    return {
        'score_features': score_features,
        'performance_features': performance_features,
        'labels': labels,
        'score_mask': score_mask,
        'perf_mask': perf_mask,
        'composer_ids': composer_ids
    }


import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import seaborn as sns


def analyze_deviation_distributions(dataset, batch_size=32, num_workers=4):
    """
    Analyze the distribution of onset and duration deviations across the entire dataset.

    Args:
        dataset: ScorePerformanceDataset instance
        batch_size: Batch size for data loading
        num_workers: Number of workers for data loading

    Returns:
        Dictionary containing statistics and arrays of all deviations
    """
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers
    )

    # Collect all deviations
    all_onset_deviations = []
    all_duration_deviations = []

    print("Collecting deviation data from dataset...")
    for batch in tqdm(dataloader, desc="Processing batches"):
        # Get labels (which contain the performance features)
        labels = batch['labels']

        # Extract deviations (indices based on PERFORMANCE_FEATURE_NAMES)
        # onset_deviation_in_seconds is at index 5
        # duration_deviation_in_seconds is at index 6
        onset_devs = labels[:, :, 5].cpu().numpy().flatten()
        duration_devs = labels[:, :, 6].cpu().numpy().flatten()

        all_onset_deviations.extend(onset_devs)
        all_duration_deviations.extend(duration_devs)

    # Convert to numpy arrays
    all_onset_deviations = np.array(all_onset_deviations)
    all_duration_deviations = np.array(all_duration_deviations)

    # Calculate statistics
    statistics = {
        'onset_deviation': {
            'mean': np.mean(all_onset_deviations),
            'std': np.std(all_onset_deviations),
            'min': np.min(all_onset_deviations),
            'max': np.max(all_onset_deviations),
            'median': np.median(all_onset_deviations),
            'q25': np.percentile(all_onset_deviations, 25),
            'q75': np.percentile(all_onset_deviations, 75),
            'q5': np.percentile(all_onset_deviations, 5),
            'q95': np.percentile(all_onset_deviations, 95),
            'q1': np.percentile(all_onset_deviations, 1),
            'q99': np.percentile(all_onset_deviations, 99),
        },
        'duration_deviation': {
            'mean': np.mean(all_duration_deviations),
            'std': np.std(all_duration_deviations),
            'min': np.min(all_duration_deviations),
            'max': np.max(all_duration_deviations),
            'median': np.median(all_duration_deviations),
            'q25': np.percentile(all_duration_deviations, 25),
            'q75': np.percentile(all_duration_deviations, 75),
            'q5': np.percentile(all_duration_deviations, 5),
            'q95': np.percentile(all_duration_deviations, 95),
            'q1': np.percentile(all_duration_deviations, 1),
            'q99': np.percentile(all_duration_deviations, 99),
        }
    }

    return statistics, all_onset_deviations, all_duration_deviations


def plot_deviation_distributions(onset_deviations, duration_deviations, statistics, save_path=None):
    """
    Create comprehensive visualization of deviation distributions.

    Args:
        onset_deviations: Array of onset deviations
        duration_deviations: Array of duration deviations
        statistics: Dictionary of statistics
        save_path: Optional path to save the figure
    """
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))

    # 1. Onset Deviation Histogram
    ax1 = plt.subplot(2, 3, 1)
    counts, bins, _ = ax1.hist(onset_deviations, bins=100, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(statistics['onset_deviation']['mean'], color='red', linestyle='--', linewidth=2,
                label=f"Mean: {statistics['onset_deviation']['mean']:.3f}")
    ax1.axvline(statistics['onset_deviation']['median'], color='green', linestyle='--', linewidth=2,
                label=f"Median: {statistics['onset_deviation']['median']:.3f}")
    ax1.set_xlabel('Onset Deviation (seconds)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Onset Deviation Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Duration Deviation Histogram
    ax2 = plt.subplot(2, 3, 2)
    counts, bins, _ = ax2.hist(duration_deviations, bins=100, alpha=0.7, color='orange', edgecolor='black')
    ax2.axvline(statistics['duration_deviation']['mean'], color='red', linestyle='--', linewidth=2,
                label=f"Mean: {statistics['duration_deviation']['mean']:.3f}")
    ax2.axvline(statistics['duration_deviation']['median'], color='green', linestyle='--', linewidth=2,
                label=f"Median: {statistics['duration_deviation']['median']:.3f}")
    ax2.set_xlabel('Duration Deviation (seconds)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Duration Deviation Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Box plots
    ax3 = plt.subplot(2, 3, 3)
    bp = ax3.boxplot([onset_deviations, duration_deviations],
                     labels=['Onset', 'Duration'],
                     patch_artist=True,
                     showmeans=True,
                     meanline=True)
    for patch, color in zip(bp['boxes'], ['blue', 'orange']):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax3.set_ylabel('Deviation (seconds)', fontsize=12)
    ax3.set_title('Deviation Box Plots', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. Q-Q plots for onset deviation
    ax4 = plt.subplot(2, 3, 4)
    from scipy import stats
    stats.probplot(onset_deviations, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot: Onset Deviation', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # 5. Q-Q plots for duration deviation
    ax5 = plt.subplot(2, 3, 5)
    stats.probplot(duration_deviations, dist="norm", plot=ax5)
    ax5.set_title('Q-Q Plot: Duration Deviation', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # 6. 2D Density plot
    ax6 = plt.subplot(2, 3, 6)
    # Sample if too many points
    if len(onset_deviations) > 50000:
        sample_idx = np.random.choice(len(onset_deviations), 50000, replace=False)
        onset_sample = onset_deviations[sample_idx]
        duration_sample = duration_deviations[sample_idx]
    else:
        onset_sample = onset_deviations
        duration_sample = duration_deviations

    hexbin = ax6.hexbin(onset_sample, duration_sample, gridsize=50, cmap='YlOrRd', mincnt=1)
    ax6.set_xlabel('Onset Deviation (seconds)', fontsize=12)
    ax6.set_ylabel('Duration Deviation (seconds)', fontsize=12)
    ax6.set_title('2D Density: Onset vs Duration Deviations', fontsize=14, fontweight='bold')
    plt.colorbar(hexbin, ax=ax6, label='Count')

    plt.suptitle('Deviation Analysis for Performance Dataset', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def print_statistics(statistics):
    """
    Print formatted statistics.

    Args:
        statistics: Dictionary of statistics
    """
    print("\n" + "=" * 80)
    print("DEVIATION STATISTICS SUMMARY")
    print("=" * 80)

    for deviation_type in ['onset_deviation', 'duration_deviation']:
        stats = statistics[deviation_type]
        print(f"\n📊 {deviation_type.replace('_', ' ').title()} (in seconds):")
        print("-" * 50)
        print(f"  Mean:     {stats['mean']:8.4f}")
        print(f"  Std Dev:  {stats['std']:8.4f}")
        print(f"  Median:   {stats['median']:8.4f}")
        print(f"  Min:      {stats['min']:8.4f}")
        print(f"  Max:      {stats['max']:8.4f}")
        print(f"  Range:    {stats['max'] - stats['min']:8.4f}")
        print(f"\n  Percentiles:")
        print(f"    1%:     {stats['q1']:8.4f}")
        print(f"    5%:     {stats['q5']:8.4f}")
        print(f"   25%:     {stats['q25']:8.4f}")
        print(f"   50%:     {stats['median']:8.4f}")
        print(f"   75%:     {stats['q75']:8.4f}")
        print(f"   95%:     {stats['q95']:8.4f}")
        print(f"   99%:     {stats['q99']:8.4f}")
        print(f"\n  IQR:      {stats['q75'] - stats['q25']:8.4f} (Q75 - Q25)")
        print(f"  90% CI:   [{stats['q5']:8.4f}, {stats['q95']:8.4f}]")
        print(f"  98% CI:   [{stats['q1']:8.4f}, {stats['q99']:8.4f}]")

    print("\n" + "=" * 80)

    # Additional analysis
    print("\n📈 ADDITIONAL INSIGHTS:")
    print("-" * 50)

    # Check for outliers using IQR method
    for deviation_type in ['onset_deviation', 'duration_deviation']:
        stats = statistics[deviation_type]
        iqr = stats['q75'] - stats['q25']
        lower_bound = stats['q25'] - 1.5 * iqr
        upper_bound = stats['q75'] + 1.5 * iqr

        print(f"\n{deviation_type.replace('_', ' ').title()}:")
        print(f"  IQR Method Outlier Bounds: [{lower_bound:.4f}, {upper_bound:.4f}]")
        print(f"  Values outside bounds: Min={stats['min']:.4f}, Max={stats['max']:.4f}")

        if stats['min'] < lower_bound or stats['max'] > upper_bound:
            print(f"  ⚠️  Outliers detected!")
        else:
            print(f"  ✅  No extreme outliers by IQR method")

    print("\n" + "=" * 80)


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Analyze deviation distributions in the dataset')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing JSON files')
    parser.add_argument('--sequence_length', type=int, default=512,
                        help='Sequence length for dataset')
    parser.add_argument('--stride', type=int, default=256,
                        help='Stride for sliding window')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for data loading')
    parser.add_argument('--save_fig', type=str, default='deviation_analysis.png',
                        help='Path to save the figure')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')

    args = parser.parse_args()

    # Create dataset
    print(f"Loading dataset from {args.data_dir}...")
    dataset = ScorePerformanceDataset(
        data_dir=args.data_dir,
        sequence_length=args.sequence_length,
        stride=args.stride
    )

    print(f"Total sequences: {len(dataset)}")
    print(f"Starting deviation analysis...")

    # Analyze deviations
    statistics, onset_deviations, duration_deviations = analyze_deviation_distributions(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Print statistics
    print_statistics(statistics)

    # Plot distributions
    print("\nGenerating visualization...")
    plot_deviation_distributions(
        onset_deviations,
        duration_deviations,
        statistics,
        save_path=args.save_fig
    )

    # Save statistics to JSON
    stats_file = args.save_fig.replace('.png', '_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(statistics, f, indent=2)
    print(f"\nStatistics saved to {stats_file}")

    print("\nAnalysis complete!")

# python dataset.py \
#     --data_dir enhanced_full_tokens_output_arithmetic \
#     --sequence_length 512 \
#     --stride 256 \
#     --batch_size 16 \
#     --save_fig deviation_analysis.png \
#     --num_workers 4
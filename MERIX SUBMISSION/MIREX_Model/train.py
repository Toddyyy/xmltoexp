import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import argparse
import yaml
import os
import json
from datetime import datetime
import sys
import math

# Import your modules
from dataset import ScorePerformanceDataset, collate_fn, PERFORMANCE_FEATURE_NAMES
from model.model import ScorePerformer, PERFORMANCE_TOKEN_TYPES, SCORE_TOKEN_TYPES, CONTINUOUS_TOKEN_TYPES
from trainer import AdvancedTrainer
from model.custom_boundaries import generate_binned_config_with_boundaries


class WarmupCosineAnnealingLR:
    def __init__(self, optimizer, warmup_epochs=10, max_epochs=1000,
                 min_lr=1e-6, warmup_start_lr=1e-7, auto_adjust=True):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.auto_adjust = auto_adjust
        self.current_epoch = 0

        self.patience_counter = 0
        self.best_loss = float('inf')
        self.reduction_factor = 0.5
        self.patience = 5

        print(f"📈 WarmupCosineAnnealingLR initialized:")
        print(f"   Base LR: {self.base_lr}")
        print(f"   Warmup epochs: {warmup_epochs}")
        print(f"   Min LR: {min_lr}")
        print(f"   Auto adjust: {auto_adjust}")

    def step(self, epoch=None, val_loss=None):
        """Update learning rate"""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1

        # Calculate learning rate
        if self.current_epoch < self.warmup_epochs:
            # Warmup phase: linear growth
            lr = self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * self.current_epoch / self.warmup_epochs
        else:
            # Cosine annealing phase
            progress = (self.current_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            progress = min(progress, 1.0)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

        # Adaptive adjustment
        if self.auto_adjust and val_loss is not None:
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

                # If validation loss no longer improves, reduce learning rate
                if self.patience_counter >= self.patience:
                    self.base_lr *= self.reduction_factor
                    self.patience_counter = 0
                    print(f"🔽 Auto-adjusting learning rate: {self.base_lr:.2e}")

                    # Recalculate current learning rate
                    if self.current_epoch < self.warmup_epochs:
                        lr = self.warmup_start_lr + (
                                    self.base_lr - self.warmup_start_lr) * self.current_epoch / self.warmup_epochs
                    else:
                        progress = (self.current_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
                        progress = min(progress, 1.0)
                        lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

        # Apply learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self):
        return {
            'current_epoch': self.current_epoch,
            'base_lr': self.base_lr,
            'best_loss': self.best_loss,
            'patience_counter': self.patience_counter
        }

    def load_state_dict(self, state_dict):
        self.current_epoch = state_dict.get('current_epoch', 0)
        self.base_lr = state_dict.get('base_lr', self.base_lr)
        self.best_loss = state_dict.get('best_loss', float('inf'))
        self.patience_counter = state_dict.get('patience_counter', 0)


def setup_model_config():
    """Setup model configuration based on dataset features."""
    # Score token configuration - based on dataset features
    num_score_tokens = {
        'note_idx': 7,  # Number of unique pitch strings in dataset
        'accidental_idx': 5,
        'octave_idx': 8,
        'position': 1,  # Special positional embedding
        'duration': 1,  # Continuous value
        'is_staccato': 2,  # Binary: 0, 1
        'is_accent': 2,  # Binary: 0, 1
        'part_id': 2,  # Binary: 0, 1
    }

    # Performance token configuration - based on dataset features
    num_tokens = {
        'pitch_int': 128,  # MIDI pitch range 0-127
        'duration': 1,  # Continuous value
        'is_staccato': 2,  # Binary: 0, 1
        'is_accent': 2,  # Binary: 0, 1
        'part_id': 2,  # Binary: 0, 1
        'onset_deviation_in_seconds': 1,  # Continuous embedding and Will be binned
        'duration_deviation_in_seconds': 1,  # Continuous embedding and Will be binned
        'local_tempo': 1,  # Continuous embedding and Will be binned
        'velocity': 1,  # Continuous embedding and Will be binned
        'sustain_level': 2  # Continuous embedding and Will be binned
    }

    return num_tokens, num_score_tokens


def create_dataloaders(config):
    """Create train and validation dataloaders."""
    # Create full dataset
    full_dataset = ScorePerformanceDataset(
        data_dir=config['data']['data_dir'],
        sequence_length=config['data']['sequence_length'],
        stride=config['data']['stride']
    )

    # Split dataset
    total_size = len(full_dataset)
    train_size = int(config['data']['train_split'] * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config['training']['seed'])
    )

    print(f"Dataset Split:")
    print(f"   Total samples: {total_size}")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=False
    )

    return train_dataloader, val_dataloader


def create_model(config):
    """Create Enhanced ScorePerformer model."""
    num_tokens, num_score_tokens = setup_model_config()
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
        },
    }

    # Create enhanced model with tempo prediction
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
        # New tempo prediction parameters
        use_tempo_prediction=config['model'].get('use_tempo_prediction', True),
        tempo_predictor_type=config['model'].get('tempo_predictor_type', 'standard'),
        tempo_hidden_dim=config['model'].get('tempo_hidden_dim', 256),
        tempo_fusion_type=config['model'].get('tempo_fusion_type', 'concat'),
        condition_dim=config['model'].get('condition_dim', 128),
        tempo_loss_weight=config['training'].get('tempo_loss_weight', 1.0)
    )

    return model


def create_optimizer_scheduler(model, config):
    """Create optimizer and improved learning rate scheduler."""

    learning_rate = float(config['training']['learning_rate'])
    weight_decay = float(config['training']['weight_decay'])
    adam_eps = float(config['training']['adam_eps'])

    # Optimizer
    if config['training']['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=config['training']['adam_betas'],
            eps=adam_eps,
            weight_decay=weight_decay
        )
    elif config['training']['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=config['training']['adam_betas'],
            eps=adam_eps,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config['training']['optimizer']}")

    scheduler = None
    if config['training']['use_scheduler']:
        scheduler_type = config['training'].get('scheduler_type', 'warmup_cosine')

        if scheduler_type == 'warmup_cosine':
            scheduler = WarmupCosineAnnealingLR(
                optimizer=optimizer,
                warmup_epochs=int(config['training'].get('warmup_epochs', 10)),
                max_epochs=int(config['training']['max_epochs']),
                min_lr=float(config['training'].get('min_lr', 1e-6)),
                warmup_start_lr=float(config['training'].get('warmup_start_lr', 1e-7)),
                auto_adjust=bool(config['training'].get('auto_adjust_lr', True))
            )
        elif scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(config['training']['max_epochs']),
                eta_min=float(config['training']['min_lr'])
            )
        elif scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=float(config['training']['lr_factor']),
                patience=int(config['training']['lr_patience']),
                min_lr=float(config['training']['min_lr'])
            )
        elif scheduler_type == 'cosine_warmup':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=int(config['training']['warmup_epochs']),
                T_mult=2,
                eta_min=float(config['training']['min_lr'])
            )

    return optimizer, scheduler


def load_pretrained_weights(model, pretrained_path, device):
    """
    Load pretrained weights (only model parameters)
    """
    print(f"Loading pretrained weights from: {pretrained_path}")

    try:
        checkpoint = torch.load(pretrained_path, map_location=device)

        # Only load model weights
        if 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
        else:
            # If checkpoint is directly model weights
            model_state_dict = checkpoint

        # Ensure all weights are on the correct device
        model_state_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in model_state_dict.items()}

        # Load weights (ignore mismatched keys)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in model_state_dict.items() if k in model_dict}

        # Check which weights were successfully loaded
        loaded_keys = set(pretrained_dict.keys())
        model_keys = set(model_dict.keys())

        print(f"Successfully loaded {len(loaded_keys)}/{len(model_keys)} parameters")

        if len(loaded_keys) < len(model_keys):
            missing_keys = model_keys - loaded_keys
            print(f"Missing keys: {list(missing_keys)[:5]}...")  # Only show first 5

        if len(set(model_state_dict.keys()) - model_keys) > 0:
            extra_keys = set(model_state_dict.keys()) - model_keys
            print(f"Extra keys in checkpoint: {list(extra_keys)[:5]}...")  # Only show first 5

        # Update model weights
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        print("Pretrained weights loaded successfully!")
        return True

    except Exception as e:
        print(f"❌ Error loading pretrained weights: {e}")
        return False


def check_model_stability(model, dataloader, device):
    """
    Check model numerical stability
    """
    print("Checking model stability...")
    model.eval()

    try:
        with torch.no_grad():
            batch = next(iter(dataloader))
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Ensure model is on correct device
            model = model.to(device)

            # Check if model parameters are on correct device
            param_devices = set()
            for param in model.parameters():
                param_devices.add(param.device)

            if len(param_devices) > 1:
                print(f"Model parameters on multiple devices: {param_devices}")
                # Force move all parameters to target device
                model = model.to(device)
                print(f"Force moved all model parameters to {device}")

            # Check input data device
            print(f"📍 Input data devices:")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"   {key}: {value.device}")

            # Try forward pass
            if hasattr(model, 'use_tempo_prediction') and model.use_tempo_prediction:
                # For models with tempo prediction, need to handle score_features
                score_features = batch.get('score_features')
                if score_features is not None:
                    # Simplified handling, remove last two columns (if tempo info exists)
                    if score_features.shape[-1] > 8:  # Assume tempo info exists
                        batch['score_features'] = score_features[:, :, :-2]

                output = model(
                    perf_tokens=batch['performance_features'],
                    score_tokens=batch['score_features'],
                    labels=batch['labels'],
                    score_mask=batch.get('score_mask'),
                    perf_mask=batch.get('perf_mask'),
                    composer_ids=batch.get('composer_ids'),
                    target_tempo=None,  # Don't use tempo during stability check
                    use_predicted_tempo=False,
                    teacher_force_tempo=True
                )
            else:
                output = model(
                    perf_tokens=batch['performance_features'],
                    score_tokens=batch['score_features'],
                    labels=batch['labels'],
                    score_mask=batch.get('score_mask'),
                    perf_mask=batch.get('perf_mask'),
                    composer_ids=batch.get('composer_ids')
                )

            if torch.isnan(output.loss) or torch.isinf(output.loss):
                print("❌ Model produces NaN/Inf loss!")
                print(f"   Loss value: {output.loss}")
                if output.losses:
                    print("   Individual losses:")
                    for key, val in output.losses.items():
                        print(f"     {key}: {val}")
                return False

            print(f"✅ Model stability check passed. Loss: {output.loss.item():.6f}")

            # Check individual loss components
            if output.losses:
                print("📊 Individual loss components:")
                for key, val in output.losses.items():
                    if torch.isnan(val) or torch.isinf(val):
                        print(f"   ❌ {key}: {val} (NaN/Inf detected)")
                        return False
                    else:
                        print(f"   ✅ {key}: {val.item():.6f}")

            return True

    except Exception as e:
        print(f"❌ Model stability check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='Train Enhanced ScorePerformer Model')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (loads all states)')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained model weights (loads only model parameters)')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name (overrides config)')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable wandb logging')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')

    args = parser.parse_args()

    # Check parameter conflicts
    if args.resume and args.pretrained:
        raise ValueError("Cannot use both --resume and --pretrained arguments simultaneously")

    if args.config:
        try:
            config = load_config(args.config)
            print(f"Loaded config from {args.config}")
        except Exception as e:
            print(f"Error loading config from {args.config}: {e}")
            raise FileNotFoundError(f"Configuration file {args.config} not found.")
    else:
        raise ValueError("No configuration file specified. Please provide a valid config file.")

    # Override config with command line arguments
    if args.experiment_name:
        config['trainer']['experiment_name'] = args.experiment_name
    if args.wandb:
        config['trainer']['use_wandb'] = True

    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # device = 'cpu'
    else:
        device = args.device

    print(f" Using device: {device}")

    # Set seed for reproducibility
    set_seed(config['training']['seed'])
    print(f"Set random seed to {config['training']['seed']}")

    # Create experiment directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"{config['trainer']['experiment_name']}_{timestamp}"
    config['trainer']['experiment_name'] = experiment_name

    save_dir = os.path.join(config['trainer']['save_dir'], experiment_name)
    config['trainer']['save_dir'] = save_dir
    os.makedirs(save_dir, exist_ok=True)

    # Save config
    config_save_path = os.path.join(save_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    print(f"Saved config to {config_save_path}")

    # Create dataloaders
    print("Creating dataloaders...")
    train_dataloader, val_dataloader = create_dataloaders(config)

    # Create model
    print("Creating enhanced model with tempo prediction...")
    model = create_model(config)

    # Load pretrained weights if specified
    if args.pretrained:
        success = load_pretrained_weights(model, args.pretrained, device)
        if not success:
            print("  Continuing with random initialization...")

    # Move model to device AFTER loading pretrained weights
    model = model.to(device)
    print(f" Model moved to device: {device}")

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📈 Enhanced model created:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Tempo prediction: {model.use_tempo_prediction}")
    print(f"   Composer conditioning: {model.use_composer_conditioning}")

    # Check model stability
    if not check_model_stability(model, train_dataloader, device):
        print(" Model failed stability check. Exiting...")
        return

    # Create optimizer and scheduler
    print("⚙️ Creating optimizer and improved scheduler...")
    optimizer, scheduler = create_optimizer_scheduler(model, config)

    # Define evaluation loss weights (performance-focused)
    eval_loss_weights = {
        'onset_deviation_in_seconds': 2.0,
        'duration_deviation_in_seconds': 2.0,
        'local_tempo': 1.5,
        'velocity': 1.0,
        'sustain_level': 1.0
    }

    # Create enhanced trainer with tempo training capabilities
    trainer = AdvancedTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        max_epochs=config['training']['max_epochs'],
        grad_clip_value=config['training']['grad_clip_value'],
        accumulation_steps=config['training']['accumulation_steps'],
        mixed_precision=config['training']['mixed_precision'],
        patience=config['trainer']['patience'],
        min_delta=config['trainer']['min_delta'],
        save_dir=config['trainer']['save_dir'],
        experiment_name=config['trainer']['experiment_name'],
        eval_every=config['trainer']['eval_every'],
        save_every=config['trainer']['save_every'],
        use_wandb=config['trainer']['use_wandb'],
        wandb_project=config['trainer']['wandb_project'],
        eval_loss_weights=eval_loss_weights,
        # New tempo training parameters
        tempo_schedule_type=config['training'].get('tempo_schedule_type', 'linear'),
        start_predicted_epoch=config['training'].get('start_predicted_epoch', 20),
        final_predicted_ratio=config['training'].get('final_predicted_ratio', 0.7)
    )

    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)

    # Start training
    print(f" Starting enhanced training from epoch {start_epoch}...")
    print(f" Experiment: {experiment_name}")
    print(f" Save directory: {save_dir}")
    print(f" Tempo prediction enabled: {model.use_tempo_prediction}")
    print(f" Composer conditioning: {model.use_composer_conditioning}")
    if args.pretrained:
        print(f"🔄 Loaded pretrained weights from: {args.pretrained}")
    print("=" * 80)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        trainer.save_checkpoint(len(trainer.train_history.get('total', [])), is_last=True)
        print("💾 Saved checkpoint before exit")
    except Exception as e:
        print(f"\n❌ Training failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        trainer.save_checkpoint(len(trainer.train_history.get('total', [])), is_last=True)
        print("💾 Saved checkpoint before exit")
        sys.exit(1)

    print("🎉 Enhanced training completed successfully!")


if __name__ == '__main__':
    main()

# python train.py --config config.yaml  # Train from scratch
# python train.py --config config.yaml --resume ./check/regression_20250813_093907/regression_20250813_093907_best.pt  # Resume training
# python train.py --config config.yaml --pretrained ./check/regression_20250813_093907/regression_20250813_093907_best.pt  # Using pre-trained weights
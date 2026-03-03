import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import time
import random
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import wandb
from tqdm import tqdm
from typing import Dict, Union


class TempoTrainingScheduler:
    """
    Manages tempo training schedule for multi-task learning.
    Controls when to use predicted vs ground truth tempo.
    """

    def __init__(
            self,
            schedule_type: str = "linear",  # "linear", "exponential", "step", "cosine"
            total_epochs: int = 1000,
            start_predicted_epoch: int = 150,  # When to start using predicted tempo
            final_predicted_ratio: float = 0.7  # Final ratio of predicted tempo usage
    ):
        self.schedule_type = schedule_type
        self.total_epochs = total_epochs
        self.start_predicted_epoch = start_predicted_epoch
        self.final_predicted_ratio = final_predicted_ratio

    def get_tempo_strategy(self, epoch: int) -> Dict[str, Union[bool, float]]:
        """
        Get tempo training strategy for current epoch.

        Returns:
            strategy: Dict containing training flags and probabilities
        """
        if epoch < self.start_predicted_epoch:
            # Early training: only use ground truth tempo
            return {
                "use_predicted_tempo": False,
                "teacher_force_tempo": True,
                "predicted_tempo_prob": 0.0
            }

        # Calculate progression
        progress = (epoch - self.start_predicted_epoch) / (self.total_epochs - self.start_predicted_epoch)
        progress = max(0.0, min(1.0, progress))

        if self.schedule_type == "linear":
            predicted_prob = progress * self.final_predicted_ratio
        elif self.schedule_type == "exponential":
            predicted_prob = (1 - np.exp(-3 * progress)) * self.final_predicted_ratio
        elif self.schedule_type == "cosine":
            predicted_prob = 0.5 * (1 - np.cos(np.pi * progress)) * self.final_predicted_ratio
        elif self.schedule_type == "step":
            predicted_prob = self.final_predicted_ratio if progress > 0.5 else 0.0
        else:
            predicted_prob = progress * self.final_predicted_ratio

        # Random decision for this batch
        use_predicted = random.random() < predicted_prob

        return {
            "use_predicted_tempo": use_predicted,
            "teacher_force_tempo": not use_predicted,
            "predicted_tempo_prob": predicted_prob
        }


class DataProcessor:
    """
    Processes data for the enhanced ScorePerformer model.
    Handles tempo extraction and conditioning setup.
    """

    def __init__(
            self,
            score_tempo_keys: Optional[List[str]] = None
    ):
        # Keys where tempo information might be stored in the original data
        self.score_tempo_keys = score_tempo_keys or ['global_tempo_mean', 'global_tempo_std']

    def extract_tempo_from_data(self, batch: Dict) -> Optional[torch.Tensor]:
        """
        Extract tempo information from batch data.

        Args:
            batch: Data batch from dataloader

        Returns:
            tempo_params: [batch, 2] tensor with (mean, std) or None if not available
        """
        # Try to extract from score features first
        score_features = batch.get('score_features')
        if score_features is not None and len(self.score_tempo_keys) >= 2:
            try:
                # Extract tempo values (assume they're in the last columns)
                tempo_mean = score_features[:, 0, -2]  # Assume second to last column
                tempo_std = score_features[:, 0, -1]  # Assume last column
                return torch.stack([tempo_mean, tempo_std], dim=1)
            except (IndexError, RuntimeError):
                pass

        return None

    def remove_tempo_from_score(self, score_features: torch.Tensor) -> torch.Tensor:
        """
        Remove tempo information from score features for tempo prediction training.

        Args:
            score_features: [batch, seq_len, num_features] original score features

        Returns:
            cleaned_features: Score features without tempo information
        """
        if len(self.score_tempo_keys) > 0:
            # Remove last N columns where N is number of tempo keys
            num_tempo_cols = len(self.score_tempo_keys)
            return score_features[:, :, :-num_tempo_cols]
        return score_features


class AdvancedTrainer:
    """
    Enhanced trainer with tempo prediction and modern training techniques.
    """

    def __init__(
            self,
            model: nn.Module,
            train_dataloader: DataLoader,
            val_dataloader: DataLoader,
            optimizer: optim.Optimizer,
            scheduler: Optional[object] = None,
            device: str = 'cuda',
            # Training parameters
            max_epochs: int = 100,
            grad_clip_value: float = 1.0,
            accumulation_steps: int = 1,
            mixed_precision: bool = True,
            # Early stopping parameters
            patience: int = 10,
            min_delta: float = 1e-6,
            # Save parameters
            save_dir: str = './check',
            experiment_name: str = 'score_performer',
            # Evaluation parameters
            eval_every: int = 1,
            save_every: int = 5,
            # Logging
            use_wandb: bool = False,
            wandb_project: str = 'score_performer',
            # Loss weights for evaluation
            eval_loss_weights: Optional[Dict[str, float]] = None,
            # New tempo training parameters
            tempo_schedule_type: str = "linear",
            start_predicted_epoch: int = 20,
            final_predicted_ratio: float = 0.7
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        # Training settings
        self.max_epochs = max_epochs
        self.grad_clip_value = grad_clip_value
        self.accumulation_steps = accumulation_steps
        self.mixed_precision = mixed_precision

        # Early stopping
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.early_stopped = False

        # Save settings
        self.save_dir = save_dir
        self.experiment_name = experiment_name
        os.makedirs(save_dir, exist_ok=True)

        # Evaluation settings
        self.eval_every = eval_every
        self.save_every = save_every

        # Key performance metrics for evaluation (包括tempo mean)
        self.eval_token_types = [
            'onset_deviation_in_seconds',
            'duration_deviation_in_seconds',
            'local_tempo',
            'velocity',
            'sustain_level',
            'tempo_mean'  # 添加tempo mean作为performance metric
        ]

        self.eval_loss_weights = eval_loss_weights or {
            'onset_deviation_in_seconds': 2.0,  # More important
            'duration_deviation_in_seconds': 2.0,  # More important
            'local_tempo': 1.5,
            'velocity': 1.0,
            'sustain_level': 1.0,
            'tempo_mean': 1.5  # Tempo mean也是重要的performance指标
        }

        # Initialize mixed precision scaler
        self.scaler = GradScaler() if mixed_precision else None

        # Logging
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project=wandb_project, name=experiment_name)
            wandb.watch(model)

        # Training history
        self.train_history = defaultdict(list)
        self.val_history = defaultdict(list)
        self.eval_loss_history = defaultdict(list)  # For key performance metrics

        # NEW: Tempo training components
        self.tempo_scheduler = TempoTrainingScheduler(
            schedule_type=tempo_schedule_type,
            total_epochs=max_epochs,
            start_predicted_epoch=start_predicted_epoch,
            final_predicted_ratio=final_predicted_ratio
        )
        self.data_processor = DataProcessor()

        # Model to device
        self.model.to(device)

        # Check if model supports tempo prediction
        self.has_tempo_prediction = hasattr(model, 'use_tempo_prediction') and model.use_tempo_prediction

        print(f"  trainer initialized:")
        print(f"   Device: {device}")
        print(f"   Mixed precision: {mixed_precision}")
        print(f"   Gradient clipping: {grad_clip_value}")
        print(f"   Accumulation steps: {accumulation_steps}")
        print(f"   Early stopping patience: {patience}")
        print(f"   Evaluation token types: {self.eval_token_types}")
        print(f"   Tempo prediction: {self.has_tempo_prediction}")
        if self.has_tempo_prediction:
            print(f"   Tempo schedule: {tempo_schedule_type}")
            print(f"   Start predicted epoch: {start_predicted_epoch}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch with tempo strategy."""
        self.model.train()
        epoch_losses = defaultdict(list)

        # Get tempo training strategy for this epoch
        tempo_strategy = self.tempo_scheduler.get_tempo_strategy(epoch) if self.has_tempo_prediction else {}

        pbar = tqdm(self.train_dataloader, desc=f"Training (Epoch {epoch})", leave=False)

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Process batch for tempo training
            target_tempo = None
            score_tokens = batch.get('score_features')

            if self.has_tempo_prediction and score_tokens is not None:
                # Extract tempo from original data
                target_tempo = self.data_processor.extract_tempo_from_data(batch)

                # Remove tempo from score features for prediction training
                batch['score_features'] = self.data_processor.remove_tempo_from_score(score_tokens)

            # Forward pass with mixed precision
            if self.mixed_precision:
                with autocast():
                    if self.has_tempo_prediction:
                        output = self.model(
                            perf_tokens=batch['performance_features'],
                            score_tokens=batch['score_features'],
                            labels=batch['labels'],
                            score_mask=batch.get('score_mask'),
                            perf_mask=batch.get('perf_mask'),
                            composer_ids=batch.get('composer_ids'),
                            target_tempo=target_tempo,
                            use_predicted_tempo=tempo_strategy.get('use_predicted_tempo', False),
                            teacher_force_tempo=tempo_strategy.get('teacher_force_tempo', True)
                        )
                    else:
                        # Fallback for models without tempo prediction
                        output = self.model(
                            perf_tokens=batch['performance_features'],
                            score_tokens=batch['score_features'],
                            labels=batch['labels'],
                            score_mask=batch.get('score_mask'),
                            perf_mask=batch.get('perf_mask'),
                            composer_ids=batch.get('composer_ids')
                        )
                    loss = output.loss / self.accumulation_steps
            else:
                if self.has_tempo_prediction:
                    output = self.model(
                        perf_tokens=batch['performance_features'],
                        score_tokens=batch['score_features'],
                        labels=batch['labels'],
                        score_mask=batch.get('score_mask'),
                        perf_mask=batch.get('perf_mask'),
                        composer_ids=batch.get('composer_ids'),
                        target_tempo=target_tempo,
                        use_predicted_tempo=tempo_strategy.get('use_predicted_tempo', False),
                        teacher_force_tempo=tempo_strategy.get('teacher_force_tempo', True)
                    )
                else:
                    # Fallback for models without tempo prediction
                    output = self.model(
                        perf_tokens=batch['performance_features'],
                        score_tokens=batch['score_features'],
                        labels=batch['labels'],
                        score_mask=batch.get('score_mask'),
                        perf_mask=batch.get('perf_mask'),
                        composer_ids=batch.get('composer_ids')
                    )
                loss = output.loss / self.accumulation_steps

            # Backward pass
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Accumulate losses
            epoch_losses['total'].append(loss.item() * self.accumulation_steps)
            if output.losses:
                for key, val in output.losses.items():
                    epoch_losses[key].append(val.item())

            # Add tempo-specific metrics
            if self.has_tempo_prediction:
                epoch_losses['tempo_prob'].append(tempo_strategy.get('predicted_tempo_prob', 0.0))

            # Update weights every accumulation_steps
            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.mixed_precision:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)

                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)

                    # Optimizer step
                    self.optimizer.step()

                self.optimizer.zero_grad()

            # Update progress bar
            current_loss = np.mean(epoch_losses['total'][-10:]) if epoch_losses['total'] else 0
            tempo_info = f", tempo_prob: {tempo_strategy.get('predicted_tempo_prob', 0):.2f}" if self.has_tempo_prediction else ""
            pbar.set_postfix({'loss': f'{current_loss:.4f}' + tempo_info})

        # Average losses
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        return avg_losses

    def validate_epoch(self) -> Tuple[Dict[str, float], float]:
        """Validate for one epoch and return both all losses and evaluation loss."""
        self.model.eval()
        epoch_losses = defaultdict(list)

        with torch.no_grad():
            pbar = tqdm(self.val_dataloader, desc="Validation", leave=False)

            for batch in pbar:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                # Process batch for tempo validation
                target_tempo = None
                score_tokens = batch.get('score_features')

                if self.has_tempo_prediction and score_tokens is not None:
                    # Extract tempo from original data
                    target_tempo = self.data_processor.extract_tempo_from_data(batch)

                    # Remove tempo from score features
                    batch['score_features'] = self.data_processor.remove_tempo_from_score(score_tokens)

                # Forward pass
                if self.mixed_precision:
                    with autocast():
                        if self.has_tempo_prediction:
                            output = self.model(
                                perf_tokens=batch['performance_features'],
                                score_tokens=batch['score_features'],
                                labels=batch['labels'],
                                score_mask=batch.get('score_mask'),
                                perf_mask=batch.get('perf_mask'),
                                composer_ids=batch.get('composer_ids'),
                                target_tempo=target_tempo,
                                use_predicted_tempo=True,  # Use predicted tempo for validation
                                teacher_force_tempo=False
                            )
                        else:
                            # Fallback for models without tempo prediction
                            output = self.model(
                                perf_tokens=batch['performance_features'],
                                score_tokens=batch['score_features'],
                                labels=batch['labels'],
                                score_mask=batch.get('score_mask'),
                                perf_mask=batch.get('perf_mask'),
                                composer_ids=batch.get('composer_ids')
                            )
                else:
                    if self.has_tempo_prediction:
                        output = self.model(
                            perf_tokens=batch['performance_features'],
                            score_tokens=batch['score_features'],
                            labels=batch['labels'],
                            score_mask=batch.get('score_mask'),
                            perf_mask=batch.get('perf_mask'),
                            composer_ids=batch.get('composer_ids'),
                            target_tempo=target_tempo,
                            use_predicted_tempo=True,  # Use predicted tempo for validation
                            teacher_force_tempo=False
                        )
                    else:
                        # Fallback for models without tempo prediction
                        output = self.model(
                            perf_tokens=batch['performance_features'],
                            score_tokens=batch['score_features'],
                            labels=batch['labels'],
                            score_mask=batch.get('score_mask'),
                            perf_mask=batch.get('perf_mask'),
                            composer_ids=batch.get('composer_ids')
                        )

                # Accumulate losses
                if output.loss is not None:
                    epoch_losses['total'].append(output.loss.item())
                if output.losses:
                    for key, val in output.losses.items():
                        epoch_losses[key].append(val.item())

        # Average losses
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}

        # Calculate evaluation loss (包括tempo mean)
        eval_loss = 0.0
        eval_count = 0
        for token_type in self.eval_token_types:
            if token_type in avg_losses:
                weight = self.eval_loss_weights.get(token_type, 1.0)
                eval_loss += avg_losses[token_type] * weight
                eval_count += weight

        if eval_count > 0:
            eval_loss = eval_loss / eval_count
        else:
            eval_loss = avg_losses.get('total', float('inf'))

        return avg_losses, eval_loss

    def save_checkpoint(self, epoch: int, is_best: bool = False, is_last: bool = False):
        """Save model checkpoint with tempo training state."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_val_loss': self.best_val_loss,
            'train_history': dict(self.train_history),
            'val_history': dict(self.val_history),
            'eval_loss_history': dict(self.eval_loss_history),
            # Add tempo training state
            'has_tempo_prediction': self.has_tempo_prediction,
            'tempo_scheduler_state': {
                'schedule_type': self.tempo_scheduler.schedule_type,
                'total_epochs': self.tempo_scheduler.total_epochs,
                'start_predicted_epoch': self.tempo_scheduler.start_predicted_epoch,
                'final_predicted_ratio': self.tempo_scheduler.final_predicted_ratio
            } if self.has_tempo_prediction else None
        }

        # Save different types of checkpoints
        if is_best:
            save_path = os.path.join(self.save_dir, f'{self.experiment_name}_best.pt')
            torch.save(checkpoint, save_path)
            print(f" Saved best checkpoint: {save_path}")
        else:
            save_path = os.path.join(self.save_dir, f'{self.experiment_name}_last.pt')
            torch.save(checkpoint, save_path)
            print(f" Saved last checkpoint: {save_path}")

    def log_metrics(self, epoch: int, train_losses: Dict[str, float],
                    val_losses: Dict[str, float], eval_loss: float):
        """Log metrics to console and wandb with detailed tempo information."""

        # Console logging
        print(f"\n Epoch {epoch} Results:")
        print(f"   Train Total Loss: {train_losses.get('total', 0):.6f}")
        print(f"   Val Total Loss: {val_losses.get('total', 0):.6f}")
        print(f"   Eval Loss (Performance): {eval_loss:.6f}")

        # Add tempo-specific logging with separate mean and std
        if self.has_tempo_prediction:
            tempo_strategy = self.tempo_scheduler.get_tempo_strategy(epoch)
            print(f"   Tempo Prediction Probability: {tempo_strategy['predicted_tempo_prob']:.3f}")

            # Show tempo mean and std losses separately
            if 'tempo_mean' in val_losses:
                print(f"   Tempo Mean Loss: {val_losses['tempo_mean']:.6f}")
            if 'tempo_std' in val_losses:
                print(f"   Tempo Std Loss: {val_losses['tempo_std']:.6f}")

            # original tempo prediction
            if 'tempo_prediction' in val_losses:
                print(f"   Tempo Prediction Loss (combined): {val_losses['tempo_prediction']:.6f}")

        print("\n Detailed Train Losses:")
        for key, val in train_losses.items():
            if key not in ['total', 'tempo_prob']:
                print(f"   {key}: {val:.6f}")

        print("\n Detailed Val Losses:")
        for key, val in val_losses.items():
            if key != 'total':
                print(f"   {key}: {val:.6f}")

        # Highlight evaluation metrics
        print(f"\n Key Performance Metrics (Validation):")
        for token_type in self.eval_token_types:
            if token_type in val_losses:
                weight = self.eval_loss_weights.get(token_type, 1.0)
                print(f"   {token_type}: {val_losses[token_type]:.6f} (weight: {weight})")

        # Wandb logging
        if self.use_wandb:
            log_dict = {
                'epoch': epoch,
                'train_total_loss': train_losses.get('total', 0),
                'val_total_loss': val_losses.get('total', 0),
                'eval_loss': eval_loss
            }

            # Add tempo-specific metrics
            if self.has_tempo_prediction:
                tempo_strategy = self.tempo_scheduler.get_tempo_strategy(epoch)
                log_dict['tempo_prediction_prob'] = tempo_strategy['predicted_tempo_prob']

            # Add detailed losses
            for key, val in train_losses.items():
                log_dict[f'train_{key}'] = val
            for key, val in val_losses.items():
                log_dict[f'val_{key}'] = val

            wandb.log(log_dict)

    def train(self):
        """Main training loop with tempo strategy support."""
        print(f" Starting enhanced training for {self.max_epochs} epochs...")
        if self.has_tempo_prediction:
            print(f"🎼 Tempo prediction enabled with {self.tempo_scheduler.schedule_type} schedule")
        start_time = time.time()

        for epoch in range(1, self.max_epochs + 1):
            epoch_start = time.time()

            # Training with current epoch for tempo strategy
            train_losses = self.train_epoch(epoch)

            # Validation first to get validation loss for scheduler
            val_losses = {}
            eval_loss = float('inf')
            if epoch % self.eval_every == 0:
                val_losses, eval_loss = self.validate_epoch()

            # Update learning rate scheduler
            if self.scheduler:
                if hasattr(self.scheduler, 'step'):
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        # ReduceLROnPlateau needs the metric
                        self.scheduler.step(eval_loss if eval_loss != float('inf') else train_losses['total'])
                    elif hasattr(self.scheduler, 'auto_adjust'):
                        # Our custom WarmupCosineAnnealingLR scheduler
                        current_lr = self.scheduler.step(epoch, eval_loss if eval_loss != float('inf') else None)
                        if epoch % self.eval_every == 0:
                            print(f"📊 Current learning rate: {current_lr:.2e}")
                    else:
                        self.scheduler.step()

            # Store training history
            for key, val in train_losses.items():
                self.train_history[key].append(val)
            for key, val in val_losses.items():
                self.val_history[key].append(val)

            # Store evaluation metrics history (包括tempo mean)
            if epoch % self.eval_every == 0:
                for token_type in self.eval_token_types:
                    if token_type in val_losses:
                        self.eval_loss_history[token_type].append(val_losses[token_type])

            # Logging
            if epoch % self.eval_every == 0:
                self.log_metrics(epoch, train_losses, val_losses, eval_loss)

            # Early stopping and checkpointing
            if epoch % self.eval_every == 0:
                # Check for improvement
                if eval_loss < self.best_val_loss - self.min_delta:
                    self.best_val_loss = eval_loss
                    self.epochs_without_improvement = 0
                    self.save_checkpoint(epoch, is_best=True)
                else:
                    self.epochs_without_improvement += self.eval_every

                # Early stopping check
                if self.epochs_without_improvement >= self.patience:
                    print(f" Early stopping triggered after {epoch} epochs")
                    print(f"   No improvement for {self.epochs_without_improvement} epochs")
                    self.early_stopped = True
                    break

            # Save last checkpoint
            self.save_checkpoint(epoch, is_last=True)

            # Timing
            epoch_time = time.time() - epoch_start
            if epoch % self.eval_every == 0:
                print(f"⏱️ Epoch {epoch} completed in {epoch_time:.2f}s")
                print("-" * 80)

        # Training completed
        total_time = time.time() - start_time
        print(f"\n training completed!")
        print(f"   Total time: {total_time / 3600:.2f} hours")
        print(f"   Best validation loss: {self.best_val_loss:.6f}")
        print(f"   Early stopped: {self.early_stopped}")

        # Plot final training curves
        self.plot_training_curves()

        # Save final training summary
        self.save_training_summary()

        if self.use_wandb:
            wandb.finish()

    def plot_training_curves(self):
        """Plot and save training curves for key performance metrics."""
        if len(self.eval_loss_history) == 0:
            return

        num_metrics = len(self.eval_token_types)
        # Add tempo std if available (tempo mean已经在eval_token_types中)
        if self.has_tempo_prediction and 'tempo_std' in self.val_history:
            num_metrics += 1

        # Determine subplot layout
        if num_metrics <= 3:
            rows, cols = 1, num_metrics
        elif num_metrics <= 6:
            rows, cols = 2, 3
        else:
            rows, cols = 3, 3

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
        if num_metrics == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.flatten() if cols > 1 else [axes]
        else:
            axes = axes.flatten()

        fig.suptitle('Training Progress: Performance + Tempo Metrics', fontsize=16)

        # Plot evaluation metrics
        plot_idx = 0
        for token_type in self.eval_token_types:
            if token_type in self.eval_loss_history and plot_idx < len(axes):
                ax = axes[plot_idx]
                epochs = range(1, len(self.eval_loss_history[token_type]) + 1)
                ax.plot(epochs, self.eval_loss_history[token_type], 'b-', linewidth=2)
                ax.set_title(f'{token_type}')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.grid(True, alpha=0.3)

                # Add best loss annotation
                best_loss = min(self.eval_loss_history[token_type])
                best_epoch = self.eval_loss_history[token_type].index(best_loss) + 1
                ax.annotate(f'Best: {best_loss:.6f}\nEpoch: {best_epoch}',
                            xy=(best_epoch, best_loss),
                            xytext=(10, 10), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
                plot_idx += 1

        # Plot tempo std loss if available (tempo mean已经在上面绘制了)
        if (self.has_tempo_prediction and 'tempo_std' in self.val_history
                and plot_idx < len(axes)):
            ax = axes[plot_idx]
            epochs = range(1, len(self.val_history['tempo_std']) + 1)
            ax.plot(epochs, self.val_history['tempo_std'], 'r-', linewidth=2)
            ax.set_title('Tempo Std Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)

            # Add best loss annotation
            best_loss = min(self.val_history['tempo_std'])
            best_epoch = self.val_history['tempo_std'].index(best_loss) + 1
            ax.annotate(f'Best: {best_loss:.6f}\nEpoch: {best_epoch}',
                        xy=(best_epoch, best_loss),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            plot_idx += 1

        # Remove empty subplots
        for i in range(plot_idx, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{self.experiment_name}_training_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📈 Enhanced training curves saved: {save_path}")

    def save_training_summary(self):
        """Save training summary with key performance metrics and detailed tempo info."""
        summary = {
            'experiment_name': self.experiment_name,
            'total_epochs': len(self.train_history.get('total', [])),
            'best_val_loss': self.best_val_loss,
            'early_stopped': self.early_stopped,
            'eval_token_types': self.eval_token_types,
            'eval_loss_weights': self.eval_loss_weights,
            'has_tempo_prediction': self.has_tempo_prediction,
            'final_performance_metrics': {}
        }

        # Add tempo training summary with separate mean and std
        if self.has_tempo_prediction:
            summary['tempo_training'] = {
                'schedule_type': self.tempo_scheduler.schedule_type,
                'start_predicted_epoch': self.tempo_scheduler.start_predicted_epoch,
                'final_predicted_ratio': self.tempo_scheduler.final_predicted_ratio
            }

            # Add tempo prediction performance - separate mean and std
            if 'tempo_mean' in self.val_history:
                tempo_mean_losses = self.val_history['tempo_mean']
                summary['tempo_training']['mean_final_loss'] = tempo_mean_losses[-1] if tempo_mean_losses else None
                summary['tempo_training']['mean_best_loss'] = min(tempo_mean_losses) if tempo_mean_losses else None

            if 'tempo_std' in self.val_history:
                tempo_std_losses = self.val_history['tempo_std']
                summary['tempo_training']['std_final_loss'] = tempo_std_losses[-1] if tempo_std_losses else None
                summary['tempo_training']['std_best_loss'] = min(tempo_std_losses) if tempo_std_losses else None

            # 保持兼容性
            if 'tempo_prediction' in self.val_history:
                tempo_losses = self.val_history['tempo_prediction']
                summary['tempo_training']['combined_final_loss'] = tempo_losses[-1] if tempo_losses else None
                summary['tempo_training']['combined_best_loss'] = min(tempo_losses) if tempo_losses else None

        # Add final loss values for key performance metrics (包括tempo mean)
        for token_type in self.eval_token_types:
            if token_type in self.eval_loss_history and self.eval_loss_history[token_type]:
                losses = self.eval_loss_history[token_type]
                summary['final_performance_metrics'][token_type] = {
                    'final_loss': losses[-1],
                    'best_loss': min(losses),
                    'best_epoch': losses.index(min(losses)) + 1,
                    'improvement': losses[0] - min(losses) if len(losses) > 1 else 0.0
                }

        # Save summary
        summary_path = os.path.join(self.save_dir, f'{self.experiment_name}_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"📄 Enhanced training summary saved: {summary_path}")

        # Print final performance summary
        print(f"\n🎯 Final Performance Summary:")
        for token_type, metrics in summary['final_performance_metrics'].items():
            print(f"   {token_type}:")
            print(f"     Best: {metrics['best_loss']:.6f} (epoch {metrics['best_epoch']})")
            print(f"     Final: {metrics['final_loss']:.6f}")
            print(f"     Improvement: {metrics['improvement']:.6f}")

        if self.has_tempo_prediction and 'tempo_training' in summary:
            print(f"\n🎼 Tempo Prediction Summary:")
            tempo_info = summary['tempo_training']
            if 'mean_best_loss' in tempo_info and tempo_info['mean_best_loss'] is not None:
                print(f"   Tempo Mean:")
                print(f"     Best loss: {tempo_info['mean_best_loss']:.6f}")
                print(f"     Final loss: {tempo_info['mean_final_loss']:.6f}")
            if 'std_best_loss' in tempo_info and tempo_info['std_best_loss'] is not None:
                print(f"   Tempo Std:")
                print(f"     Best loss: {tempo_info['std_best_loss']:.6f}")
                print(f"     Final loss: {tempo_info['std_final_loss']:.6f}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint for resuming training with tempo state."""
        print(f" Loading enhanced checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.train_history = defaultdict(list, checkpoint.get('train_history', {}))
        self.val_history = defaultdict(list, checkpoint.get('val_history', {}))
        self.eval_loss_history = defaultdict(list, checkpoint.get('eval_loss_history', {}))

        # Restore tempo training state if available
        if checkpoint.get('has_tempo_prediction') and self.has_tempo_prediction:
            tempo_state = checkpoint.get('tempo_scheduler_state')
            if tempo_state:
                self.tempo_scheduler = TempoTrainingScheduler(
                    schedule_type=tempo_state['schedule_type'],
                    total_epochs=tempo_state['total_epochs'],
                    start_predicted_epoch=tempo_state['start_predicted_epoch'],
                    final_predicted_ratio=tempo_state['final_predicted_ratio']
                )
                print(f"✅ Tempo training state restored")

        start_epoch = checkpoint['epoch'] + 1
        print(f"✅ Enhanced checkpoint loaded. Resuming from epoch {start_epoch}")
        return start_epoch
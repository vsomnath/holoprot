"""
Abstract class for handling model training and checkpointing
"""
import torch
import torch.nn as nn
import numpy as np
import gc
import os
import traceback
from typing import List, Dict, Tuple, Optional
from scipy import stats
import wandb
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

from holoprot.utils.metrics import METRICS, DATASET_METRICS, EVAL_METRICS

class Trainer:
    """Trainer class for training models and storing summaries."""

    def __init__(self,
                 model: nn.Module,
                 dataset: str = 'pdbbind',
                 task_type: str = 'amino_complex',
                 add_grad_noise: bool = False,
                 print_every: int = 100,
                 eval_every: int = None,
                 save_every: int = None,
                 **kwargs):
        """
        Parameters
        ----------
        model: nn.Module,
            Model to train and evaluate
        lr: float, default 0.001
            Learning rate, used only when optimizer is None
        optimizer: torch.optim.Optimizer, default None
            Optimizer used
        scheduler: torch.optim.lr_scheduler, default None,
            Learning rate scheduler used.
        print_every: int, default 100
            Print stats every print_every iterations
        eval_every: int, default None,
            Frequency of evaluation during training. If None, evaluation done
            only every epoch
        """
        self.model = model
        if add_grad_noise:
            for param in self.model.parameters():
                param.register_hook(self.grad_with_noise)
        self.print_every = print_every
        self.eval_every = eval_every
        self.save_every = save_every
        self.global_step = 0
        self.epoch_start = 0

        self._init_metric(dataset, task_type)

    def _init_metric(self, dataset, task_type):
        metrics = DATASET_METRICS.get(dataset)
        eval_metrics = EVAL_METRICS.get(dataset)
        eval_metric = eval_metrics.get(task_type)
        if not isinstance(metrics, list):
            metrics = [metrics]

        self.metrics = {}
        for metric in metrics:
            self.metrics[metric], _, _ = METRICS.get(metric)
        _, self.best_metric, self.compare_fn = METRICS.get(eval_metric)
        self.eval_metric = eval_metric

    def _update_metric(self, eval_metric: float, verbose: bool = False):
        improvement = self.compare_fn(eval_metric, self.best_metric)
        if improvement:
            self.best_metric = eval_metric
            self._save_checkpoint()
            if verbose:
                print(f"Global Step: {self.global_step}. Best eval {self.eval_metric} so far. Saving model.", flush=True)
        if verbose:
            print(flush=True)

    def build_optimizer(self, optimizer: torch.optim.Optimizer = None, learning_rate: float = 0.001, weight_decay: float = 0.0):
        if optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            self.optimizer = optimizer

    def build_scheduler(self,
                        type: str,
                        anneal_rate: float,
                        step_after: Optional[int] = None,
                        patience: Optional[int] = None,
                        thresh: Optional[float] = None):
        if type == 'exp_lr':
            self.scheduler = lr_scheduler.ExponentialLR(self.optimizer,
                                                        anneal_rate)
        elif type == 'plateau':
            if self.best_metric == np.inf:
                mode = 'min'
            elif self.best_metric == 0.0:
                mode = 'max'
            else:
                pass
            self.scheduler = lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=mode,
                patience=patience,
                factor=anneal_rate,
                threshold=thresh,
                min_lr=1e-6,
                threshold_mode='abs')
        elif type == 'step_lr':
            self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=step_after,
                                                 gamma=anneal_rate)
        else:
            self.scheduler = None

    def _save_checkpoint(self, name: str = None) -> None:
        """Saves checkpoint.

        Parameters
        ----------
        name: str, default None
            Name of the checkpoint.
        """
        save_dict = {'state': self.model.state_dict()}
        if hasattr(self.model, 'get_saveables'):
            save_dict['saveables'] = self.model.get_saveables()

        name = "best_ckpt.pt"
        save_file = os.path.join(wandb.run.dir, name)
        torch.save(save_dict, save_file)


    def log_metrics(self, metrics, mode='train'):
        metric_dict = {}
        metric_dict['iteration'] = self.global_step
        for metric in metrics:
            if metrics[metric] is not None:
                metric_dict[f"{mode}_{metric}"] = metrics[metric]

        wandb.log(metric_dict)

    def train_epochs(self,
                     train_data: DataLoader,
                     eval_data: DataLoader,
                     epochs: int = 10,
                     **kwargs) -> None:
        """Train model for given number of epochs.

        Parameters
        ----------
        data: MolGraphDataset
            Dataset to generate batches from.
        batch_size: int, default 16
            Batch size used for training
        epochs: int, default 10
            Number of epochs used for training
        """
        for epoch in range(epochs):
            print(
                f"--------- Starting Epoch: {self.epoch_start + epoch+1} ----------------",
                flush=True)
            print(flush=True)

            epoch_metrics = self._train_epoch(train_data, eval_data, **kwargs)
            for metric, val in epoch_metrics.items():
                epoch_metrics[metric] = np.round(np.mean(val), 4)
            eval_metrics = self._evaluate(eval_data, **kwargs)
            self.log_metrics(eval_metrics, mode='valid')

            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                eval_metric = eval_metrics.get(self.eval_metric, None)
                if eval_metric is not None:
                    self.scheduler.step(eval_metric)

            else:
                self.scheduler.step()

            eval_metric = eval_metrics.get(self.eval_metric, None)
            if eval_metric is not None:
                self._update_metric(eval_metric)

            print(
                f"-------- Completed Epoch: {self.epoch_start + epoch+1} Global Step: {self.global_step} ----------------",
                flush=True)
            print(f"Train Metrics: {epoch_metrics}", flush=True)
            print(f"Eval Metrics: {eval_metrics}", flush=True)
            print(
                "-----------------------------------------------------",
                flush=True)
            print(flush=True)

    def _train_epoch(self,
                     train_data: DataLoader,
                     eval_data: DataLoader = None,
                     **kwargs) -> List[np.ndarray]:
        """Train a single epoch of data.

        Parameters
        ----------
        data: MolGraphDataset
            Dataset to generate batches from, mode == eval
        batch_size: int, default 16
            batch size used for training
        """
        epoch_losses = []
        count = 0
        epoch_metrics = {}

        for inputs in train_data:
            if inputs is None:
                continue

            try:
                step_metrics = self._train_step(inputs=inputs, step_count=count, **kwargs)
                for metric, metric_val in step_metrics.items():
                    if metric not in epoch_metrics:
                        epoch_metrics[metric] = [metric_val]
                    else:
                        epoch_metrics[metric].append(metric_val)
                self.global_step += 1

                if count % self.print_every == 0:
                    print(
                        f"After {count+1} steps, Global Step: {self.global_step}",
                        flush=True)

                    metrics = epoch_metrics.copy()
                    for metric, metric_vals in metrics.items():
                        metrics[metric] = np.round(np.mean(metric_vals), 4)

                    print(f"Train Metrics so far: {metrics}", flush=True)
                    print(flush=True)
                    self.log_metrics(metrics, mode='train')

                if self.eval_every is not None:
                    if count % self.eval_every == 0 and count:
                        is_best = False
                        eval_metrics = self._evaluate(eval_data, **kwargs)
                        self.log_metrics(eval_metrics, mode='valid')
                        print(
                            f"Evaluating after {count+1} steps, Global Step: {self.global_step}",
                            flush=True)
                        print(f"Eval Metrics: {eval_metrics}", flush=True)

                        eval_metric = eval_metrics.get(self.eval_metric)
                        if eval_metric is not None:
                            self._update_metric(eval_metric, verbose=True)

                if self.save_every is not None:
                    if idx % self.save_every == 0 and idx:
                        print(
                            f"Saving model after global step {self.global_step}",
                            flush=True)
                        print(flush=True)
                        self._save_checkpoint()
                count += 1

            except Exception as e:
                print(f"Exception: {e}", flush=True)
                traceback.print_exc()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

        return epoch_metrics

    def _train_step(self, inputs: Tuple[Tuple[torch.Tensor, ...], ...],
                    step_count: int, **kwargs) -> Dict[str, float]:
        """Runs a train step.

        Parameters
        ----------
        inputs: tuple of tuples of torch.Tensors
            Inputs to the WLNDisconnect forward pass
        optimizer: torch.optim.Optimizer:
            optimizer used for gradient computation
        """
        total_loss, metrics = self.model.train_step(inputs)
        assert torch.isfinite(total_loss).all()

        accum_every = kwargs.get('accum_every', None)
        if accum_every is not None:
            apply_grad = (step_count % accum_every) == 0
            total_loss /= accum_every
            total_loss.backward()

            if apply_grad:
                if "clip_norm" in kwargs:
                    nn.utils.clip_grad_norm_(self.model.parameters(),
                                             kwargs["clip_norm"])

                self.optimizer.step()
                self.optimizer.zero_grad()

            if step_count % self.print_every == 0:
                if torch.cuda.is_available():
                    alloc_memory = torch.cuda.memory_allocated(
                    ) / 1024.0 / 1024.0
                    cached_memory = torch.cuda.memory_reserved() / 1024.0 / 1024.0
                    print(
                        f"Memory: Allocated: {alloc_memory:.3f} MB, Cache: {cached_memory:.3f} MB",
                        flush=True)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        else:
            self.optimizer.zero_grad()
            total_loss.backward()

            if "clip_norm" in kwargs:
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         kwargs["clip_norm"])

            self.optimizer.step()

            if step_count % self.print_every == 0:
                if torch.cuda.is_available():
                    alloc_memory = torch.cuda.memory_allocated(
                    ) / 1024.0 / 1024.0
                    cached_memory = torch.cuda.memory_reserved() / 1024.0 / 1024.0
                    print(
                        f"Memory: Allocated: {alloc_memory:.3f} MB, Cache: {cached_memory:.3f} MB",
                        flush=True)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for metric in metrics:
            metrics[metric] = np.round(metrics[metric], 4)

        return metrics

    def _evaluate(self, eval_data: DataLoader, **kwargs) -> Dict[str, float]:
        """Computes metrics on eval dataset.

        Parameters
        ----------
        data: MolGraphDataset
            Dataset to generate batches from, mode == eval
        batch_size: int, default 1
            batch size used for evaluation
        """
        eval_metrics = self.model.eval_step(eval_data)
        return eval_metrics

    def grad_with_noise(self, grad):
        std = np.sqrt(1.0 / (1 + self.global_step)**0.55)
        noise = std * torch.randn(tuple(grad.shape), device=grad.device)
        return grad + noise

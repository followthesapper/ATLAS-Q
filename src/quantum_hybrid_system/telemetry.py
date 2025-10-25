"""
Telemetry and Monitoring for Quantum-AQED Hybrid System
=======================================================

Tracks performance metrics, routing decisions, and MPS statistics
during training and inference.

Features:
- Real-time performance monitoring
- Routing pattern analysis
- MPS compression statistics
- Anomaly detection
- Export to TensorBoard/wandb

Author: Claude Code
Date: October 24, 2025
"""

import torch
import time
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
import json


class HybridAQEDTelemetry:
    """
    Comprehensive telemetry for hybrid AQED system.

    Tracks:
    - Layer-wise latency
    - Router decisions (tokens routed to full vs MPS)
    - MPS compression ratios
    - Memory usage
    - Gradient statistics

    Args:
        enabled: Enable/disable telemetry
        window_size: Rolling window for statistics
        log_interval: Steps between logging

    Example:
        >>> telemetry = HybridAQEDTelemetry()
        >>> with telemetry.timer('forward'):
        ...     output = model(input)
        >>> telemetry.log_router_decision(layer=0, num_routed=100, total=1000)
        >>> stats = telemetry.get_stats()
    """

    def __init__(
        self,
        enabled: bool = True,
        window_size: int = 100,
        log_interval: int = 10,
        tensorboard_writer=None,
        wandb_run=None,
    ):
        self.enabled = enabled
        self.window_size = window_size
        self.log_interval = log_interval
        self.tensorboard = tensorboard_writer
        self.wandb = wandb_run

        # Counters
        self.step = 0

        # Timers
        self.timers = defaultdict(lambda: deque(maxlen=window_size))
        self._timer_starts = {}

        # Router statistics
        self.router_stats = defaultdict(lambda: {
            'num_routed': deque(maxlen=window_size),
            'total_tokens': deque(maxlen=window_size),
            'route_frac': deque(maxlen=window_size),
        })

        # MPS statistics
        self.mps_stats = defaultdict(lambda: {
            'compression_ratio': deque(maxlen=window_size),
            'avg_bond_dim': deque(maxlen=window_size),
            'max_bond_dim': deque(maxlen=window_size),
        })

        # Memory tracking
        self.memory_stats = deque(maxlen=window_size)

        # Performance metrics
        self.throughput = deque(maxlen=window_size)  # tokens/sec

        # Anomalies
        self.anomalies = []

    def timer(self, name: str):
        """Context manager for timing operations."""
        return _Timer(self, name)

    def log_time(self, name: str, duration: float):
        """Log a timed duration."""
        if not self.enabled:
            return
        self.timers[name].append(duration)

    def log_router_decision(
        self,
        layer: int,
        num_routed: int,
        total_tokens: int,
    ):
        """Log router decision statistics."""
        if not self.enabled:
            return

        stats = self.router_stats[layer]
        stats['num_routed'].append(num_routed)
        stats['total_tokens'].append(total_tokens)
        stats['route_frac'].append(num_routed / max(total_tokens, 1))

        # Anomaly detection: sudden routing changes
        if len(stats['route_frac']) >= 2:
            prev_frac = stats['route_frac'][-2]
            curr_frac = stats['route_frac'][-1]
            if abs(curr_frac - prev_frac) > 0.3:  # 30% change
                self.anomalies.append({
                    'step': self.step,
                    'layer': layer,
                    'type': 'routing_change',
                    'prev': prev_frac,
                    'curr': curr_frac,
                })

    def log_mps_stats(
        self,
        layer: int,
        compression_ratio: float,
        avg_bond_dim: float,
        max_bond_dim: float,
    ):
        """Log MPS compression statistics."""
        if not self.enabled:
            return

        stats = self.mps_stats[layer]
        stats['compression_ratio'].append(compression_ratio)
        stats['avg_bond_dim'].append(avg_bond_dim)
        stats['max_bond_dim'].append(max_bond_dim)

    def log_memory(self, allocated_mb: float):
        """Log GPU memory usage."""
        if not self.enabled:
            return
        self.memory_stats.append(allocated_mb)

    def log_throughput(self, tokens: int, duration: float):
        """Log throughput (tokens/sec)."""
        if not self.enabled:
            return
        self.throughput.append(tokens / max(duration, 1e-9))

    def step_increment(self):
        """Increment step counter and potentially log to external systems."""
        self.step += 1

        if self.step % self.log_interval == 0:
            self._log_to_external()

    def _log_to_external(self):
        """Log current statistics to TensorBoard/wandb."""
        stats = self.get_stats()

        if self.tensorboard is not None:
            # Log to TensorBoard
            for key, value in self._flatten_dict(stats).items():
                if isinstance(value, (int, float)):
                    self.tensorboard.add_scalar(key, value, self.step)

        if self.wandb is not None:
            # Log to wandb
            self.wandb.log(self._flatten_dict(stats), step=self.step)

    def _flatten_dict(self, d: dict, parent_key: str = '', sep: str = '/') -> dict:
        """Flatten nested dict for logging."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics summary."""
        import numpy as np

        stats = {
            'step': self.step,
            'timers': {},
            'router': {},
            'mps': {},
            'memory': {},
            'throughput': {},
        }

        # Timers
        for name, times in self.timers.items():
            if len(times) > 0:
                stats['timers'][name] = {
                    'mean': float(np.mean(times)) * 1000,  # ms
                    'std': float(np.std(times)) * 1000,
                    'min': float(np.min(times)) * 1000,
                    'max': float(np.max(times)) * 1000,
                }

        # Router stats
        for layer, layer_stats in self.router_stats.items():
            if len(layer_stats['route_frac']) > 0:
                stats['router'][f'layer_{layer}'] = {
                    'avg_route_frac': float(np.mean(layer_stats['route_frac'])),
                    'std_route_frac': float(np.std(layer_stats['route_frac'])),
                }

        # MPS stats
        for layer, layer_stats in self.mps_stats.items():
            if len(layer_stats['compression_ratio']) > 0:
                stats['mps'][f'layer_{layer}'] = {
                    'avg_compression': float(np.mean(layer_stats['compression_ratio'])),
                    'avg_bond_dim': float(np.mean(layer_stats['avg_bond_dim'])),
                    'max_bond_dim': float(np.max(layer_stats['max_bond_dim'])),
                }

        # Memory
        if len(self.memory_stats) > 0:
            stats['memory'] = {
                'mean_mb': float(np.mean(self.memory_stats)),
                'max_mb': float(np.max(self.memory_stats)),
            }

        # Throughput
        if len(self.throughput) > 0:
            stats['throughput'] = {
                'mean_tokens_per_sec': float(np.mean(self.throughput)),
                'std': float(np.std(self.throughput)),
            }

        return stats

    def print_stats(self):
        """Print statistics to console."""
        stats = self.get_stats()

        print(f"\n{'='*60}")
        print(f"Hybrid AQED Telemetry (Step {self.step})")
        print(f"{'='*60}")

        # Timers
        if stats['timers']:
            print("\nTimers:")
            for name, timer_stats in stats['timers'].items():
                print(f"  {name:20s}: {timer_stats['mean']:6.2f} ± {timer_stats['std']:5.2f} ms")

        # Router
        if stats['router']:
            print("\nRouter:")
            for layer, router_stats in stats['router'].items():
                print(f"  {layer:20s}: {router_stats['avg_route_frac']*100:5.1f}% routed to full attn")

        # MPS
        if stats['mps']:
            print("\nMPS Compression:")
            for layer, mps_stats in stats['mps'].items():
                print(f"  {layer:20s}: {mps_stats['avg_compression']:5.1f}× compression, "
                      f"χ_avg={mps_stats['avg_bond_dim']:.1f}")

        # Memory
        if stats['memory']:
            print(f"\nMemory: {stats['memory']['mean_mb']:.1f} MB (max: {stats['memory']['max_mb']:.1f} MB)")

        # Throughput
        if stats['throughput']:
            print(f"\nThroughput: {stats['throughput']['mean_tokens_per_sec']:.0f} tokens/sec")

        # Anomalies
        if self.anomalies:
            print(f"\nAnomalies detected: {len(self.anomalies)}")
            for anomaly in self.anomalies[-3:]:  # Show last 3
                print(f"  Step {anomaly['step']}: {anomaly['type']} in layer {anomaly['layer']}")

        print(f"{'='*60}\n")

    def export_json(self, filepath: str):
        """Export statistics to JSON file."""
        stats = self.get_stats()
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)

    def reset(self):
        """Reset all statistics."""
        self.step = 0
        self.timers.clear()
        self.router_stats.clear()
        self.mps_stats.clear()
        self.memory_stats.clear()
        self.throughput.clear()
        self.anomalies.clear()


class _Timer:
    """Context manager for timing."""

    def __init__(self, telemetry: HybridAQEDTelemetry, name: str):
        self.telemetry = telemetry
        self.name = name
        self.start = None

    def __enter__(self):
        if self.telemetry.enabled:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.telemetry.enabled and self.start is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            duration = time.perf_counter() - self.start
            self.telemetry.log_time(self.name, duration)


class PerformanceMonitor:
    """
    Lightweight performance monitor for inference.

    Tracks only essential metrics with minimal overhead.
    """

    def __init__(self):
        self.total_tokens = 0
        self.total_time = 0.0
        self.layer_times = defaultdict(float)

    def log_batch(self, num_tokens: int, duration: float, layer_durations: Optional[dict] = None):
        """Log a batch processing."""
        self.total_tokens += num_tokens
        self.total_time += duration

        if layer_durations:
            for layer, dur in layer_durations.items():
                self.layer_times[layer] += dur

    def get_throughput(self) -> float:
        """Get tokens per second."""
        return self.total_tokens / max(self.total_time, 1e-9)

    def print_summary(self):
        """Print summary."""
        print(f"\nPerformance Monitor:")
        print(f"  Total tokens: {self.total_tokens}")
        print(f"  Total time: {self.total_time:.2f}s")
        print(f"  Throughput: {self.get_throughput():.0f} tokens/sec")

        if self.layer_times:
            print(f"\n  Layer times:")
            for layer, dur in sorted(self.layer_times.items()):
                pct = (dur / self.total_time) * 100
                print(f"    {layer}: {dur:.2f}s ({pct:.1f}%)")


# Global telemetry instance (can be configured)
_global_telemetry = None


def get_telemetry() -> Optional[HybridAQEDTelemetry]:
    """Get global telemetry instance."""
    return _global_telemetry


def set_telemetry(telemetry: HybridAQEDTelemetry):
    """Set global telemetry instance."""
    global _global_telemetry
    _global_telemetry = telemetry


def configure_telemetry(
    enabled: bool = True,
    tensorboard_writer=None,
    wandb_run=None,
    **kwargs,
) -> HybridAQEDTelemetry:
    """
    Configure and set global telemetry.

    Args:
        enabled: Enable telemetry
        tensorboard_writer: TensorBoard SummaryWriter
        wandb_run: Wandb run object
        **kwargs: Additional args for HybridAQEDTelemetry

    Returns:
        Configured telemetry instance
    """
    telemetry = HybridAQEDTelemetry(
        enabled=enabled,
        tensorboard_writer=tensorboard_writer,
        wandb_run=wandb_run,
        **kwargs,
    )
    set_telemetry(telemetry)
    return telemetry


__all__ = [
    'HybridAQEDTelemetry',
    'PerformanceMonitor',
    'get_telemetry',
    'set_telemetry',
    'configure_telemetry',
]

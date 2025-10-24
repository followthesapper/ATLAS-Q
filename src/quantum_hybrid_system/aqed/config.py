"""AQED Configuration"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class AQEDConfig:
    """Configuration for AQED Transformer

    Args:
        vocab_size: Vocabulary size
        seq_len: Maximum sequence length
        d_model: Model dimension
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout probability

        attn_keep_every: Full attention every N layers (AQED parameter)
                        1 = all layers have attention (traditional transformer)
                        4 = attention every 4 layers (2-4× speedup)
                        8 = attention every 8 layers (4-6× speedup)
        use_flash: Use memory-efficient attention (PyTorch SDPA)
        compile: Enable torch.compile for additional 2-3× speedup

    Example:
        >>> # Fast configuration (6-10× speedup)
        >>> config = AQEDConfig(
        ...     vocab_size=32000,
        ...     seq_len=4096,
        ...     attn_keep_every=8,
        ...     compile=True
        ... )
    """
    # Model architecture
    vocab_size: int = 32000
    seq_len: int = 2048
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1

    # AQED parameters (the magic!)
    attn_keep_every: int = 4  # Attention skipping - higher = faster
    use_flash: bool = True     # Memory-efficient attention
    compile: bool = True       # torch.compile optimization

    # Training (optional, for scripts)
    batch_size: int = 16
    epochs: int = 1
    lr: float = 3e-4
    weight_decay: float = 0.01
    amp: bool = True  # Automatic mixed precision

    # Logging
    log_csv: Optional[str] = None
    log_every: int = 50

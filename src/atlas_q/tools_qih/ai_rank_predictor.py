"""
AI-assisted rank predictor for tensor network compression.
Uses a lightweight neural network to predict optimal truncation rank.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class RankPredictor(nn.Module):
    """
    Neural network that predicts optimal truncation fraction given singular value spectrum.

    Input: First 64 singular values (log-normalized)
    Output: Predicted fraction of rank to keep (0.0 to 1.0)
    """

    def __init__(self, input_dim=64, hidden_dim=512):
        super().__init__()
        self.input_dim = input_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),  # Output between 0 and 1
        )

    def forward(self, sigmas, max_ranks):
        """
        Args:
            sigmas: Singular values tensor (batch_size, max_rank) or (max_rank,)
            max_ranks: Actual number of non-zero SVs for each sample (batch_size,) or scalar
        Returns:
            Predicted fraction (batch_size,) or scalar
        """
        # Handle both batched and single input
        is_batched = sigmas.dim() == 2
        if not is_batched:
            sigmas = sigmas.unsqueeze(0)
            max_ranks = torch.tensor([max_ranks], device=sigmas.device)

        batch_size = sigmas.size(0)
        x = torch.zeros(batch_size, self.input_dim, device=sigmas.device, dtype=torch.float32)

        for b in range(batch_size):
            s = sigmas[b]
            n = min(int(max_ranks[b].item()), self.input_dim)
            if n > 0 and s[0] > 0:
                normalized = s[:n] / s[0]
                x[b, :n] = torch.log1p(normalized)

        result = self.net(x).squeeze(-1)
        return result if is_batched else result.squeeze(0)

    def predict(self, sigmas):
        """
        Predict rank for given singular values.

        Args:
            sigmas: Singular values tensor (1D)
        Returns:
            Integer rank
        """
        with torch.no_grad():
            # Count non-zero singular values
            actual_rank = (sigmas > 1e-10).sum().item()
            if actual_rank == 0:
                return 1

            fraction = self.forward(sigmas, actual_rank).item()
            predicted_rank = max(1, min(int(fraction * actual_rank), actual_rank))
        return predicted_rank


class RankPredictorWrapper:
    """Wrapper for easy loading/saving and training of rank predictor."""

    def __init__(self, model_path=None, device="cuda"):
        self.device = device
        self.model = RankPredictor().to(device)

        if model_path and Path(model_path).exists():
            self.load(model_path)

    def load(self, path):
        """Load trained model from file."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        print(f"✅ Loaded rank predictor from {path}")

    def save(self, path):
        """Save model to file."""
        torch.save(self.model.state_dict(), path)
        print(f"✅ Saved rank predictor to {path}")

    def predict(self, sigmas):
        """Predict rank for given singular values."""
        return self.model.predict(sigmas)

    def train(self, train_loader, val_loader=None, epochs=100, lr=1e-3):
        """
        Train the rank predictor with batched data.

        Args:
            train_loader: DataLoader with training data
            val_loader: Optional DataLoader for validation
            epochs: Number of training epochs
            lr: Learning rate
        """
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        self.model.train()

        print(f"\nTraining rank predictor for {epochs} epochs...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training batches: {len(train_loader)}")

        best_val_loss = float("inf")

        for epoch in range(epochs):
            total_loss = 0
            self.model.train()

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch_sigmas, batch_max_ranks, batch_targets in pbar:
                batch_sigmas = batch_sigmas.to(self.device)
                batch_max_ranks = batch_max_ranks.to(self.device)
                batch_targets = batch_targets.to(self.device)

                optimizer.zero_grad()
                predictions = self.model(batch_sigmas, batch_max_ranks)
                # Use BCE loss since output is between 0 and 1
                loss = nn.functional.mse_loss(predictions, batch_targets)
                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.6f}"})

            scheduler.step()
            avg_loss = total_loss / len(train_loader)

            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_sigmas, batch_max_ranks, batch_targets in val_loader:
                        batch_sigmas = batch_sigmas.to(self.device)
                        batch_max_ranks = batch_max_ranks.to(self.device)
                        batch_targets = batch_targets.to(self.device)
                        predictions = self.model(batch_sigmas, batch_max_ranks)
                        val_loss += nn.functional.mse_loss(predictions, batch_targets).item()
                val_loss /= len(val_loader)

                # Track best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(
                        f"  ⭐ Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f} (BEST), LR: {scheduler.get_last_lr()[0]:.6f}"
                    )
                else:
                    print(
                        f"  Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}"
                    )
            else:
                print(
                    f"  Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}"
                )

        self.model.eval()
        print("✅ Training complete")
        print(f"   Best validation loss: {best_val_loss:.6f}")


def generate_training_data(n_samples=10000, max_rank=2048, device="cuda", batch_size=256):
    """
    Generate synthetic training data for rank predictor.

    Creates random singular value spectra and computes optimal truncation fractions.
    Returns DataLoader for batched training.
    """
    print(f"Generating {n_samples} training samples with max_rank={max_rank}...")

    all_sigmas = []
    all_max_ranks = []
    all_targets = []

    for _ in tqdm(range(n_samples), desc="Generating samples"):
        # Generate random spectrum (exponential decay + noise)
        rank = np.random.randint(20, max_rank)
        decay_rate = np.random.uniform(0.3, 5.0)

        # Create exponentially decaying spectrum
        sigmas = np.exp(-decay_rate * np.arange(rank) / rank)
        # Add noise to make it more realistic
        noise_level = np.random.uniform(0.01, 0.15)
        sigmas += np.random.uniform(0, noise_level, size=rank) * sigmas

        # Sort in descending order and copy to avoid negative strides
        sigmas = np.sort(sigmas)[::-1].copy()

        # Pad to max_rank
        padded_sigmas = np.zeros(max_rank, dtype=np.float32)
        padded_sigmas[:rank] = sigmas

        # Compute optimal rank for a random tolerance
        tol = np.random.choice([1e-3, 1e-4, 1e-5, 5e-5, 1e-6])
        sigmas_torch = torch.tensor(sigmas, dtype=torch.float32)
        S_squared = sigmas_torch**2
        total = S_squared.sum()
        cumsum = torch.cumsum(S_squared, dim=0)
        kept_weight = cumsum / total

        # Find rank that keeps (1-tol) of weight
        mask = kept_weight >= (1 - tol)
        if mask.any():
            target_rank = mask.nonzero(as_tuple=True)[0][0].item() + 1
        else:
            target_rank = rank

        # Store as FRACTION instead of absolute rank
        target_fraction = target_rank / rank

        all_sigmas.append(padded_sigmas)
        all_max_ranks.append(float(rank))
        all_targets.append(target_fraction)

    # Convert to tensors
    sigmas_tensor = torch.tensor(np.array(all_sigmas), dtype=torch.float32)
    max_ranks_tensor = torch.tensor(all_max_ranks, dtype=torch.float32)
    targets_tensor = torch.tensor(all_targets, dtype=torch.float32)

    # Create dataset and dataloader
    dataset = torch.utils.data.TensorDataset(sigmas_tensor, max_ranks_tensor, targets_tensor)

    # Split into train and validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    print(f"✅ Generated {len(dataset)} samples ({train_size} train, {val_size} val)")
    print(f"   Target range: {min(all_targets):.3f} to {max(all_targets):.3f} (fractions)")
    return train_loader, val_loader


# Example usage
if __name__ == "__main__":
    # Generate training data
    train_loader, val_loader = generate_training_data(n_samples=20000, batch_size=512)

    # Train predictor
    predictor = RankPredictorWrapper(device="cuda")
    predictor.train(train_loader, val_loader, epochs=50, lr=1e-3)

    # Save model
    Path("models").mkdir(exist_ok=True)
    predictor.save("models/rank_predictor.pt")

    # Test prediction
    test_sigmas = torch.linspace(1.0, 0.01, 100).cuda()
    predicted_rank = predictor.predict(test_sigmas)
    print(f"\nTest prediction: {predicted_rank} (out of {len(test_sigmas)} SVs)")

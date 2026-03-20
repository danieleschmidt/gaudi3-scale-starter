"""Tests for GaudiTrainer."""
import pytest
from gaudi3_scale.trainer import GaudiTrainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class SimpleModel:
    """Minimal model stub for testing."""
    def __init__(self, fixed_loss: float = 0.5):
        self.fixed_loss = fixed_loss
        self.steps_called = 0

    def training_step(self, batch, batch_idx: int) -> float:
        self.steps_called += 1
        return self.fixed_loss

    def validation_step(self, batch, batch_idx: int) -> float:
        return self.fixed_loss + 0.1


class FakeDataloader:
    """Simple list-backed dataloader."""
    def __init__(self, num_batches: int = 5):
        self.num_batches = num_batches

    def __iter__(self):
        for i in range(self.num_batches):
            yield [i, i + 1]


class TensorLikeModel:
    """Model that returns objects with .item() like real tensors."""
    class _Loss:
        def __init__(self, v): self.v = v
        def item(self): return self.v

    def training_step(self, batch, batch_idx): return self._Loss(0.3)
    def validation_step(self, batch, batch_idx): return self._Loss(0.4)


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

def test_init_basic():
    t = GaudiTrainer(max_epochs=5, devices=2)
    assert t.max_epochs == 5
    assert t.devices == 2
    assert t.accelerator == "hpu"


def test_init_custom_accelerator():
    t = GaudiTrainer(max_epochs=1, devices=1, accelerator="cpu")
    assert t.accelerator == "cpu"


def test_init_invalid_max_epochs():
    with pytest.raises(ValueError, match="max_epochs"):
        GaudiTrainer(max_epochs=0, devices=1)


def test_init_invalid_devices():
    with pytest.raises(ValueError, match="devices"):
        GaudiTrainer(max_epochs=1, devices=0)


def test_repr():
    t = GaudiTrainer(max_epochs=3, devices=8)
    r = repr(t)
    assert "GaudiTrainer" in r
    assert "max_epochs=3" in r
    assert "devices=8" in r


# ---------------------------------------------------------------------------
# fit() tests
# ---------------------------------------------------------------------------

def test_fit_runs_correct_epochs():
    model = SimpleModel(0.5)
    trainer = GaudiTrainer(max_epochs=3, devices=1)
    loader = FakeDataloader(num_batches=4)
    trainer.fit(model, loader)
    # 3 epochs × 4 batches = 12 calls
    assert model.steps_called == 12


def test_fit_records_train_losses():
    trainer = GaudiTrainer(max_epochs=2, devices=1)
    model = SimpleModel(fixed_loss=1.0)
    loader = FakeDataloader(num_batches=3)
    trainer.fit(model, loader)
    assert len(trainer.train_losses) == 2
    assert all(abs(l - 1.0) < 1e-9 for l in trainer.train_losses)


def test_fit_handles_tensor_loss():
    trainer = GaudiTrainer(max_epochs=1, devices=1)
    model = TensorLikeModel()
    loader = FakeDataloader(num_batches=2)
    trainer.fit(model, loader)
    assert len(trainer.train_losses) == 1
    assert abs(trainer.train_losses[0] - 0.3) < 1e-9


def test_fit_updates_current_epoch():
    trainer = GaudiTrainer(max_epochs=4, devices=1)
    loader = FakeDataloader(1)
    trainer.fit(SimpleModel(), loader)
    assert trainer.current_epoch == 3  # 0-indexed, last epoch


# ---------------------------------------------------------------------------
# validate() tests
# ---------------------------------------------------------------------------

def test_validate_returns_val_loss():
    trainer = GaudiTrainer(max_epochs=1, devices=1)
    model = SimpleModel(fixed_loss=0.7)
    loader = FakeDataloader(num_batches=3)
    metrics = trainer.validate(model, loader)
    assert "val_loss" in metrics
    assert abs(metrics["val_loss"] - 0.8) < 1e-9  # validation_step returns 0.7 + 0.1


def test_validate_records_val_losses():
    trainer = GaudiTrainer(max_epochs=1, devices=1)
    model = SimpleModel()
    loader = FakeDataloader(2)
    trainer.validate(model, loader)
    trainer.validate(model, loader)
    assert len(trainer.val_losses) == 2


def test_validate_falls_back_to_training_step():
    """When model has no validation_step, trainer should fall back to training_step."""
    class ModelNoVal:
        def training_step(self, batch, batch_idx): return 0.9

    trainer = GaudiTrainer(max_epochs=1, devices=1)
    metrics = trainer.validate(ModelNoVal(), FakeDataloader(3))
    assert abs(metrics["val_loss"] - 0.9) < 1e-9


def test_validate_tensor_loss():
    trainer = GaudiTrainer(max_epochs=1, devices=1)
    model = TensorLikeModel()
    metrics = trainer.validate(model, FakeDataloader(2))
    assert abs(metrics["val_loss"] - 0.4) < 1e-9

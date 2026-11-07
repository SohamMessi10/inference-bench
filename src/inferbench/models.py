"""The models the server can run: two real digit classifiers and a synthetic stand-in.

Every model says what it is (`kind` is "real" or "synthetic"), so a benchmark result can
never pass off a sleep call as model inference.

Real models are small neural networks trained here, on the handwritten digits dataset that
ships with scikit-learn (no download). Training uses scikit-learn; serving uses only numpy,
running the learned weights as plain matrix products. That keeps the server light and makes
the cost of each batch size easy to reason about.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

INPUT_SIZE = 64          # 8×8 pixels
PIXEL_MAX = 16.0         # the dataset's pixel values run from 0 to 16
TEST_FRACTION = 0.25
SPLIT_SEED = 0
TRAIN_SEED = 0

DIGITS_MODELS = {
    "digits-small": (256, 256),
    "digits-large": (1024, 1024),
}


class ModelError(Exception):
    """A model can't be trained, saved, or loaded."""


@dataclass(frozen=True)
class ModelInfo:
    name: str
    kind: str                      # "real" or "synthetic"
    input_size: int
    description: str
    details: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)


class Model:
    """Anything the server can run: takes a batch of inputs, returns one digit per input."""

    info: ModelInfo

    def predict(self, batch: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def load_digits_split():
    """The fixed train/test split every part of the project uses: (X_train, X_test, y_train, y_test).

    Pixels are left as 0–16, the form clients send; models scale them themselves.
    """
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    X, y = load_digits(return_X_y=True)
    return train_test_split(X, y, test_size=TEST_FRACTION, random_state=SPLIT_SEED, stratify=y)


class DigitsMLP(Model):
    """A trained multi-layer perceptron, run with numpy: ReLU hidden layers, then a linear layer."""

    def __init__(self, name: str, weights: list[np.ndarray], biases: list[np.ndarray], details: dict):
        if len(weights) != len(biases) or not weights:
            raise ModelError(f"{name}: weights and biases don't match")
        self.weights = [np.ascontiguousarray(w, dtype=np.float32) for w in weights]
        self.biases = [np.ascontiguousarray(b, dtype=np.float32) for b in biases]
        if self.weights[0].shape[0] != INPUT_SIZE or self.weights[-1].shape[1] != 10:
            raise ModelError(f"{name}: expected 64 inputs and 10 outputs, got "
                             f"{self.weights[0].shape[0]} and {self.weights[-1].shape[1]}")
        hidden = [w.shape[1] for w in self.weights[:-1]]
        parameters = sum(w.size + b.size for w, b in zip(self.weights, self.biases))
        self.info = ModelInfo(
            name=name, kind="real", input_size=INPUT_SIZE,
            description=f"neural network classifying 8×8 handwritten digits, hidden layers {hidden}",
            details={**details, "hidden_layers": hidden, "parameters": parameters},
        )

    def logits(self, batch: np.ndarray) -> np.ndarray:
        h = np.asarray(batch, dtype=np.float32) / np.float32(PIXEL_MAX)
        last = len(self.weights) - 1
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ w + b
            if i < last:
                np.maximum(h, 0, out=h)
        return h

    def predict(self, batch: np.ndarray) -> np.ndarray:
        return self.logits(batch).argmax(axis=1)

    def save(self, path: Path) -> None:
        arrays = {f"w{i}": w for i, w in enumerate(self.weights)} | {f"b{i}": b for i, b in enumerate(self.biases)}
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, **arrays, details=np.array(json.dumps(self.info.details)))
        tmp.replace(path)  # never leave a half-written weights file behind

    @classmethod
    def load(cls, name: str, path: Path) -> DigitsMLP:
        if not path.exists():
            raise ModelError(f"{path} doesn't exist; train it first with: ibench train {name}")
        try:
            with np.load(path, allow_pickle=False) as data:
                layers = sum(1 for key in data.files if key.startswith("w"))
                weights = [data[f"w{i}"] for i in range(layers)]
                biases = [data[f"b{i}"] for i in range(layers)]
                details = json.loads(str(data["details"]))
        except (OSError, ValueError, KeyError) as exc:
            raise ModelError(f"{path} isn't a readable weights file ({exc}); train it again") from None
        details = {k: v for k, v in details.items() if k not in ("hidden_layers", "parameters")}
        return cls(name, weights, biases, details)


def train_digits(name: str, models_dir: Path) -> DigitsMLP:
    """Train one of DIGITS_MODELS, measure it on the held-out split, save it, and return it."""
    if name not in DIGITS_MODELS:
        raise ModelError(f"unknown model {name!r}; choose from {', '.join(DIGITS_MODELS)}")
    import sklearn
    from sklearn.neural_network import MLPClassifier

    X_train, X_test, y_train, y_test = load_digits_split()
    started = time.perf_counter()
    trained = MLPClassifier(hidden_layer_sizes=DIGITS_MODELS[name], max_iter=300,
                            random_state=TRAIN_SEED).fit(X_train / PIXEL_MAX, y_train)
    seconds = time.perf_counter() - started
    details = {
        "train_images": len(X_train), "test_images": len(X_test),
        "split_seed": SPLIT_SEED, "train_seed": TRAIN_SEED, "train_seconds": round(seconds, 1),
        "scikit_learn": sklearn.__version__, "numpy": np.__version__,
    }
    model = DigitsMLP(name, trained.coefs_, trained.intercepts_, details)
    # Accuracy of the numpy version that will actually be served, not of the scikit-learn object.
    accuracy = float((model.predict(X_test) == y_test).mean())
    model = DigitsMLP(name, model.weights, model.biases, details | {"test_accuracy": round(accuracy, 4)})
    models_dir.mkdir(parents=True, exist_ok=True)
    model.save(models_dir / f"{name}.npz")
    return model


class SyntheticModel(Model):
    """A stand-in with a controlled cost: fixed_ms per call plus per_item_ms per input.

    It waits with time.sleep, which releases Python's lock the way a call to an accelerator
    does, so it behaves like a device where batching spreads a fixed cost over many inputs.
    Its "prediction" is a fixed function of the input, so responses can be checked.
    """

    def __init__(self, fixed_ms: float = 5.0, per_item_ms: float = 0.5):
        for label, value in (("fixed_ms", fixed_ms), ("per_item_ms", per_item_ms)):
            if not 0 <= value <= 10_000:
                raise ModelError(f"{label} must be between 0 and 10000, got {value}")
        self.fixed_ms, self.per_item_ms = fixed_ms, per_item_ms
        self.info = ModelInfo(
            name="synthetic", kind="synthetic", input_size=INPUT_SIZE,
            description=f"NOT A MODEL: waits {fixed_ms} ms per call + {per_item_ms} ms per input",
            details={"fixed_ms": fixed_ms, "per_item_ms": per_item_ms},
        )

    def predict(self, batch: np.ndarray) -> np.ndarray:
        batch = np.asarray(batch, dtype=np.float32)
        time.sleep((self.fixed_ms + self.per_item_ms * len(batch)) / 1000)
        return (batch.sum(axis=1).astype(np.int64)) % 10


class InjectedFailure(RuntimeError):
    """A failure FlakyModel raised on purpose."""


class FlakyModel(Model):
    """Wraps a model and makes a fraction of its calls fail, for testing error handling.

    Failures come from a seeded random generator, so a run with the same calls in the same order
    fails the same calls. The wrapped model's description says it is failing on purpose.
    """

    def __init__(self, model: Model, fail_rate: float, seed: int = 0):
        if not 0 <= fail_rate <= 1:
            raise ModelError(f"fail_rate must be between 0 and 1, got {fail_rate}")
        self.model, self.fail_rate = model, fail_rate
        self.rng = np.random.default_rng(seed)
        self.lock = threading.Lock()  # several model threads call predict; numpy's generator isn't thread-safe
        info = model.info
        self.info = ModelInfo(info.name, info.kind, info.input_size,
                              f"{info.description} (failing {fail_rate:.0%} of calls on purpose)",
                              {**info.details, "injected_fail_rate": fail_rate})

    def predict(self, batch: np.ndarray) -> np.ndarray:
        with self.lock:
            fail = self.rng.random() < self.fail_rate
        if fail:
            raise InjectedFailure("injected failure")
        return self.model.predict(batch)


def load_model(name: str, models_dir: Path, **synthetic) -> Model:
    if name == "synthetic":
        return SyntheticModel(**synthetic)
    if name in DIGITS_MODELS:
        return DigitsMLP.load(name, models_dir / f"{name}.npz")
    raise ModelError(f"unknown model {name!r}; choose from {', '.join([*DIGITS_MODELS, 'synthetic'])}")

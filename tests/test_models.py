import time

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from inferbench.models import (
    DIGITS_MODELS, PIXEL_MAX, TRAIN_SEED, DigitsMLP, ModelError, SyntheticModel, load_model, train_digits,
)


@pytest.fixture(scope="module")
def small(models_dir):
    return load_model("digits-small", models_dir)


def test_small_model_is_accurate_on_held_out_images(small, digits):
    _, X_test, _, y_test = digits
    accuracy = (small.predict(X_test) == y_test).mean()
    assert accuracy >= 0.95
    assert small.info.details["test_accuracy"] == round(float(accuracy), 4)
    assert small.info.details["test_images"] == len(X_test) == 450


def test_numpy_inference_matches_scikit_learn(small, digits):
    # The served model is plain numpy; it must compute what the trained scikit-learn network
    # computes. Comparing only the top answer once let a ReLU on the output layer slip
    # through (it rarely changes the top answer), so the probabilities are compared too.
    from sklearn.neural_network import MLPClassifier

    X_train, X_test, y_train, _ = digits
    reference = MLPClassifier(hidden_layer_sizes=DIGITS_MODELS["digits-small"], max_iter=300,
                              random_state=TRAIN_SEED).fit(X_train / PIXEL_MAX, y_train)
    assert (small.predict(X_test) == reference.predict(X_test / PIXEL_MAX)).all()
    logits = small.logits(X_test).astype(np.float64)
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(probabilities, reference.predict_proba(X_test / PIXEL_MAX), atol=1e-4)


def test_model_describes_itself(small):
    info = small.info
    assert (info.name, info.kind, info.input_size) == ("digits-small", "real", 64)
    assert info.details["hidden_layers"] == [256, 256]
    assert info.details["parameters"] == 64 * 256 + 256 + 256 * 256 + 256 + 256 * 10 + 10


def test_save_and_load_round_trip(small, tmp_path, digits):
    path = tmp_path / "copy.npz"
    small.save(path)
    again = DigitsMLP.load("digits-small", path)
    assert again.info == small.info
    assert (again.predict(digits[1]) == small.predict(digits[1])).all()
    assert not list(tmp_path.glob("*.tmp*"))


def test_loading_problems_are_explained(tmp_path):
    with pytest.raises(ModelError, match="train it first with: ibench train digits-large"):
        load_model("digits-large", tmp_path)
    (tmp_path / "digits-small.npz").write_bytes(b"not a weights file")
    with pytest.raises(ModelError, match="isn't a readable weights file"):
        load_model("digits-small", tmp_path)
    with pytest.raises(ModelError, match="unknown model"):
        load_model("gpt", tmp_path)
    with pytest.raises(ModelError, match="unknown model"):
        train_digits("gpt", tmp_path)


def test_weights_with_the_wrong_shape_are_refused():
    with pytest.raises(ModelError, match="64 inputs and 10 outputs"):
        DigitsMLP("x", [np.zeros((63, 10))], [np.zeros(10)], {})
    with pytest.raises(ModelError, match="don't match"):
        DigitsMLP("x", [np.zeros((64, 10))], [], {})


pixels = arrays(np.float32, st.tuples(st.integers(1, 40), st.just(64)),
                elements=st.floats(0, 16, width=32))


@settings(deadline=None)
@given(pixels)
def test_a_batch_gives_the_same_answers_as_one_image_at_a_time(small, batch):
    # Batching must not change results, or a batching experiment would trade accuracy for
    # speed without saying so. Float32 sums can differ in the last bits between a batch and a
    # single row, so compare logits closely and predictions wherever the top two aren't tied.
    together = small.logits(batch)
    alone = np.vstack([small.logits(row[None, :]) for row in batch])
    np.testing.assert_allclose(together, alone, rtol=1e-4, atol=1e-4)
    top2 = np.sort(alone, axis=1)[:, -2:]
    clear = (top2[:, 1] - top2[:, 0]) > 1e-3
    assert (together.argmax(1)[clear] == alone.argmax(1)[clear]).all()


def test_large_model_trains_and_is_accurate(tmp_path, digits):
    model = train_digits("digits-large", tmp_path)  # about 20 seconds
    assert model.info.details["hidden_layers"] == [1024, 1024]
    assert (model.predict(digits[1]) == digits[3]).mean() >= 0.95
    assert (tmp_path / "digits-large.npz").exists()


def test_synthetic_model_is_labelled_and_takes_the_time_it_says():
    model = SyntheticModel(fixed_ms=20, per_item_ms=2)
    assert model.info.kind == "synthetic" and model.info.description.startswith("NOT A MODEL")
    batch = np.full((10, 64), 3.0)
    started = time.perf_counter()
    predictions = model.predict(batch)
    elapsed_ms = (time.perf_counter() - started) * 1000
    assert 40 <= elapsed_ms < 80  # 20 + 2 × 10 = 40 ms, plus scheduling slack
    assert predictions.tolist() == [192 % 10] * 10  # 64 × 3 = 192: a fixed, checkable answer


@pytest.mark.parametrize("fixed,per_item", [(-1, 0), (0, -0.5), (10_001, 0)])
def test_synthetic_costs_must_be_sensible(fixed, per_item):
    with pytest.raises(ModelError):
        SyntheticModel(fixed, per_item)

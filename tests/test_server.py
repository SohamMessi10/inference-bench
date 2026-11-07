import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from inferbench.models import SyntheticModel, load_model
from inferbench.server import MAX_BODY_BYTES, MAX_IMAGES, create_app


@pytest.fixture(scope="module")
def model(models_dir):
    return load_model("digits-small", models_dir)


@pytest.fixture(scope="module")
def client(model):
    return TestClient(create_app(model), base_url="http://localhost")


def post(client, body, **headers):
    return client.post("/v1/predict", content=json.dumps(body), headers={"content-type": "application/json", **headers})


def test_predicts_real_images(client, model, digits):
    images = digits[1][:16]
    response = post(client, {"inputs": images.tolist()})
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "digits-small"
    assert body["predictions"] == model.predict(images).tolist()
    timing = body["timing"]
    assert timing["batch_size"] == 16 and timing["inference_ms"] > 0 and timing["queue_ms"] >= 0


def test_whole_number_pixels_are_accepted(client):
    assert post(client, {"inputs": [[0] * 32 + [16] * 32]}).status_code == 200


def test_model_info_and_health(client):
    info = client.get("/v1/model").json()
    assert (info["name"], info["kind"], info["input_size"]) == ("digits-small", "real", 64)
    assert info["details"]["test_accuracy"] >= 0.95
    assert info["server"] == {"threads": 4, "max_images": 64}
    assert client.get("/health").json() == {"status": "ok"}


def test_synthetic_server_says_it_is_synthetic():
    client = TestClient(create_app(SyntheticModel(fixed_ms=0, per_item_ms=0)), base_url="http://localhost")
    assert client.get("/v1/model").json()["kind"] == "synthetic"
    assert post(client, {"inputs": [[1.0] * 64]}).json()["predictions"] == [4]


image = [1.0] * 64


@pytest.mark.parametrize("body", [
    {},
    {"inputs": []},
    {"inputs": [image] * (MAX_IMAGES + 1)},
    {"inputs": [[1.0] * 63]},
    {"inputs": [[1.0] * 65]},
    {"inputs": [[17.0] + [1.0] * 63]},
    {"inputs": [[-0.5] + [1.0] * 63]},
    {"inputs": [["1"] + [1.0] * 63]},
    {"inputs": [[True] + [1.0] * 63]},
    {"inputs": [[None] + [1.0] * 63]},
    {"inputs": image},
    {"inputs": [image], "extra": 1},
    [image],
])
def test_invalid_inputs_are_refused(client, body):
    assert post(client, body).status_code == 422


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_pixels_are_refused(client, value):
    # Regression: FastAPI's default error response echoed the NaN back, the JSON encoder
    # failed on it, and the server answered 500 instead of 422.
    raw = '{"inputs": [[' + value + ', ' + ", ".join(["1.0"] * 63) + "]]}"
    response = client.post("/v1/predict", content=raw, headers={"content-type": "application/json"})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "inputs", "0", "0"]


def test_error_responses_point_at_the_problem_without_echoing_inputs(client):
    response = post(client, {"inputs": [[99.0] * 64] * 64})
    body = response.json()
    assert response.status_code == 422 and body["errors"] == 64 * 64
    assert len(body["detail"]) == 20 and "99" not in response.text
    assert body["detail"][0]["msg"] == "Input should be less than or equal to 16"


def test_body_that_is_not_json_is_refused(client):
    assert client.post("/v1/predict", content="{nope", headers={"content-type": "application/json"}).status_code == 422


@pytest.mark.parametrize("threads", [0, 65])
def test_thread_count_must_be_sensible(model, threads):
    with pytest.raises(ValueError, match="threads"):
        create_app(model, threads=threads)


def test_request_guards(client):
    huge = "x" * (MAX_BODY_BYTES + 1)
    assert client.post("/v1/predict", content=huge, headers={"content-type": "application/json"}).status_code == 413
    assert post(client, {"inputs": [image]}, origin="https://evil.example").status_code == 403
    assert client.get("/health", headers={"host": "evil.example"}).status_code == 400
    assert post(client, {"inputs": [image]}, origin="http://localhost").status_code == 200


def test_requests_without_a_length_are_refused(client):
    def chunks():
        yield b'{"inputs": [['
        yield b", ".join([b"1.0"] * 64) + b"]]}"

    response = client.post("/v1/predict", content=chunks(), headers={"content-type": "application/json"})
    assert response.status_code == 411


def test_prediction_matches_one_image_at_a_time(client, model, digits):
    images = digits[1][:MAX_IMAGES]
    together = post(client, {"inputs": images.tolist()}).json()["predictions"]
    alone = [post(client, {"inputs": [row.tolist()]}).json()["predictions"][0] for row in images]
    assert together == alone == model.predict(np.asarray(images)).tolist()

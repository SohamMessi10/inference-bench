import os

from hypothesis import settings

# Local runs stay fast; CI tries more random cases to search harder for counterexamples.
settings.register_profile("dev", max_examples=100)
settings.register_profile("ci", max_examples=1000)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))


import pytest  # noqa: E402

from inferbench.models import load_digits_split, train_digits  # noqa: E402


@pytest.fixture(scope="session")
def models_dir(tmp_path_factory):
    """A folder with digits-small trained once for the whole test run (about 3 seconds)."""
    folder = tmp_path_factory.mktemp("models")
    train_digits("digits-small", folder)
    return folder


@pytest.fixture(scope="session")
def digits():
    """The fixed split: (X_train, X_test, y_train, y_test)."""
    return load_digits_split()

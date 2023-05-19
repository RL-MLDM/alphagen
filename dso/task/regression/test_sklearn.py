"""Tests for sklearn interface."""

import pytest
import numpy as np

from dso import DeepSymbolicRegressor
from dso.test.generate_test_data import CONFIG_TRAINING_OVERRIDE


@pytest.fixture
def model():
    return DeepSymbolicRegressor()


def test_task(model):
    """Test regression for various configs."""

    # Generate some data
    np.random.seed(0)
    X = np.random.random(size=(10, 3))
    y = np.random.random(size=(10,))

    model.config_training.update(CONFIG_TRAINING_OVERRIDE)
    model.fit(X, y)

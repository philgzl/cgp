import matlab.engine
import numpy as np
import pytest


@pytest.fixture(scope="session")
def eng():
    eng = matlab.engine.start_matlab()
    eng.addpath(eng.genpath("tests/matlab"))
    return eng


@pytest.fixture(scope="function")
def rng():
    return np.random.default_rng(seed=0)

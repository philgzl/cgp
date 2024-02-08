import matlab.engine
import pytest


@pytest.fixture(scope="session")
def eng():
    eng = matlab.engine.start_matlab()
    eng.addpath(eng.genpath('tests/matlab'))
    return eng

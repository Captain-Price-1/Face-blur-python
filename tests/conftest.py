from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_video() -> Path:
    p = FIXTURES / "sample_5s.mp4"
    if not p.exists():
        pytest.skip("tests/fixtures/sample_5s.mp4 not present — see plan Task 4")
    return p

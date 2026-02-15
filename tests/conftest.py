from collections.abc import Iterator, MutableMapping

import structlog
import pytest


@pytest.fixture(autouse=True)
def log_events() -> Iterator[list[MutableMapping[str, object]]]:

    with structlog.testing.capture_logs() as captured:
        yield captured

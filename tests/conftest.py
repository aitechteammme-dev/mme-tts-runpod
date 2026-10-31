import inspect
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import handler as h


@pytest.fixture(autouse=True)
def unwrap_generator_for_legacy_tests(monkeypatch):
    """RunPod streaming handler yields dict responses.
    Legacy unit tests call handler(job) synchronously and expect the response dict directly.
    Unwrap single-item generator responses for legacy tests.
    """
    orig_handler = h.handler

    def wrapped_handler(job):
        res = orig_handler(job)
        if inspect.isgenerator(res):
            items = list(res)
            return items[0] if items else {}
        return res

    monkeypatch.setattr(h, "handler", wrapped_handler)

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


@pytest.fixture(autouse=True)
def _patch_plt_show(monkeypatch):
    """Patch plt.show so the test suite never blocks on interactive plots."""
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

"""The offline path must import without the optional `data` extra.

`pyproject.toml` says the extra is "not needed for tests, the demo, or CI".
That was documentation rather than fact: `data.providers.__init__` imports
every provider eagerly so the `@register` decorators run, several providers
import the HTTP client, and the client imported `requests` at module scope. So
the fully synthetic, network-free path could not be imported without a network
library it never calls.

The consequence was not subtle and it was still missed for a week: CI was red
on both supported Python versions and the Pages build failed on every push,
while the same suite passed on developer machines that happened to have
`requests` installed for other reasons. A claim about dependencies that only
holds where the dependency is present is not a claim, so it is now a test.

The check runs in a subprocess because it works by making modules unimportable,
which cannot be undone cleanly inside the running interpreter.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

#: Everything in the `data` extra. Blocking the lot is the point: the offline
#: path should need none of it, not merely need less of it.
OPTIONAL_MODULES = ("requests", "yfinance", "lxml", "bs4", "pyarrow", "dotenv", "tqdm", "openpyxl")

#: What must import with the extra absent. The CLI is here because `eee demo`
#: and `eee holdout` are the entry points CI and the Pages workflow call.
OFFLINE_MODULES = (
    "earnings_engine",
    "earnings_engine.cli",
    "earnings_engine.pipeline",
    "earnings_engine.data.providers",
    "earnings_engine.data.providers.synthetic",
    "earnings_engine.data.providers.local",
)

_SCRIPT = textwrap.dedent(
    """
    import sys

    BLOCKED = {blocked!r}

    class Blocker:
        def find_spec(self, name, path=None, target=None):
            root = name.split(".")[0]
            if root in BLOCKED:
                raise ModuleNotFoundError(f"No module named {{root!r}}")
            return None

    sys.meta_path.insert(0, Blocker())

    for name in {modules!r}:
        __import__(name)
    print("ok")
    """
)


def _run_with_blocked(modules: tuple[str, ...]) -> subprocess.CompletedProcess:
    script = _SCRIPT.format(blocked=set(OPTIONAL_MODULES), modules=modules)
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=180
    )


def test_the_offline_path_imports_without_the_data_extra():
    result = _run_with_blocked(OFFLINE_MODULES)
    assert result.returncode == 0, (
        "the synthetic offline path needs a module from the optional data extra:\n"
        + result.stderr
    )
    assert "ok" in result.stdout


def test_the_guard_actually_blocks_something():
    """A blocker that blocks nothing would make the test above vacuous."""
    result = _run_with_blocked(("requests",))
    assert result.returncode != 0
    assert "No module named 'requests'" in result.stderr


@pytest.mark.parametrize("module", OFFLINE_MODULES)
def test_each_offline_module_individually(module):
    """Named separately so a failure says which import pulled the extra in."""
    result = _run_with_blocked((module,))
    assert result.returncode == 0, result.stderr


def test_asking_for_a_network_client_names_the_extra():
    """The failure, when it is genuine, should say what to install."""
    script = textwrap.dedent(
        """
        import sys

        class Blocker:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] == "requests":
                    raise ModuleNotFoundError("No module named 'requests'")
                return None

        sys.meta_path.insert(0, Blocker())

        from earnings_engine.data.http import HttpClient
        try:
            HttpClient(user_agent="test")
        except ModuleNotFoundError as exc:
            print(exc)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0, result.stderr
    assert "data extra" in result.stdout

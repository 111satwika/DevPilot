"""Shared pytest fixtures.

Gap-fix (Entry 41): this project had zero automated tests before this --
every verification through Entry 40 was manual, live, ad-hoc (real, but
nothing catches a future change silently breaking the sandbox or an
approval gate). This is a first pass, scoped to what's fast and reliable
in CI: no WSL/Docker/Ollama dependency, since none of those are available
in a generic CI runner (or reliably in this environment either). Docker/
git-bridge behavior stays covered by the manual testing discipline logged
in DevPilot_AI_Implementation_Log.html.

Each sandboxed MCP server module (filesystem/terminal/git/docker) resolves
its own WORKSPACE_ROOT once at import time -- correct for how the real app
runs (one long-lived process per workspace, Entry 25), but it means a test
can't retarget the sandbox by setting DEVPILOT_WORKSPACE_ROOT after the
module is already imported. patch_workspace_root() below monkeypatches the
already-imported module's WORKSPACE_ROOT attribute directly instead --
that's still a genuine global lookup at call time, so it actually takes
effect.
"""

from pathlib import Path

import pytest


@pytest.fixture()
def workspace(tmp_path) -> Path:
    """A throwaway directory standing in for a real project workspace."""
    return tmp_path


def patch_workspace_root(monkeypatch: pytest.MonkeyPatch, *modules, root: Path) -> None:
    for module in modules:
        monkeypatch.setattr(module, "WORKSPACE_ROOT", root)

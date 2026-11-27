from pathlib import Path

import pytest

AGENTS = Path(__file__).resolve().parents[2] / "src/mobilerun_agents"
MODULES = sorted(path.parent.relative_to(AGENTS).as_posix() for path in AGENTS.rglob("README.md"))
HEADINGS = [
    "**Purpose:**",
    "**What belongs here / what doesn't:**",
    "**Contents:**",
    "**Examples:**",
    "**Depends on / used by:**",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_readme_template(module):
    folder = AGENTS / module
    text = (folder / "README.md").read_text(encoding="utf-8")
    positions = [text.index(heading) for heading in HEADINGS]
    assert positions == sorted(positions)
    assert all(text.count(heading) == 1 for heading in HEADINGS)
    contents = text[positions[2] : positions[3]]
    for path in folder.glob("*.py"):
        assert f"`{path.name}`" in contents


@pytest.mark.parametrize(
    "package",
    sorted(
        path.parent
        for root in ("agent", "macro_agent")
        for path in (AGENTS / root).rglob("__init__.py")
    ),
)
def test_agent_packages_have_readmes(package):
    assert (package / "README.md").is_file(), package

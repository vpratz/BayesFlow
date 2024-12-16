from functools import partial
from pathlib import Path

from sphinx_polyversion.api import apply_overrides
from sphinx_polyversion.driver import DefaultDriver
from sphinx_polyversion.git import Git, file_predicate, refs_by_type, closest_tag
from sphinx_polyversion.pyvenv import Pip
from sphinx_polyversion.sphinx import SphinxBuilder, Placeholder

#: CodeRegex matching the branches to build docs for
BRANCH_REGEX = r"(doc-polyversion)"

#: Regex matching the tags to build docs for
TAG_REGEX = r"v1.1.6.1"

#: Output dir relative to project root
OUTPUT_DIR = "_polybuild"

#: Source directory
SOURCE_DIR = "docsrc/"

#: Arguments to pass to `sphinx-build`
SPHINX_ARGS = "-a -v"

#: Extra packages for building docs
SPHINX_DEPS = [
    "sphinx",
    "numpydoc",
    "myst-nb",
    "sphinx_design",
    "sphinx-book-theme",
    "sphinxcontrib-bibtex",
    "sphinx-polyversion==1.0.0",
]

BACKEND_DEPS = [
    "jax",
    "torch",
    "tensorflow",
]


#: Data passed to templates
def data(driver, rev, env):
    revisions = driver.targets
    branches, tags = refs_by_type(revisions)
    latest = max(tags or branches)
    return {
        "current": rev,
        "tags": tags,
        "branches": branches,
        "revisions": revisions,
        "latest": latest,
    }


def root_data(driver):
    revisions = driver.builds
    branches, tags = refs_by_type(revisions)
    latest = max(tags or branches)
    return {"revisions": revisions, "latest": latest}


# Load overrides read from commandline to global scope
apply_overrides(globals())
# Determine repository root directory
root = Git.root(Path(__file__).parent)


async def selector(f, a, b):
    return a.name


# Setup environments for the different versions

ENVIRONMENT = {
    None: Pip.factory(venv=Path(".venv"), args=["-vv", "bayesflow==1.1.6"] + SPHINX_DEPS),
    "doc-polyversion": Pip.factory(venv=Path(".venv/dev"), args=["-vv", "-e", "."] + SPHINX_DEPS + BACKEND_DEPS),
    "v1.1.6": Pip.factory(venv=Path(".venv/v1.1.6"), args=["-vv", "bayesflow==1.1.6"] + SPHINX_DEPS),
}

# Setup driver and run it
src = Path(SOURCE_DIR)
DefaultDriver(
    root,
    OUTPUT_DIR,
    vcs=Git(
        branch_regex=BRANCH_REGEX,
        tag_regex=TAG_REGEX,
        buffer_size=1 * 10**9,  # 1 GB
        predicate=file_predicate([src]),  # exclude refs without source dir
    ),
    builder=SphinxBuilder(
        src / "source",
        args=SPHINX_ARGS.split(),
        pre_cmd=["python", root / src / "pre-build.py", Placeholder.SOURCE_DIR],
    ),
    env=ENVIRONMENT,
    selector=partial(selector, partial(closest_tag, root)),
    template_dir=root / src / "polyversion/templates",
    static_dir=root / src / "polyversion/static",
    data_factory=data,
    root_data_factory=root_data,
).run(False)

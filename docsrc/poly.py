from datetime import datetime
from pathlib import Path

from sphinx_polyversion.api import apply_overrides
from sphinx_polyversion.driver import DefaultDriver
from sphinx_polyversion.git import Git, GitRef, GitRefType, file_predicate, refs_by_type
from sphinx_polyversion.pyvenv import Pip
from sphinx_polyversion.sphinx import SphinxBuilder, Placeholder

#: Regex matching the branches to build docs for
BRANCH_REGEX = r"master-doctest"

#: Regex matching the tags to build docs for
TAG_REGEX = r"v1.1.6"

#: Output dir relative to project root
OUTPUT_DIR = "_polybuild"

#: Source directory
SOURCE_DIR = "docsrc/"

#: Arguments to pass to `pip install`
POETRY_ARGS = "bayesflow"

#: Arguments to pass to `sphinx-build`
SPHINX_ARGS = "-a -v"

#: Mock data used for building local version
MOCK_DATA = {
    "revisions": [
        GitRef("v1.1.6", "", "", GitRefType.TAG, datetime.fromtimestamp(0)),
        GitRef("master-doctest", "", "", GitRefType.BRANCH, datetime.fromtimestamp(3)),
    ],
    "current": GitRef("master-doctest", "", "", GitRefType.TAG, datetime.fromtimestamp(6)),
}


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
print("root", root)

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
        src / "source", args=SPHINX_ARGS.split(), pre_cmd=["python", "pre-build.py", Placeholder.SOURCE_DIR]
    ),
    env=Pip.factory(args=POETRY_ARGS.split(), venv="buildvenv"),
    template_dir=root / src / "polyversion/templates",
    static_dir=root / src / "polyversion/static",
    data_factory=data,
    root_data_factory=root_data,
).run(False)

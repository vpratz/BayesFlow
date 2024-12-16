import os
import shutil
import sys
from pathlib import Path


def copy_files(sourcedir):
    basedir = Path(sourcedir).parent.parent
    print(basedir, sourcedir)

    examples_src = os.path.join(basedir, "examples")
    examples_dst = os.path.join(sourcedir, "_examples")
    if os.path.exists(examples_src):
        print("Copying examples")
        shutil.copytree(examples_src, examples_dst, dirs_exist_ok=True)
    examples_in_progress = os.path.join(examples_dst, "in_progress")
    if os.path.exists(examples_in_progress):
        shutil.rmtree(examples_in_progress)
    contributing_src = os.path.join(basedir, "CONTRIBUTING.md")
    contributing_dst = os.path.join(sourcedir, "contributing.md")
    if os.path.exists(contributing_src):
        shutil.copy2(contributing_src, contributing_dst)
    installation_src = os.path.join(basedir, "INSTALL.rst")
    installation_dst = os.path.join(sourcedir, "installation.rst")
    if os.path.exists(installation_src):
        shutil.copy2(installation_src, installation_dst)
    print(os.listdir(sourcedir))
    print(os.listdir(examples_dst))


if __name__ == "__main__":
    print("Running pre-build script")  # move files around if necessary
    sourcedir = sys.argv[1]
    copy_files(sourcedir)

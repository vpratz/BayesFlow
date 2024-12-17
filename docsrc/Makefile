# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# from https://github.com/acerbilab/pyvbmc/blob/main/docsrc/Makefile
# copy example notebooks to source folder
# render docs
# copy rendered docs and .nojekyll (for github.io) to docs folder
# remove example notebooks from source folder
dev:
	@python pre-build.py source
	@make html
	@echo 'Copying docs to ../docs'
	@cp -a _build/html/. ../docs
	@cp .nojekyll ../docs/.nojekyll

github:
	@sphinx-polyversion -vv poly.py
	@echo 'Copying docs to ../docs'
	@cp -a _build_polyversion/. ../docs

clean:
	@rm -rf ../docs/*
	@rm -rf _build
	@rm -rf _build_polyversion
	@rm -rf source/_examples
	@rm source/installation.rst
	@rm source/contributing.md

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

SPHINXOPTS    ?=
SPHINXBUILD   ?=sphinx-build
SOURCEDIR     ?=source
BUILDDIR      ?=_build
CONFDIR       ?=.

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" -c "$(CONFDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" -c "$(CONFDIR)" $(SPHINXOPTS) $(O)

# You can set these variables from the command line, and also
# from the environment for the first two.

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

autopep:
	# Running autopep8
	@autopep8 -r --in-place teaspoon/

unittests:
	# Running unittests
	@python -m unittest

%: Makefile
	# Running sphinx-build to build html files.
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	
	@mkdir $(shell pwd)/docs/temp
	# Removing all files from docs folder to be replaced with new docs.
	@rm -rf $(shell pwd)/docs/
	@mkdir $(shell pwd)/docs/
	# Copying over contents of build/html to docs
	@mkdir $(shell pwd)/docs/.doctrees
	@cp -a $(shell pwd)/build/doctrees/. $(shell pwd)/docs/.doctrees/
	@cp -a $(shell pwd)/build/html/. $(shell pwd)/docs/


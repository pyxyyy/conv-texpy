
PYTHON := poetry run python

typecheck:
	MYPYPATH="stubs/" $(PYTHON) -m mypy texpy/

docs: help

help:
	$(info "Generating documentation")
	make -C docs html


.PHONY: typecheck docs help

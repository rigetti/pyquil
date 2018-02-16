# top-level pyquil Makefile

PACKAGENAME = pyquil

# Kudos: Adapted from Auto-documenting default target 
# https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-12s\033[0m %s\n", $$1, $$2}'

test:		## Run unittests with current enviroment
	@pytest $(PACKAGENAME)/tests

testall:	## Run full test suite
	@tox

coverage:	## Report test coverage
	@pytest --cov=$(PACKAGENAME) --cov-report term-missing $(PACKAGENAME)/tests

lint:		## Delint python source
	@flake8 $(PACKAGENAME)

typecheck:	## Static typechecking 
	@mypy pyquil/*.py pyquil/*/*.py  --ignore-missing-imports --follow-imports=skip

untyped:	## Report type errors and untyped functions
	@mypy pyquil/*.py pyquil/api/*.py pyquil/latex/*.py pyquil/_parser/*.py --ignore-missing-imports --follow-imports=skip --disallow-untyped-defs

docs:		## Build documentation
	$(MAKE) -C docs html

antlr: 		## Rebuild antlr parser (Run after modification to antlr grammar ./pyquil/_parser/Quil.g4)
	(cd pyquil/_parser && antlr4 -Dlanguage=Python2 -o gen2 Quil.g4)
	(cd pyquil/_parser && antlr4 -Dlanguage=Python3 -o gen3 Quil.g4)


.PHONY: help
.PHONY: docs

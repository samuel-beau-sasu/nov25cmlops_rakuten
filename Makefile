#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = nov25cmlops_rakuten
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Delete processed and interim data
.PHONY: clean-data
clean-data:
	rm -rf data/processed/*
	rm -rf data/interim/*

## Delete all generated artifacts (be careful)
.PHONY: clean-all
clean-all: clean clean-data
	rm -rf models/*
	rm -rf reports/*


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"
	

#################################################################################
# DATA VERSIONING                                                               #
#################################################################################

## Create data versions (v1.0, v2.0, v3.0) - ONE TIME ONLY
.PHONY: create-versions
create-versions:
	@echo " Creating data versions..."
	$(PYTHON_INTERPRETER) -m mlops_rakuten.main create-versions
	@echo " Data versions created in data/versions/"

## Show information about current data version
.PHONY: data-info
data-info:
	@echo " Current data configuration:"
	$(PYTHON_INTERPRETER) -m mlops_rakuten.main info

## Quick check of data versions
.PHONY: check-versions
check-versions:
	@echo " Available data versions:"
	@for version in data/versions/*/; do \
		if [ -d "$$version" ]; then \
			version_name=$$(basename $$version); \
			if [ -f "$$version/metadata.json" ]; then \
				samples=$$(grep -o '"n_samples": [0-9]*' "$$version/metadata.json" | grep -o '[0-9]*'); \
				echo "  ✓ $$version_name: $$samples samples"; \
			else \
				echo "  ✗ $$version_name: metadata missing"; \
			fi \
		fi \
	done

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Train model
.PHONY: train
train: requirements
	$(PYTHON_INTERPRETER) mlops_rakuten/main.py train


## Make prediction
.PHONY: predict
predict: requirements
	$(PYTHON_INTERPRETER) mlops_rakuten/main.py predict "$(TEXT)"


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)

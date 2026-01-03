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
	rm -rf data/raw/rakuten/seeds
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
# PROJECT RULES                                                                 #
#################################################################################


## Seed data and train model
.PHONY: seed
seed: requirements
	$(PYTHON_INTERPRETER) mlops_rakuten/main.py seed


## Ingest data and train model
.PHONY: ingest
ingest: requirements
	$(PYTHON_INTERPRETER) mlops_rakuten/main.py ingest $(CSV_PATH)


## Train model
.PHONY: train
train: requirements
	$(PYTHON_INTERPRETER) mlops_rakuten/main.py train


## Make prediction
.PHONY: predict
predict: requirements
	$(PYTHON_INTERPRETER) mlops_rakuten/main.py predict "$(TEXT)"


#################################################################################
# DVC & DAGHUB                                             						#
#################################################################################

## Initialize DVC config.local with credentials interactively
.PHONY: dvc-credentials
dvc-credentials:
	@echo "Création du fichier .dvc/config.local avec les identifiants DagsHub"
	@read -p "Enter DagsHub Access Key ID: " ACCESS_KEY; \
	read -p "Enter DagsHub Secret Access Key: " SECRET_KEY; \
	dvc remote modify origin --local access_key_id $$ACCESS_KEY; \
	dvc remote modify origin --local secret_access_key $$SECRET_KEY;
	@echo ".dvc/config.local crée avec succès."

## Test DVC connection to Dagshub s3 remote storage
.PHONY: dvc-test
dvc-test:
	@echo "Test connexion DVC vers DagsHub..."
	@dvc status && echo "Connected to DagsHub" || echo "Connection failed"

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

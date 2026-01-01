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
	uv pip install -r requirements-dev.txt


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
# DATA / ML JOBS
#################################################################################


.PHONY: docker-bootstrap
docker-bootstrap:
	$(COMPOSE_CMD) up -d --build api-train
	docker run --rm \
		-v rakuten_data:/app/data \
		-v $(PWD)/data/raw:/bootstrap/raw:ro \
		alpine \
		sh -c "mkdir -p /app/data/raw && cp -r /bootstrap/raw/* /app/data/raw/"


.PHONY: docker-seed
docker-seed:
	$(COMPOSE_CMD) run --rm api-train \
		python -m mlops_rakuten.main seed


#################################################################################
# DOCKER COMPOSE
#################################################################################

COMPOSE_CMD := $(shell \
	if command -v docker-compose >/dev/null 2>&1; then \
		echo docker-compose; \
	else \
		echo docker compose; \
	fi \
)


.PHONY: docker-start
docker-start:
	$(COMPOSE_CMD) up -d --build nginx api-inference


.PHONY: docker-init
docker-init: docker-bootstrap docker-seed docker-start


.PHONY: docker-stop
docker-stop:
	$(COMPOSE_CMD) down


.PHONY: docker-rerun
docker-rerun: docker-stop docker-start


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

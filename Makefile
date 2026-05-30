# Makefile for verb-gloss-wsd

UV=uv
FIND=find
MAKE=make
SRC=aidu.ai.llm
APP=app/

.PHONY: help install clean wipe serve run smoke test curl web.build lint format check-format pre-commit-install pre-commit-run

help:                                     ## Show this help
	@grep -h "##" $(MAKEFILE_LIST) | grep -v grep | sed -e "s/\$$//" -e "s/##//"

# install targets

server.install:                                  ## Install python dependencies and set up environment
	@echo "Installing dependencies"
	@$(UV) sync

	@echo "Upgrading pip"
	@$(UV) run python -m ensurepip --upgrade

# Cleanup targets

server.clean:                             ## Clean temporary and cache files
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf .venv
	$(FIND) . -type f -name '*~' -delete
	$(FIND) . -type f -name '*.pyc' -delete
	$(FIND) . -type d -name '__pycache__' -delete

wipe:                                     ## Delete all uv-related files for a fresh start
wipe: server.clean
	@echo "Removing uv.lock"
	rm -f uv.lock


# Application targets

server.run:	                               ## Run the web server for the application
	$(UV) run python -m serve.app

app.run:                                   ## Run the analysis application (default)
	@$(UV) run python -m aidu.ai.controller.main shell
# Smoke test targets

smoke.controller.controller:           ## Run a quick smoke test for the simple agent controller
	$(UV) run python -m $(SRC).controller.controller



smoke:									  ## Run all smoke tests
	$(MAKE) smoke.controller.controller


# Testing targets
	
test:                                     ## Run all tests
	@echo "Running tests..."
	$(UV) run pytest

# Linting and formatting targets
lint:                                      ## Run ruff linting across the project
	$(UV) run ruff check src tests || true

format:                                    ## Format code: run Black then Ruff (Ruff will apply final fixes, e.g. remove trailing commas)
	$(UV) run black .
	$(UV) run ruff format .

format-black-only:                          ## Format code with Black only (preserves Black's trailing-comma style)
	$(UV) run black .

check-format:                              ## Check formatting (black + ruff) without modifying files
	$(UV) run black --check .
	$(UV) run ruff check .

pre-commit-install:                        ## Install pre-commit and set up hooks
	$(UV) run pip install pre-commit
	$(UV) run pre-commit install

pre-commit-run:                            ## Run pre-commit checks on all files
	$(UV) run pre-commit run --all-files

curl:	                                  ## Runs curl tests against the server
	@echo "Running curl tests..."
	test/curl_tests.sh

# Web frontend targets

web.clean:                                ## Clean up the web frontend
	cd web && $(MAKE) clean
web.install:	                          ## Install web frontend dependencies
	cd web && $(MAKE) install
web.build:                                ## Build the web frontend
	cd web && $(MAKE) build

jupyter:        ## Start a jupyter notebook server
	@if [ ! -d ".venv" ]; then uv venv; fi
	uv pip install jupyter
	uv run jupyter lab

clean: server.clean web.clean
	@echo "Cleaned server and web frontend"

install: server.install web.install
	@echo "Installed server and web frontend"

serve: server.run
	@echo "Running the application"
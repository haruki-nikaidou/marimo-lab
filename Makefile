UV ?= uv
NOTEBOOKS := notebooks
EXPORTS := exports

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk -F':.*?## ' '{printf "  \033[1m%-12s\033[0m %s\n", $$1, $$2}'

.PHONY: sync
sync: ## Install/refresh the project environment from uv.lock
	$(UV) sync

.PHONY: lock
lock: ## Re-resolve dependencies and update uv.lock
	$(UV) lock --upgrade

.PHONY: edit
edit: ## Open the notebook workspace in the browser
	$(UV) run marimo edit $(NOTEBOOKS)

.PHONY: new
new: ## Create/open a notebook: make new N=my_experiment
	@test -n "$(N)" || { echo "usage: make new N=<notebook_name>"; exit 2; }
	$(UV) run marimo edit $(NOTEBOOKS)/$(N).py

.PHONY: run
run: ## Serve a notebook as a read-only app: make run N=00_welcome
	@test -n "$(N)" || { echo "usage: make run N=<notebook_name>"; exit 2; }
	$(UV) run marimo run $(NOTEBOOKS)/$(N).py

.PHONY: script
script: ## Execute a notebook headlessly: make script N=00_welcome
	@test -n "$(N)" || { echo "usage: make script N=<notebook_name>"; exit 2; }
	$(UV) run python $(NOTEBOOKS)/$(N).py

.PHONY: export
export: ## Export every notebook to static HTML under exports/
	@mkdir -p $(EXPORTS)
	@for nb in $(NOTEBOOKS)/*.py; do \
		out=$(EXPORTS)/$$(basename $$nb .py).html; \
		echo "$$nb -> $$out"; \
		$(UV) run marimo export html "$$nb" -o "$$out" || exit 1; \
	done

.PHONY: check
check: ## Lint notebooks and sources, then run tests
	$(UV) run marimo check $(NOTEBOOKS)
	$(UV) run ruff check .
	$(UV) run ruff format --check .
	$(UV) run pytest -q

.PHONY: fmt
fmt: ## Format sources and notebooks
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

.PHONY: clean
clean: ## Remove exports and caches
	rm -rf $(EXPORTS) .ruff_cache .pytest_cache __marimo__
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

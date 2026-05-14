VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
BIN := $(VENV)/bin
LOG := pipeline_run.log

.PHONY: all setup install run run-detached run-dry stop logs logs-ollama logs-all \
        docker-build docker-run docker-run-cloud docker-logs docker-stop \
        test coverage typecheck clean reset

all: setup

$(VENV)/bin/activate:
	python3 -m venv $(VENV)

setup: $(VENV)/bin/activate
	$(PIP) install -e ".[dev]"

install: setup

run: setup
	@set -a; [ -f .env ] && . ./.env; set +a; \
	$(BIN)/lora-pipeline 2>&1 | tee $(LOG)

run-detached: setup
	@set -a; [ -f .env ] && . ./.env; set +a; \
	nohup $(BIN)/lora-pipeline > $(LOG) 2>&1 & echo $$! > .pipeline.pid; \
	echo "Pipeline running (PID $$(cat .pipeline.pid)). Logs: $(LOG)"

run-dry: setup
	@set -a; [ -f .env ] && . ./.env; set +a; \
	$(BIN)/lora-pipeline --dry-run 2>&1 | tee $(LOG)

stop:
	@if [ -f .pipeline.pid ]; then \
		kill $$(cat .pipeline.pid) 2>/dev/null && echo "Stopped PID $$(cat .pipeline.pid)" || echo "Process not running"; \
		rm -f .pipeline.pid; \
	else \
		echo "No PID file found"; \
	fi

logs:
	tail -f $(LOG)

logs-ollama:
	tail -f ~/.ollama/logs/server.log

logs-all:
	tail -f $(LOG) ~/.ollama/logs/server.log

test: setup
	$(BIN)/pytest

coverage: setup
	$(BIN)/pytest --cov=lora_pipeline -q

typecheck: setup
	$(BIN)/mypy src/

docker-build:
	docker compose build

docker-run:
	IMAGE_ROOT=$(IMAGE_ROOT) docker compose up

docker-run-cloud:
	IMAGE_ROOT=$(IMAGE_ROOT) docker compose -f docker-compose.yml -f docker-compose.cloud.yml up

docker-logs:
	docker compose logs -f pipeline

docker-stop:
	docker compose down

clean:
	rm -rf $(VENV) __pycache__ .coverage htmlcov .pipeline.pid
	find . -name "*.pyc" -delete

reset: setup
	@set -a; [ -f .env ] && . ./.env; set +a; \
	$(BIN)/lora-pipeline --reset 2>&1 | tee $(LOG)

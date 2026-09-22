UV ?= uv
ARTIFACT_PATH ?= artifacts/model.joblib
ARTIFACT_OVERRIDES ?=

.PHONY: install test lint train artifact web docker-build docker-up docker-down docker-logs

install:
	$(UV) sync

test:
	$(UV) run pytest -q

lint:
	$(UV) run ruff check src tests
	$(UV) run ruff format --check src tests

train:
	$(UV) run train

artifact:
	mkdir -p $(dir $(ARTIFACT_PATH))
	$(UV) run python -m serving.build_artifact \
		--output $(ARTIFACT_PATH) $(ARTIFACT_OVERRIDES)

web:
	MODEL_BUNDLE_PATH=$(ARTIFACT_PATH) $(UV) run streamlit run src/serving/app.py

docker-build:
	docker compose build

docker-up:
	mkdir -p artifacts
	docker compose up --build

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f web

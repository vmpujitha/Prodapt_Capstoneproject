# AI-Powered Agriculture Crop Advisory Assistant
# Quick start (local):  make install && make setup && make dev
PYTHON ?= .venv/bin/python
PIP    ?= .venv/bin/pip

.PHONY: help venv install env data index train setup backend frontend dev test eval diagram build docker-up docker-down clean-index

help:
	@echo "make install    - create .venv, install backend + frontend dependencies"
	@echo "make setup      - .env, dataset (Kaggle/real or synthetic), BM25 + vector index, ML models"
	@echo "make dev        - run backend (:8000) and frontend (:5173) together"
	@echo "make backend    - run only the FastAPI backend with reload"
	@echo "make frontend   - run only the Vite dev server"
	@echo "make test       - run the pytest suite (no LLM needed)"
	@echo "make eval       - run the DeepEval golden-set evaluation"
	@echo "make diagram    - regenerate docs/architecture.{png,jpg,pdf}"
	@echo "make docker-up  - build and start backend + frontend containers"

venv:
	test -d .venv || python3 -m venv .venv

install: venv
	$(PIP) install -r backend/requirements.txt
	cd frontend && npm ci

env:
	test -f .env || cp .env.example .env

data: env
	$(PYTHON) scripts/download_data.py

index: env
	-$(PYTHON) scripts/build_index.py

train: env
	$(PYTHON) scripts/train_models.py

setup: data train index

backend: env
	$(PYTHON) -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

frontend:
	cd frontend && npm run dev

dev: env
	@trap 'kill 0' INT TERM; \
	$(PYTHON) -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 & \
	(cd frontend && npm run dev) & \
	wait

test:
	$(PYTHON) -m pytest backend/tests -q

eval:
	$(PYTHON) scripts/run_eval.py

diagram:
	$(PYTHON) scripts/make_architecture_diagram.py

build:
	cd frontend && npm run build

docker-up: env
	docker compose up --build -d

docker-down:
	docker compose down

clean-index:
	rm -rf data/chroma/* data/processed/bm25_index.pkl

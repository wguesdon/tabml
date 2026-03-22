.PHONY: help setup train predict submit clean docker-build docker-run terraform-init terraform-apply

help:
	@echo "Available commands:"
	@echo "  setup          - Install dependencies and set up environment"
	@echo "  train          - Train models locally"
	@echo "  predict        - Generate predictions"
	@echo "  submit         - Submit to Kaggle"
	@echo "  docker-build   - Build Docker image"
	@echo "  docker-run     - Run pipeline in Docker"
	@echo "  terraform-init - Initialize Terraform"
	@echo "  terraform-apply- Deploy AWS infrastructure"
	@echo "  clean          - Clean up generated files"

setup:
	pip install -r requirements.txt
	pre-commit install
	mkdir -p data/raw data/processed data/features models submissions logs

train:
	python scripts/run_pipeline.py --mode train --optimize

predict:
	python scripts/run_pipeline.py --mode predict

submit: predict
	python scripts/submit_to_kaggle.py \
		--competition $(COMPETITION_NAME) \
		--submission-file submissions/submission_$(COMPETITION_NAME).csv \
		--message "Automated submission from pipeline"

docker-build:
	cd infrastructure/docker && docker build -t kaggle-tabular-pipeline:latest ../..

docker-run:
	cd infrastructure/docker && docker-compose up kaggle-pipeline

docker-mlflow:
	cd infrastructure/docker && docker-compose up -d mlflow

docker-jupyter:
	cd infrastructure/docker && docker-compose up -d jupyter

terraform-init:
	cd infrastructure/terraform && terraform init

terraform-apply:
	cd infrastructure/terraform && terraform apply -var="competition_name=$(COMPETITION_NAME)"

clean:
	rm -rf __pycache__ .pytest_cache .coverage htmlcov
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
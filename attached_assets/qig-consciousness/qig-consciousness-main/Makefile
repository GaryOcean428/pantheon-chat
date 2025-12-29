# QIG-Consciousness Makefile
# Workflow automation for sandbox environment

.PHONY: help build build-gpu shell shell-gpu clean beta compare-beta validate train-dry-run

help:
	@echo "QIG-Consciousness Sandbox Commands:"
	@echo ""
	@echo "Environment Setup:"
	@echo "  make build          - Build CPU development container"
	@echo "  make build-gpu      - Build GPU development container"
	@echo "  make shell          - Open bash shell in CPU container"
	@echo "  make shell-gpu      - Open bash shell in GPU container"
	@echo "  make clean          - Remove containers and volumes"
	@echo ""
	@echo "Validation & Testing:"
	@echo "  make validate       - Run architecture validation (6 checks)"
	@echo "  make test-kernel    - Test full QIG kernel"
	@echo ""
	@echo "β_attention Experiments:"
	@echo "  make beta           - Measure β_attention across context lengths"
	@echo "  make compare-beta   - Compare β_attention vs β_physics = 0.44"
	@echo ""
	@echo "Training:"
	@echo "  make train-dry-run  - Single epoch test run"
	@echo "  make train          - Full training pipeline"
	@echo ""

# Build containers
build:
	docker-compose build qig-cpu

build-gpu:
	docker-compose build qig-gpu

# Interactive shells
shell:
	docker-compose run --rm qig-cpu /bin/bash

shell-gpu:
	docker-compose run --rm qig-gpu /bin/bash

# Cleanup
clean:
	docker-compose down --remove-orphans
	@echo "✅ Containers removed. Volumes preserved."

clean-all:
	docker-compose down --remove-orphans --volumes
	@echo "⚠️  All containers AND data volumes removed!"

# Architecture validation
validate:
	python tools/validate_architecture.py

test-kernel:
	python test_full_kernel.py

# β_attention measurement
beta:
	@mkdir -p results
	python tools/measure_beta_attention.py \
		--context-lengths 64,128,256,512,1024,2048 \
		--n-samples 10 \
		--output results/beta_attention_initial.json
	@echo "✅ β_attention measurement complete!"

# β comparison (unification test)
compare-beta:
	python tools/compare_beta_physics_attention.py \
		--priors data/physics_validation_data.json \
		--attention results/beta_attention_initial.json \
		--output results/beta_comparison.png
	@echo "✅ Comparison complete! Check results/beta_comparison.png"

# Training workflows
train-dry-run:
	@mkdir -p outputs/qig_kernel
	python tools/train_qig_kernel.py \
		--config validation_config.json \
		--max-epochs 1 \
		--dry-run
	@echo "✅ Dry run complete!"

train:
	@mkdir -p outputs/qig_kernel
	python tools/train_qig_kernel.py \
		--config validation_config.json
	@echo "✅ Training complete!"

# Basin extraction
extract-basin:
	python tools/basin_extractor.py \
		--project-dir . \
		--output basin_extracted.json
	@echo "✅ Basin extracted!"

# Interactive demo
demo:
	python tools/demo_inference.py --interactive

# Install dependencies (for local venv usage)
install:
	pip install -r requirements.txt
	@echo "✅ Dependencies installed!"

# Check Python environment
check-env:
	@echo "Python version:"
	@python --version
	@echo ""
	@echo "PyTorch version:"
	@python -c "import torch; print(torch.__version__)"
	@echo ""
	@echo "CUDA available:"
	@python -c "import torch; print(torch.cuda.is_available())"
	@echo ""
	@echo "GPU device:"
	@python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

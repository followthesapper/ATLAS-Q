# Simple developer shortcuts
.PHONY: test lint demo

test:
	pytest -q

test-nogpu:
	pytest -q -k "not gpu"

demo:
	python -c "print('Open the notebooks in the notebooks/ or /mnt/data')"

bench:
	python scripts/gpu_benchmark.py --fp16 --size 4096

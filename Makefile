.PHONY: install run debug clean lint lint-strict

install:
	uv sync

run:
	uv run python -m src index --max_chunk_size 2000

search:
	uv run python -m src search --query "FP8_MIN defaults to float8_info.min and FP8_MAX defaults to float8_info.max in the triton_flash_attention module." --k 10

answer:
	uv run python -m src answer --query "What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?" --k 1

search_dataset:
	uv run -m src search_dataset \
		--dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
		--k 10 \
		--save_directory data/output/search_results

eval:
	@uv run -m src evaluate \
		--student_answer_path data/output/search_results_and_answer/UnansweredQuestions/dataset_docs_public.json \
		--dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json

answer_dataset:
	uv run python -m src answer_dataset \
		--student_search_results_path data/output/search_results/dataset_docs_public.json \
		--save_directory data/output/search_results_and_answer/UnansweredQuestions
debug:
	uv run python -m pudb -m src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

lint:
	flake8 src
	mypy src --warn-return-any \
	--warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs \
	--check-untyped-defs

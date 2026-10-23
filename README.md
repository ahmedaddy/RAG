*This project has been created as part of the 42 curriculum by aaddy.*

# RAG against the Machine

## Description

This project implements a Retrieval-Augmented Generation (RAG) system for
answering questions about the vLLM codebase. Rather than relying solely on a
language model's training data, the system retrieves relevant source-code and
documentation excerpts first, then gives that evidence to a local language
model to produce a grounded answer.

The pipeline has four stages:

1. Index the supplied vLLM repository.
2. Retrieve the most relevant source chunks for a question.
3. Add the retrieved context to a prompt.
4. Generate an answer with `Qwen/Qwen3-0.6B`.

Retrieval quality is measured with recall@k against the supplied reference
datasets.

## System architecture

```text
data/raw/vllm-0.10.1/
          |
          v
    Python / Markdown chunking
          |
          v
   BM25 index + chunk metadata
          |
          v
 data/processed/bm25_all.pkl
 data/processed/chunks_all.pkl
          |
          v
 Question -> BM25 retrieval -> top-k source locations
          |                              |
          |                              v
          +---------------------- Qwen/Qwen3-0.6B
                                         |
                                         v
                              grounded natural-language answer
```

- `src/chunking.py` reads supported files and produces bounded source chunks.
- `src/indexer.py` tokenizes chunks, builds the BM25 index, saves it, and
  ranks results for a query.
- `src/generator.py` loads Qwen and asks it to answer only from retrieved
  context.
- `src/models.py` defines the Pydantic models exchanged between pipeline
  stages.
- `src/evaluator.py` calculates local recall@k for iteration.
- `src/__main__.py` exposes the pipeline through the Python Fire CLI.

## Chunking strategy

The maximum chunk length is configurable with `--max_chunk_size` and defaults
to 2,000 characters. Chunks never exceed this limit, which keeps every
retrieved source valid for the evaluation context limit.

Two different strategies are used:

- Python files are parsed with the `ast` module. Top-level syntax nodes are
  used as natural code boundaries; oversized nodes are split on line
  boundaries.
- Markdown and text files use a sliding window with a small overlap. The
  splitter prefers paragraph and line boundaries before falling back to the
  maximum length.

Each chunk retains its repository-relative path and its original start and end
character offsets. Those offsets are emitted with retrieval results so they
can be compared to the ground-truth source locations.

## Retrieval method

The project uses the BM25 lexical ranking algorithm through `rank-bm25`.
During indexing, each chunk is tokenized from both its path and its content.
Tokens are lowercased, punctuation is normalized, and underscore-separated
identifiers are also split to make code identifiers easier to match.

At query time, the question is tokenized with the same function. BM25 scores
every indexed chunk and returns the highest-scoring top-k source locations.

## Instructions

### Requirements

- Python 3.10 or later
- `uv`
- Enough disk space for the vLLM source corpus and the Qwen model weights

### Install dependencies

```bash
make install
```

or:

```bash
uv sync
```

### Index the codebase

```bash
uv run python -m src index --max_chunk_size 2000
```

The index is saved under `data/processed/`.

### Search one question

```bash
uv run python -m src search \
  --query "What is the default value for mlp_bias?" \
  --k 10
```

### Search a dataset

```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions
```

### Generate one answer

```bash
uv run python -m src answer \
  --query "How is an OpenAI-compatible server configured?" \
  --k 10
```

### Generate answers for a dataset

```bash
uv run python -m src answer_dataset \
  --student_search_results_path \
  data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

### Evaluate retrieval

```bash
uv run python -m src evaluate \
  --student_answer_path \
  data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 10
```

### Useful Makefile targets

```bash
make run
make lint
make clean
make debug
```

## Performance analysis

The current public-dataset search outputs were validated with the provided
moulinette using `k=10` and a maximum context length of 2,000 characters.

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---|---:|---:|---:|---:|
| Documentation | 55.0% | 73.0% | 80.0% | 85.0% |
| Code | 29.3% | 51.5% | 57.6% | 66.7% |

These results meet the mandatory recall@5 thresholds: at least 80% for
documentation questions and 50% for code questions. Performance should be
measured again after any change to chunking, tokenization, or ranking.

## Design decisions

- **BM25 instead of embeddings:** BM25 is lightweight, deterministic, and
  effective for matching documentation wording and code identifiers without
  requiring an additional embedding model.
- **AST-aware Python chunking:** Splitting code at top-level syntax boundaries
  preserves meaningful units such as classes and functions whenever possible.
- **Overlapping text chunks:** The overlap reduces the risk of separating a
  heading or explanation from the details that follow it.
- **Pydantic data models:** Models validate datasets and output structures at
  stage boundaries.
- **Grounded prompt:** The model is instructed to answer only from the
  retrieved source context and to state when that context is insufficient.

## Challenges faced

- A source chunk must be no longer than 2,000 characters or the evaluator
  rejects the complete result. Large Python nodes therefore need a secondary
  line-based split.
- Questions often paraphrase source material, while code contains exact
  identifiers. Including both paths and content in the BM25 corpus helps
  lexical matching in both cases.
- Retrieval needs exact repository-relative file paths and original character
  offsets. Any path-prefix mismatch prevents a source from receiving credit
  during evaluation.
- The small required language model has limited reasoning ability, so answer
  quality depends heavily on retrieving focused and relevant context.

## Resources

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [Okapi BM25 and the `rank-bm25` package](https://github.com/dorianbrown/rank_bm25)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [Python Fire documentation](https://google.github.io/python-fire/)
- [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers/)
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)

### AI usage
<!-- This parts is build with Ai,I mean i asked him in what he helped me and that was he's answer -->
AI tools were used as a learning and review aid: to clarify RAG concepts,
review the assignment requirements, identify possible edge cases, and improve
documentation. All implementation choices, testing, and final code review
remain the responsibility of the project author.

try:
    import fire
    import json
    from pathlib import Path
    from tqdm import tqdm
    from .chunking import get_files
    from .indexer import indexing_chunks, load_index, search_bm25
    from .generator import RAGGenerator
    from .models import RagDataset, StudentSearchResults, \
        MinimalSearchResults, MinimalAnswer
    from .evaluator import evaluate_results
except ImportError as e:
    print(f"Error importing modules: {e}")
    raise
except Exception as e:
    print(f"Unexpected error: {e}")
    raise


def validate_input(query: str, k: int):
    if not isinstance(query, str) or \
            not query.strip() or not isinstance(k, int) or k <= 0:
        return False
    return True


class RagPipeline:
    def index(
            self,
            max_chunk_size: int = 2000,
            rep_path: str = "data/raw/vllm-0.10.1"
            ):
        if max_chunk_size <= 0 or max_chunk_size > 2000:
            print("Error: max_chunk_size must be between 1 and 2000.")
            return
        try:
            chunks = get_files(max_chunk_size, rep_path)
            indexing_chunks(chunks)
            # print(chunks)
        except Exception as e:
            print(f"Error while getting indexing files: {e}")

    def search(self, query: str, k: int = 10):
        if not validate_input(query, k):
            return
        bm25, chunks = load_index()
        ranked = search_bm25(bm25, chunks, query, k, True)

        ranked.sort(reverse=True, key=lambda x: x[1])
        results = (chunk for chunk, s in ranked[:k])
        print(f"\nTop results for: '{query}'\n" + "=" * 40)
        for i, chunk in enumerate(results):
            print(
                f"{i + 1}. {chunk.file_path}"
                f" (Chars {chunk.first_character_index}-\
{chunk.last_character_index})"
            )

    def answer(self, query: str, k: int = 10):
        try:
            bm25, chunks = load_index()
            generator = RAGGenerator()
            if not validate_input(query, k):
                return False
            ranked = search_bm25(bm25, chunks, query, k, True)
            ranked.sort(reverse=True, key=lambda x: x[1])
            results = [chunk for chunk, s in ranked[:k]]
            # print(results)
            answer_text = generator.generate_answer(query, results)
            print("Question: ", query)
            print("Answer: ", answer_text)
        except Exception as e:
            print(f"Error while answering: {e}")

    def search_dataset(
            self,
            dataset_path: str,
            k: int = 10,
            save_directory: str = "data/output/search_results"
            ):
        if not validate_input("valid", k):
            print("Error: k must be a positive integer.")
            return
        try:
            bm25, chunks = load_index()

            with open(dataset_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            dataset = RagDataset.model_validate(raw_data)
            search_results = []
            for question in tqdm(
                    dataset.rag_questions, desc="Searching Dataset"
                    ):
                ranked = search_bm25(bm25, chunks, question.question, k, True)
                results = [chunk for chunk, _ in ranked]
                search_result = MinimalSearchResults(
                    question_id=question.question_id,
                    question=question.question,
                    retrieved_sources=[c for c in results]
                )
                search_results.append(search_result)
            student_results = StudentSearchResults(
                search_results=search_results,
                k=k
            )
            save_dir = Path(save_directory)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / Path(dataset_path).name
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(student_results.model_dump_json(indent=2))
        except Exception as e:
            print(f"Error while searching dataset: {e}")

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
    ) -> None:
        try:
            with open(student_search_results_path, "r", encoding="utf-8") as f:
                search_data = StudentSearchResults.model_validate(json.load(f))
            generator = RAGGenerator()
            answers = []
            for result in tqdm(
                search_data.search_results, desc="Generating Answers"
            ):
                answer_text = generator.generate_answer(
                    result.question, result.retrieved_sources
                )
                print("Question: ", result.question)
                print("Answer", answer_text)
                answers.append(MinimalAnswer(
                    question_id=result.question_id,
                    question=result.question,
                    retrieved_sources=result.retrieved_sources,
                    answer=answer_text
                    ))
            output = StudentSearchResults(
                search_results=answers,
                k=search_data.k
            )
            save_directory = Path(save_directory)
            save_directory.mkdir(parents=True, exist_ok=True)
            save_path = save_directory / Path(student_search_results_path).name
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(output.model_dump_json(indent=2))
        except Exception as e:
            print(f"Error while answering dataset: {e}")

    def evaluate(
        self,
        student_answer_path: str,
        dataset_path: str,
        k: int = 10,
    ):
        try:
            evaluate_results(student_answer_path, dataset_path, k)
        except Exception as e:
            print(f"Error while evaluating: {e}")


if __name__ == "__main__":
    fire.Fire(RagPipeline)

import sys
import argparse
import fire
import json
import time
from pathlib import Path
from tqdm import tqdm
from .chunking import get_files
from .indexer import indexing_chunks, load_index, search_bm25
from .generator import RAGGenerator
from .models import RagDataset, StudentSearchResults, MinimalSearchResults, MinimalSource, MinimalAnswer, StudentSearchResults


def validate_input(query: str, k: int):
    if not isinstance(query, str) or not query.strip() or not isinstance(k, int) or k <= 0:
        return False
    return True

class RagPipeline:
    def index(self, max_chunk_size: int, rep_path: str = "data/raw/vllm-0.10.1"):
        if max_chunk_size <= 0 or max_chunk_size > 2000:
            print("Error: max_chunk_size must be between 1 and 2000.")
            return
        # try:
        chunks = get_files(max_chunk_size, rep_path)
        indexing_chunks(chunks)
        # print(chunks)
        # except Exception as e:
        #     print(f"Error while getting indexing files: {e}")

    def search(self, query: str, k: int = 10):
        if not validate_input(query, k):
            return
        bm25_docs, chunks_docs, bm25_code, chunks_code = load_index()
        docs_results = search_bm25(bm25_docs, chunks_docs, query, k, True)
        code_results = search_bm25(bm25_code, chunks_code, query, k, True)

        all_results = (docs_results + code_results)
        all_results.sort(reverse=True, key=lambda x: x[1])
        results = (chunk for chunk, s in all_results[:k])
        print(f"\nTop results for: '{query}'\n" + "=" * 40)
        for i, chunk in enumerate(results):
            print(
                f"{i + 1}. {chunk.file_path}"
                f" (Chars {chunk.first_character_index}-{chunk.last_character_index})"
            )
        
    def answer(self, query: str, k: int = 10):

        bm25_docs, chunks_docs, bm25_code, chunks_code = load_index()
        generator = RAGGenerator()
        if not validate_input(query, k):
            return False
        docs_results = search_bm25(bm25_docs, chunks_docs, query, k, True)
        code_results = search_bm25(bm25_code, chunks_code, query, k, True)
        all_results = (docs_results + code_results)
        all_results.sort(reverse=True, key=lambda x: x[1])
        results = [chunk for chunk, s in all_results[:k]]
        print(results)
        answer_text = generator.generate_answer(query, results)
        print("Question: ", query)
        print("Answer: ", answer_text)

    def search_dataset(self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results"):
        if not validate_input("valid", k):
            print("")
            return
        bm25_docs, chunks_docs, bm25_code, chunks_code = load_index()

        with open(dataset_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        dataset = RagDataset.model_validate(raw_data)
        search_results = []
        for question in dataset.rag_questions:
            docs_results = search_bm25(bm25_docs, chunks_docs, question.question, k, True)
            code_results = search_bm25(bm25_code, chunks_code, question.question, k, True)
            all_results = (docs_results + code_results)
            all_results.sort(reverse=True, key=lambda x: x[1])
            results = [chunk for chunk, s in all_results[:k]]

            search_result = MinimalSearchResults(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=[c for c in results]
            )
            search_results.append(search_result)
        student_results = StudentSearchResults(
            search_results=search_results,
            k = k
        )
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / Path(dataset_path).name
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(student_results.model_dump_json(indent=2))

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
    ) -> None:
        with open(student_search_results_path, "r", encoding="utf-8") as f:
            search_data = StudentSearchResults.model_validate(json.load(f))
        generator = RAGGenerator()
        # print(search_data.search_results)
        answers = []
        for result in tqdm(
            search_data.search_results, desc="Generating Answers"
        ):
            start = time.perf_counter()
            answer_text = generator.generate_answer(result.question, result.retrieved_sources)
            # print("Question: ", result.question)
            # print("Answer", answer_text)
            answers.append(MinimalAnswer(
                question_id=result.question_id,
                question=result.question,
                retrieved_sources=result.retrieved_sources,
                answer=answer_text
                ))
            elapsed = time.perf_counter() - start
        output = StudentSearchResults(
            search_results=answers,
            k=search_data.k
        )
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        save_path = save_directory / Path(student_search_results_path).name
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(output.model_dump_json(indent=2))


if __name__ == "__main__":
    fire.Fire(RagPipeline)

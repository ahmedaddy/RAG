try:
    from pathlib import Path
    import string
    import pickle
    from rank_bm25 import BM25Okapi
except ImportError:
    print("Please install the 'rank_bm25' package to use the indexer.")

index_directory = Path("data/processed")


def tokenize(text: str) -> list[str]:
    """
    Tokenize the input text into a list of tokens.
    Args:
        text (str): The input text to tokenize.
    Returns:
        list[str]: A list of tokens extracted from the input text.
    """
    text = text.lower()
    punct = string.punctuation.replace('_', '')
    clean_text = text.translate(str.maketrans(punct, ' ' * len(punct)))
    tokens = clean_text.split()
    extended_tokens = []
    for t in tokens:
        extended_tokens.append(t)
        if '_' in t:
            extended_tokens.extend(t.split('_'))
    return extended_tokens


def indexing_chunks(chunks: list) -> None:
    """
    Index the provided chunks using the BM25 algorithm.
    Args:
        chunks (list): A list of DataSource objectsrepresenting
        the chunks to index.
    Returns:
        None
    """
    try:
        from rank_bm25 import BM25Okapi
        print("INDEXER:")
        print("#" * 40, "\n")
        print(f"Building index for {len(chunks)} chunks...")
        print("\n", "#" * 40)
        index_directory.mkdir(parents=True, exist_ok=True)

        docs_chunks = [
            c for c in chunks
            if Path(c.file_path).suffix in {".md", ".txt"}
        ]
        code_chunks = [
            c for c in chunks
            if Path(c.file_path).suffix in {".py"}
        ]

        print("docs:", len(docs_chunks))
        print("code:", len(code_chunks))
        print("doc type:", type(docs_chunks[0]))
        print("code type:", type(code_chunks[0]))
        docs_corpus = [
            tokenize(f"{c.file_path}\n{c.text}") for c in docs_chunks
        ]
        code_corpus = [
            tokenize(f"{c.file_path}\n{c.text}") for c in code_chunks
        ]

        all_chunks = docs_chunks + code_chunks
        all_corpus = docs_corpus + code_corpus
        bm25_all = BM25Okapi(all_corpus)
        with open(index_directory / "bm25_all.pkl", "wb") as f:
            pickle.dump(bm25_all, f)
        with open(index_directory / "chunks_all.pkl", "wb") as f:
            pickle.dump(all_chunks, f)
    except Exception as e:
        print(f"Error while indexing chunks: {e}")


def load_index() -> tuple:
    """
    Load the BM25 index and chunks from the specified directory.
    Returns:
        tuple: A tuple containing the BM25 index and the list of chunks.
    """
    try:
        with open(index_directory / "bm25_all.pkl", "rb") as f:
            bm25 = pickle.load(f)
        with open(index_directory / "chunks_all.pkl", "rb") as f:
            chunks = pickle.load(f)
        return bm25, chunks
    except FileNotFoundError:
        raise ValueError("Index files not found. Please run \
the indexing process first.")
    except PermissionError:
        raise ValueError("Permission denied while accessing index files.")
    except Exception as e:
        raise ValueError(f"Error while loading index: {e}")
    return None, None


def search_bm25(
        bm25: BM25Okapi,
        chunks: list,
        query: str,
        k: int,
        with_scores: bool = False
        ) -> list:
    """
    Search for the top k results for the given query using the BM25 index.
    Args:
        bm25 (BM25Okapi): The BM25 index.
        chunks (list): A list of DataSource objects representing the chunks.
        query (str): The search query.
        k (int): The number of top results to retrieve.
        with_scores (bool): Whether to return scores along with the results.
    Returns:
        list: A list of top k DataSource objects or tuples of
        (DataSource, score) if with_scores is True.
    """
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)
    score_index_pairs = [(scores[i], i) for i in range(len(scores))]
    score_index_pairs.sort(reverse=True, key=lambda x: x[0])
    top_scores_index = [(i, s) for s, i in score_index_pairs[:k] if s > 0]
    if with_scores:
        return [(chunks[i], s) for i, s in top_scores_index]
    return [chunks[i] for i, s in top_scores_index]

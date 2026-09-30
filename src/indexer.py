from rank_bm25 import BM25Okapi
from pathlib import Path
import string
import pickle

index_directory = Path("data/processed")

def tokenize(text: str):
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

def indexing_chunks(chunks):
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
    docs_corpus = [tokenize(c.text) for c in docs_chunks]
    code_corpus = [tokenize(c.text) for c in code_chunks]

    bm25_docs = BM25Okapi(docs_corpus)
    bm25_code = BM25Okapi(docs_corpus)
    # print(bm25_docs)
    with open(index_directory / "bm25_docs.pkl", "wb") as f:
        pickle.dump(bm25_docs, f)
    with open(index_directory / "chunks_docs.pkl", "wb") as f:
        pickle.dump(docs_chunks, f)

    with open(index_directory / "bm25_code.pkl", "wb") as f:
        pickle.dump(bm25_code, f)
    with open(index_directory / "chunks_code.pkl", "wb") as f:
        pickle.dump(code_chunks, f)

def load_index(index_type: str = "all"):
    try:
        if index_type == "doc":
            with open(index_directory / "bm25_docs.pkl", "rb") as f:
                bm25_docs = pickle.load(f)
            with open(index_directory / "chunks_docs.pkl", "rb") as f:
                chunks_docs = pickle.load(f)
            return bm25_docs, chunks_docs
        if index_type == "code":
            with open(index_directory / "bm25_code.pkl", "rb") as f:
                bm25_code = pickle.load(f)
            with open(index_directory / "chunks_code.pkl", "rb") as f:
                chunks_code= pickle.load(f)
            return bm25_code, chunks_code
        else:
            with open(index_directory / "bm25_docs.pkl", "rb") as f:
                bm25_docs = pickle.load(f)
            with open(index_directory / "chunks_docs.pkl", "rb") as f:
                chunks_docs = pickle.load(f)
            with open(index_directory / "bm25_code.pkl", "rb") as f:
                bm25_code = pickle.load(f)
            with open(index_directory / "chunks_code.pkl", "rb") as f:
                chunks_code= pickle.load(f)
            return bm25_docs, chunks_docs, bm25_code, chunks_code
    except Exception as e:
        print(f"Error loading index: {e}")

def search_bm25(bm25, chunks, query, k):
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)
    score_index_pairs = [(scores[i], i) for i in range(len(scores))]
    score_index_pairs.sort(reverse=True, key=lambda x: x[0])
    top_scores_index = [i for s, i in score_index_pairs[:k] if s > 0]
    return [chunks[i] for i in top_scores_index]

from pathlib import Path
from .models import MinimalSource
import ast


def split_point_finder(file_path, file_text, start, max_chunk_size):
    if start + max_chunk_size >= len(file_text):
        return len(file_text)
    chunk = file_text[start:start + max_chunk_size]
    if file_path.suffix == ".md":
        splits = ["\n\n", "\n"]
    else:
        splits = ["\n#", "\n##"]
    for s in splits:
        index = chunk.rfind(s)
        if index > 0:
            return start + index
    return start + max_chunk_size

# Sliding Window Chunking
def chunk_md_txt_files(file_path, file_text, max_chunk_size):
    overlap = int(max_chunk_size * 0.05)
    chunks = []
    start = 0
    while start < len(file_text):
        end = split_point_finder(file_path, file_text, start, max_chunk_size)
        chunk_content = file_text[start:end]
        if chunk_content:
            chunks.append(MinimalSource(
                file_path=str(file_path),
                first_character_index=start,
                last_character_index=end,
                text=chunk_content
                ))

        start = end
    return chunks

def split_large_chunk(file_path, chunk, max_chunk_size, start):
    lines = chunk.splitlines(keepends=True)

    chunks = []
    current = []
    current_size = 0
    curr_start = start
    for line in lines:
        if current_size + len(line) > max_chunk_size:
            if current:
                chunks_text = "".join(current)
                chunks.append(
                    MinimalSource(
                    file_path=str(file_path),
                    first_character_index=curr_start,
                    last_character_index=curr_start + len(chunks_text),
                    text=chunks_text,
                ))
                curr_start += len(chunks_text)
            current = []
            current_size = 0

        current.append(line)
        current_size += len(line)
    if current:
        chunks_text = "\n".join(current)
        chunks.append(
            MinimalSource(
            file_path=str(file_path),
            first_character_index=curr_start,
            last_character_index=curr_start + len(chunks_text),
            text=chunks_text,
        ))
    return chunks

def chunk_py_files(file_path, code, max_chunk_size):
    tree = ast.parse(code)
    chunks = []
    lines = code.splitlines(keepends=True)
    for node in tree.body:
        chunks_text = ""
        start = sum(len(line) for line in lines[:node.lineno - 1]) + node.col_offset
        end = sum(len(line) for line in lines[:node.end_lineno - 1]) + node.end_col_offset
        chunks_text = code[start:end]
        
        if len(chunks_text) <= max_chunk_size:
            chunks.append(MinimalSource(
                file_path=str(file_path),
                first_character_index=start,
                last_character_index=end,
                text=chunks_text,
            ))
            print(chunks_text)
        else:
            chunks.extend(split_large_chunk(file_path, chunks_text, max_chunk_size, start))
    return chunks


def read_file(path: Path):
    try:
        with open(path, "r", encoding="UTF-8") as f:
            return f.read()
    except Exception:
        print(f"Error reading file: {path}")
        return None

def chunk_files(file_path, max_chunk_size):
    file_data = read_file(file_path)
    if file_path.suffix == ".py":
        return chunk_py_files(file_path, file_data, max_chunk_size)
    if file_path.suffix in [".md", ".txt"]:
        return chunk_md_txt_files(file_path, file_data, max_chunk_size)

def get_files(max_chunk_size, rep_path):
    # file = Path("data/r.py")
    # chunk_files(file, max_chunk_size)
    # return
    rep_path = Path(rep_path)
    valid_suffixes = [".md", ".txt", ".py"]
    chunks = []
    for file in rep_path.rglob("*"):
        if file.is_file() and file.suffix in valid_suffixes:
            chunks.extend(chunk_files(file, max_chunk_size))
    return chunks
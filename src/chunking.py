try:
    from pathlib import Path
    from .models import DataSource
    import ast
except ImportError as e:
    print(f"Error importing modules: {e}")
    raise
except Exception as e:
    print(f"Unexpected error: {e}")
    raise


def split_point_finder(
        file_path: Path, file_text: str, start: int, max_chunk_size: int
        ) -> int:
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
def chunk_md_txt_files(
        file_path: Path, file_text: str, max_chunk_size: int
        ) -> list:
    overlap = int(max_chunk_size * 0.05)
    chunks = []
    start = 0
    while start < len(file_text):
        end = split_point_finder(file_path, file_text, start, max_chunk_size)
        chunk_content = file_text[start:end]
        if chunk_content:
            chunks.append(DataSource(
                file_path=str(file_path),
                first_character_index=start,
                last_character_index=end,
                text=chunk_content
                ))
        if end - overlap > start:
            start = end - overlap
        else:
            start = end
    return chunks


def split_large_chunk(
        file_path: Path, chunk: str, max_chunk_size: int, start: int
        ) -> list:
    lines = chunk.splitlines(keepends=True)
    chunks: list = []
    current: list = []
    current_size: int = 0
    current_start: int = start

    for line in lines:
        if current_size + len(line) > max_chunk_size:
            if current:
                chunk_text = "".join(current)
                chunks.append(
                    DataSource(
                        file_path=str(file_path),
                        first_character_index=current_start,
                        last_character_index=current_start + len(chunk_text),
                        text=chunk_text,
                    )
                )
                current_start += len(chunk_text)
                current_size = 0
                current = []
        if len(line) > max_chunk_size:
            for i in range(0, len(line), max_chunk_size):
                sub_chunk = line[i:i + max_chunk_size]
                chunks.append(
                    DataSource(
                        file_path=str(file_path),
                        first_character_index=current_start,
                        last_character_index=current_start + len(sub_chunk),
                        text=sub_chunk,
                    )
                )
            current_start += len(sub_chunk)
        else:
            current.append(line)
            current_size += len(line)
    if current:
        chunk_text = "".join(current)
        chunks.append(
            DataSource(
                file_path=str(file_path),
                first_character_index=current_start,
                last_character_index=current_start + len(chunk_text),
                text=chunk_text,
            )
        )
    return chunks


def chunk_py_files(file_path: Path, code: str, max_chunk_size: int) -> list:
    # print(file_path)
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        print(f"Error parsing Python file: {file_path} - {e}")
        return []
    chunks = []
    lines = code.splitlines(keepends=True)
    for node in tree.body:
        chunks_text = ""
        start = (
            sum(len(line) for line in lines[:node.lineno - 1])
            + node.col_offset
            )
        if node.end_lineno is None or node.end_col_offset is None:
            continue
        end = (
            sum(len(line) for line in lines[:node.end_lineno - 1])
            + node.end_col_offset
            )
        chunks_text = code[start:end]

        if len(chunks_text) <= max_chunk_size:
            chunks.append(DataSource(
                file_path=str(file_path),
                first_character_index=start,
                last_character_index=end,
                text=chunks_text,
            ))
            # print(chunks_text)
            if end - start > 2000:
                print(end - start)
                break
        else:
            chunks.extend(split_large_chunk(
                file_path, chunks_text, max_chunk_size, start))
    return chunks


def read_file(path: Path) -> str:
    try:
        with open(path, "r", encoding="UTF-8") as f:
            return f.read()
    except Exception:
        print(f"Error reading file: {path}")
        return ""


def chunk_files(file_path: Path, max_chunk_size: int) -> list:
    file_data = read_file(file_path)
    if file_path.suffix == ".py":
        return chunk_py_files(file_path, file_data, max_chunk_size)
    if file_path.suffix in [".md", ".txt"]:
        return chunk_md_txt_files(file_path, file_data, max_chunk_size)
    return []


def get_files(max_chunk_size: int, rep_path: str) -> list:
    # file = Path("data/r.py")
    # chunk_files(file, max_chunk_size)
    # return
    repository_path = Path(rep_path)
    valid_suffixes = [".md", ".txt", ".py"]
    chunks = []
    for file in repository_path.rglob("*"):
        if file.is_file() and file.suffix in valid_suffixes:
            chunks.extend(chunk_files(file, max_chunk_size))
    return chunks

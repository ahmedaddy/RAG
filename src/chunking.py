try:
    from pathlib import Path
    from .models import DataSource
    from tqdm import tqdm
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
    """
    Find the optimal split point for a chunk of text.
    Args:
        file_path (Path): The path to the file.
        file_text (str): The text of the file.
        start (int): The starting index of the chunk.
        max_chunk_size (int): The maximum size of the chunk.
    Returns:
        int: The index where the chunk should be split.
    """
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
    """
    Chunk markdown and text files into smaller pieces.
    Args:
        file_path (Path): The path to the file.
        file_text (str): The text of the file.
        max_chunk_size (int): The maximum size of each chunk.
    Returns:
        list: A list of DataSource objects representing the chunks.
    """
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
    """
    Split a large chunk into smaller pieces.
    Args:
        file_path (Path): The path to the file.
        chunk (str): The text of the chunk.
        max_chunk_size (int): The maximum size of each sub-chunk.
        start (int): The starting index of the chunk.
    Returns:
        list: A list of DataSource objects representing the sub-chunks.
    """
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
    """
    Chunk Python files into smaller pieces based on AST nodes.
    Args:
        file_path (Path): The path to the Python file.
        code (str): The code of the Python file.
        max_chunk_size (int): The maximum size of each chunk.
    Returns:
        list: A list of DataSource objects representing the chunks.
    """
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
    """
    Read the contents of a file.
    Args:
        path (Path): The path to the file.
    Returns:
        str: The contents of the file.
    """
    try:
        with open(path, "r", encoding="UTF-8") as f:
            return f.read()
    except FileNotFoundError as e:
        print(f"File not found: {path} - {e}")
        return ""
    except UnicodeDecodeError as e:
        print(f"Unicode decode error for file: {path} - {e}")
        return ""
    except PermissionError as e:
        print(f"Permission denied for file: {path} - {e}")
        return ""
    except IsADirectoryError as e:
        print(f"Expected a file but found a directory: {path} - {e}")
        return ""
    except Exception as e:
        print(f"Error reading file: {path} - {e}")
        return ""


def chunk_files(file_path: Path, max_chunk_size: int) -> list:
    """
    Chunk files based on their type (Python, Markdown, or text).
    Args:
        file_path (Path): The path to the file.
        max_chunk_size (int): The maximum size of each chunk.
    Returns:
        list: A list of DataSource objects representing the chunks.
    """
    file_data = read_file(file_path)
    if file_path.suffix == ".py":
        return chunk_py_files(file_path, file_data, max_chunk_size)
    if file_path.suffix in [".md", ".txt"]:
        return chunk_md_txt_files(file_path, file_data, max_chunk_size)
    return []


def get_files(max_chunk_size: int, rep_path: str) -> list:
    """
    Get all files from the specified repository path and chunk them.
    Args:
        max_chunk_size (int): The maximum size of each chunk.
        rep_path (str): The path to the repository containing files to index.
    Returns:
        list: A list of DataSource objects
        representing the chunks from all files.
    """
    # file = Path("data/r.py")
    # chunk_files(file, max_chunk_size)
    # return
    repository_path = Path(rep_path)
    if not repository_path.exists():
        raise FileNotFoundError(f"Repository path does not exist: {rep_path}")
    valid_suffixes = [".md", ".txt", ".py"]
    chunks = []
    files = [
        file for file in repository_path.rglob("*")
        if file.is_file() and file.suffix in valid_suffixes
    ]
    for file in tqdm(
        files,
        desc="Indexing files"
    ):
        chunks.extend(chunk_files(file, max_chunk_size))
    return chunks

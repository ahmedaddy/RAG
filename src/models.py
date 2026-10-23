from pydantic import BaseModel, Field
from typing import List
import uuid


class MinimalSource(BaseModel):
    """
    Represents a minimal source of
    information with file path and character indices.
    Attributes:
        file_path (str): The path to the source file.
        first_character_index (int): The index of
        the first character in the source.
        last_character_index (int): The index of
        the last character in the source.
    """
    file_path: str
    first_character_index: int
    last_character_index: int


class DataSource(BaseModel):
    """
    Represents a data source with file path and character indices.
    Attributes:
        file_path (str): The path to the source file.
        first_character_index (int): The index of
        the first character in the source.
        last_character_index (int): The index of
        the last character in the source.
        text (str): The text content of the source.
    """
    file_path: str
    first_character_index: int
    last_character_index: int
    text: str


class UnansweredQuestion(BaseModel):
    """
    Represents an unanswered question.
    Attributes:
        question_id (str): The ID of the question.
        question (str): The question text.
    """
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """
    Represents an answered question.
    Attributes:
        sources (List[MinimalSource]): The sources used to answer the question.
        answer (str): The answer to the question.
    """
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """
    Represents a dataset of RAG questions.
    Attributes:
        rag_questions (List[AnsweredQuestion | UnansweredQuestion]):
        The list of RAG questions.
    """
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """
    Represents the minimal search results for a question.
    Attributes:
        question_id (str): The ID of the question.
        question (str): The question text.
        retrieved_sources (List[DataSource]):
        The list of retrieved data sources.
    """
    question_id: str
    question: str
    retrieved_sources: List[DataSource]


class MinimalAnswer(MinimalSearchResults):
    """
    Represents the minimal answer for a question.
    Attributes:
        question_id (str): The ID of the question.
        question (str): The question text.
        retrieved_sources (List[DataSource]):
        The list of retrieved data sources.
        answer (str): The answer to the question.
    """
    answer: str


class StudentSearchResults(BaseModel):
    """
    Represents the search results for a student.
    Attributes:
        search_results (List[MinimalSearchResults]):
        The list of search results.
        k (int): The number of top results to retrieve.
    """
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """
    Represents the search results and answer for a student.
    Attributes:
        search_results (List[MinimalAnswer]):
        The list of search results with answers.
        k (int): The number of top results to retrieve.
    """
    search_results: List[MinimalAnswer]
    k: int

from .models import AnsweredQuestion, RagDataset, StudentSearchResults
import json


def get_overlap(
        start1: int,
        end1: int,
        start2: int,
        end2: int
        ) -> float:
    """
    Calculate the overlap between two ranges.
    Args:
        start1 (int): The starting index of the first range.
        end1 (int): The ending index of the first range.
        start2 (int): The starting index of the second range.
        end2 (int): The ending index of the second range.
    Returns:
        float: The overlap between the two ranges.
    """
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection = max(0, intersection_end - intersection_start)
    union_area1 = end1 - start1
    union_area2 = end2 - start2
    union = union_area1 + union_area2 - intersection
    if union == 0:
        return 0.0
    return intersection / union


def evaluate_results(
        student_answer_path: str,
        dataset_path: str,
        k: int
        ) -> None:
    """
    Evaluate the student answers against the dataset.
    Args:
        student_answer_path (str): The path to the student answers.
        dataset_path (str): The path to the dataset.
        k (int): The number of top results to consider.
    Returns:
        None
    """
    try:
        with open(student_answer_path, "r", encoding="utf-8") as f:
            student_data = StudentSearchResults.model_validate(json.load(f))
        with open(dataset_path, "r", encoding="utf-8") as f:
            correct_data = RagDataset.model_validate(json.load(f))
    except Exception as e:
        print(f"Error: Could not load student results or dataset: {e}")
        return
    # print(student_data)
    # print(correct_data)
    correct_data_map = {
        q.question_id: q
        for q in correct_data.rag_questions
        if isinstance(q, AnsweredQuestion)
    }
    k_values = [1, 3, 5, 10]
    recall = {kval: 0.0 for kval in k_values}
    qustion_count = 0
    for result in student_data.search_results:
        if result.question_id not in correct_data_map:
            continue
        qustion_count += 1
        correct_question = correct_data_map[result.question_id]
        correct_source = correct_question.sources
        # print(result)
        for kval in k_values:
            top_k_retreived = result.retrieved_sources[:kval]
            found_count = 0
            for s in correct_source:
                found = False
                for retreived in top_k_retreived:
                    if (s.file_path == retreived.file_path):
                        overlap = get_overlap(
                                s.first_character_index,
                                s.last_character_index,
                                retreived.first_character_index,
                                retreived.last_character_index
                            )
                        if overlap >= 0.05:
                            found = True
                            break
                if found:
                    found_count += 1
            try:
                qustion_recall = found_count / len(correct_source)
            except Exception:
                raise ValueError(f"Error: The correct source \
list is empty for question_id: {result.question_id}")
            recall[kval] += qustion_recall
    for kval in k_values:
        recall[kval] /= qustion_count
    print("Evaluation Results:")
    for kval in k_values:
        print(f"Recall at {kval}: {recall[kval]:.2f}")

from .models import AnsweredQuestion, RagDataset, StudentSearchResults
import json


def get_overlap(start1, end1, start2, end2):
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    overlap_length = max(0, overlap_end - overlap_start + 1)
    range1_length = end1 - start1 + 1
    range2_length = end2 - start2 + 1
    if range1_length == 0 or range2_length == 0:
        return 0.0
    return overlap_length / min(range1_length, range2_length)


def evaluate_results(student_answer_path, dataset_path, k: int):
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
            overlaps = []
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
                        overlaps.append(overlap)
                        if overlap >= 0.05:
                            found = True
                            break
            if found:
                found_count += 1
            print("found count: ", found_count)
            qustion_recall = found_count
            recall[kval] += qustion_recall
            # print(found_count)
        recall[kval] /= qustion_count
    print(recall)

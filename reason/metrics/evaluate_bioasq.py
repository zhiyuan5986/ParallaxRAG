"""
BioASQ evaluation script.
Supports yes/no, factoid, list, and summary questions.
"""

import json
import re
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import numpy as np


def extract_answer_from_prediction(prediction: str) -> str:
    if '</think>' in prediction:
        # Extract content after </think>.
        match = re.search(r'</think>\s*(.*)', prediction, re.DOTALL)
        if match:
            return match.group(1).strip()
    return prediction.strip()


def extract_key_entities_from_ideal(ideal_text: str) -> List[str]:
    entities = []

    # Percent or numeric in parentheses (e.g., "13%").
    percent_matches = re.findall(r'\((\d+%)\)', ideal_text)
    entities.extend(percent_matches)

    # Standalone percentages.
    percent_matches = re.findall(r'\b(\d+(?:\.\d+)?%)\b', ideal_text)
    entities.extend(percent_matches)

    # Proper nouns (2-5 capitalized words).
    proper_noun_matches = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4})\b', ideal_text)
    entities.extend(proper_noun_matches)

    # Uppercase acronyms (e.g., FOLFOXIRI, MHC).
    acronym_matches = re.findall(r'\b([A-Z]{2,}(?:-[A-Z][a-z]+)?)\b', ideal_text)
    entities.extend(acronym_matches)

    # De-duplicate.
    return list(set(entities))


def normalize_answer(answer: str) -> str:
    answer = answer.lower().strip()
    # Remove punctuation.
    answer = re.sub(r'[^\w\s]', '', answer)
    # Normalize whitespace.
    answer = re.sub(r'\s+', ' ', answer)
    return answer


def extract_yes_no_from_text(text: str) -> str:
    text = text.lower().strip()

    # Direct prefix match.
    if text.startswith('yes'):
        return 'yes'
    elif text.startswith('no'):
        return 'no'

    # First occurrence match.
    yes_match = re.search(r'\byes\b', text)
    no_match = re.search(r'\bno\b', text)
    
    if yes_match and no_match:
        # Choose the earliest occurrence.
        if yes_match.start() < no_match.start():
            return 'yes'
        else:
            return 'no'
    elif yes_match:
        return 'yes'
    elif no_match:
        return 'no'
    
    return 'unknown'


def extract_list_answers_from_text(text: str, num_expected: int = None) -> List[str]:
    answers = []

    # Method 1: numbered list.
    numbered_pattern = r'(?:^|\n)\s*(?:\d+[\.\)]\s*|[-\*]\s*)([^\n]+)'
    matches = re.findall(numbered_pattern, text, re.MULTILINE)
    if matches:
        answers = [normalize_answer(m.strip()) for m in matches if m.strip()]

    # Method 2: comma-separated.
    if not answers:
        # Find a short sentence that contains commas.
        sentences = re.split(r'[.!?]\s+', text)
        for sent in sentences:
            if ',' in sent and len(sent) < 200:
                parts = sent.split(',')
                if len(parts) >= 2:
                    answers = [normalize_answer(p.strip()) for p in parts if p.strip()]
                    break

    # Method 3: simple entity extraction.
    if not answers:
        # Parentheses or capitalized phrases.
        entity_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        matches = re.findall(entity_pattern, text)
        if matches:
            answers = [normalize_answer(m) for m in matches[:10]]

    # De-dup while preserving order.
    seen = set()
    unique_answers = []
    for ans in answers:
        if ans and ans not in seen:
            seen.add(ans)
            unique_answers.append(ans)
    
    return unique_answers


def calculate_f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def evaluate_yesno(predictions: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Evaluate yes/no questions using macro F1 over Yes/No classes.
    """
    # Map ID to ground truth.
    gt_map = {item['id']: item for item in ground_truth}

    # Count TP/FP/FN for yes and no.
    yes_tp = yes_fp = yes_fn = 0
    no_tp = no_fp = no_fn = 0
    total = 0
    correct = 0
    
    for pred_item in predictions:
        qid = pred_item['id']
        if qid not in gt_map:
            continue
        
        gt_item = gt_map[qid]
        # Support both type and question_type fields.
        gt_type = gt_item.get('type') or gt_item.get('question_type')
        if gt_type != 'yesno':
            continue
        
        total += 1
        
        # Extract ground truth answer (prefer exact_answer, fallback to ideal_answer).
        gt_answer_text = gt_item.get('exact_answer', '')
        if not gt_answer_text and 'ideal_answer' in gt_item:
            # ideal_answer can be list or string.
            ideal_ans = gt_item['ideal_answer']
            if isinstance(ideal_ans, list) and ideal_ans:
                gt_answer_text = ideal_ans[0]
            elif isinstance(ideal_ans, str):
                gt_answer_text = ideal_ans
        
        if isinstance(gt_answer_text, str):
            gt_answer_text = gt_answer_text.lower().strip()
        else:
            gt_answer_text = str(gt_answer_text).lower().strip()
        
        # Extract yes/no from ground truth text.
        gt_answer = extract_yes_no_from_text(gt_answer_text)

        # Extract prediction (strip <think>).
        pred_text = pred_item.get('prediction', '')
        pred_text = extract_answer_from_prediction(pred_text)
        pred_answer = extract_yes_no_from_text(pred_text)

        # Update counts.
        if pred_answer == gt_answer and gt_answer != 'unknown':
            correct += 1
            if gt_answer == 'yes':
                yes_tp += 1
            elif gt_answer == 'no':
                no_tp += 1
        else:
            if gt_answer == 'yes':
                yes_fn += 1
                if pred_answer == 'no':
                    no_fp += 1
            elif gt_answer == 'no':
                no_fn += 1
                if pred_answer == 'yes':
                    yes_fp += 1
    
    # Precision/recall/F1 for each class.
    yes_precision = yes_tp / (yes_tp + yes_fp) if (yes_tp + yes_fp) > 0 else 0.0
    yes_recall = yes_tp / (yes_tp + yes_fn) if (yes_tp + yes_fn) > 0 else 0.0
    yes_f1 = calculate_f1_score(yes_precision, yes_recall)
    
    no_precision = no_tp / (no_tp + no_fp) if (no_tp + no_fp) > 0 else 0.0
    no_recall = no_tp / (no_tp + no_fn) if (no_tp + no_fn) > 0 else 0.0
    no_f1 = calculate_f1_score(no_precision, no_recall)
    
    # Macro F1.
    macro_f1 = (yes_f1 + no_f1) / 2

    # Accuracy.
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        'macro_f1': macro_f1,
        'yes_f1': yes_f1,
        'no_f1': no_f1,
        'yes_precision': yes_precision,
        'yes_recall': yes_recall,
        'no_precision': no_precision,
        'no_recall': no_recall,
        'accuracy': accuracy,
        'total': total
    }


def evaluate_factoid(predictions: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]]) -> Dict[str, float]:
    gt_map = {item['id']: item for item in ground_truth}
    
    reciprocal_ranks = []
    strict_correct = 0
    lenient_correct = 0
    total = 0
    
    for pred_item in predictions:
        qid = pred_item['id']
        if qid not in gt_map:
            continue
        
        gt_item = gt_map[qid]
        # Support both type and question_type fields.
        gt_type = gt_item.get('type') or gt_item.get('question_type')
        if gt_type != 'factoid':
            continue
        
        total += 1
        
        # Extract ground truth answers (may contain multiple acceptable answers).
        exact_answers = gt_item.get('exact_answer', [[]])

        # Fallback to ideal_answer if exact_answer is missing.
        if not exact_answers or not exact_answers[0]:
            if 'ideal_answer' in gt_item:
                ideal_ans = gt_item['ideal_answer']
                if isinstance(ideal_ans, list) and ideal_ans:
                    # Convert ideal_answer to exact_answer format.
                    exact_answers = [[ideal_ans[0]]]
                elif isinstance(ideal_ans, str) and ideal_ans:
                    exact_answers = [[ideal_ans]]
        
        if not exact_answers or not exact_answers[0]:
            continue
        
        # Normalize ground truth answers.
        gt_answers_normalized = set()
        for answer_list in exact_answers:
            for ans in answer_list:
                gt_answers_normalized.add(normalize_answer(str(ans)))
        
        # Extract key entities for lenient matching.
        gt_key_entities = []
        for answer_list in exact_answers:
            for ans in answer_list:
                entities = extract_key_entities_from_ideal(str(ans))
                gt_key_entities.extend([normalize_answer(e) for e in entities])
        gt_key_entities = set(gt_key_entities)
        
        # Extract prediction (strip <think>, take first sentence).
        pred_text = pred_item.get('prediction', '').strip()
        pred_text = extract_answer_from_prediction(pred_text)

        # Strip "Answer:" prefix if present.
        pred_text = re.sub(r'^answer:\s*', '', pred_text, flags=re.IGNORECASE).strip()

        pred_normalized = normalize_answer(pred_text.split('.')[0])

        # Match checks.
        rank = -1
        matched = False

        # Method 1: strict full match.
        for gt_ans in gt_answers_normalized:
            if pred_normalized == gt_ans:
                strict_correct += 1
                lenient_correct += 1
                rank = 1
                matched = True
                break

        # Method 2: key-entity match.
        if not matched and gt_key_entities:
            for entity in gt_key_entities:
                if entity in pred_normalized or pred_normalized in entity:
                    lenient_correct += 1
                    if rank == -1:
                        rank = 1
                    matched = True
                    break

        # Method 3: substring match.
        if not matched:
            for gt_ans in gt_answers_normalized:
                if gt_ans in pred_normalized or pred_normalized in gt_ans:
                    lenient_correct += 1
                    if rank == -1:
                        rank = 1
                    break

        # Reciprocal rank.
        if rank > 0:
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
    
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    strict_accuracy = strict_correct / total if total > 0 else 0.0
    lenient_accuracy = lenient_correct / total if total > 0 else 0.0
    
    return {
        'mrr': mrr,
        'strict_accuracy': strict_accuracy,
        'lenient_accuracy': lenient_accuracy,
        'total': total
    }


def evaluate_list(predictions: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]]) -> Dict[str, float]:
    gt_map = {item['id']: item for item in ground_truth}
    
    f1_scores = []
    precision_scores = []
    recall_scores = []
    total = 0
    
    for pred_item in predictions:
        qid = pred_item['id']
        if qid not in gt_map:
            continue
        
        gt_item = gt_map[qid]
        # Support both type and question_type fields.
        gt_type = gt_item.get('type') or gt_item.get('question_type')
        if gt_type != 'list':
            continue
        
        total += 1
        
        # Extract ground truth answers.
        exact_answers = gt_item.get('exact_answer', [[]])

        # Fallback to ideal_answer if exact_answer is missing.
        if not exact_answers or not exact_answers[0]:
            if 'ideal_answer' in gt_item:
                ideal_ans = gt_item['ideal_answer']
                if isinstance(ideal_ans, list) and ideal_ans:
                    # For list questions, extract entities from ideal_answer.
                    ideal_text = ideal_ans[0] if isinstance(ideal_ans[0], str) else str(ideal_ans[0])
                    extracted_items = extract_list_answers_from_text(ideal_text)
                    if extracted_items:
                        exact_answers = [extracted_items]
                    else:
                        exact_answers = [ideal_ans]
                elif isinstance(ideal_ans, str) and ideal_ans:
                    extracted_items = extract_list_answers_from_text(ideal_ans)
                    if extracted_items:
                        exact_answers = [extracted_items]
                    else:
                        exact_answers = [[ideal_ans]]
        
        if not exact_answers or not exact_answers[0]:
            continue
        
        # Normalize ground truth answers.
        gt_answers_normalized = set()
        for answer_list in exact_answers:
            for ans in answer_list:
                gt_answers_normalized.add(normalize_answer(str(ans)))
        
        # Extract predictions (strip <think>).
        pred_text = pred_item.get('prediction', '')
        pred_text = extract_answer_from_prediction(pred_text)
        pred_answers = extract_list_answers_from_text(pred_text, len(gt_answers_normalized))
        pred_answers_set = set(pred_answers)

        # Precision/recall/F1 for this question.
        if not pred_answers_set:
            f1_scores.append(0.0)
            precision_scores.append(0.0)
            recall_scores.append(0.0)
            continue

        # Count matches.
        correct = 0
        for pred_ans in pred_answers_set:
            for gt_ans in gt_answers_normalized:
                # Lenient match.
                if pred_ans == gt_ans or pred_ans in gt_ans or gt_ans in pred_ans:
                    correct += 1
                    break
        
        precision = correct / len(pred_answers_set) if pred_answers_set else 0.0
        recall = correct / len(gt_answers_normalized) if gt_answers_normalized else 0.0
        f1 = calculate_f1_score(precision, recall)
        
        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)
    
    mean_f1 = np.mean(f1_scores) if f1_scores else 0.0
    mean_precision = np.mean(precision_scores) if precision_scores else 0.0
    mean_recall = np.mean(recall_scores) if recall_scores else 0.0
    
    return {
        'mean_f1': mean_f1,
        'mean_precision': mean_precision,
        'mean_recall': mean_recall,
        'total': total
    }


def evaluate_summary(predictions: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]]) -> Dict[str, float]:
    gt_map = {item['id']: item for item in ground_truth}
    
    total = 0
    
    for pred_item in predictions:
        qid = pred_item['id']
        if qid not in gt_map:
            continue
        
        gt_item = gt_map[qid]
        # Support both type and question_type fields.
        gt_type = gt_item.get('type') or gt_item.get('question_type')
        if gt_type != 'summary':
            continue
        
        total += 1
    
    return {
        'total': total,
        'note': 'Summary questions require ROUGE or expert evaluation'
    }


def evaluate_bioasq(pred_file: str, gt_file: str) -> Dict[str, Any]:
    # Load predictions.
    predictions = []
    with open(pred_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))
    
    # Load ground truth.
    with open(gt_file, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
        if isinstance(gt_data, dict) and 'questions' in gt_data:
            ground_truth = gt_data['questions']
        elif isinstance(gt_data, list):
            ground_truth = gt_data
        else:
            raise ValueError(f"Invalid ground truth format in {gt_file}")
    
    print(f"Loaded {len(predictions)} predictions and {len(ground_truth)} ground truth items")
    
    # Evaluate by question type.
    results = {}

    # Yes/No.
    yesno_results = evaluate_yesno(predictions, ground_truth)
    results['yesno'] = yesno_results
    print(f"\nYes/No Questions ({yesno_results['total']} questions):")
    print(f"  Macro F1: {yesno_results['macro_f1']:.4f}")
    print(f"  Accuracy: {yesno_results['accuracy']:.4f}")
    print(f"  Yes F1: {yesno_results['yes_f1']:.4f}")
    print(f"  No F1: {yesno_results['no_f1']:.4f}")
    
    # Factoid.
    factoid_results = evaluate_factoid(predictions, ground_truth)
    results['factoid'] = factoid_results
    print(f"\nFactoid Questions ({factoid_results['total']} questions):")
    print(f"  MRR: {factoid_results['mrr']:.4f}")
    print(f"  Strict Accuracy: {factoid_results['strict_accuracy']:.4f}")
    print(f"  Lenient Accuracy: {factoid_results['lenient_accuracy']:.4f}")
    
    # List.
    list_results = evaluate_list(predictions, ground_truth)
    results['list'] = list_results
    print(f"\nList Questions ({list_results['total']} questions):")
    print(f"  Mean F1: {list_results['mean_f1']:.4f}")
    print(f"  Mean Precision: {list_results['mean_precision']:.4f}")
    print(f"  Mean Recall: {list_results['mean_recall']:.4f}")
    
    # Summary.
    summary_results = evaluate_summary(predictions, ground_truth)
    results['summary'] = summary_results
    print(f"\nSummary Questions ({summary_results['total']} questions):")
    print(f"  {summary_results['note']}")
    
    # Overall stats.
    total_evaluated = (yesno_results['total'] + factoid_results['total'] + 
                      list_results['total'] + summary_results['total'])
    results['overall'] = {
        'total_evaluated': total_evaluated,
        'total_predictions': len(predictions),
        'total_ground_truth': len(ground_truth)
    }
    
    print(f"\nOverall: {total_evaluated} questions evaluated")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate BioASQ predictions")
    parser.add_argument('pred_file', type=str, help='Prediction file (JSONL)')
    parser.add_argument('gt_file', type=str, help='Ground truth file (JSON)')
    parser.add_argument('--output', type=str, help='Output results to JSON file')
    
    args = parser.parse_args()
    
    results = evaluate_bioasq(args.pred_file, args.gt_file)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")


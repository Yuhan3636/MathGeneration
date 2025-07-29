import os
import json
import random
import logging
import base64
from collections import defaultdict
from fractions import Fraction

try:
    from tqdm import tqdm
    use_tqdm = True
except ImportError:
    use_tqdm = False

import openai

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise RuntimeError('Please set the OPENAI_API_KEY environment variable.')
openai.api_key = OPENAI_API_KEY

INPUT_JSON = os.path.join('data', 'testmini.json')
OUTPUT_JSON = os.path.join(os.path.dirname(__file__), 'testmini_only_question.json')

with open(INPUT_JSON, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Filter to exclude 'Text Only' and 'Vision Only' questions
filtered_data = []
for item in data:
    problem_version = item.get('problem_version', '')
    # Exclude 'Text Only' and 'Vision Only' questions
    if problem_version not in ['Text Only', 'Vision Only']:
        filtered_data.append(item)

math_targeted_data = {str(i): item for i, item in enumerate(filtered_data)}


def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logging.error(f"Failed to encode image {image_path}: {e}")
        return None

def generate_new_question(orig_question, orig_answer, image_path):
    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        return None, None, None, None
    prompt = (
        "Given the image, question and answer from a visual math question, your task is to create a *slightly but meaningfully* modified question that stays true to the original theme yet is *not trivially identical*. This requires you to generate 4 things:\n"
        "(1)New question description: a short explanation of the change.\n"
        "(2)New question: A single clear sentence posing the new math question about the image.\n"
        "(3)New answer description: A concise explanation or computation that justifies new answer\n"
        "(4)New answer: the ground-truth answer to the new question. If the answer is numeric, provide only the number. If the answer is not numeric, provide the correct entity, value, or word answer (e.g., a name, object, or label), not just a number.\n\n"
        "Constraints & Style rules:\n"
        "Consistency – New question and answer must match the image\n"
        "Non-trivial – Change something meaningful (dimensions, counts, angles…)\n"
        "Mathematical Accuracy – Ensure the new answer is mathematically correct based on the modified question\n"
        "Clear Reasoning – The answer description should clearly explain the mathematical steps\n"
        "For multiple choice questions, provide the letter (A, B, C, D) as the answer\n"
        "For numeric answers, provide only the number without units unless the question specifically asks for units\n\n"
        "Here are the original question and original answer:\n"
        f"orig_question: {{{orig_question}}}\n"
        f"orig_answer: {{{orig_answer}}}\n\n"
        "Output your response in this format exactly:\n"
        "new_question_description: < short explanation>\n"
        "new_question: <your new question>\n"
        "new_answer_description: <short explanation>\n"
        "new_answer: <your answer>"
    )
    try:
        response = openai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens=256,
            temperature=0.7,
        )
        content = response.choices[0].message.content.strip()
        new_question_description, new_question, new_answer_description, new_answer = None, None, None, None
        for line in content.split('\n'):
            if line.lower().startswith("new_question_description:"):
                new_question_description = line[len("new_question_description:"):].strip()
            elif line.lower().startswith("new_question:"):
                new_question = line[len("new_question:"):].strip()
            elif line.lower().startswith("new_answer_description:"):
                new_answer_description = line[len("new_answer_description:"):].strip()
            elif line.lower().startswith("new_answer:"):
                new_answer = line[len("new_answer:"):].strip()
        return new_question_description, new_question, new_answer_description, new_answer
    except Exception as e:
        logging.error(f"Step 1 (new_question pipeline) generation failed: {e}")
        return None, None, None, None

def solution_output(new_question, image_path):
    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        return None, None
    prompt = (
        "Your task is to solve the newly generated question given the original image. This requires you to provide two things:\n"
        "(1)check_answer_description: a concise, step-by-step explanation of your reasoning.\n"
        "(2)check_answer: the answer to the new question (numbers only if numeric).\n"
        "Constraints & Style rules:\n"
        "Accuracy - Derive the answer directly from the image and new generated question.\n"
        "Mathematical Precision - Ensure your calculations are mathematically correct\n"
        "For label/entity questions, do not extract a number from the description, even if the description contains one. The answer should be the correct label, entity, or word, not a number.\n"
        "For numeric questions, provide only the number as the answer.\n"
        "For multiple choice questions, provide only the letter (A, B, C, D) as the answer.\n"
        "Do not leave either line blank. \n"
        "Double-check your mathematical reasoning to ensure accuracy.\n\n"
        f"Here is the question:\n"
        f"new_question: {{{new_question}}}\n\n"
        "Output exactly two lines, no extra text, use the exact format below:\n"
        "check_answer_description: <short explanation>\n"
        "check_answer: <your answer>"
    )
    import re
    def parse_solution_output(content, new_question=None):
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if len(lines) >= 2:
            check_answer_description = lines[0]
            check_answer = lines[1]
            def strip_prefix(s, prefix):
                return re.sub(rf'^{prefix}[:\s-]*', '', s, flags=re.IGNORECASE)
            check_answer_description = strip_prefix(check_answer_description, 'check_answer_description')
            check_answer = strip_prefix(check_answer, 'check_answer')
            answer_candidate = check_answer.strip()
            if answer_candidate and not answer_candidate.lower().startswith('no answer') and not answer_candidate.lower().startswith('description'):
                check_answer = answer_candidate
            else:
                # Only extract a number if the question is numeric
                if new_question and re.search(r'\b(how many|what is the value|calculate|amount|number|sum|difference|total|product|quotient|average|mean|median|max|min|area|length|distance|perimeter|radius|diameter|volume|weight|mass|count|score|price|cost|probability|percent|percentage|fraction|decimal|integer|add|subtract|multiply|divide)\b', new_question, re.IGNORECASE):
                    num_match_desc = re.search(r'-?\d+(?:\.\d+)?', check_answer_description)
                    if num_match_desc:
                        check_answer = num_match_desc.group(0)
                    else:
                        check_answer = ''
                else:
                    check_answer = ''
            if not check_answer_description.strip():
                check_answer_description = 'No explanation provided'
            return check_answer_description, check_answer
        elif len(lines) == 1:
            check_answer_description = lines[0]
            check_answer_description = re.sub(r'^check_answer_description[:\s-]*', '', check_answer_description, flags=re.IGNORECASE)
            if not check_answer_description.strip():
                check_answer_description = 'No explanation provided'
            return check_answer_description, ''
        else:
            return 'No explanation provided', ''
    tries = 0
    while tries < 3:
        try:
            response = openai.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                max_tokens=256,
                temperature=0.0,
            )
            content = response.choices[0].message.content.strip()
            check_answer_description, check_answer = parse_solution_output(content, new_question)
            # Only accept check_answer if it contains a number or a non-empty string
            if check_answer_description != 'No explanation provided' and check_answer:
                return check_answer_description, check_answer
            tries += 1
        except Exception as e:
            logging.error(f"Solution output failed: {e}")
            return 'No explanation provided', 'No answer provided'
    # If all attempts fail, set check_answer to 'No answer provided'
    if not check_answer_description or check_answer_description == 'No explanation provided':
        return 'No explanation provided', 'No answer provided'
    return check_answer_description, 'No answer provided'

def generate_multi_choice_options(new_question, new_answer, check_answer, image_path):
    """
    Generate multi-choice options (A, B, C, D) for the question and convert answers to option letters.
    """
    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        return None, None, None
    
    prompt = (
        "Given `new_question`, `new_answer`, and `check_answer`, your task is to generate four options (A, B, C, D) for the multi-choice question, and convert `new_answer` and `check_answer` to option letters. This requires you to generate 3 things:\n"
        "(1) new_question: <new question with options included>\n"
        "(2) new_answer: <A, B, C, or D>\n"
        "(3) check_answer: <A, B, C, or D>\n\n"
        "Constraints & Style rules:\n"
        "1. Generate exactly 4 options labeled A, B, C, D.\n"
        "2. `new_answer` and `check_answer` must be included in the options.\n"
        "3. If `new_answer` and `check_answer` are equal, only include one of them.\n"
        "4. All options are plausible but only one is correct.\n"
        "5. Updated `new_question` maintains the same format and style as the original`new_question`, but adding 4 options.\n"
        "6. Convert `new_answer` to the corresponding option letter.\n"
        "7. Convert `check_answer` to the corresponding option letter.\n\n"
        "Here are `new_question`, `new_answer`, and `check_answer`:\n"
        f"new_question: {{{new_question}}}\n"
        f"new_answer: {{{new_answer}}}\n"
        f"check_answer: {{{check_answer}}}\n\n"
        "Output exactly 3 lines, use the exact format below:\n"
        "new_question: <new question with options included>\n"
        "new_answer: <A, B, C, or D>\n"
        "check_answer: <A, B, C, or D>"
    )
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens=512,
            temperature=0.7,
        )
        content = response.choices[0].message.content.strip()
        
        # Parse the response
        new_question_with_options = None
        new_answer_letter = None
        check_answer_letter = None
        
        lines = content.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.lower().startswith("new_question:"):
                # Start with the question text
                new_question_with_options = line[len("new_question:"):].strip()
                # Look for options in subsequent lines
                i += 1
                while i < len(lines) and not lines[i].lower().startswith(("new_answer:", "check_answer:")):
                    current_line = lines[i].strip()
                    if current_line and (current_line.startswith(('A.', 'B.', 'C.', 'D.')) or 
                                       current_line.startswith(('A:', 'B:', 'C:', 'D:')) or
                                       current_line.startswith(('A)', 'B)', 'C)', 'D)')) or
                                       current_line.startswith(('A ', 'B ', 'C ', 'D '))):
                        new_question_with_options += "\n" + current_line
                    i += 1
                continue
            elif line.lower().startswith("new_answer:"):
                new_answer_letter = line[len("new_answer:"):].strip().upper()
            elif line.lower().startswith("check_answer:"):
                check_answer_letter = line[len("check_answer:"):].strip().upper()
            i += 1
        
        # Validate that we got valid option letters
        if new_answer_letter and new_answer_letter in ['A', 'B', 'C', 'D'] and check_answer_letter and check_answer_letter in ['A', 'B', 'C', 'D']:
            return new_question_with_options, new_answer_letter, check_answer_letter
        else:
            logging.error(f"Invalid option letters generated: new_answer={new_answer_letter}, check_answer={check_answer_letter}")
            return None, None, None
            
    except Exception as e:
        logging.error(f"Multi-choice option generation failed: {e}")
        return None, None, None

def solution_comparison(new_answer, check_answer):
    import re
    def is_number(s):
        try:
            float(s)
            return True
        except (ValueError, TypeError):
            return False
    
    def convert_fraction_to_decimal(fraction_str):
        try:
            if '/' in fraction_str:
                num, denom = map(float, fraction_str.split('/'))
                return num / denom
            return None
        except:
            return None
    
    # If one is number and the other is not, they are not equal
    if (is_number(new_answer) and not is_number(check_answer)) or (not is_number(new_answer) and is_number(check_answer)):
        return "FALSE"
    
    # Strict numeric comparison 
    if is_number(new_answer) and is_number(check_answer):
        try:
            return "TRUE" if abs(float(new_answer) - float(check_answer)) < 1e-6 else "FALSE"
        except Exception:
            pass  # fallback to LLM if conversion fails
    
    # Check for fraction vs decimal equivalence
    new_decimal = convert_fraction_to_decimal(new_answer)
    check_decimal = convert_fraction_to_decimal(check_answer)
    
    if new_decimal is not None and is_number(check_answer):
        try:
            return "TRUE" if abs(new_decimal - float(check_answer)) < 1e-6 else "FALSE"
        except:
            pass
    
    if check_decimal is not None and is_number(new_answer):
        try:
            return "TRUE" if abs(check_decimal - float(new_answer)) < 1e-6 else "FALSE"
        except:
            pass
    
    prompt = (
        "Compare new_answer and check_answer to verify the consistency of these two solutions. If they are the same, print answer_check as 'TRUE'; if not, print 'FALSE'.\n\n"
        "Constraints & Style rules:\n"
        "Treat answers as equal if they match under any of the following:\n"
        "- Case-insensitive match (e.g., \"Quarter\" == \"quarter\")\n"
        "- Leading-zero-insensitive numeric match (e.g., \"05\" == \"5\")\n"
        "- Fraction vs decimal equivalence (e.g., \"1/2\" == \"0.5\")\n"
        "- Numeric match where one includes **common units** like \"square feet\", \"inches\", \"cm\", etc. (e.g., \"15\" == \"15 square feet\")\n"
        "- Treat answers as equal if one is a substring of the other (case-insensitive, ignoring whitespace).\n\n"
        "Here is the new answer and check answer:\n"
        f"new_answer: {{{new_answer}}}\n"
        f"check_answer: {{{check_answer}}}\n\n"
        "Output exactly one line, no extra text:\n"
        "answer_check: <TRUE or FALSE>"
    )
    try:
        response = openai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=16,
            temperature=0.0,
        )
        content = response.choices[0].message.content.strip()
        answer_check = None
        for line in content.split('\n'):
            if line.lower().startswith("answer_check:"):
                answer_check = line[len("answer_check:"):].strip().upper()
        return answer_check if answer_check in ["TRUE", "FALSE"] else "FALSE"
    except Exception as e:
        logging.error(f"Solution comparison failed: {e}")
        return "FALSE"

def question_check(new_question):
    prompt = (
        "Your task is to determine whether the new question can be answered without viewing the image. If it can, the question lacks visual dependency and should be filtered from the dataset. If not, it remains valid.\n\n"
        "Constraints & Style rules:\n"
        "(1) Only questions answerable from text alone should produce an output.\n"
        "(2) The output must be a single line.\n\n"
        "Input:\n"
        f"new_question: {{{new_question}}}\n\n"
        "Output (solvable):\n"
        "question_only_answer: <ans>\n\n"
        "Output (not solvable):\n"
    )
    try:
        import re
        response = openai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=64,
            temperature=0.0,
        )
        content = response.choices[0].message.content.strip()
        if content.lower().startswith("question_only_answer:"):
            answer = content[len("question_only_answer:"):].strip()
            print(answer)
            return answer
        else:
            print('not solvable with text')
            return 'not solvable with text'
    except Exception as e:
        logging.error(f"Question check failed: {e}")
        return 'not solvable with text'




# Group questions by subfield, case-insensitive
groups = defaultdict(list)
for qid, q in math_targeted_data.items():
    # Use subfield as category, fallback to subject if subfield is not available
    category = q.get('metadata', {}).get('subfield', q.get('metadata', {}).get('subject', 'Unknown')).strip().lower()
    groups[category].append((qid, q))


# Process all categories
all_results = {}
category_stats = {}

for selected_category, questions in groups.items():
    # Select more than 25 questions to account for processing failures
    # If we have enough questions, select 35 to ensure we get 25 successful ones
    if len(questions) >= 35:
        sampled = random.sample(questions, 35)
    elif len(questions) >= 25:
        sampled = random.sample(questions, len(questions))  # Use all available
    else:
        sampled = questions  # Use all available if less than 25
    
    modified_dataset = []
    for qid, q in sampled:
        # Stop if we have 25 successful questions
        if len(modified_dataset) >= 25:
            break
            
        orig_question = q['question']
        orig_image = q['image']
        orig_answer = q.get('answer', '')
        metadata = q.get('metadata', {})
        # Add problem_version to metadata for multi-choice check
        metadata['problem_version'] = q.get('problem_version', '')
        metadata['question_type'] = q.get('question_type', '')
        # Handle image path for testmini dataset
        if orig_image.startswith('images_version_'):
            image_path = os.path.join('images', orig_image)
        elif orig_image.startswith('images/'):
            image_path = orig_image
        else:
            image_path = os.path.join('images', orig_image)

        # Step 1: New question generation (full pipeline)
        new_question_description, new_question, new_answer_description, new_answer = generate_new_question(orig_question, orig_answer, image_path)
        if not (new_question and new_answer):
            continue

        # Step 2a: Solution output
        check_answer_description, check_answer = solution_output(new_question, image_path)
        
        # Check if original question is multi-choice type
        question_type = metadata.get('question_type', '').lower()
        
        # Step 2b: Multi-choice option generation (only if original question is multi-choice)
        if question_type == 'multi-choice':
            print(f"Processing multi-choice question: {question_type}")
            new_question_with_options, new_answer_letter, check_answer_letter = generate_multi_choice_options(new_question, new_answer, check_answer, image_path)
            print(f"Multi-choice result: new_question_with_options={new_question_with_options is not None}, new_answer_letter={new_answer_letter}, check_answer_letter={check_answer_letter}")
            # Overwrite the original fields with multi-choice versions
            if new_question_with_options and new_answer_letter and check_answer_letter:
                print(f"Overwriting with multi-choice versions")
                new_question = new_question_with_options
                new_answer = new_answer_letter
                check_answer = check_answer_letter
            else:
                print(f"Multi-choice generation failed, keeping original")
        else:
            print(f"Not multi-choice: {question_type}")
        
        # Step 2c: Solution comparison with validation
        if new_answer and check_answer:
            # Additional validation for mathematical consistency
            if question_type == 'multi-choice':
                # For multiple choice, ensure both answers are valid letters
                valid_letters = ['A', 'B', 'C', 'D']
                if new_answer in valid_letters and check_answer in valid_letters:
                    answer_comparison = solution_comparison(new_answer, check_answer)
                else:
                    answer_comparison = "FALSE"  # Invalid multiple choice format
            else:
                # For numeric/free-form questions, proceed with comparison
                answer_comparison = solution_comparison(new_answer, check_answer)
        else:
            answer_comparison = "FALSE"  # Missing answers

        # Step 3: Question check (text-only)
        question_only_answer = question_check(new_question)

        entry = {
            'orig_question': orig_question,
            'orig_answer': orig_answer,
            'original_image': orig_image,
            'new_question_description': new_question_description,
            'new_question': new_question,
            'new_answer_description': new_answer_description,
            'new_answer': new_answer,
            'check_answer_description': check_answer_description,
            'check_answer': check_answer,
            'answer_comparison': answer_comparison,
            'question_only_answer': question_only_answer,
            'metadata': metadata,
        }
        modified_dataset.append(entry)
        logging.info(f"Processed question from {selected_category}: {qid}")

    all_results[selected_category] = modified_dataset
    # Print statistics for this category
    total = len(modified_dataset)
    if total > 0:
        num_true = sum(1 for entry in modified_dataset if entry.get('answer_comparison') == 'TRUE')
        num_visual = sum(1 for entry in modified_dataset if entry.get('question_only_answer') == 'not solvable with text')
        # Count multi-choice questions by checking if the new_answer is a letter (A, B, C, D)
        num_multi_choice = sum(1 for entry in modified_dataset if entry.get('new_answer', '') in ['A', 'B', 'C', 'D'])
        print(f"[{selected_category}] Proportion of questions with same answers: {num_true}/{total} = {num_true/total:.2%}")
        print(f"[{selected_category}] Proportion of visually-dependent questions: {num_visual}/{total} = {num_visual/total:.2%}")
        print(f"[{selected_category}] Multi-choice questions processed: {num_multi_choice}/{total} = {num_multi_choice/total:.2%}")
        category_stats[selected_category] = {
            'num_questions': total,
            'num_true': num_true,
            'proportion_true': num_true/total,
            'num_visual': num_visual,
            'proportion_visual': num_visual/total,
            'num_multi_choice': num_multi_choice,
            'proportion_multi_choice': num_multi_choice/total
        }
    else:
        print(f"[{selected_category}] No questions processed.")
        category_stats[selected_category] = {
            'num_questions': 0,
            'num_true': 0,
            'proportion_true': 0.0,
            'num_visual': 0,
            'proportion_visual': 0.0,
            'num_multi_choice': 0,
            'proportion_multi_choice': 0.0
        }

# Save output grouped by category
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

# Print statistics for each category
print("\nCategory Statistics:")
for category, entries in all_results.items():
    total = len(entries)
    num_true = sum(1 for entry in entries if entry.get('answer_comparison') == 'TRUE')
    num_visual = sum(1 for entry in entries if entry.get('question_only_answer') == 'not solvable with text')
    # Count multi-choice questions by checking if the new_answer is a letter (A, B, C, D)
    num_multi_choice = sum(1 for entry in entries if entry.get('new_answer', '') in ['A', 'B', 'C', 'D'])
    prop_true = num_true / total if total > 0 else 0
    prop_visual = num_visual / total if total > 0 else 0
    prop_multi_choice = num_multi_choice / total if total > 0 else 0
    print(f"Category: {category}")
    print(f"  Total questions: {total}")
    print(f"  # TRUE answer_comparison: {num_true} ({prop_true:.2%})")
    print(f"  # Visually-dependent questions: {num_visual} ({prop_visual:.2%})")
    print(f"  # Multi-choice questions: {num_multi_choice} ({prop_multi_choice:.2%})\n")

logging.info(f"Saved modified dataset grouped by category to {OUTPUT_JSON}")

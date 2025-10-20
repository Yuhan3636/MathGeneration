import os
import json
import random
import logging
import base64
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
random.seed(42)

try:
    from tqdm import tqdm
    use_tqdm = True
except ImportError:
    use_tqdm = False

import openai

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


# Get the directory of this script
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent

# Load the selected text-only questions
INPUT_JSON = SCRIPT_DIR / 'selected_text_only_questions.json'
OUTPUT_JSON = SCRIPT_DIR / 'text_only_gpt5_results.json'

with open(INPUT_JSON, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Group questions by subfield
groups = defaultdict(list)
for item in data:
    subfield = item.get('metadata', {}).get('subfield', 'Unknown').strip().lower()
    groups[subfield].append(item)

def solution_output_text_only(question):
    """
    Solve the text-only question using GPT-5
    """
    prompt = (
        "Your task is to solve the mathematical question. This requires you to provide two things:\n"
        "(1) rationales: a concise, step-by-step explanation of your reasoning.\n"
        "(2) final answer: the answer to the question.\n"
        "Constraints & Style rules:\n"
        "Accuracy - Derive the answer directly from the question text.\n"
        "Mathematical Precision - Ensure your calculations are mathematically correct.\n"
        "For multiple choice questions, provide only the letter (A, B, C, D) as the answer.\n"
        "For numeric questions, provide only the number or expression as the answer (units are optional).\n"
        "Do not leave either line blank.\n"
        "IMPORTANT: You must provide a specific answer. Do NOT say 'cannot be determined', 'insufficient information', or similar phrases.\n"
        "If the problem seems unclear, make reasonable assumptions and solve it based on standard mathematical interpretations.\n"
        "Double-check your mathematical reasoning to ensure accuracy.\n\n"
        f"Here is the question:\n"
        f"{question}\n\n"
        "Output exactly two lines, use the exact format below:\n"
        "rationales: <step-by-step reasoning>\n"
        "final answer: <your answer>"
    )
    
    import re
    def parse_solution_output(content, question=None):
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if len(lines) >= 2:
            rationales = lines[0]
            final_answer = lines[1]
            def strip_prefix(s, prefix):
                return re.sub(rf'^{prefix}[:\s-]*', '', s, flags=re.IGNORECASE)
            rationales = strip_prefix(rationales, 'rationales')
            final_answer = strip_prefix(final_answer, 'final answer')
            answer_candidate = final_answer.strip()
            # Check if answer is invalid (cannot be determined, insufficient, etc.)
            invalid_phrases = ['cannot be determined', 'insufficient', 'not enough', 'missing', 'no answer', 'rationales']
            is_invalid = any(phrase in answer_candidate.lower() for phrase in invalid_phrases)
            
            if answer_candidate and not is_invalid:
                final_answer = answer_candidate
            else:
                # Only extract a number if the question is numeric
                if question and re.search(r'\b(how many|what is the value|calculate|amount|number|sum|difference|total|product|quotient|average|mean|median|max|min|area|length|distance|perimeter|radius|diameter|volume|weight|mass|count|score|price|cost|probability|percent|percentage|fraction|decimal|integer|add|subtract|multiply|divide)\b', question, re.IGNORECASE):
                    num_match_desc = re.search(r'-?\d+(?:\.\d+)?', rationales)
                    if num_match_desc:
                        final_answer = num_match_desc.group(0)
                    else:
                        final_answer = ''
                else:
                    final_answer = ''
            if not rationales.strip():
                rationales = 'No explanation provided'
            return rationales, final_answer
        elif len(lines) == 1:
            rationales = lines[0]
            rationales = re.sub(r'^rationales[:\s-]*', '', rationales, flags=re.IGNORECASE)
            if not rationales.strip():
                rationales = 'No explanation provided'
            return rationales, ''
        else:
            return 'No explanation provided', ''
    
    tries = 0
    while tries < 3:
        try:
            response = openai.responses.create(
                model="gpt-5",
                input=[
                    {"role": "user", "content": [{"type": "input_text", "text": prompt}]}
                ],
                max_output_tokens=2048,
                reasoning={"effort": "low"},
            )
            content = response.output_text.strip()
            rationales, final_answer = parse_solution_output(content, question)
            # Only accept final_answer if it contains a value
            if rationales != 'No explanation provided' and final_answer:
                return rationales, final_answer
            tries += 1
        except Exception as e:
            logging.error(f"Solution output failed: {e}")
            return 'No explanation provided', 'No answer provided'
    # If all attempts fail
    if not rationales or rationales == 'No explanation provided':
        return 'No explanation provided', 'No answer provided'
    return rationales, 'No answer provided'

def solution_comparison(generated_answer, ground_truth_answer):
    """
    Compare the generated answer with the ground truth answer
    """
    import re
    # Normalize and perform strict checks first
    if generated_answer is not None and ground_truth_answer is not None:
        a = str(generated_answer).strip()
        b = str(ground_truth_answer).strip()
        # Multiple-choice strict check
        if len(a) == 1 and len(b) == 1 and a.upper() in ['A', 'B', 'C', 'D'] and b.upper() in ['A', 'B', 'C', 'D']:
            return "TRUE" if a.upper() == b.upper() else "FALSE"
        # Case-insensitive exact match
        if a.lower() == b.lower():
            return "TRUE"
        
        # Strip common units and LaTeX formatting before comparison
        def strip_units_and_latex(s):
            # Remove LaTeX math mode delimiters
            s = re.sub(r'\$+', '', s)
            # Remove LaTeX commands with units and their exponents: \mathrm{~cm}^{3} or \mathrm{m}^3
            s = re.sub(r'\\mathrm\{[^}]*\}\^?\{?[0-9]*\}?', '', s)
            s = re.sub(r'\\text\{[^}]*\}\^?\{?[0-9]*\}?', '', s)
            # Remove Volume = or similar prefixes
            s = re.sub(r'^(?:Volume|Area|Length|Answer|Result)\s*=\s*', '', s, flags=re.IGNORECASE)
            # Remove common standalone units with optional exponents
            s = re.sub(r'\s*(?:litres?|meters?|metre|cm|km|mm|ft|feet|in|inches|yd|yards?|mi|miles?|km/h|mph|m/h|grams?|kg|kilograms?|lb|pounds?|°|degrees?|°C|°F|seconds?|minutes?|hours?|hrs?|L|mL|milliliters?|gal|gallons?)(?:\^?\{?[0-9]+\}?)?\s*', '', s, flags=re.IGNORECASE)
            # Remove standalone exponent markers that might be left (like ^{3})
            s = re.sub(r'\^\{?[0-9]+\}?\s*$', '', s)
            # Remove LaTeX formatting characters
            s = s.replace('\\', '')
            s = re.sub(r'[\{\}]', '', s)
            # Normalize whitespace
            s = re.sub(r'\s+', '', s)
            # Remove tilde (~) often used in LaTeX
            s = s.replace('~', '')
            # Normalize pi symbol
            s = s.replace('π', 'pi')
            return s.strip()
        
        # Try comparing after stripping units and formatting
        a_stripped = strip_units_and_latex(a)
        b_stripped = strip_units_and_latex(b)
        if a_stripped and b_stripped and a_stripped.lower() == b_stripped.lower():
            return "TRUE"
    
    def normalize_for_comparison(s):
        """Further normalize mathematical expressions for comparison"""
        if not s:
            return s
        s = str(s)
        # Strip variable assignments like "y = ", "x = ", "answer = ", etc.
        s = re.sub(r'^[a-zA-Z]\s*=\s*', '', s)
        # Normalize spaces around operators
        s = re.sub(r'\s*([+\-*/=<>(),])\s*', r'\1', s)
        # Normalize parentheses around coordinates
        s = re.sub(r'^\s*\(?\s*', '', s)
        s = re.sub(r'\s*\)?\s*$', '', s)
        # Normalize degree symbols
        s = s.replace('°', '')
        # Normalize sqrt notations
        s = s.replace('√', 'sqrt')
        return s.strip().lower()
    
    # Additional flexible comparison after stripping
    gen_norm = normalize_for_comparison(generated_answer)
    gt_norm = normalize_for_comparison(ground_truth_answer)
    if gen_norm and gt_norm and gen_norm == gt_norm:
        return "TRUE"
    
    # Check if one is substring of the other (after normalization)
    if gen_norm and gt_norm:
        if gen_norm in gt_norm or gt_norm in gen_norm:
            return "TRUE"
    
    def is_number(s):
        try:
            float(s)
            return True
        except (ValueError, TypeError):
            return False
    
    def convert_fraction_to_decimal(fraction_str):
        try:
            # Normalize first (remove variable assignments)
            s = re.sub(r'^[a-zA-Z]\s*=\s*', '', str(fraction_str).strip())
            if '/' in s:
                # Handle simple fractions
                parts = s.split('/')
                if len(parts) == 2:
                    num, denom = map(float, parts)
                    return num / denom
            return None
        except:
            return None
    
    def extract_numeric_value(s):
        """Extract numeric value from string, handling variable assignments"""
        if not s:
            return None
        s = str(s).strip()
        # Remove variable assignments
        s = re.sub(r'^[a-zA-Z]\s*=\s*', '', s)
        # Try to convert to float
        try:
            return float(s)
        except:
            # Try fraction
            return convert_fraction_to_decimal(s)
    
    # Try to extract numeric values and compare (handles fractions, decimals, variable assignments)
    gen_value = extract_numeric_value(generated_answer)
    gt_value = extract_numeric_value(ground_truth_answer)
    
    if gen_value is not None and gt_value is not None:
        try:
            return "TRUE" if abs(gen_value - gt_value) < 1e-4 else "FALSE"
        except:
            pass
    
    # Fallback: Strict numeric comparison for simple cases
    if is_number(generated_answer) and is_number(ground_truth_answer):
        try:
            return "TRUE" if abs(float(generated_answer) - float(ground_truth_answer)) < 1e-4 else "FALSE"
        except Exception:
            pass  # fallback to LLM if conversion fails
    
    prompt = (
        "Compare generated_answer and ground_truth_answer to determine if they are MATHEMATICALLY EQUIVALENT. If they are mathematically equal, print answer_check as 'TRUE'; if not, print 'FALSE'.\n\n"
        "IMPORTANT: Be VERY FLEXIBLE with comparison. Treat answers as equal if they are mathematically the same, even if formatted differently.\n\n"
        "Treat answers as equal under ANY of the following:\n"
        "- Mathematically equivalent expressions (e.g., \"x^2-1\" == \"(x-1)(x+1)\", \"2x+3x\" == \"5x\")\n"
        "- Variable assignments vs values (e.g., \"y = 77/10\" == \"7.7\", \"x = 5\" == \"5\")\n"
        "- Different orderings (e.g., \"x+y\" == \"y+x\", \"AB\" == \"BA\" for multiplication)\n"
        "- Case-insensitive match (e.g., \"Quarter\" == \"quarter\")\n"
        "- Leading-zero-insensitive numeric match (e.g., \"05\" == \"5\")\n"
        "- Fraction vs decimal equivalence (e.g., \"1/2\" == \"0.5\", \"3/4\" == \"0.75\", \"77/10\" == \"7.7\")\n"
        "- Numeric values match regardless of units or notation (e.g., \"298°T\" == \"298°\" == \"298\", \"15 square feet\" == \"15\", \"5cm\" == \"5\")\n"
        "- Coordinate pairs match regardless of formatting (e.g., \"(135°, -1)\" == \"135, -1\" == \"(135,-1)\")\n"
        "- Different square root notations (e.g., \"sqrt(2)\" == \"√2\")\n"
        "- Equivalent trigonometric values (e.g., \"sin(90°)\" == \"1\")\n"
        "- Polynomial expansions (e.g., \"(x+2)^2\" == \"x^2+4x+4\")\n"
        "- Simplified vs unsimplified forms (e.g., \"6/8\" == \"3/4\", \"0.50\" == \"0.5\")\n"
        "- Different but mathematically identical representations\n"
        "- Treat as equal if one answer is contained within the other (ignoring extra text/formatting)\n\n"
        "Here is the generated answer and ground truth answer:\n"
        f"generated_answer: {{{generated_answer}}}\n"
        f"ground_truth_answer: {{{ground_truth_answer}}}\n\n"
        "Output exactly one line, no extra text:\n"
        "answer_check: <TRUE or FALSE>"
    )
    try:
        response = openai.responses.create(
            model="gpt-5",
            input=[
                {"role": "user", "content": [{"type": "input_text", "text": prompt}]}
            ],
            max_output_tokens=128,
            reasoning={"effort": "medium"},
        )
        content = response.output_text.strip()
        answer_check = None
        for line in content.split('\n'):
            if line.lower().startswith("answer_check:"):
                answer_check = line[len("answer_check:"):].strip().upper()
        return answer_check if answer_check in ["TRUE", "FALSE"] else "FALSE"
    except Exception as e:
        logging.error(f"Solution comparison failed: {e}")
        return "FALSE"

# Process all categories
all_results = {}
category_stats = {}

for category, questions in groups.items():
    print(f"\n=== Processing category: {category} ===")
    print(f"Number of questions available: {len(questions)}")

    modified_dataset = []
    for i, question_data in enumerate(questions[:15]):
        try:
            question = question_data['question']
            ground_truth_answer = question_data.get('answer', '')
            metadata = question_data.get('metadata', {})
            
            print(f"Processing question {i+1}/{len(questions)} from {category}")
            
            # Solve the question using GPT-5
            rationales, generated_answer = solution_output_text_only(question)
            
            if not generated_answer or generated_answer == 'No answer provided':
                logging.info(f"Skipping question {i+1} - failed to generate answer")
                continue
            
            # Compare answers
            answer_comparison = solution_comparison(generated_answer, ground_truth_answer)
            
            entry = {
                'question': question,
                'ground_truth_answer': ground_truth_answer,
                'rationales': rationales,
                'generated_answer': generated_answer,
                'answer_comparison': answer_comparison,
                'metadata': metadata,
            }
            modified_dataset.append(entry)
            logging.info(f"Processed question {i+1} from {category}")
            print(f"Successfully processed question {i+1} from {category}. Answer comparison: {answer_comparison}")
            
        except Exception as e:
            logging.error(f"Error processing question {i+1} from {category}: {e}")
            print(f"Error processing question {i+1} from {category}: {e}")
            continue

    # Store results for this category
    all_results[category] = modified_dataset
    
    # Print statistics for this category
    total = len(modified_dataset)
    if total > 0:
        num_correct = sum(1 for entry in modified_dataset if entry.get('answer_comparison') == 'TRUE')
        print(f"\n=== {category.upper()} SUMMARY STATISTICS ===")
        print(f"Total questions processed: {total}")
        print(f"Correct answers: {num_correct}/{total} = {num_correct/total:.2%}")
        category_stats[category] = {
            'num_questions': total,
            'num_correct': num_correct,
            'accuracy': num_correct/total
        }
    else:
        print(f"\n=== {category.upper()} SUMMARY STATISTICS ===")
        print(f"No questions processed for {category}")
        category_stats[category] = {
            'num_questions': 0,
            'num_correct': 0,
            'accuracy': 0.0
        }

# Save output grouped by category
with open(str(OUTPUT_JSON), 'w', encoding='utf-8') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

# Print overall statistics
print("\n" + "="*50)
print("OVERALL STATISTICS")
print("="*50)
total_questions = sum(stats['num_questions'] for stats in category_stats.values())
total_correct = sum(stats['num_correct'] for stats in category_stats.values())
overall_accuracy = total_correct / total_questions if total_questions > 0 else 0

print(f"Total questions processed: {total_questions}")
print(f"Total correct answers: {total_correct}")
print(f"Overall accuracy: {overall_accuracy:.2%}")

print("\nCategory Statistics:")
for category, stats in category_stats.items():
    print(f"Category: {category}")
    print(f"  Questions: {stats['num_questions']}")
    print(f"  Correct: {stats['num_correct']}")
    print(f"  Accuracy: {stats['accuracy']:.2%}\n")

logging.info(f"Saved results to {OUTPUT_JSON}")

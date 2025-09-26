Folder 9/14

1. Comparison.py: 
   new question:
   - new question generation: GPT-5
   - new answer:GPT-5, GPT-4.1, O3, GPT-4o
   - new answer consistency: compare with GPT-5(ans1)
   - visual dependency check
   original question:
   - orginal answer: GPT-5, GPT-4.1, O3, GPT-4o
   - orginal answer accuracy: compare with original ground truth
 
 2. Comparison.js
    results generated from Comparison.py including the answers of new and 
    original questions from all models except gemini
 
 3. Gemini.py: 
    - Load comparison.json 
    new question:
      a. original answer: gemini 2.5 pro
      b. original answer accuracy: compare with orignal ground truth
    original question:
      a. new answer: gemini 2.5 pro
      b. new answer consistency: compare with GPT-5

4. Gemini.json:
   results generated from Gemini.py including the answers of new and 
   original questions using gemini

5. Evaluate copy.py (9/21):
   a. train a classifer based on labels
   b. consistency: compare ans1(gpt-5) with ans2(gpt5), gemini, gpt4.1, o3, gpt4o
   c. correctness: compare actual with ans1(gpt-5), ans2(gpt5), gemini, gpt4.1, o3, gpt4o
   d. Ensemble: use majority vote
   e. F1 score of consistency and correctness: 
      - precision, recall, F1 for incorrect class
      - precision, recall, F1 for correct class
   f. Visually dependent questions (LLM vs human)
   g. F1 score of visually dependent questions
      - minority class
      - majority class
   h. Different from orignal questions (LLM vs human)
   i. F1 score of different from orignal questions 
      - minority class
      - majority class
   j. Question quality (LLM vs human)
   
 6. Quality.py:
    - check if new question is different with orig question 
    - question quality score
 
 7. Quality.json:
    results generated from Quality.py

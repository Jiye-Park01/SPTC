from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import csv
import re

# 모델 및 토크나이저 로딩
phi2 = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(phi2)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    phi2, device_map="auto", torch_dtype=torch.float16
)
model.eval()

# GSM8K 데이터셋 로드
dataset = load_dataset("openai/gsm8k", "main")
small_dataset = dataset["train"].shuffle(seed=42).select(range(100))
questions = [item["question"] for item in small_dataset]
answers = [item["answer"] for item in small_dataset]

# 정답 숫자 추출 함수
def extract_final_answer(text):
    match = re.search(r"####\s*(\d+)", text)
    return match.group(1) if match else None

# 프롬프트 함수들
def no_prompt(question):
    return f"Q: {question}\nAnswer:"

def few_shot_cot_prompt(question):
    few_shot = (
        "Q: Jason had 5 apples. He bought 3 more. How many apples does he have now?\n"
        "Let's think step-by-step.\n"
        "Jason started with 5 apples.\n"
        "He bought 3 more apples.\n"
        "5 + 3 = 8\n"
        "#### 8\n\n"
    )
    return few_shot + f"Q: {question}\nLet's think step-by-step.\n"

def sptc_prompt(question):
    few_shot = (
        "Q: Sarah read 12 pages on Monday and 15 pages on Tuesday. How many pages did she read in total?\n"
        "Step-by-step reasoning:\n"
        "(1) Sarah read 12 pages on Monday.\n"
        "(2) She read 15 pages on Tuesday.\n"
        "(3) Adding these gives: 12 + 15 = 27.\n"
        "Evaluation:\n"
        "- The addition is straightforward.\n"
        "- No subtraction or multiplication involved.\n"
        "Final Answer: 27\n"
        "#### 27\n\n"
    )
    return few_shot + f"Q: {question}\nStep-by-step reasoning:\n"

# 응답 생성 함수
def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            synced_gpus=False
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.strip()

# 평가 함수 (정답 숫자 직접 비교)
def final_answer_evaluate(questions, answers, prompt_func, csv_path=None):
    correct = 0
    logs = []

    for i, (q, a) in enumerate(zip(questions, answers)):
        prompt = prompt_func(q)
        prediction = generate_answer(prompt)

        pred_ans = extract_final_answer(prediction)
        true_ans = extract_final_answer(a)

        is_correct = pred_ans == true_ans
        if is_correct:
            correct += 1

        logs.append([i, q, true_ans, pred_ans, prediction])

    if csv_path:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "question", "ground_truth", "predicted_answer", "full_prediction"])
            writer.writerows(logs)

    acc = correct / len(questions) * 100
    return acc

# 실험 실행
print("=== GSM8K 실험 시작 ===")

start_time = time.time()
print("▶ No-Prompt 실험 중...")
no_cot_accuracy = final_answer_evaluate(
    questions, answers, no_prompt,
    csv_path="gsm8k_no_prompt_log.csv"
)
print(f"No-Prompt Accuracy: {no_cot_accuracy:.2f}%")

start_time = time.time()
print("▶ Few-shot CoT 실험 중...")
few_cot_accuracy = final_answer_evaluate(
    questions, answers, few_shot_cot_prompt,
    csv_path="gsm8k_few_shot_cot_log.csv"
)
print(f"Few-shot CoT Accuracy: {few_cot_accuracy:.2f}%")

start_time = time.time()
print("▶ SPTC 실험 중...")
sptc_accuracy = final_answer_evaluate(
    questions, answers, sptc_prompt,
    csv_path="gsm8k_sptc_log.csv"
)
print(f"SPTC Accuracy: {sptc_accuracy:.2f}%")

print("=== GSM8K 실험 완료 ===")

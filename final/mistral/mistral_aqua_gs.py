from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import re

# 모델 및 토크나이저 로딩
qwen = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(qwen, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.unk_token_id

model = AutoModelForCausalLM.from_pretrained(
    qwen,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
model.eval()

# GSM8K 데이터셋 로드
dataset = load_dataset("openai/gsm8k", "main")
small_dataset = dataset["train"].shuffle(seed=42).select(range(100))
questions = [item["question"] for item in small_dataset]
answers = [item["answer"] for item in small_dataset]

# 정답 추출 함수
def extract_final_answer(text):
    match = re.search(r"####\s*(\d+)", text)
    return match.group(1) if match else None

# 프롬프트 함수들
def no_prompt(question):
    return f"Q: {question}\nAnswer:"

def zero_shot_cot_prompt(question):
    return f"Q: {question}\nLet's think step-by-step.\n"

def few_shot_cot_prompt(question):
    few_shot = (
        "Q: Jason had 5 apples. He bought 3 more. How many apples does he have now?\n"
        "Let's think step-by-step.\n"
        "He started with 5 apples.\n"
        "He bought 3 more.\n"
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
        "(3) Adding: 12 + 15 = 27.\n"
        "Evaluation: Addition is straightforward.\n"
        "Final Answer: 27\n"
        "#### 27\n\n"
    )
    return few_shot + f"Q: {question}\nStep-by-step reasoning:\n"

# 응답 생성
def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.strip()

# 정답 비교 평가 함수
def final_answer_evaluate(questions, answers, prompt_func):
    correct = 0
    for i, (q, a) in enumerate(zip(questions, answers)):
        prompt = prompt_func(q)
        prediction = generate_answer(prompt)
        pred_ans = extract_final_answer(prediction)
        true_ans = extract_final_answer(a)
        if pred_ans == true_ans:
            correct += 1
    acc = correct / len(questions) * 100
    return acc

# 실험 실행
print("=== Qwen 실험 시작 ===")

start_time = time.time()
print("▶ No-Prompt 실험 중...")
acc = final_answer_evaluate(questions, answers, no_prompt)
print(f"No-Prompt Accuracy: {acc:.2f}%")
print(f"[No-Prompt] Execution time: {time.time() - start_time:.2f} sec")

start_time = time.time()
print("▶ CoT (Zero-shot) 실험 중...")
acc = final_answer_evaluate(questions, answers, zero_shot_cot_prompt)
print(f"Zero-shot CoT Accuracy: {acc:.2f}%")
print(f"[Zero-shot CoT] Execution time: {time.time() - start_time:.2f} sec")

start_time = time.time()
print("▶ CoT (Few-shot) 실험 중...")
acc = final_answer_evaluate(questions, answers, few_shot_cot_prompt)
print(f"Few-shot CoT Accuracy: {acc:.2f}%")
print(f"[Few-shot CoT] Execution time: {time.time() - start_time:.2f} sec")

start_time = time.time()
print("▶ SPTC 실험 중...")
acc = final_answer_evaluate(questions, answers, sptc_prompt)
print(f"SPTC Accuracy: {acc:.2f}%")
print(f"[SPTC] Execution time: {time.time() - start_time:.2f} sec")

print("=== Qwen 실험 완료 ===")

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import csv
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# 모델 및 토크나이저 로딩
phi2 = "HuggingFaceTB/SmolLM-1.7B"
tokenizer = AutoTokenizer.from_pretrained(phi2)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    phi2, device_map="auto", torch_dtype=torch.float16
)
model.eval()

# SentenceBERT 임베딩 모델
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# CommonsenseQA 데이터셋 로드
dataset = load_dataset("commonsense_qa")
small_dataset = dataset["train"].shuffle(seed=42).select(range(100))
questions = [item["question"] for item in small_dataset]
answers = [item["answerKey"] for item in small_dataset]
choices = [item["choices"] for item in small_dataset]

# 프롬프트 함수
def sptc_prompt(question, choices):
    choices_text = "\n".join([f"{label}: {text}" for label, text in zip(choices["label"], choices["text"])])
    
    few_shot = (
        "Q: Why do people wear sunglasses?\n"
        "Choices:\n"
        "A: To protect eyes from sunlight\n"
        "B: To reduce glare\n"
        "C: To look fashionable\n"
        "D: To hide tired eyes\n"
        "E: To protect from wind and dust\n"
        "Let's think through the choices carefully:\n"
        "(1) A is essential for eye protection.\n"
        "(2) B is useful for driving.\n"
        "(3) C is stylish but not critical.\n"
        "(4) D and E are minor reasons.\n"
        "Final answer: \\boxed{A}\n\n"
    )

    instruction = (
        "For each question, evaluate the 5 choices provided.\n"
        "You must choose the final answer using ONLY the format \\boxed{A}, \\boxed{B}, ..., \\boxed{E}.\n"
        "DO NOT write the full sentence, only output one boxed letter.\n"
    )

    return (
        instruction
        + few_shot
        + f"Q: {question}\nChoices:\n{choices_text}\n"
        "Let's think through the choices carefully.\n"
        "Final answer: \\boxed{...}\nA:"
    )

# 응답 생성 함수
def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=True,
            temperature=0.3,
            pad_token_id=tokenizer.pad_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 의미 유사도 평가
def is_semantically_correct(predicted, ground_truth, threshold=0.75):
    embeddings = embedder.encode([predicted.lower(), ground_truth.lower()])
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return sim >= threshold, sim

# 평가 함수
def evaluate(questions, answers, prompt_func, choices, csv_path=None):
    correct = 0
    logs = []

    for i, (q, a) in enumerate(zip(questions, answers)):
        prompt = prompt_func(q, choices[i])
        output = generate_answer(prompt)

        match = re.search(r"\\boxed\{([A-E])\}", output)
        if not match:
            print(f"[Warning] {i}번 문제: 형식 불일치 → {output}")
            continue

        predicted_label = match.group(1).strip().upper()
        labels = choices[i]["label"]
        texts = choices[i]["text"]

        try:
            pred_idx = labels.index(predicted_label)
            predicted_text = texts[pred_idx]
            true_idx = labels.index(a)
            ground_truth = texts[true_idx]
        except Exception as e:
            print(f"[Warning] {i}번 인덱싱 오류 → {e}")
            continue

        is_correct, sim_score = is_semantically_correct(predicted_text, ground_truth)

        if is_correct:
            correct += 1
            result = "Correct"
            print(f"[{i}] ✅ 정답 | 예측: {predicted_text} / 정답: {ground_truth} (sim: {sim_score:.2f})")
        else:
            result = "Wrong"
            print(f"[{i}] ❌ 오답 | 예측: {predicted_text} / 정답: {ground_truth} (sim: {sim_score:.2f})")

        logs.append([i, q, predicted_label, predicted_text, a, ground_truth, f"{sim_score:.4f}", result])

    if csv_path:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "index", "question", "pred_label", "predicted_text",
                "answer_label", "ground_truth", "similarity", "result"
            ])
            writer.writerows(logs)

    return correct / len(logs) * 100 if logs else 0.0

# 실행
print("=== phi2 SPTC 평가 시작 ===")
start = time.time()

accuracy = evaluate(
    questions, answers, sptc_prompt, choices,
    csv_path="sptc_result.csv"
)

print(f"SPTC Accuracy: {accuracy:.2f}%")
print(f"총 소요 시간: {time.time() - start:.2f}초")

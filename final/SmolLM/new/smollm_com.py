from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import csv
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
choices_list = [item["choices"] for item in small_dataset]

# 프롬프트 함수들
def no_prompt(question, choices):
    choices_text = "\n".join([f"{label}: {text}" for label, text in zip(choices["label"], choices["text"])])
    
    few_shot = (
        "Q: Why do people wear sunglasses?\n"
        "Choices:\n"
        "A: To protect eyes from sunlight\n"
        "B: To reduce glare\n"
        "C: To look fashionable\n"
        "D: To hide tired eyes\n"
        "E: To protect from wind and dust\n"
        "Final answer: A\n\n"
        "Q: Where do you send letters?\n"
        "Choices:\n"
        "A: Restaurant\n"
        "B: Supermarket\n"
        "C: Park\n"
        "D: Post office\n"
        "E: School\n"
        "Final answer: D\n\n"
    )
    
    return (
        few_shot
        + f"Q: {question}\nChoices:\n{choices_text}\n"
        "Choose only one of the choices (A, B, C, D, or E) and write your final answer in the format: \"Final answer: X\""
    )


def zero_shot_cot_prompt(question, choices):
    choices_text = "\n".join([f"{label}: {text}" for label, text in zip(choices["label"], choices["text"])])
    
    few_shot = (
        "Q: Why do people wear sunglasses?\n"
        "Choices:\n"
        "A: To protect eyes from sunlight\n"
        "B: To reduce glare\n"
        "C: To look fashionable\n"
        "D: To hide tired eyes\n"
        "E: To protect from wind and dust\n"
        "Let's think step-by-step.\n"
        "Sunlight can hurt our eyes, so we wear sunglasses.\n"
        "Final answer: A\n\n"
        "Q: Where do you send letters?\n"
        "Choices:\n"
        "A: Restaurant\n"
        "B: Supermarket\n"
        "C: Park\n"
        "D: Post office\n"
        "E: School\n"
        "Let's think step-by-step.\n"
        "We send letters at the post office.\n"
        "Final answer: D\n\n"
    )

    return (
        few_shot
        + f"Q: {question}\nChoices:\n{choices_text}\n"
        "Let's think step-by-step.\n"
        "Choose only one of the choices (A, B, C, D, or E) and write your final answer in the format: \"Final answer: X\":"
    )


def sptc_prompt(question, choices):
    choices_text = "\n".join([f"{label}: {text}" for label, text in zip(choices["label"], choices["text"])])
    
    # Few-shot 예시
    few_shot = (
        "Q: Why do people wear sunglasses?\n"
        "Choices:\n"
        "A: To protect eyes from sunlight\n"
        "B: To reduce glare\n"
        "C: To look fashionable\n"
        "D: To hide tired eyes\n"
        "E: To protect from wind and dust\n"
        "Let's think through different possibilities carefully:\n"
        "(1) To protect eyes from sunlight.\n"
        "(2) To reduce glare while driving.\n"
        "(3) To look fashionable.\n"
        "(4) To hide tired eyes.\n"
        "(5) To protect from wind and dust.\n"
        "Evaluation:\n"
        "- Protecting from sunlight is critical.\n"
        "- Reducing glare is very useful.\n"
        "- Fashion is secondary.\n"
        "- Hiding tired eyes is less important.\n"
        "Final Answer: To protect eyes from sunlight\n\n"
    )

    # 시스템 프롬프트
    system_prompt = (
        "For each question, list 5 possible reasons or explanations.\n"
        "Evaluate which are most reasonable.\n"
        "Pick the final best answer based on careful evaluation.\n"
    )

    # 실제 질문 + 선택지 포함
    return (
        system_prompt
        + "\n\n"
        + few_shot
        + f"Q: {question}\nChoices:\n{choices_text}\nLet's think through different possibilities carefully and Choose only one of the choices (A, B, C, D, or E) and write your final answer:\nA:"
    )



# 응답 생성 함수
def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            synced_gpus=False
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "A:" in decoded:
        return decoded.split("A:")[-1].strip()
    else:
        return decoded.strip()

# 평가 및 CSV 저장 함수
import re

def extract_answer_after_final_answer(text):
    """
    모델 응답에서 'Final answer:' 다음에 나오는 텍스트를 추출
    """
    match = re.search(r"A:\s*(.*)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        return text.strip()

def embedding_similarity_evaluate(questions, answers, prompt_func, choices_list, csv_path=None):
    correct = 0
    logs = []

    for i, (q, a) in enumerate(zip(questions, answers)):
        choices = choices_list[i]
        prompt = prompt_func(q, choices)
        prediction = generate_answer(prompt)

        try:
            labels = [label.lower() for label in choices["label"]]
            texts = choices["text"]
            idx = labels.index(a.lower())
            ground_truth = texts[idx]
        except Exception as e:
            print(f"[Warning] {i}번째 문제에서 정답 추출 실패 → {e}")
            continue

        predicted_text = extract_answer_after_final_answer(prediction)

        if predicted_text.lower() == ground_truth.lower():
            correct += 1
            result = "Correct"
            print(f"[{i}] 정답 | 정답: {ground_truth} / 예측: {predicted_text}")
        else:
            result = "Wrong"
            print(f"[{i}] 틀림 | 정답: {ground_truth} / 예측: {predicted_text}")

        logs.append([i, q, ground_truth, predicted_text, result])

    if csv_path:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "question", "ground_truth", "prediction", "accuracy_score"])
            writer.writerows(logs)

    acc = correct / len(questions) * 100
    return acc



# 실험 실행
print("===phi2 실험 시작 ===")

# start_time = time.time()
# print("▶ No-Prompt 실험 중...")
# no_cot_accuracy = embedding_similarity_evaluate(
#     questions, answers, no_prompt, choices_list,
#     csv_path="no_prompt_log.csv"
# )
# print(f"No-Prompt Accuracy: {no_cot_accuracy:.2f}%")
# print(f"[No-Prompt] Execution time: {time.time() - start_time:.6f} seconds")

# start_time = time.time()
# print("▶ CoT (Zero-shot, Step-by-Step) 실험 중...")
# zero_cot_accuracy = embedding_similarity_evaluate(
#     questions, answers, zero_shot_cot_prompt, choices_list,
#     csv_path="zero_shot_cot_log.csv"
# )
# print(f"CoT (Zero-shot) Accuracy: {zero_cot_accuracy:.2f}%")
# print(f"[Zero-CoT] Execution time: {time.time() - start_time:.6f} seconds")

start_time = time.time()
print("▶ SPTC (structured reasoning + evaluation) 실험 중...")
sptc_accuracy = embedding_similarity_evaluate(
    questions, answers, sptc_prompt, choices_list,
    csv_path="sptc_log.csv"
)
print(f"SPTC Accuracy: {sptc_accuracy:.2f}%")
print(f"[SPTC] Execution time: {time.time() - start_time:.6f} seconds")

print("=== phi2 실험 완료 ===")

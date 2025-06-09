from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import csv
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# 모델 및 토크나이저 로딩
phi2 = "meta-llama/Llama-2-7b-chat-hf"
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

def format_choices(choices):
    return "\n".join([f"{label}: {text}" for label, text in zip(choices["label"], choices["text"])])

def no_prompt(question, choices):
    choices_text = format_choices(choices)
    few_shot = (
        "Q: Why do people wear sunglasses?\n"
        "Choices:\n"
        "A: To protect eyes from sunlight\n"
        "B: To reduce glare\n"
        "C: To look fashionable\n"
        "D: To hide tired eyes\n"
        "E: To protect from wind and dust\n"
        "Final Answer: To protect eyes from sunlight and reduce glare.\n\n"
    )
    return  few_shot + f"Q: {question}\nChoices:\n{choices_text}\nA:"

# def zero_shot_cot_prompt(question, choices):
#     choices_text = format_choices(choices)
#     return f"Q: {question}\nChoices:\n{choices_text}\nLet's think step-by-step.\nA:"

def few_shot_cot_prompt(question, choices):
    choices_text = format_choices(choices)
    few_shot = (
        "Q: Why do people wear sunglasses?\n"
        "Choices:\n"
        "A: To protect eyes from sunlight\n"
        "B: To reduce glare\n"
        "C: To look fashionable\n"
        "D: To hide tired eyes\n"
        "E: To protect from wind and dust\n"
        "Let's think step-by-step.\n"
        "First, sunglasses protect eyes from sunlight.\n"
        "Second, they reduce glare.\n"
        "Third, they are a fashion accessory.\n"
        "Final Answer: To protect eyes from sunlight and reduce glare.\n\n"
    )
    return few_shot + f"Q: {question}\nChoices:\n{choices_text}\nLet's think step-by-step.\nA:"


def sptc_prompt(question, choices):
    choices_text = format_choices(choices)
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
        "Final Answer: To protect eyes from sunlight and reduce glare.\n\n"
    )
    system_prompt = (
        "For each question, list 5 possible reasons or explanations.\n"
        "Evaluate which are most reasonable.\n"
        "Pick the final best answer based on careful evaluation.\n"
    )
    return system_prompt + "\n\n" + few_shot + f"Q: {question}\nChoices:\n{choices_text}\nLet's think through different possibilities carefully:\nA:"

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
    if "A:" in decoded:
        return decoded.split("A:")[-1].strip()
    else:
        return decoded.strip()

# 평가 및 CSV 저장 함수
def embedding_similarity_evaluate(questions, answers, prompt_func, choices_list, threshold=0.3, csv_path=None):
    correct = 0
    logs = []

    for i, (q, a) in enumerate(zip(questions, answers)):
        choices = choices_list[i]  
        prompt = prompt_func(q, choices)
        prediction = generate_answer(prompt)

        try:
            labels = [label.lower() for label in choices_list[i]["label"]]
            texts = choices_list[i]["text"]
            idx = labels.index(a.lower())
            ground_truth = texts[idx]
        except Exception as e:
            print(f"[Warning] {i}번째 문제에서 정답 추출 실패 → {e}")
            continue

        embeddings = embedder.encode([ground_truth.lower(), prediction.lower()])
        sim_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

        if sim_score >= threshold:
            correct += 1

        logs.append([i, q, ground_truth, prediction, sim_score])

    if csv_path:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "question", "ground_truth", "prediction", "similarity_score"])
            writer.writerows(logs)

    acc = correct / len(questions) * 100
    return acc

# 실험 실행
print("===Llama 실험 시작 ===")

start_time = time.time()
print("▶ No-Prompt 실험 중...")
no_cot_accuracy = embedding_similarity_evaluate(
    questions, answers, no_prompt, choices_list,
    csv_path="/home/jhrew/jiye/SPTC/final/mistral/no_prompt_log_open.csv"
)
print(f"No-Prompt Accuracy: {no_cot_accuracy:.2f}%")
print(f"[No-Prompt] Execution time: {time.time() - start_time:.6f} seconds")

start_time = time.time()
print("▶ CoT (Zero-shot, Step-by-Step) 실험 중...")
few_cot_accuracy = embedding_similarity_evaluate(
    questions, answers, few_shot_cot_prompt, choices_list,
    csv_path="/home/jhrew/jiye/SPTC/final/mistral/zero_shot_cot_log_open.csv"
)
print(f"CoT (Zero-shot) Accuracy: {few_cot_accuracy:.2f}%")
print(f"[Zero-CoT] Execution time: {time.time() - start_time:.6f} seconds")

start_time = time.time()
print("▶ SPTC (structured reasoning + evaluation) 실험 중...")
sptc_accuracy = embedding_similarity_evaluate(
    questions, answers, sptc_prompt, choices_list,
    csv_path="/home/jhrew/jiye/SPTC/final/mistral/sptc_log_open.csv"
)
print(f"SPTC Accuracy: {sptc_accuracy:.2f}%")
print(f"[SPTC] Execution time: {time.time() - start_time:.6f} seconds")

print("=== Llama 실험 완료 ===")

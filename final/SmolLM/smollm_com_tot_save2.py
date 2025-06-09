import openai
import time
import csv
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re

# OpenAI API 키 설정
openai.api_key = ""

# OpenAI GPT 호출 함수
def call_gpt(prompt, temperature=0.7, model="gpt-4"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message["content"].strip()

# Step 1: 생각 생성
def generate_thoughts(question, num_thoughts=5):
    prompt = (
        f"Q: {question}\n"
        f"Step 1: Generate {num_thoughts} distinct thoughts or hypotheses.\n"
        f"Label them as (1), (2), ..., (n).\n"
        f"A:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # "A:" 뒤의 응답만 추출
    if "A:" in decoded:
        decoded= decoded.split("A:")[-1].strip()
    else:
        decoded= decoded.strip()

    # (1), (2), ... 형태로 시작하는 줄만 추출
    thoughts = [line.strip() for line in decoded.split('\n') if re.match(r"\(\d+\)", line.strip())]

    return thoughts
    

# Step 2: 생각 평가
def evaluate_thought(thought):
    prompt = (
        "Evaluate the plausibility and usefulness of the following thought.\n"
        f"Thought: {thought}\n"
        "Is this thought (a) sure, (b) maybe, or (c) impossible?\n"
        "Answer with only one word: sure / maybe / impossible"
    )
    evaluation = call_gpt(prompt, temperature=0.2)
    return evaluation.lower()

# Step 3: 가지치기
def prune_thoughts(thoughts, keep_labels=("sure", "maybe")):
    pruned = []
    for thought in thoughts:
        label = evaluate_thought(thought)
        print(f"Evaluated → {thought} → {label}")
        if label in keep_labels:
            pruned.append(thought)
        time.sleep(1)
    return pruned

# Step 4: 최종 결론 생성
def generate_final_answer(question, best_thoughts):
    joined = '\n'.join(best_thoughts)
    prompt = (
        f"Q: {question}\n"
        f"The following thoughts were considered most promising:\n"
        f"{joined}\n"
        "Based on these, write a final, concise answer that best explains the question.\n"
        "A:"
    )
    return call_gpt(prompt)

# Multi-Step ToT 전체 파이프라인
def multi_step_tot(question, num_thoughts=5):
    print("\nStep 1: 후보 생각 생성")
    thoughts = generate_thoughts(question, num_thoughts)

    print("\nStep 2: 각 생각 평가")
    pruned = prune_thoughts(thoughts)

    if not pruned:
        return "No promising thoughts found."

    print("\nStep 3: 가지치기 완료 → 다음 단계 확장")

    print("\nStep 4: 최종 결론 생성")
    answer = generate_final_answer(question, pruned)

    return answer

# 프롬프트 래퍼
def prompt_func(q):
    return q

# 응답 생성 함수
def generate_answer(prompt):
    return multi_step_tot(prompt)

# 평가 함수 + CSV 저장
def embedding_similarity_evaluate(questions, answers, prompt_func, choices_list, threshold=0.6, csv_path=None):
    correct = 0
    logs = []

    for i, (q, a) in enumerate(zip(questions, answers)):
        print(f"\n[{i}] 질문 처리 중: {q}")
        prompt = prompt_func(q)
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

# 모델 및 임베딩 초기화
embedder = SentenceTransformer('all-MiniLM-L6-v2')
model_name = "HuggingFaceTB/SmolLM-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
model.eval()

# 실행
if __name__ == "__main__":
    start_time = time.time()

    dataset = load_dataset("commonsense_qa")
    small_dataset = dataset["train"].shuffle(seed=42).select(range(100))  # 실험은 우선 10개만
    questions = [item["question"] for item in small_dataset]
    answers = [item["answerKey"] for item in small_dataset]
    choices_list = [item["choices"] for item in small_dataset]

    accuracy = embedding_similarity_evaluate(
        questions, answers, prompt_func, choices_list,
        threshold=0.3,
        csv_path="tot_results.csv"
    )

    print(f"\nTotal Accuracy: {accuracy:.2f}%")
    print(f"\nExecution Time: {time.time() - start_time:.2f} seconds")

import openai
import time
import csv
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

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

# Step 1: 생각 여러 개 생성
def generate_thoughts(question, num_thoughts=5):
    prompt = f"Q: {question}\nA:"
    thoughts = []
    for _ in range(num_thoughts):
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=60,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                synced_gpus=False
            )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "A:" in decoded:
            answer_part = decoded.split("A:")[-1].strip()
        else:
            answer_part = decoded.strip()

        first_sentence = answer_part.split(".")[0].strip()
        if first_sentence:
            thoughts.append(first_sentence)
        else:
            thoughts.append(answer_part)  # 문장 나눌 수 없으면 전체 사용

    return thoughts



# Step 2: GPT로 생각들 한번에 평가
def evaluate_thoughts_bulk(question, choices, thoughts):
    choices_text = "\n".join([f"{label}: {text}" for label, text in zip(choices["label"], choices["text"])])
    joined = "\n".join([f"{i+1}. {t}" for i, t in enumerate(thoughts)])
    prompt = (
        f"Question:\n{question}\n\n"
        f"Choices:\n{choices_text}\n\n"
        "You are given some reasoning statements (called 'thoughts') for the above multiple-choice question.\n"
        "Label each one as follows:\n"
        "- sure: if the thought clearly supports a correct answer\n"
        "- maybe: if it provides some plausible reasoning\n"
        "- impossible: if it is irrelevant or wrong\n\n"
        "Answer in the format: 1: sure, 2: maybe, ...\n\n"
        f"{joined}\n"
    )
    evaluation_text = call_gpt(prompt, temperature=0.2)
    labels = {}
    for line in evaluation_text.splitlines():
        if ":" in line:
            try:
                index, label = line.split(":")
                labels[int(index.strip()) - 1] = label.strip().lower()
            except:
                continue
    return labels


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



# Step 3~4: 가지치기 + 결론 생성
def multi_step_tot(question, choices, num_thoughts=5):
    print("\nStep 1: 후보 생각 생성")
    thoughts = generate_thoughts(question, num_thoughts)
    for idx, t in enumerate(thoughts):
        print(f"Thought {idx+1}: {t}")

    print("\nStep 2: 한번에 평가 시작")
    labels = evaluate_thoughts_bulk(question, choices, thoughts)

    pruned = [thoughts[i] for i in range(len(thoughts)) if labels.get(i) in ("sure", "maybe")]

    for i in range(len(thoughts)):
        label = labels.get(i, "unknown")
        print(f"Thought {i+1}: {thoughts[i]} → {label}")

    if not pruned:
        return "No promising thoughts found."

    print("\nStep 3: 가지치기 완료 → 다음 단계 확장")

    print("\nStep 4: 최종 결론 생성")
    return generate_final_answer(question, pruned)

# 프롬프트 래퍼
def prompt_func(q):
    return q

# 응답 생성 함수
def generate_answer(prompt, choices):
    return multi_step_tot(prompt, choices)


# 평가 함수 + CSV 저장
def embedding_similarity_evaluate(questions, answers, prompt_func, choices_list, threshold=0.6, csv_path=None):
    correct = 0
    logs = []

    for i, (q, a) in enumerate(zip(questions, answers)):
        print(f"\n[{i}] 질문 처리 중: {q}")
        prompt = prompt_func(q)
        prediction = generate_answer(prompt, choices_list[i])


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
        is_correct = sim_score >= threshold


        print(f" Prediction: {prediction}")
        print(f" Ground Truth: {ground_truth}")
        print(f" Similarity Score: {sim_score:.3f}")
        print(f" Result: {' Correct' if is_correct else ' Wrong'}")

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

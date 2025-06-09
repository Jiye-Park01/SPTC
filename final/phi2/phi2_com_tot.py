import openai
import time
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# 모델 이름
model_name = "mistralai/Mistral-7B-v0.1"


embedder = SentenceTransformer('all-MiniLM-L6-v2') 
# OpenAI GPT API Key
openai.api_key = ""

# GPT 호출 함수
def call_gpt(prompt, temperature=0.7, model="gpt-4"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message["content"].strip()

# Step 1: 후보 생각 생성
def generate_thoughts(question, num_thoughts=5):
    prompt = (
        f"Q: {question}\n"
        f"Step 1: Generate {num_thoughts} distinct thoughts or hypotheses.\n"
        f"Label them as (1), (2), ..., (n).\n"
        f"A:"
    )
    output = call_gpt(prompt)
    thoughts = [line.strip() for line in output.split('\n') if line.strip()]
    return thoughts

# Step 2: 각 생각 평가
def evaluate_thought(thought):
    prompt = (
        "Evaluate the plausibility and usefulness of the following thought.\n"
        f"Thought: {thought}\n"
        "Is this thought (a) sure, (b) maybe, or (c) impossible?\n"
        "Answer with only one word: sure / maybe / impossible"
    )
    evaluation = call_gpt(prompt, temperature=0.2)
    return evaluation.lower()

# Step 3: 가지치기 (sure/maybe 유지)
def prune_thoughts(thoughts, keep_labels=("sure", "maybe")):
    pruned = []
    for thought in thoughts:
        label = evaluate_thought(thought)
        print(f"Evaluated → {thought} → {label}")
        if label in keep_labels:
            pruned.append(thought)
        time.sleep(1)  # API rate limit 방지
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

# Multi-Step ToT 수행
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

# 기존 evaluate 구조를 그대로 사용하기 위한 설정
def prompt_func(q):
    return q  # dummy 프롬프트 반환 (사실상 질문)

def generate_answer(prompt):
    return multi_step_tot(prompt)  # 실제로는 prompt = question

# evaluate 함수 (기존 구조 유지)
def embedding_similarity_evaluate(questions, answers, prompt_func, choices_list, threshold=0.6):
    correct = 0

    for i, (q, a) in enumerate(zip(questions, answers)):
        prompt = prompt_func(q)
        prediction = generate_answer(prompt)

        try:
            # label과 정답키(a) 비교 시 대소문자 처리
            labels = [label.lower() for label in choices_list[i]["label"]]
            texts = choices_list[i]["text"]
            idx = labels.index(a.lower())
            ground_truth = texts[idx]
        except Exception as e:
            print(f"[Warning] {i}번째 문제에서 정답 추출 실패 → {e}")
            continue  # 그냥 스킵

        embeddings = embedder.encode([ground_truth.lower(), prediction.lower()])
        sim_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

        if sim_score >= threshold:
            correct += 1

    acc = correct / len(questions) * 100
    return acc

embedder = SentenceTransformer('all-MiniLM-L6-v2')  

# 모델 로딩
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
    start_time=0
    end_time=0
    elapsed_time=0
    start_time = time.time()

    dataset = load_dataset("commonsense_qa")
    small_dataset = dataset["train"].shuffle(seed=42).select(range(100))
    questions = [item["question"] for item in small_dataset]
    answers = [item["answerKey"] for item in small_dataset]
    choices_list = [item["choices"] for item in small_dataset] 
    accuracy = embedding_similarity_evaluate(questions, answers, prompt_func, choices_list, threshold=0.3)
    print(f"\nTotal Accuracy: {accuracy:.2f}%")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nExecution Time: {elapsed_time:.2f} seconds")

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


phi2 = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(phi2)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    phi2, device_map="auto", torch_dtype=torch.float16
)
model.eval()

#commensense_qa
dataset = load_dataset("commonsense_qa")
small_dataset = dataset["train"].shuffle(seed=42).select(range(100))
questions = [item["question"] for item in small_dataset]
answers = [item["answerKey"] for item in small_dataset]
choices_list = [item["choices"] for item in small_dataset]

# 프롬프트 함수들
def no_prompt(question):
    return f"Q: {question}\nA:"

def zero_shot_cot_prompt(question):
    tot_intro = (
        "Let's think step-by-step.\n"
    )
    return f"Q: {question}\n{tot_intro}A:"

def few_shot_cot_prompt(question):
    few_shot = (
        "Q: Why do people wear sunglasses?\n"
        "Let's think step-by-step.\n"
        "First, sunglasses protect eyes from sunlight.\n"
        "Second, they reduce glare.\n"
        "Third, they are a fashion accessory.\n"
        "A: To protect eyes, reduce glare, and for fashion.\n\n"
    )
    return few_shot + f"Q: {question}\nLet's think step-by-step.\nA:"

def zero_shot_tot_prompt(question):
    tot_intro = (
        "Let's reason step-by-step and consider multiple possible thoughts before choosing the best answer.\n"
    )
    return f"Q: {question}\n{tot_intro}A:"

def few_shot_tot_prompt(question):
    few_shot = (
        "Q: Why do people wear sunglasses?\n"
        "Let's think of 5 possible reasons:\n"
        "(1) To block sunlight.\n"
        "(2) To look fashionable.\n"
        "(3) To hide emotions.\n"
        "(4) To reduce eye strain.\n"
        "(5) To protect eyes from wind and dust.\n"
        "Now, among these, which is the most reasonable? To block sunlight and protect eyes.\n"
        "A: To block sunlight and protect eyes.\n\n"
    )
    return few_shot + f"Q: {question}\nLet's think of 5 possible reasons:\nA:"

def sptc_prompt(question):
    few_shot = (
        "Q: Why do people wear sunglasses?\n"
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
    return system_prompt + "\n\n" + few_shot + f"Q: {question}\nLet's think through different possibilities carefully:\nA:"

# 답변 생성 함수
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

# 평가 함수
def embedding_similarity_evaluate(questions, answers, prompt_func, choices_list, threshold=0.3):
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

# 실험 실행
print("===phi2 실험 시작 ===")



start_time=0
end_time=0
elapsed_time=0
start_time = time.time()
print("▶ No-Prompt 실험 중...")
no_cot_accuracy = embedding_similarity_evaluate(questions, answers, no_prompt,choices_list)
print(f"No-Prompt Accuracy: {no_cot_accuracy:.2f}%")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"[No-Prompt] Execution time: {elapsed_time:.6f} seconds")

start_time=0
end_time=0
elapsed_time=0
start_time = time.time()
print("▶ CoT (Zero-shot, Step-by-Step) 실험 중...")
zero_cot_accuracy = embedding_similarity_evaluate(questions, answers, zero_shot_cot_prompt,choices_list)
print(f"CoT (Zero-shot Accuracy: {zero_cot_accuracy:.2f}%")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"[Zero-CoT] Execution time: {elapsed_time:.6f} seconds")


start_time=0
end_time=0
elapsed_time=0
start_time = time.time()
print("▶ SPTC (structured reasoning + evaluation) 실험 중...")
SPTC_accuracy = embedding_similarity_evaluate(questions, answers, sptc_prompt, choices_list)
print(f"SPTC Accuracy: {SPTC_accuracy:.2f}%")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"[SPTC] Execution time: {elapsed_time:.6f} seconds")

print("=== phi2 실험 완료 ===")
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


qwen = "Qwen/Qwen2.5-Math-1.5B"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B", use_auth_token=True)

model = AutoModelForCausalLM.from_pretrained(
    qwen,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
model.eval()

# aqua_rat
dataset = load_dataset("deepmind/aqua_rat")
small_dataset = dataset["train"].shuffle(seed=42).select(range(100))
questions = [item["question"] for item in small_dataset]
answers = [item["correct"] for item in small_dataset]
choices_list = [item["options"] for item in small_dataset]

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
        "Q: A car is being driven, in a straight line and at a uniform speed, towards the base of a vertical tower. "
        "The top of the tower is observed from the car and, in the process, it takes 10 minutes for the angle of elevation to change from 45° to 60°. "
        "After how much more time will this car reach the base of the tower?\n"
        "Let's think step-by-step.\n"
        "First, we assume the height of the tower is h.\n"
        "Second, at the first position, tan 45° = h / d → so d = h.\n"
        "Third, after 10 minutes, tan 60° = h / x → so x = h / √3.\n"
        "Fourth, the car moved distance d - x = h - h / √3 = h(1 - 1/√3) in 10 minutes.\n"
        "Fifth, we compute time to travel remaining x = h / √3 using same speed → time = 10 * (1 / √3) / (1 - 1/√3).\n"
        "Sixth, rationalizing gives: time = 10(√3 + 1)/2 = 5(√3 + 1)\n"
        "A: (A) 5(√3 + 1)\n\n"
    )
    return few_shot + f"Q: {question}\nLet's think step-by-step.\nA:"

def zero_shot_tot_prompt(question):
    tot_intro = (
        "Let's reason step-by-step and consider multiple possible thoughts before choosing the best answer.\n"
    )
    return f"Q: {question}\n{tot_intro}A:"

def few_shot_tot_prompt(question):
    few_shot = (
        "Q: A car is being driven, in a straight line and at a uniform speed, towards the base of a vertical tower. "
        "The top of the tower is observed from the car and, in the process, it takes 10 minutes for the angle of elevation to change from 45° to 60°. "
        "After how much more time will this car reach the base of the tower?\n"
        "Choices:\n"
        "(A) 5(√3 + 1)\n"
        "(B) 6(√3 + √2)\n"
        "(C) 7(√3 – 1)\n"
        "(D) 8(√3 – 2)\n"
        "(E) None of these\n"
        "Let's think of 5 possible ways to approach the problem:\n"
        "(1) Use tan 45° = h/d → gives h = d₁.\n"
        "(2) Use tan 60° = h/x → gives x = h/√3.\n"
        "(3) Compute distance moved in 10 min = h - h/√3 = h(1 - 1/√3).\n"
        "(4) Compute speed = h(1 - 1/√3) / 10.\n"
        "(5) Use speed to find time to travel x = h/√3 → time = [h/√3] / [h(1 - 1/√3)/10] = 10(1 + √3)/2.\n"
        "Now, among these, which leads to the correct final value? The travel time is 5(√3 + 1), matching choice A.\n"
        "A: (A) 5(√3 + 1)\n\n"
    )
    return few_shot + f"Q: {question}\nLet's think of 5 possible reasons:\nA:"

def sptc_prompt(question):
    few_shot = (
        "Q: A car is being driven, in a straight line and at a uniform speed, towards the base of a vertical tower. "
        "The top of the tower is observed from the car and, in the process, it takes 10 minutes for the angle of elevation to change from 45° to 60°. "
        "After how much more time will this car reach the base of the tower?\n"
        "Choices:\n"
        "(A) 5(√3 + 1)\n"
        "(B) 6(√3 + √2)\n"
        "(C) 7(√3 – 1)\n"
        "(D) 8(√3 – 2)\n"
        "(E) None of these\n"
        "Let's think through different possibilities carefully:\n"
        "(1) Let the height of the tower be h.\n"
        "(2) From the first position: tan 45° = h / d → d = h.\n"
        "(3) After 10 minutes, tan 60° = h / x → x = h / √3.\n"
        "(4) So in 10 minutes, car moved distance d - x = h - h / √3 = h(1 - 1/√3).\n"
        "(5) Let t be the time to travel remaining distance x = h / √3.\n"
        "(6) Since distance ∝ time (uniform speed), use ratio: t / 10 = (h / √3) / [h(1 - 1/√3)] → t = 10 * (1 / √3) / (1 - 1/√3)\n"
        "(7) Rationalize the denominator: t = 10(√3 + 1)/2 = 5(√3 + 1)\n"
        "Evaluation:\n"
        "- All calculations align with option A.\n"
        "- Other options do not match the algebraic result.\n"
        "Final Answer: (A) 5(√3 + 1)\n\n"
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
            labels = [opt.split(")")[0].strip().lower() for opt in choices_list[i]]
            texts = [opt.split(")", 1)[1].strip() for opt in choices_list[i]]

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
print("===qwen 실험 시작 ===")



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
print("▶ CoT (Few-shot, Step-by-Step) 실험 중...")
few_cot_accuracy = embedding_similarity_evaluate(questions, answers, few_shot_cot_prompt, choices_list)
print(f"CoT Accuracy: {few_cot_accuracy:.2f}%")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"[Few-CoT] Execution time: {elapsed_time:.6f} seconds")

# start_time=0
# end_time=0
# elapsed_time=0
# start_time = time.time()
# print("▶ ToT (Zero-shot) 실험 중...")
# zero_tot_accuracy = embedding_similarity_evaluate(questions, answers, zero_shot_tot_prompt, choices_list)
# print(f"ToT Accuracy: {zero_tot_accuracy:.2f}%")
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"[Zero-ToT] Execution time: {elapsed_time:.6f} seconds")

# start_time=0
# end_time=0
# elapsed_time=0
# start_time = time.time()
# print("▶ ToT (Few-shot, 5-branches Tree-of-Thought) 실험 중...")
# few_tot_accuracy = embedding_similarity_evaluate(questions, answers, few_shot_tot_prompt, choices_list)
# print(f"ToT Accuracy: {few_tot_accuracy:.2f}%")
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"[Few-ToT] Execution time: {elapsed_time:.6f} seconds")

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

print("=== qwen 실험 완료 ===")
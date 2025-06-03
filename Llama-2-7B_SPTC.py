from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

llama2 = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(llama2, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    llama2,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
model.eval()

dataset = load_dataset("commonsense_qa")
small_dataset = dataset["train"].shuffle(seed=42).select(range(100))
questions = [item["question"] for item in small_dataset]
answers = [item["answerKey"] for item in small_dataset]

# 프롬프트 함수들
def no_cot_prompt(question):
    return f"Q: {question}\nA:"

def cot_prompt(question):
    few_shot = (
        "Q: Why do people wear sunglasses?\n"
        "Let's think step-by-step.\n"
        "First, sunglasses protect eyes from sunlight. Second, they can reduce glare. Third, they can be a fashion accessory.\n"
        "A: To protect eyes, reduce glare, and for fashion.\n\n"
    )
    return few_shot + f"Q: {question}\nLet's think step-by-step.\nA:"

def tot_prompt(question):
    tot_intro = (
        "Let's reason step-by-step and consider multiple possible thoughts before choosing the best answer.\n"
    )
    return f"Q: {question}\n{tot_intro}A:"

def one_step_tot_prompt(question):
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
            synced_gpus=False
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "A:" in decoded:
        return decoded.split("A:")[-1].strip()
    else:
        return decoded.strip()

# 평가 함수
def evaluate(questions, answers, prompt_func, method_name=""):
    correct = 0
    for q, a in zip(questions, answers):
        prompt = prompt_func(q)
        prediction = generate_answer(prompt)
        if a.lower() in prediction.lower():
            correct += 1
    acc = correct / len(questions) * 100
    print(f"{method_name} Accuracy: {acc:.2f}%")

def best_first_one_step_tot_prompt(question, num_options=5):
    # Step 1: 다양한 아이디어 나열
    idea_prompt = (
        f"Q: {question}\n"
        "Let's think through different possible explanations or answers.\n"
        f"List {num_options} numbered possibilities like this:\n"
        "(1) ...\n(2) ...\n(3) ...\nA:"
    )
    ideas_raw = generate_answer(idea_prompt)

    # 아이디어 파싱
    idea_lines = [line.strip() for line in ideas_raw.split("\n") if line.strip().startswith("(")]
    ideas = idea_lines[:num_options]

    if len(ideas) == 0:
        return "Unable to generate ideas."

    # Step 2: 각 아이디어 평가
    scored_ideas = []
    for idea in ideas:
        eval_prompt = (
            f"Evaluate this idea for the question: \"{question}\"\n"
            f"Idea: {idea}\n"
            "Score it from 1 (poor) to 10 (excellent) based on plausibility, relevance, and informativeness.\n"
            "Score:"
        )
        score_text = generate_answer(eval_prompt)
        try:
            score = float(score_text.strip().split()[0])
        except:
            score = 0
        scored_ideas.append((score, idea))

    if not scored_ideas:
        return "Failed to evaluate any reasoning path."

    # Step 3: 점수순 정렬
    scored_ideas.sort(reverse=True, key=lambda x: x[0])
    top_idea = scored_ideas[0][1]

    # Step 4: 최상위 아이디어 기반 확장
    final_prompt = (
        f"Q: {question}\n"
        f"Selected idea: {top_idea}\n"
        "Explain why this idea is the best and provide the final answer.\n"
        "A:"
    )
    final_answer = generate_answer(final_prompt)
    return final_answer

    
def best_first_one_step_tot_wrapper(q):
    return best_first_one_step_tot_prompt(q)



# 실험 시작
print("=== Llama2 실험 시작 ===")

print("\n▶ No-CoT 실험 중...")
evaluate(questions, answers, no_cot_prompt, "No-CoT")

print("\n▶ CoT (Few-shot) 실험 중...")
evaluate(questions, answers, cot_prompt, "CoT")

print("\n▶ ToT (Tree of Thoughts) 스타일 실험 중...")
evaluate(questions, answers, tot_prompt, "ToT")

# print("▶ One-Step ToT (structured reasoning + evaluation) 실험 중...")
# one_step_tot_accuracy = evaluate(questions, answers, one_step_tot_prompt)
# print(f"One-Step ToT Accuracy: {one_step_tot_accuracy:.2f}%")

print("▶ One-Step + Best-First ToT 실험 중...")
evaluate(questions, answers, best_first_one_step_tot_wrapper, "One-Step + Best-First ToT")

print("\n=== Llama2 실험 완료 ===")
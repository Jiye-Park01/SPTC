from datasets import load_dataset
import os

# 캐시 경로 설정
os.environ["HF_DATASETS_CACHE"] = "./gsm8k"

# # main 버전 다운로드
# gsm8k_main = load_dataset("openai/gsm8k", "main")

# socratic 버전 다운로드
ds = load_dataset("openai/gsm8k", "socratic")
print(ds["train"][0])
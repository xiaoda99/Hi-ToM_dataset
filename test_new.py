
from dotenv import load_dotenv
import json
import os
import requests
import pandas as pd

from min_tom1 import Engine
load_dotenv()

with open('data2.json', 'r', encoding='utf-8') as f:
    data_list = json.load(f)


def get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable is required to call the API."
        )
    return key


SYSTEM_PROMPT = (
    "You answer theory-of-mind questions about short stories. "
    "Respond with only the most likely answer as a short phrase. Do not provide reasoning, explanations, or chain-of-thought. "
    "If the answer is unknown, reply exactly with 'Unknown'."
)
PROMPT_TEMPLATE = """Story:\n{story}\n\nQuestion: {question}\n\nReply with the best answer. Do not add explanations.\nAnswer:"""
# 配置模型 API
# 测试用的轨迹流动的免费模型
# API_URL = "https://api.siliconflow.cn/v1/chat/completions"
# API_KEY = ""
# MODEL = "Qwen/Qwen2.5-7B-Instruct"


# API_URL = "https://aihubmix.com/v1/chat/completions"
# API_KEY = ""
# MODEL = "claude-sonnet-4-0"
# claude-opus-4-0

API_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = get_api_key()
MODEL = "anthropic/claude-sonnet-4.5"
# MODEL = "anthropic/claude-sonnet-4"
# MODEL = "anthropic/claude-opus-4"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}


# 构造提问 Prompt 并调用模型
def build_prompt(story: str, question: str) -> str:
    return PROMPT_TEMPLATE.format(story=story, question=question)


def ask_model(story: str, question: str) -> str:
    prompt = build_prompt(story, question)
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/openrouter/documentation",  # polite default
        "X-Title": "Hi-ToM QA Evaluation",
    }
    payload = {
        "model": MODEL,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        model_answer = result["choices"][0]["message"]["content"].strip()
        return model_answer
    except Exception as e:
        print(f"[错误] 调用模型失败: {e}")
        return "ERROR"


# 主流程：遍历数据，调用模型，比对答案，统计结果

# 保存每条数据的结果
# 主流程：遍历数据，调用模型，比对答案，统计结果
results = []

for idx, sample in enumerate(data_list):
    #if 11 <= idx <= 20:
    if idx == 114:
        print(idx)
        id = sample.get("id",'')
        print(id)
        story_id=sample.get("story_id", "")
        story_text = sample.get("story_text", "")
        question = sample.get("question", "")
        expected = sample.get("expected", "").strip()
        story_length = sample.get("story_length", "")
        qa_order = sample.get("qa_order", "")
        #  调用模型，获取模型答案
        model_answer_letter = ask_model(story_text, question)

        print(f"模型输出: {model_answer_letter}")

        # 比对模型答案内容与正确答案内容
        is_correct = (model_answer_letter == expected)

        # 保存结果
        results.append({
            "id": id,
            "story_id": story_id,
            "story_length": story_length,
            "story_text": story_text,
            "qa_order": qa_order,
            "question": question,
            "expected": expected,
            "answer": model_answer_letter,
            "is_correct": is_correct
        })

# 转 DataFrame
df = pd.DataFrame(results)

output_excel = "1.xlsx"
sheet_name = 'Sheet1'

file_exists = os.path.exists(output_excel)

if file_exists:
    with pd.ExcelWriter(output_excel, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
        max_row = writer.sheets[sheet_name].max_row
        df.to_excel(writer, index=False, header=False, startrow=max_row)
else:
    with pd.ExcelWriter(output_excel, mode='w', engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)

print(f"结果已保存到 Excel 文件：{output_excel}")

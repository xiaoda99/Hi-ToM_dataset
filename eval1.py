"""Utilities for evaluating Hi-ToM QA generations with OpenRouter LLMs.

Produces a pandas DataFrame with per-question results and story metadata so you can
analyze accuracy by model, question order, narrative traits, etc., directly in a notebook.

Example
-------
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests
import pandas as pd

from min_tom1 import Engine

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# OPENROUTER_URL = "https://aihubmix.com/v1/chat/completions"
DEFAULT_MODELS = [
    "anthropic/claude-sonnet-4"
]
SYSTEM_PROMPT = (
    "You answer theory-of-mind questions about short stories. "
    "Respond with only the most likely answer as a short phrase. Do not provide reasoning, explanations, or chain-of-thought. "
    "If the answer is unknown, reply exactly with 'Unknown'."
)
PROMPT_TEMPLATE = """Story:\n{story}\n\nQuestion: {question}\n\nReply with the best answer. Do not add explanations.\nAnswer:"""
load_dotenv()

@dataclass
class FlatResult:
    id: int
    seed: int
    story_id: int
    story_length: int
    story_text: str
    model: str
    # story_steps: int
    # n_events: int
    # events: List[Event]
    n_peeks: int
    n_distractions: int
    qa_order: int
    qa_object: str
    question: str
    expected: str
    answer: Optional[str] = None
    is_correct: Optional[bool] = None
    latency_sec: float = 0.0
    retries: int = 0

def get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable is required to call the API."
        )
    return key


def build_engine_example() -> Engine:
    """Create a basic engine instance used when eval.py is run directly."""
    rooms = {
        "garden": {"objects": ["marble"], "containers": ["basket", "box"]},
        # "kitchen": {"objects": ["cup"], "containers": ["cabinet", "drawer"]},
    }
    engine = Engine(
        agents=["Sally", "Anne", "Emma"],
        rooms=rooms,
        peek_prob=0.25,
        distracted_prob=0.25,
        exit_without_move_prob=0.2,
        reenter_prob=0.2,
        seed=42,
    )
    return engine


def match_answer(answer: str, expected: str, max_answer_tokens: int = 10) -> bool:
    if len(answer.split()) > max_answer_tokens: return False
    return expected.lower() in answer.lower()


def build_prompt(story: str, question: str) -> str:
    return PROMPT_TEMPLATE.format(story=story, question=question)


def call_openrouter(
        api_key: str,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        max_retries: int = 3,
        timeout: int = 60,
) -> tuple[str, float, int]:

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/openrouter/documentation",
        "X-Title": "Hi-ToM QA Evaluation",
    }
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    if False and model in ['anthropic/claude-sonnet-4.5', "openai/gpt-5", "google/gemini-2.5-pro"]:
        payload["reasoning"] = {"effort": "medium"}

    start_time = time.perf_counter()
    backoff = 2.0

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)

            if response.status_code == 200:
                try:
                    data = response.json()
                except Exception as err:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                try:
                    content = data["choices"][0]["message"]["content"].strip()
                    elapsed = time.perf_counter() - start_time
                    retries = attempt - 1
                    return content, elapsed, retries
                except (KeyError, IndexError) as err:
                    error_msg = f"Unexpected API response format: {data}"
                    content = f"error: {error_msg}"
                    elapsed = time.perf_counter() - start_time
                    retries = attempt - 1
                    return content, elapsed, retries

            else:
                # HTTP 请求失败（比如 403、500、429 等）
                error_msg = f"HTTP Error {response.status_code}: {response.text[:200]}"
                time.sleep(backoff)
                backoff *= 2
        except Exception as e:
            error_msg = f"请求异常: {type(e).__name__} - {str(e)}"
            time.sleep(backoff)
            backoff *= 2

    # 如果所有重试都失败了，返回错误信息存储
    error_msg = f"所有 {max_retries} 次尝试均失败"
    return f"error: {error_msg}", 0.0, max_retries


def evaluate_model_on_qas(
        api_key: str,
        model: str,
        items: List[FlatResult],
        sleep: float = 0.5,
        max_workers: int = 5,
) -> List[FlatResult]:
    def run_one(item: FlatResult) -> FlatResult:
        prompt = build_prompt(item.story_text, item.question)
        answer, latency, retries = call_openrouter(api_key, model, prompt)
        is_correct = match_answer(answer, item.expected)
        if sleep:
            time.sleep(sleep)
        item.answer = answer
        item.is_correct = is_correct
        print(is_correct)
        item.latency_sec = latency
        item.retries = retries
        return item

    results: List[FlatResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_one, item) for item in items]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Evaluating {model}", unit="QA"):
            results.append(future.result())
    return results


def evaluate_models(
        models: Iterable[str],
        story_results: List[FlatResult],
        sleep: float = 0.5,
        max_workers: int = 5,
) -> List[FlatResult]:
    api_key = get_api_key()
    all_results: List[FlatResult] = []
    for model in models:
        print(f"Evaluating {len(story_results)} QAs with model {model} (max_workers={max_workers})...")
        items = [FlatResult(**{**item.__dict__, "model": model}) for item in story_results]
        all_results.extend(evaluate_model_on_qas(api_key, model, items, sleep=sleep, max_workers=max_workers))
    return all_results


# 从json文件中获取测试数据
def get_data():
    with open("data1.json", "r", encoding="utf-8") as f:
        flat_items = [FlatResult(**item) for item in json.load(f)]
    return flat_items


def run_evaluation(
        models=DEFAULT_MODELS,
        sleep: float = 0,
        max_workers: int = 5,
):
    flat_items = get_data()
    results = evaluate_models(models, flat_items, sleep=sleep, max_workers=max_workers)
    df_new = pd.DataFrame([r.__dict__ for r in results])
    excel_filename = 'results_4.5.xlsx'
    if os.path.exists(excel_filename):
        df_old = pd.read_excel(excel_filename)
    else:
        df_old = pd.DataFrame()

    df_combined = pd.concat([df_old, df_new], ignore_index=True)
    df_combined.to_excel(excel_filename, index=False)
    print("测评完成")


if __name__ == "__main__":
    # Example usage (replace with your own engine factory and parameters)
    models = [
        'anthropic/claude-sonnet-4',
        'anthropic/claude-sonnet-4.5',
        'anthropic/claude-opus-4.1',
        # 'openai/gpt-5',
        # 'google/gemini-2.5-pro',
    ]
    model=['anthropic/claude-sonnet-4.5']
    run_evaluation(
        models=model,
        sleep=0,
        max_workers=1,
    )

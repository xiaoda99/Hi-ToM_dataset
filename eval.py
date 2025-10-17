"""Utilities for evaluating Hi-ToM QA generations with OpenRouter LLMs.

Produces a pandas DataFrame with per-question results and story metadata so you can
analyze accuracy by model, question order, narrative traits, etc., directly in a notebook.

Example
-------
>>> df = run_evaluations(build_engine_example, num_stories=3, steps=8, models=["openai/gpt-5"])
>>> df.groupby(["model", "qa_order"]).is_correct.mean()
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests
import pandas as pd

from min_tom import Engine, Event


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODELS = [
    "anthropic/claude-3.5-sonnet",
    "openai/gpt-5",
]
SYSTEM_PROMPT = (
    "You answer theory-of-mind questions about short stories. "
    "Respond with only the most likely answer as a short phrase. Do not provide reasoning, explanations, or chain-of-thought. "
    "If the answer is unknown, reply exactly with 'Unknown'."
)
PROMPT_TEMPLATE = """Story:\n{story}\n\nQuestion: {question}\n\nReply with the best answer. Do not add explanations.\nAnswer:"""


@dataclass
class QAExample:
    order: int
    obj: str
    question: str
    expected: str


@dataclass
class FlatResult:
    id: int
    seed: int
    story_id: int
    story_text: str
    model: str
    story_steps: int
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
    key = os.getenv("OPENROUTER_API_KEY")
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


def flatten_qas(qas: Dict[int, Dict[str, List[tuple[str, str]]]]) -> List[QAExample]:
    """Convert the nested QA dictionary into a flat list."""
    flat: List[QAExample] = []
    for order, object_map in sorted(qas.items()):
        for obj, qa_list in object_map.items():
            for question, expected in qa_list:
                flat.append(QAExample(order=order, obj=obj, question=question, expected=expected))
    return flat


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
        "HTTP-Referer": "https://github.com/openrouter/documentation",  # polite default
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


def gather_story_data(engine: Engine, steps: int, story_id: int) -> tuple[str, List[QAExample], Dict[str, int]]:
    story = engine.generate_story(steps=steps)
    qas_dict = engine.generate_QAs()
    qas = flatten_qas(qas_dict)
    metadata = {
        "story_id": story_id,
        "story_steps": steps,
        # "n_events": len(engine.events),
        # 'events': engine.events,
        "n_peeks": getattr(engine, "n_peeks", 0),
        "n_distractions": getattr(engine, "n_distractions", 0),
    }
    return story, qas, metadata


def run_evaluations(
    engine_factory: Callable[[], Engine],
    num_stories: int,
    steps: int,
    models: Iterable[str],
    max_qas: Optional[int] = None,
    sleep: float = 0.0,
    max_workers: int = 5,
) -> pd.DataFrame:
    flat_items: List[FlatResult] = []

    for story_id in range(num_stories):
        engine = engine_factory(story_id)
        story, qas, metadata = gather_story_data(engine, steps, story_id)
        if max_qas is not None:
            qas = qas[:max_qas]
        print(
            f"Story {story_id}: {len(qas)} QA items, "
            f"{metadata['n_peeks']} peeks, {metadata['n_distractions']} distractions."
        )
        for qa in qas:
            flat_items.append(
                FlatResult(
                    **metadata,
                    seed=engine.seed,
                    story_text=story,
                    model="",
                    qa_order=qa.order,
                    qa_object=qa.obj,
                    question=qa.question,
                    expected=qa.expected,
                )
            )

    results = evaluate_models(models, flat_items, sleep=sleep, max_workers=max_workers)
    df = pd.DataFrame([r.__dict__ for r in results])
    return df


def run_evaluation(
    engine: Engine,
    steps: int = 10,
    models: Iterable[str] = DEFAULT_MODELS,
    max_qas: int = None,
    sleep: float = 0.5,
    max_workers: int = 5,
) -> pd.DataFrame:
    story, qas, metadata = gather_story_data(engine, steps, story_id=0)
    if max_qas is not None:
        qas = qas[:max_qas]
    flat_items = [
        FlatResult(
            **metadata,
            story_text=story,
            model="",
            qa_order=qa.order,
            qa_object=qa.obj,
            question=qa.question,
            expected=qa.expected,
        )
        for qa in qas
    ]
    results = evaluate_models(models, flat_items, sleep=sleep, max_workers=max_workers)
    df = pd.DataFrame([r.__dict__ for r in results])
    return df

# 从json文件中获取测试数据
def get_data():
    with open("data1.json", "r", encoding="utf-8") as f:
        flat_items = [FlatResult(**item) for item in json.load(f)]
    return flat_items

def run_evaluation_new(
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
    run_evaluation_new(
        models=model,
        sleep=0,
        max_workers=1,
    )


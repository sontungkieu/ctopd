import asyncio
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional

from tqdm import tqdm

logger = logging.getLogger(__name__)


def extract_last_boxed(text: str) -> Optional[str]:
    """Return the content of the last complete ``\\boxed`` or ``\\fbox``."""
    if not isinstance(text, str):
        return None

    boxed_index = text.rfind("\\boxed")
    fbox_index = text.rfind("\\fbox")
    start = max(boxed_index, fbox_index)
    if start < 0:
        return None

    left_brace = text.find("{", start)
    if left_brace < 0:
        return None

    depth = 0
    for index in range(left_brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[left_brace + 1:index]
    return None


def verify_math_answer(prediction: Optional[str], label: Any) -> bool:
    """Compare a boxed prediction with a reference answer using math-verify."""
    try:
        from math_verify import parse, verify
    except ImportError as error:
        raise RuntimeError(
            "Math evaluation requires the 'math-verify' package. "
            "Install the project dependencies before starting training."
        ) from error

    if prediction is None:
        return False

    try:
        prediction = prediction[:300]
        return bool(
            verify(
                parse(f"\\boxed{{{label}}}"),
                parse(f"\\boxed{{{prediction}}}"),
            )
        )
    except Exception:
        return False


def score_math_responses(
    responses: List[List[str]],
    labels: List[Any],
    verifier: Callable[[Optional[str], Any], bool] = verify_math_answer,
) -> tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Score multiple generated responses per problem and compute pass@k."""
    if len(responses) != len(labels):
        raise ValueError("responses and labels must contain the same number of problems")

    total_correct = 0
    total_responses = 0
    passed_problems = 0
    results = []

    for index, (problem_responses, label) in enumerate(zip(responses, labels)):
        predictions = [extract_last_boxed(response) for response in problem_responses]
        correctness = [verifier(prediction, label) for prediction in predictions]
        num_correct = sum(correctness)
        total_correct += num_correct
        total_responses += len(correctness)
        passed_problems += int(any(correctness))
        results.append(
            {
                "index": index,
                "answer": str(label),
                "pred_answers": predictions,
                "responses": problem_responses,
                "acc_list": correctness,
                "num_correct": num_correct,
            }
        )

    num_problems = len(labels)
    accuracy = total_correct / total_responses if total_responses else 0.0
    pass_at_k = passed_problems / num_problems if num_problems else 0.0
    metrics = {
        "eval/accuracy": accuracy,
        "eval/num_correct": float(total_correct),
        "eval/num_total": float(total_responses),
        "eval/pass_at_k": pass_at_k,
    }
    return metrics, results


class MathEvaluator:
    """Run high-concurrency, multi-sample math evaluation through SGLang."""

    def __init__(self, strategy, rollout_group, generate_kwargs, student_processor):
        self.args = strategy.args
        self.rollout_group = rollout_group
        self.generate_kwargs = generate_kwargs or {}
        self.student_processor = student_processor
        self.dataset_path = self.args.data.eval_math_dataset_path
        self.n_samples = self.args.data.eval_math_n_samples
        self.max_concurrent = self.args.data.eval_math_max_concurrent
        self.max_tokens = self.args.data.eval_math_max_tokens
        self.timeout = self.args.data.eval_math_timeout

        # Fail before training if the optional evaluator dependency is unavailable.
        try:
            import math_verify  # noqa: F401
        except ImportError as error:
            raise RuntimeError(
                "Math evaluation requires the 'math-verify' package. "
                "Install the project dependencies before starting training."
            ) from error

    def evaluate(self, global_step: int) -> Dict[str, float]:
        eval_data = self._load_dataset()
        prompts, labels = self._build_prompts(eval_data)
        sampling_params = {
            "max_new_tokens": self.max_tokens,
            "temperature": self.generate_kwargs.get("temperature", 0.6),
            "top_p": self.generate_kwargs.get("top_p", 0.95),
            "n": self.n_samples,
        }

        logger.info(
            "[Math Eval] Starting %d problems at step %d "
            "(n_samples=%d, max_concurrent=%d, timeout=%ds)",
            len(eval_data),
            global_step,
            self.n_samples,
            self.max_concurrent,
            self.timeout,
        )

        if self.args.train.enable_sleep:
            self.rollout_group.wakeup()
        try:
            responses = self._generate(prompts, sampling_params)
        finally:
            if self.args.train.enable_sleep:
                self.rollout_group.sleep()

        metrics, results = score_math_responses(responses, labels)
        pass_at_k = metrics.pop("eval/pass_at_k")
        if self.n_samples > 1:
            metrics[f"eval/pass@{self.n_samples}"] = pass_at_k
        for result, item in zip(results, eval_data):
            result["problem"] = item["problem"]

        self._save_results(
            global_step, metrics, results, len(eval_data), pass_at_k
        )
        return metrics

    def _load_dataset(self) -> List[Dict[str, Any]]:
        data = []
        with open(self.dataset_path, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON in math evaluation dataset at line {line_number}"
                    ) from error
                missing = [key for key in ("problem", "answer") if key not in item]
                if missing:
                    raise ValueError(
                        f"Math evaluation dataset line {line_number} is missing "
                        f"field(s): {', '.join(missing)}"
                    )
                data.append(item)
        if not data:
            raise ValueError("Math evaluation dataset is empty")
        return data

    def _build_prompts(
        self, eval_data: List[Dict[str, Any]]
    ) -> tuple[List[List[int]], List[Any]]:
        tokenizer = getattr(
            self.student_processor, "tokenizer", self.student_processor
        )
        prompts = []
        labels = []
        for item in eval_data:
            messages = [{"role": "user", "content": str(item["problem"])}]
            template_kwargs = {}
            if "enable_thinking" in str(getattr(tokenizer, "chat_template", "")):
                template_kwargs["enable_thinking"] = self.args.model.enable_thinking
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
            prompts.append(tokenizer.encode(prompt, add_special_tokens=False))
            labels.append(item["answer"])
        return prompts, labels

    def _generate(
        self, prompt_token_ids: List[List[int]], sampling_params: Dict[str, Any]
    ) -> List[List[str]]:
        if not prompt_token_ids:
            return []
        return asyncio.run(
            self._generate_async(prompt_token_ids, sampling_params)
        )

    async def _generate_async(
        self, prompt_token_ids: List[List[int]], sampling_params: Dict[str, Any]
    ) -> List[List[str]]:
        import aiohttp

        concurrency = min(len(prompt_token_ids), self.max_concurrent)
        semaphore = asyncio.Semaphore(concurrency)
        results: List[Optional[List[str]]] = [None] * len(prompt_token_ids)
        generate_url = f"{self.rollout_group.router_url}/generate"

        async def generate_one(index, input_ids, session):
            payload = {
                "input_ids": input_ids,
                "sampling_params": sampling_params,
            }
            async with semaphore:
                try:
                    async with session.post(generate_url, json=payload) as response:
                        response.raise_for_status()
                        output = await response.json()
                    if isinstance(output, list):
                        texts = [item["text"] for item in output]
                    elif isinstance(output, dict):
                        texts = output.get("text", "")
                        texts = [texts] if isinstance(texts, str) else texts
                    else:
                        raise TypeError(
                            f"Unexpected SGLang response type: {type(output).__name__}"
                        )
                    if not isinstance(texts, list) or len(texts) != self.n_samples:
                        raise ValueError(
                            f"SGLang returned {len(texts) if isinstance(texts, list) else 0} "
                            f"responses, expected {self.n_samples}"
                        )
                    results[index] = [str(text).strip() for text in texts]
                except Exception as error:
                    logger.warning(
                        "[Math Eval] Request %d failed with %s: %s; using empty responses",
                        index,
                        type(error).__name__,
                        error,
                    )
                    results[index] = [""] * self.n_samples

        connector = aiohttp.TCPConnector(limit=concurrency)
        timeout = aiohttp.ClientTimeout(
            total=self.timeout if self.timeout > 0 else None
        )
        async with aiohttp.ClientSession(
            connector=connector, trust_env=False, timeout=timeout
        ) as session:
            tasks = [
                asyncio.create_task(generate_one(index, ids, session))
                for index, ids in enumerate(prompt_token_ids)
            ]
            progress = tqdm(
                total=len(tasks), desc="[Math Eval] Generating", unit="req"
            )
            try:
                for task in asyncio.as_completed(tasks):
                    await task
                    progress.update(1)
            finally:
                progress.close()

        return [result or [""] * self.n_samples for result in results]

    def _save_results(
        self,
        global_step: int,
        metrics: Dict[str, float],
        results: List[Dict[str, Any]],
        num_problems: int,
        pass_at_k: float,
    ) -> None:
        eval_dir = os.path.join(self.args.train.save_path, "eval_results")
        os.makedirs(eval_dir, exist_ok=True)

        result_path = os.path.join(eval_dir, f"step{global_step}.jsonl")
        with open(result_path, "w", encoding="utf-8") as file:
            for result in results:
                file.write(json.dumps(result, ensure_ascii=False) + "\n")

        summary = {
            "global_step": global_step,
            "eval_dataset": self.dataset_path,
            "n_samples": self.n_samples,
            "num_problems": num_problems,
            "num_total_responses": int(metrics["eval/num_total"]),
            "num_correct": int(metrics["eval/num_correct"]),
            "accuracy": metrics["eval/accuracy"],
            "pass_at_k": pass_at_k,
        }
        summary_path = os.path.join(
            eval_dir, f"step{global_step}_summary.json"
        )
        with open(summary_path, "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from random import Random
import re
from threading import Lock
from typing import Callable, Iterable

from jinja2 import Environment, TemplateSyntaxError, meta
from litellm import Router
from tqdm import tqdm

LLMMessage = dict[str, str]
MetricFn = Callable[[str, dict], tuple[float, str]]
DataPoint = dict


@dataclass(frozen=True)
class PromptCandidate:
    prompt_key: str
    test_metrics: list[float]
    test_feedback: list[str]


@dataclass(frozen=True)
class MiniBatchResult:
    inputs: dict
    prediction: str
    reward: float
    feedback: str


class PromptOptimizer:
    PROMPT_DIR = Path(__file__).resolve().parents[1] / "optimizer_prompts"
    REFLECTION_RETRY_MESSAGE = (
        "Invalid, try again. Return only the corrected prompt inside "
        "<improved_prompt>...</improved_prompt>."
    )

    def __init__(
        self,
        seed_prompt: str,
        router: Router,
        train_data: list[DataPoint],
        test_data: list[DataPoint],
        metric: MetricFn,
        compulsory_input_keys: list[str],
        student_model: str = "student",
        reflection_model: str = "reflection",
        student_max_tokens: int | None = None,
        reflection_max_tokens: int | None = None,
        minibatch_size: int = 4,
        num_threads: int = 4,
        max_rollouts: int | None = 5,
        llm_call_budget: int | None = 500,
        max_reflection_retries: int = 3,
        random_seed: int | None = None,
    ) -> None:
        if minibatch_size <= 0:
            raise ValueError("minibatch_size must be positive")
        if max_rollouts is None and llm_call_budget is None:
            raise ValueError("max_rollouts or llm_call_budget must be provided")
        if max_rollouts is not None and max_rollouts <= 0:
            raise ValueError("max_rollouts must be positive")
        if llm_call_budget is not None and llm_call_budget <= 0:
            raise ValueError("llm_call_budget must be positive")
        if max_reflection_retries <= 0:
            raise ValueError("max_reflection_retries must be positive")
        if num_threads <= 0:
            raise ValueError("num_threads must be positive")
        router_model_names = {item["model_name"] for item in router.model_list}
        missing_models = {
            model_name
            for model_name in (student_model, reflection_model)
            if model_name not in router_model_names
        }
        if missing_models:
            raise ValueError(
                "Router is missing required model declarations: "
                f"{sorted(missing_models)}"
            )

        self._prompts: dict[str, str] = {"P0": seed_prompt}
        self._prompt_counter = 1
        self._seed_prompt_key = "P0"
        self._router = router
        self._student_model = student_model
        self._reflection_model = reflection_model
        self._student_max_tokens = student_max_tokens
        self._reflection_max_tokens = reflection_max_tokens
        self._train_data = train_data
        self._test_data = test_data
        self._metric = metric
        self._compulsory_input_keys = set(compulsory_input_keys)
        self._minibatch_size = minibatch_size
        self._num_threads = num_threads
        self._max_rollouts = max_rollouts
        self._llm_call_budget = llm_call_budget
        self._max_reflection_retries = max_reflection_retries
        self._random = Random(random_seed)

        self._candidates: list[PromptCandidate] = []
        self._candidate_by_key: dict[str, PromptCandidate] = {}
        self._perf_matrix: dict[str, list[tuple[float, str]]] = {}
        self._best_prompt_key: str | None = None
        self._pruned_prompt_keys: set[str] = set()
        self._llm_calls = 0
        self._llm_lock = Lock()

        self._reflection_system_template = self._load_prompt("reflection_system.jinja")
        self._reflection_context_template = self._load_prompt(
            "reflection_context.jinja"
        )

    @property
    def candidates(self) -> list[PromptCandidate]:
        return list(self._candidates)

    @property
    def perf_matrix(self) -> dict[str, list[tuple[float, str]]]:
        return {
            prompt_index: list(entries)
            for prompt_index, entries in self._perf_matrix.items()
        }

    @property
    def best_prompt(self) -> str:
        if self._best_prompt_key is None:
            raise RuntimeError("Optimizer has not been run yet")
        return self._prompt_by_key(self._best_prompt_key)

    @property
    def best_prompt_key(self) -> str:
        if self._best_prompt_key is None:
            raise RuntimeError("Optimizer has not been run yet")
        return self._best_prompt_key

    @property
    def pruned_prompt_keys(self) -> list[str]:
        return sorted(self._pruned_prompt_keys)

    def run(self) -> None:
        self._evaluate_and_store(self._seed_prompt_key)
        seed_prompt_key = self._seed_prompt_key
        rollout_count = 0

        while True:
            if self._max_rollouts is not None and rollout_count >= self._max_rollouts:
                break
            if (
                self._llm_call_budget is not None
                and self._llm_calls >= self._llm_call_budget
            ):
                break

            total_batches = max(
                1,
                (len(self._train_data) + self._minibatch_size - 1)
                // self._minibatch_size,
            )
            for start in tqdm(
                range(0, len(self._train_data), self._minibatch_size),
                desc=f"Rollout {rollout_count + 1}",
                total=total_batches,
            ):
                minibatch = self._train_data[start : start + self._minibatch_size]
                if (
                    self._llm_call_budget is not None
                    and self._llm_calls >= self._llm_call_budget
                ):
                    break

                seed_template = self._prompt_by_key(seed_prompt_key)
                seed_results = self._evaluate_minibatch_template(
                    seed_template, minibatch
                )
                seed_score = self._average([result.reward for result in seed_results])

                candidate_prompt = self._reflect(seed_prompt_key, seed_results)
                if candidate_prompt is None:
                    continue
                print(
                    f"Proposed prompt (seed {seed_prompt_key}):\n{candidate_prompt}\n"
                )

                candidate_results = self._evaluate_minibatch_template(
                    candidate_prompt, minibatch
                )
                candidate_score = self._average(
                    [result.reward for result in candidate_results]
                )

                if candidate_score <= seed_score:
                    continue

                prompt_key = self._store_prompt(candidate_prompt)
                self._evaluate_and_store(prompt_key)
                candidate = self._candidate_by_key[prompt_key]
                test_average = self._average(candidate.test_metrics)
                print(
                    "Accepted prompt "
                    f"{prompt_key} (outperformed seed {seed_prompt_key}). "
                    f"Test average: {test_average:.4f}."
                )

            pruned_ids = self._pareto_prune()
            seed_prompt_key = self._sample_seed_prompt(pruned_ids)

            if seed_prompt_key is None:
                break

            rollout_count += 1

    def _render_prompt(self, template: str, inputs: dict) -> str:
        env = Environment(autoescape=False)
        valid, error, variables = self._check_template(template, env)
        if not valid:
            if error is not None:
                line_info = f" line {error.lineno}" if error.lineno is not None else ""
                raise ValueError(f"Template syntax error: {error.message}{line_info}.")
            missing = sorted(self._compulsory_input_keys - variables)
            extra = sorted(variables - self._compulsory_input_keys)
            raise ValueError(
                "Template variables do not match compulsory input keys. "
                f"Missing: {missing}. Extra: {extra}."
            )
        return env.from_string(template).render(inputs)

    def _validate_template(self, template: str) -> bool:
        env = Environment(autoescape=False)
        valid, _, _ = self._check_template(template, env)
        return valid

    def _check_template(
        self, template: str, env: Environment
    ) -> tuple[bool, TemplateSyntaxError | None, set[str]]:
        try:
            parsed = env.parse(template)
        except TemplateSyntaxError as exc:
            return False, exc, set()
        variables = set(meta.find_undeclared_variables(parsed))
        return variables == self._compulsory_input_keys, None, variables

    def _evaluate_minibatch_template(
        self, template: str, minibatch: list[DataPoint]
    ) -> list[MiniBatchResult]:
        def run_item(datapoint: DataPoint) -> MiniBatchResult:
            prompt = self._render_prompt(template, datapoint["inputs"])
            with self._llm_lock:
                self._llm_calls += 1
            prediction = self._call_llm(
                self._student_model,
                [{"role": "user", "content": prompt}],
                self._student_max_tokens,
            )
            reward, feedback = self._metric(prediction, datapoint)
            return MiniBatchResult(
                inputs=datapoint["inputs"],
                prediction=prediction,
                reward=reward,
                feedback=feedback,
            )

        with ThreadPoolExecutor(max_workers=self._num_threads) as executor:
            return list(executor.map(run_item, minibatch))

    def _evaluate_and_store(self, prompt_key: str) -> None:
        if prompt_key in self._candidate_by_key:
            return
        template = self._prompt_by_key(prompt_key)
        metrics: list[float] = []
        feedbacks: list[str] = []

        def run_item(datapoint: DataPoint) -> tuple[float, str]:
            prompt = self._render_prompt(template, datapoint["inputs"])
            with self._llm_lock:
                self._llm_calls += 1
            prediction = self._call_llm(
                self._student_model,
                [{"role": "user", "content": prompt}],
                self._student_max_tokens,
            )
            return self._metric(prediction, datapoint)

        with ThreadPoolExecutor(max_workers=self._num_threads) as executor:
            for reward, feedback in executor.map(run_item, self._test_data):
                metrics.append(reward)
                feedbacks.append(feedback)

        candidate = PromptCandidate(
            prompt_key=prompt_key,
            test_metrics=metrics,
            test_feedback=feedbacks,
        )
        self._candidates.append(candidate)
        self._candidate_by_key[prompt_key] = candidate
        self._perf_matrix[prompt_key] = list(zip(metrics, feedbacks, strict=True))
        self._update_best_prompt(candidate)

    def candidate_average(self, prompt_key: str) -> float:
        if prompt_key not in self._candidate_by_key:
            raise KeyError(f"Unknown candidate key {prompt_key}")
        return self._average(self._candidate_by_key[prompt_key].test_metrics)

    def _update_best_prompt(self, candidate: PromptCandidate) -> None:
        if self._best_prompt_key is None:
            self._best_prompt_key = candidate.prompt_key
            return
        current = self._candidate_by_key[self._best_prompt_key]
        if self._average(current.test_metrics) < self._average(candidate.test_metrics):
            self._best_prompt_key = candidate.prompt_key

    def _prompt_by_key(self, prompt_key: str) -> str:
        try:
            return self._prompts[prompt_key]
        except KeyError as exc:
            raise KeyError(f"Unknown prompt key {prompt_key}") from exc

    def _average(self, values: Iterable[float]) -> float:
        values_list = list(values)
        if not values_list:
            return 0.0
        return sum(values_list) / len(values_list)

    def _reflect(
        self, seed_prompt_key: str, results: list[MiniBatchResult]
    ) -> str | None:
        seed_prompt = self._prompt_by_key(seed_prompt_key)
        lines: list[str] = []
        for idx, result in enumerate(results, start=1):
            lines.append(f"Example {idx}:")
            lines.append(f"Inputs: {result.inputs}")
            lines.append(f"Prediction: {result.prediction}")
            lines.append(f"Reward: {result.reward}")
            lines.append(f"Feedback: {result.feedback}")
        feedback_context = self._reflection_context_template.format(
            seed_prompt=seed_prompt,
            examples="\n".join(lines),
        )
        messages = [
            {"role": "system", "content": self._reflection_system_template},
            {"role": "user", "content": feedback_context},
        ]
        return self._extract_improved_prompt(messages)

    def _load_prompt(self, filename: str) -> str:
        prompt_path = self.PROMPT_DIR / filename
        return prompt_path.read_text(encoding="utf-8")

    def _extract_improved_prompt(self, messages: list[LLMMessage]) -> str | None:
        retries = 0

        while retries < self._max_reflection_retries:
            with self._llm_lock:
                self._llm_calls += 1
            content = self._call_llm(
                self._reflection_model,
                messages,
                self._reflection_max_tokens,
            )
            matches = re.findall(
                r"<improved_prompt>(.*?)</improved_prompt>",
                content,
                flags=re.DOTALL,
            )
            if matches:
                extracted = matches[-1].strip()
                if extracted:
                    env = Environment(autoescape=False)
                    valid, error, variables = self._check_template(extracted, env)
                    if valid:
                        return extracted
                    messages = [
                        *messages,
                        {
                            "role": "user",
                            "content": self._format_reflection_retry_message(
                                error=error,
                                variables=variables,
                            ),
                        },
                    ]
                    retries += 1
                    continue

            messages = [
                *messages,
                {"role": "user", "content": self.REFLECTION_RETRY_MESSAGE},
            ]
            retries += 1

        return None

    def _call_llm(
        self,
        model: str,
        messages: list[LLMMessage],
        max_tokens: int | None,
    ) -> str:
        kwargs = {"model": model, "messages": messages, "max_tokens": max_tokens}
        response = self._router.completion(
            **{key: value for key, value in kwargs.items() if value is not None}
        )
        return response.choices[0].message.content or ""

    def _format_reflection_retry_message(
        self,
        error: TemplateSyntaxError | None,
        variables: set[str],
    ) -> str:
        if error is not None:
            line_info = f" line {error.lineno}" if error.lineno is not None else ""
            details = f"Jinja syntax error: {error.message}{line_info}."
        else:
            missing = sorted(self._compulsory_input_keys - variables)
            extra = sorted(variables - self._compulsory_input_keys)
            details = (
                "Template variables must match compulsory input keys. "
                f"Missing: {missing}. Extra: {extra}."
            )
        return (
            "Invalid template. "
            f"{details} Return only the corrected prompt inside "
            "<improved_prompt>...</improved_prompt>."
        )

    def _pareto_prune(self) -> list[str]:
        remaining_ids = [candidate.prompt_key for candidate in self._candidates]
        dominated: set[str] = set()

        for i, prompt_a_id in enumerate(remaining_ids):
            if prompt_a_id in dominated:
                continue
            prompt_a_metrics = self._candidate_by_key[prompt_a_id].test_metrics
            for prompt_b_id in remaining_ids[i + 1 :]:
                if prompt_b_id in dominated:
                    continue
                prompt_b_metrics = self._candidate_by_key[prompt_b_id].test_metrics
                if self._dominates(prompt_a_metrics, prompt_b_metrics):
                    dominated.add(prompt_b_id)
                elif self._dominates(prompt_b_metrics, prompt_a_metrics):
                    dominated.add(prompt_a_id)
                    break

        self._pruned_prompt_keys = set(dominated)
        return [prompt_id for prompt_id in remaining_ids if prompt_id not in dominated]

    def _dominates(self, metrics_a: list[float], metrics_b: list[float]) -> bool:
        if len(metrics_a) != len(metrics_b):
            return False
        better_or_equal = all(a >= b for a, b in zip(metrics_a, metrics_b, strict=True))
        strictly_better = any(a > b for a, b in zip(metrics_a, metrics_b, strict=True))
        return better_or_equal and strictly_better

    def _sample_seed_prompt(self, candidate_ids: list[str]) -> str | None:
        candidate_ids = [
            pid for pid in candidate_ids if pid not in self._pruned_prompt_keys
        ]
        if not candidate_ids:
            return None
        weights = [
            self._average(self._candidate_by_key[pid].test_metrics)
            for pid in candidate_ids
        ]
        if not any(weight > 0 for weight in weights):
            return self._random.choice([pid for pid in candidate_ids])
        return self._random.choices(
            [pid for pid in candidate_ids],
            weights=weights,
            k=1,
        )[0]

    def _store_prompt(self, template: str) -> str:
        prompt_key = f"P{self._prompt_counter}"
        self._prompt_counter += 1
        self._prompts[prompt_key] = template
        return prompt_key

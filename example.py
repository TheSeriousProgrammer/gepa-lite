import ast
from os import getenv
import operator
import re

from dotenv import load_dotenv
from litellm import Router

from prompt_optimizer import PromptOptimizer


def safe_eval(expr: str) -> float:
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    def eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)
            return allowed_operators[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp):
            operand = eval_node(node.operand)
            return allowed_operators[type(node.op)](operand)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        raise TypeError("Unsupported expression")

    parsed = ast.parse(expr, mode="eval")
    return eval_node(parsed.body)


def extract_answer(content: str) -> float | None:
    matches = re.findall(r"<ans>(.*?)</ans>", content, flags=re.DOTALL)
    if not matches:
        return None
    try:
        return float(matches[-1].strip())
    except ValueError:
        return None


def metric(prediction: str, datapoint: dict) -> tuple[float, str]:
    predicted_value = extract_answer(prediction)
    if predicted_value is None:
        return 0.0, "Missing or invalid <ans> tag in prediction"

    expression = datapoint["inputs"]["input_1"]
    true_value = safe_eval(expression)
    epsilon = 1e-6
    percentage_diff = (
        abs(predicted_value - true_value) / max(abs(true_value), epsilon) * 100
    )
    reward = 100 * (80 - abs(min(80, percentage_diff))) / 80
    feedback = (
        f"predicted={predicted_value}, true={true_value}, diff={percentage_diff:.2f}%"
    )
    return reward, feedback


train_data = [
    {"inputs": {"input_1": "((3 + 5) * (7 - 2)) / (4 + 1)"}, "metadata": {}},
    {"inputs": {"input_1": "(12 - 3) * (2 + 5)"}, "metadata": {}},
    {"inputs": {"input_1": "(8 / 2) + (6 * (3 - 1))"}, "metadata": {}},
    {"inputs": {"input_1": "(2 ** (1 + 2)) + (7 * (6 - 3))"}, "metadata": {}},
    {"inputs": {"input_1": "(15 - 4) / (1 + 2)"}, "metadata": {}},
    {"inputs": {"input_1": "(9 * (2 + 3)) - (8 / 4)"}, "metadata": {}},
    {"inputs": {"input_1": "(5 + 7) * (3 - (1 + 1))"}, "metadata": {}},
    {"inputs": {"input_1": "(20 / (2 + 3)) + (4 ** 2)"}, "metadata": {}},
    {"inputs": {"input_1": "((6 - 2) * (5 + 1)) / 2"}, "metadata": {}},
    {"inputs": {"input_1": "(3 + 4) * (2 + (1 * 5))"}, "metadata": {}},
    {"inputs": {"input_1": "(14 + 6) / (5 - 3)"}, "metadata": {}},
    {"inputs": {"input_1": "(11 * (4 + 1)) - (9 / 3)"}, "metadata": {}},
    {"inputs": {"input_1": "(7 + 8) * (9 - 6)"}, "metadata": {}},
    {"inputs": {"input_1": "(16 / 4) + (3 ** 3)"}, "metadata": {}},
    {"inputs": {"input_1": "((12 - 5) * (8 + 2)) / 7"}, "metadata": {}},
    {"inputs": {"input_1": "(5 ** 2) + (18 / 6)"}, "metadata": {}},
    {"inputs": {"input_1": "(13 - 4) * (2 + 6)"}, "metadata": {}},
    {"inputs": {"input_1": "((9 + 3) * (5 - 2)) / 4"}, "metadata": {}},
    {"inputs": {"input_1": "(6 * (7 - 2)) + (10 / 5)"}, "metadata": {}},
    {"inputs": {"input_1": "(21 / 3) + (4 * (2 + 1))"}, "metadata": {}},
    {"inputs": {"input_1": "(8 + (6 * 2)) - (14 / 7)"}, "metadata": {}},
    {"inputs": {"input_1": "((5 + 9) / 2) * (7 - 3)"}, "metadata": {}},
    {"inputs": {"input_1": "(2 ** 4) - (3 * (5 - 1))"}, "metadata": {}},
    {"inputs": {"input_1": "(17 - 8) + (12 / 4)"}, "metadata": {}},
    {"inputs": {"input_1": "(4 * (6 + 2)) / (3 + 1)"}, "metadata": {}},
    {"inputs": {"input_1": "(19 + 1) - (9 - 2)"}, "metadata": {}},
    {"inputs": {"input_1": "(10 / 2) * (3 + 5)"}, "metadata": {}},
    {"inputs": {"input_1": "((8 + 4) * (7 - 5)) + 6"}, "metadata": {}},
    {"inputs": {"input_1": "(22 - 7) / (3 + 2)"}, "metadata": {}},
    {"inputs": {"input_1": "(6 ** 2) - (15 / 5)"}, "metadata": {}},
]

test_data = [
    {"inputs": {"input_1": "(10 + 2) * (7 - 4)"}, "metadata": {}},
    {"inputs": {"input_1": "(18 / 3) + (2 ** 3)"}, "metadata": {}},
    {"inputs": {"input_1": "((9 - 1) * (4 + 2)) / 3"}, "metadata": {}},
    {"inputs": {"input_1": "(5 * (3 + 1)) - (6 / 2)"}, "metadata": {}},
    {"inputs": {"input_1": "(2 + 3) ** 2 + (4 - 1)"}, "metadata": {}},
    {"inputs": {"input_1": "(14 - 5) * (6 + 1)"}, "metadata": {}},
    {"inputs": {"input_1": "(24 / 6) + (3 * (4 + 2))"}, "metadata": {}},
    {"inputs": {"input_1": "((10 - 2) * (5 + 3)) / 4"}, "metadata": {}},
    {"inputs": {"input_1": "(9 ** 2) - (8 * 4)"}, "metadata": {}},
    {"inputs": {"input_1": "(16 + 8) / (6 - 2)"}, "metadata": {}},
    {"inputs": {"input_1": "(7 * (5 + 1)) - (18 / 3)"}, "metadata": {}},
    {"inputs": {"input_1": "(3 + 9) * (8 - 5)"}, "metadata": {}},
    {"inputs": {"input_1": "((12 + 6) / 3) + (2 ** 4)"}, "metadata": {}},
    {"inputs": {"input_1": "(20 - 3) + (15 / 5)"}, "metadata": {}},
    {"inputs": {"input_1": "(4 * (7 + 2)) / (5 - 1)"}, "metadata": {}},
]

seed_prompt = (
    "Solve the expression and respond only with <ans>number</ans>.\n"
    "Expression: {{ input_1 }}"
)


def build_optimizer() -> PromptOptimizer:
    api_key = getenv("OPENROUTER_API_KEY")
    api_base = "https://openrouter.ai/api/v1"
    router = Router(
        model_list=[
            {
                "model_name": "student",
                "litellm_params": {
                    "model": "openrouter/meta-llama/llama-3.2-1b-instruct",
                    "api_key": api_key,
                    "api_base": api_base,
                },
            },
            {
                "model_name": "reflection",
                "litellm_params": {
                    "model": "openrouter/google/gemini-2.5-flash-lite",
                    "api_key": api_key,
                    "api_base": api_base,
                },
            },
        ]
    )

    return PromptOptimizer(
        seed_prompt=seed_prompt,
        router=router,
        student_max_tokens=1024,
        train_data=train_data,
        test_data=test_data,
        metric=metric,
        compulsory_input_keys=["input_1"],
        minibatch_size=5,
        num_threads=5,
        max_rollouts=1,
    )


load_dotenv()

optimizer = build_optimizer()
optimizer.run()
seed_key = "P0"
seed_average = optimizer.candidate_average(seed_key)
best_key = optimizer.best_prompt_key
best_average = optimizer.candidate_average(best_key)

color = {
    "title": "\033[1;96m",
    "header": "\033[1;91m",
    "prompt": "",
    "metric_label": "\033[1;93m",
    "metric_value": "\033[1;92m",
    "reset": "\033[0m",
}

print(
    "\n".join(
        [
            "",
            f"{color['title']}=== Prompt Optimization Report ==={color['reset']}",
            "",
            f"{color['header']}Seed Prompt ({seed_key}){color['reset']}",
            f"{seed_prompt}",
            f"{color['metric_label']}Test average: {color['metric_value']}"
            f"{seed_average:.4f}{color['reset']}",
            "",
            f"{color['header']}Best Prompt ({best_key}){color['reset']}",
            f"{optimizer.best_prompt}",
            f"{color['metric_label']}Test average: {color['metric_value']}"
            f"{best_average:.4f}{color['reset']}",
        ]
    )
)

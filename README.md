# gepa-lite

`gepa-lite` is a small prompt optimizer inspired by GEPA-style reflection. It evaluates a seed prompt, asks a reflection model to propose improved prompt templates, keeps candidates that improve minibatch performance, and tracks the best prompt on a held-out test set.

The optimizer uses LiteLLM's `Router`. Declare two router model names by default:

- `student`: model used to answer task prompts
- `reflection`: model used to rewrite prompts from feedback

## Install

```bash
uv sync
```

## Configure

Create a local `.env` with your provider key:

```bash
OPENROUTER_API_KEY=...
```

The included example uses OpenRouter models through LiteLLM.

## Usage

```python
from os import getenv

from litellm import Router
from prompt_optimizer import PromptOptimizer

router = Router(
    model_list=[
        {
            "model_name": "student",
            "litellm_params": {
                "model": "openrouter/meta-llama/llama-3.2-1b-instruct",
                "api_key": getenv("OPENROUTER_API_KEY"),
                "api_base": "https://openrouter.ai/api/v1",
            },
        },
        {
            "model_name": "reflection",
            "litellm_params": {
                "model": "openrouter/google/gemini-2.5-flash-lite",
                "api_key": getenv("OPENROUTER_API_KEY"),
                "api_base": "https://openrouter.ai/api/v1",
            },
        },
    ]
)

optimizer = PromptOptimizer(
    seed_prompt="Solve {{ input_1 }} and return <ans>number</ans>.",
    router=router,
    train_data=train_data,
    test_data=test_data,
    metric=metric,
    compulsory_input_keys=["input_1"],
)
optimizer.run()
print(optimizer.best_prompt)
```

`metric` receives the model prediction as a plain string and the original datapoint:

```python
def metric(prediction: str, datapoint: dict) -> tuple[float, str]:
    return reward, feedback
```

Prompt templates are Jinja templates. The optimizer validates that every prompt candidate uses exactly the keys listed in `compulsory_input_keys`.

## Run The Example

```bash
uv run python example.py
```

The example performs one rollout to keep it usable as a smoke test. It still makes real LLM calls.

# Builder Guide: Recursive Self-Improving Agent

A practical blueprint for building an API-driven agent that rewrites its own instructions, generates its own tools, and improves across sessions — without frozen weights.

---

## What You Are Building

Most AI assistants are frozen. Their weights are fixed at training time. They can reason well but cannot grow.

This guide walks you through building a system that evolves in **prompt space** and **code space** — it rewrites its own operating instructions and generates tools it doesn't have yet. No GPU required. No fine-tuning. Just well-structured loops and persistent memory.

```
You provide:   a task, a seed prompt, and API keys
It provides:   a progressively better version of itself
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    OUTER LOOP (meta)                    │
│                                                         │
│   ┌─────────────────────────────────────────────────┐  │
│   │                INNER LOOP (task)                │  │
│   │                                                 │  │
│   │   Task ──► Generator ──► Output                 │  │
│   │                              │                  │  │
│   │                           Critic                │  │
│   │                              │                  │  │
│   │                          Reviser ◄──────────┐  │  │
│   │                              │               │  │  │
│   │                      Convergence check ──► loop  │  │
│   │                              │                  │  │
│   │                        Final output             │  │
│   └──────────────────────────┬──────────────────────┘  │
│                              │                          │
│                      Performance log                    │
│                              │                          │
│                       Meta-reflector                    │
│                              │                          │
│                  Updated system prompt ◄────────────────┘
└─────────────────────────────────────────────────────────┘
```

**Inner loop** — runs per task. Generator produces, Critic evaluates, Reviser improves. Repeats until output stabilizes or hits max iterations.

**Outer loop** — runs across tasks. After N completions, Meta-reflector analyzes patterns in what worked and rewrites the operating instructions for the next session.

---

## Components

### 1. Generator
The main reasoning model. Takes a task + system prompt and produces an output.

**Recommended:** Claude API (`claude-sonnet-4-6`), GPT-4o, or any capable chat model via API.

```python
def generate(task: str, system_prompt: str, client) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": task}]
    )
    return response.content[0].text
```

---

### 2. Critic
Evaluates the generator's output. **Critical design rule: use a different model or a different call context than the generator.** A model critiquing its own output in the same context will be sycophantic.

For an API-driven setup, the simplest approach is a separate API call with an explicit critic persona.

**Optional: use a smaller HF Inference API model as the judge for independence.**

```python
def critique(task: str, output: str, client) -> dict:
    prompt = f"""You are a strict evaluator. Given this task and response, identify flaws.

Task: {task}
Response: {output}

Return JSON with:
- score: 0-10
- flaws: list of specific problems
- suggestions: list of concrete improvements
- converged: true if score >= 8"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",  # smaller/cheaper for critic
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )
    return json.loads(response.content[0].text)
```

---

### 3. Reviser
Takes the original output + critic feedback and produces an improved version.

```python
def revise(task: str, output: str, critique: dict, system_prompt: str, client) -> str:
    revision_prompt = f"""Original task: {task}

Your previous response:
{output}

Critic feedback:
Flaws: {critique['flaws']}
Suggestions: {critique['suggestions']}

Produce an improved response addressing all identified flaws."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": revision_prompt}]
    )
    return response.content[0].text
```

---

### 4. Convergence Checker
Determines when the loop should stop. Two signals:
- **Critic score** — if score >= threshold, stop
- **Embedding similarity** — if revision is semantically close to previous output, stop (diminishing returns)

**HF resource:** `sentence-transformers/all-MiniLM-L6-v2` via the `sentence-transformers` library.

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def has_converged(prev_output: str, new_output: str, score: int,
                  score_threshold=8, similarity_threshold=0.97) -> bool:
    if score >= score_threshold:
        return True
    prev_emb = embedder.encode([prev_output])
    new_emb = embedder.encode([new_output])
    similarity = cosine_similarity(prev_emb, new_emb)[0][0]
    return float(similarity) >= similarity_threshold
```

---

### 5. Memory Layer
Persists task logs across sessions. The meta-reflector reads this to improve the system prompt.

```python
import json
from pathlib import Path
from datetime import datetime

MEMORY_FILE = "agent_memory.json"

def log_task(task: str, final_output: str, iterations: int, final_score: int):
    memory = load_memory()
    memory["tasks"].append({
        "timestamp": datetime.utcnow().isoformat(),
        "task": task,
        "final_output": final_output,
        "iterations": iterations,
        "final_score": final_score
    })
    Path(MEMORY_FILE).write_text(json.dumps(memory, indent=2))

def load_memory() -> dict:
    if not Path(MEMORY_FILE).exists():
        return {"tasks": [], "system_prompt_history": [], "current_system_prompt": SEED_PROMPT}
    return json.loads(Path(MEMORY_FILE).read_text())
```

---

### 6. Meta-Reflector
The outer loop. After N tasks, reads the performance log and rewrites the system prompt.

This is where the system becomes self-improving at the meta level.

```python
def meta_reflect(client) -> str:
    memory = load_memory()
    recent_tasks = memory["tasks"][-10:]  # last 10 tasks

    reflection_prompt = f"""You are analyzing an AI agent's recent performance to improve its operating instructions.

Recent task logs:
{json.dumps(recent_tasks, indent=2)}

Current system prompt:
{memory['current_system_prompt']}

Analyze patterns:
- What types of tasks scored low?
- What caused multiple revision iterations?
- What approaches consistently worked well?

Then rewrite the system prompt to address weaknesses and reinforce strengths.
Return ONLY the new system prompt text."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": reflection_prompt}]
    )

    new_prompt = response.content[0].text
    memory["system_prompt_history"].append(memory["current_system_prompt"])
    memory["current_system_prompt"] = new_prompt
    Path(MEMORY_FILE).write_text(json.dumps(memory, indent=2))
    return new_prompt
```

---

## The Inner Loop (Complete)

```python
MAX_ITERATIONS = 5

def run_task(task: str, client) -> dict:
    memory = load_memory()
    system_prompt = memory["current_system_prompt"]

    output = generate(task, system_prompt, client)
    iteration = 1

    while iteration < MAX_ITERATIONS:
        feedback = critique(task, output, client)

        if has_converged(output, output, feedback["score"]):
            break

        prev_output = output
        output = revise(task, output, feedback, system_prompt, client)
        iteration += 1

        if has_converged(prev_output, output, feedback["score"]):
            break

    log_task(task, output, iteration, feedback["score"])
    return {"output": output, "iterations": iteration, "score": feedback["score"]}
```

---

## The Outer Loop (Complete)

```python
META_REFLECT_EVERY = 10  # tasks

def run_agent(tasks: list[str], client):
    for i, task in enumerate(tasks):
        print(f"\n[Task {i+1}] {task[:60]}...")
        result = run_task(task, client)
        print(f"  Score: {result['score']}/10 | Iterations: {result['iterations']}")

        if (i + 1) % META_REFLECT_EVERY == 0:
            print("\n[Meta-reflection] Updating system prompt...")
            new_prompt = meta_reflect(client)
            print(f"  New prompt: {new_prompt[:100]}...")
```

---

## Seed Prompt

The starting system prompt before any self-improvement has occurred.

```python
SEED_PROMPT = """You are a careful, precise assistant. You think step by step.
You prioritize accuracy over speed. When uncertain, you say so explicitly.
You produce complete, well-structured responses."""
```

After several meta-reflection cycles, the agent will replace this with something more adapted to the tasks it has actually encountered.

---

## Adding the Code-Writing Loop

This is the step that makes the system generative in a deeper sense: the agent can write tools it doesn't have.

When a task requires a capability the agent lacks, it generates a Python function, executes it in a sandbox, and adds it to its tool library if it works.

```python
import subprocess
import tempfile

def generate_tool(capability_description: str, client) -> str:
    prompt = f"""Write a Python function that provides this capability:
{capability_description}

Requirements:
- Single self-contained function
- No external dependencies beyond stdlib and requests
- Include a brief docstring
- Return the function code only, no explanation"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

def test_tool_in_sandbox(code: str, test_input: str) -> tuple[bool, str]:
    """Execute generated code in isolated subprocess with timeout."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        f.write(f"\n\nif __name__ == '__main__':\n    print({test_input})")
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["python", tmp_path],
            capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0, result.stdout
    except subprocess.TimeoutExpired:
        return False, "timeout"

def add_to_tool_library(name: str, code: str):
    tools = load_tool_library()
    tools[name] = code
    Path("tool_library.json").write_text(json.dumps(tools, indent=2))

def load_tool_library() -> dict:
    if not Path("tool_library.json").exists():
        return {}
    return json.loads(Path("tool_library.json").read_text())
```

> **Warning:** Executing generated code is inherently risky. Always run in a sandbox (Docker container, subprocess with resource limits, or a platform with isolated execution like heyneo.so). Never run generated code with access to production credentials or file systems.

---

## Hugging Face Integration Points

| Role | Model | How to use |
|---|---|---|
| Convergence detection | `sentence-transformers/all-MiniLM-L6-v2` | `pip install sentence-transformers` |
| Independent judge | Any small chat model via HF Inference API | `huggingface_hub.InferenceClient` |
| Seed data for critic | `HuggingFaceH4/ultrafeedback_binarized` | Few-shot examples of good/bad outputs |

### HF Inference API judge example

```python
from huggingface_hub import InferenceClient

hf_client = InferenceClient(token="hf_...")

def hf_judge(output: str) -> float:
    """Use a small HF model as an independent judge."""
    result = hf_client.text_generation(
        f"Rate this response 0-10 for quality and accuracy. Reply with only a number.\n\nResponse: {output}",
        model="mistralai/Mistral-7B-Instruct-v0.3",
        max_new_tokens=5
    )
    try:
        return float(result.strip())
    except ValueError:
        return 5.0
```

---

## Failure Modes and Escape Hatches

This is the part most guides skip. Know these before you build.

### Reward hacking
The agent learns to satisfy the critic without actually improving. The critic becomes too easy to fool.

**Fix:** Rotate critic prompts. Occasionally use a completely different model as judge. Add a human-in-the-loop checkpoint every N outer loops.

### Mode collapse
The system prompt converges to something narrow and stops improving. The meta-reflector rewrites into a corner.

**Fix:** Keep the full `system_prompt_history`. Add a "diversity check" — if the last 3 prompts are too similar (cosine similarity > 0.95), inject a perturbation.

### Infinite loops
The convergence checker fails and the inner loop runs forever.

**Fix:** Hard cap on iterations (`MAX_ITERATIONS`). Log a warning when the cap is hit — this signals the critic threshold may be miscalibrated.

### Runaway tool generation
The code-writing loop generates tools that call each other recursively.

**Fix:** Whitelist allowed imports. Enforce subprocess timeout (5s). Never let generated tools write to the tool library themselves — only the outer agent can do that after human-readable inspection.

### Context bleed
Long-running sessions accumulate memory that biases the meta-reflector.

**Fix:** Use a sliding window (`recent_tasks[-10:]`). Periodically summarize old memory into a compressed "lessons learned" block rather than keeping raw logs.

---

## Minimum Viable Version (Start Here)

If you want to run this today with minimal setup:

1. **Skip the code-writing loop** — add it later once the core loop is stable
2. **Skip HF Inference judge** — use two calls to the same API with different personas
3. **Use a JSON file for memory** — no database needed
4. **Set `META_REFLECT_EVERY = 5`** — see the outer loop update quickly

```
pip install anthropic sentence-transformers scikit-learn
```

Set your API key, drop in the functions above, run with a list of 20 test tasks, and watch the system prompt evolve.

---

## What This Is Not

- **Not AGI.** This is a well-structured loop with persistent memory. It improves within the domain of tasks you give it.
- **Not weight training.** Nothing about the underlying model changes. The frozen model is still frozen — only the instructions it receives change.
- **Not autonomous.** The outer loop still runs on a schedule you control. It does not decide to run itself.

What it *is*: a foundation. The architecture above is the same one used in research systems like SELF and MARS (see references). The difference is those systems fine-tune; this one doesn't. That makes it weaker in some ways and much more accessible in others.

---

## References

- [Darwin Gödel Machine](https://hf.co/papers/2505.22954) — self-modifying agents via iterative self-improvement
- [SELF: Language-Driven Self-Evolution](https://hf.co/papers/2310.00533) — LLMs self-evolving via language feedback
- [MARS: Meta-cognitive Reflection](https://hf.co/papers/2601.11974) — reflective reasoning loops
- [Meta-Rewarding Language Models](https://hf.co/papers/2407.19594) — model judges its own judgments
- [Teaching Models to Teach Themselves](https://hf.co/papers/2601.18778) — self-generated curricula via meta-RL
- [Self-Referential Weight Matrix](https://hf.co/papers/2202.05780) — weights that learn to modify themselves

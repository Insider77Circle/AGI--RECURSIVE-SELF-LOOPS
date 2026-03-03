"""
Agent Alpha — Quality-first, comprehensive, accuracy-focused.

Alpha's strategy: cover all cases, be precise, never sacrifice correctness
for brevity. Alpha treats every task as if it will be audited.
In the adversarial loop, Alpha wins by being thorough and hard to fault.
"""

import json
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

AGENT_ID = "alpha"
MEMORY_FILE = "alpha_memory.json"
MAX_ITERATIONS = 5
SCORE_THRESHOLD = 8
SIMILARITY_THRESHOLD = 0.97

SEED_PROMPT = """You are a rigorous, comprehensive assistant.

Your priorities, in order:
1. Accuracy — never state something uncertain as fact
2. Completeness — cover edge cases and failure modes
3. Structure — organize information so it can be audited
4. Clarity — precise language, no ambiguity

When you answer, ask yourself: could a critic find a logical gap?
If yes, close it before responding."""

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ── Memory ──────────────────────────────────────────────────────────────────

def load_memory() -> dict:
    if not Path(MEMORY_FILE).exists():
        return {
            "agent_id": AGENT_ID,
            "tasks": [],
            "wins": 0,
            "losses": 0,
            "system_prompt_history": [],
            "current_system_prompt": SEED_PROMPT,
        }
    return json.loads(Path(MEMORY_FILE).read_text())


def save_memory(memory: dict):
    Path(MEMORY_FILE).write_text(json.dumps(memory, indent=2))


def log_task(task: str, final_output: str, iterations: int,
             final_score: int, won: bool | None = None):
    memory = load_memory()
    memory["tasks"].append({
        "timestamp": datetime.utcnow().isoformat(),
        "task": task,
        "final_output": final_output,
        "iterations": iterations,
        "final_score": final_score,
        "won": won,
    })
    if won is True:
        memory["wins"] += 1
    elif won is False:
        memory["losses"] += 1
    save_memory(memory)


# ── Core generation ──────────────────────────────────────────────────────────

def generate(task: str, system_prompt: str, client) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": task}],
    )
    return response.content[0].text


def critique_own(task: str, output: str, client) -> dict:
    """Alpha critiques its own output before seeing Beta's."""
    prompt = f"""You are a strict auditor reviewing this response for flaws.

Task: {task}
Response: {output}

Find every logical gap, unsupported claim, missing edge case, or ambiguity.
Be harsh — your job is to find problems.

Return JSON:
{{
  "score": <0-10>,
  "flaws": ["flaw 1", "flaw 2"],
  "suggestions": ["fix 1", "fix 2"],
  "converged": <true if score >= 8>
}}"""
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return json.loads(response.content[0].text)


def critique_opponent(task: str, beta_output: str, client) -> str:
    """Alpha critiques Beta's output in the adversarial round."""
    prompt = f"""You are reviewing a competitor's response to find weaknesses.

Task: {task}
Competitor's response: {beta_output}

Identify every flaw, gap, or mistake. Be specific and rigorous.
Your critique will be used to score the competitor."""
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def revise(task: str, output: str, feedback: dict,
           system_prompt: str, client) -> str:
    revision_prompt = f"""Original task: {task}

Your previous response:
{output}

Auditor feedback:
Flaws: {feedback['flaws']}
Suggestions: {feedback['suggestions']}

Rewrite the response. Fix every identified flaw.
Do not drop content — improve it."""
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": revision_prompt}],
    )
    return response.content[0].text


def has_converged(prev: str, new: str, score: int) -> bool:
    if score >= SCORE_THRESHOLD:
        return True
    prev_emb = embedder.encode([prev])
    new_emb = embedder.encode([new])
    sim = cosine_similarity(prev_emb, new_emb)[0][0]
    return float(sim) >= SIMILARITY_THRESHOLD


# ── Meta-reflection ──────────────────────────────────────────────────────────

def meta_reflect(client) -> str:
    """Rewrite Alpha's system prompt based on win/loss history."""
    memory = load_memory()
    recent = memory["tasks"][-15:]

    prompt = f"""You are optimizing an AI agent's operating strategy.

Agent stats: {memory['wins']} wins, {memory['losses']} losses
Recent performance:
{json.dumps(recent, indent=2)}

Current strategy:
{memory['current_system_prompt']}

Analyze:
- What types of tasks did the agent lose?
- What patterns appear in low-scoring outputs?
- What did winning outputs have in common?

Rewrite the strategy prompt to win more. Keep the agent's core identity
(rigorous, comprehensive) but sharpen its weaknesses.
Return ONLY the new strategy prompt."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    new_prompt = response.content[0].text
    memory["system_prompt_history"].append(memory["current_system_prompt"])
    memory["current_system_prompt"] = new_prompt
    save_memory(memory)
    return new_prompt


# ── Task runner ──────────────────────────────────────────────────────────────

def run_task(task: str, client) -> dict:
    """Run Alpha's inner loop on a single task."""
    memory = load_memory()
    system_prompt = memory["current_system_prompt"]

    output = generate(task, system_prompt, client)
    iteration = 1

    while iteration < MAX_ITERATIONS:
        feedback = critique_own(task, output, client)
        if has_converged(output, output, feedback["score"]):
            break
        prev = output
        output = revise(task, output, feedback, system_prompt, client)
        iteration += 1
        if has_converged(prev, output, feedback["score"]):
            break

    return {
        "agent": AGENT_ID,
        "output": output,
        "iterations": iteration,
        "score": feedback["score"],
    }

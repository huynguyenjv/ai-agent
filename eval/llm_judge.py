"""LLM-as-Judge — Phase 16.1.

Scores an AI response on multiple dimensions using a (capable) judge model.
The judge is injected as an async `complete(prompt) -> str` callable so this is
testable and model-agnostic.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, asdict
from typing import Awaitable, Callable

logger = logging.getLogger("eval.llm_judge")

JUDGE_PROMPT = """You are a strict evaluator. Rate the AI response on a 1-10 scale for:
1. correctness  — does it actually solve the task?
2. completeness — is anything important missing?
3. quality      — is the code/answer clean and idiomatic?
4. clarity      — is the explanation clear?

Task:
{task}

Response:
{response}

Output ONLY JSON: {{"correctness": X, "completeness": X, "quality": X, "clarity": X, "overall": X}}
"""


@dataclass
class JudgeResult:
    correctness: float = 0.0
    completeness: float = 0.0
    quality: float = 0.0
    clarity: float = 0.0
    overall: float = 0.0
    raw: str = ""
    parsed: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _extract_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def parse_judge_output(text: str) -> JudgeResult:
    data = _extract_json(text or "")
    if not data:
        return JudgeResult(raw=text or "", parsed=False)

    def num(key: str) -> float:
        try:
            return float(data.get(key, 0))
        except (TypeError, ValueError):
            return 0.0

    overall = num("overall")
    if not overall:
        # derive if the judge omitted it
        parts = [num("correctness"), num("completeness"), num("quality"), num("clarity")]
        overall = round(sum(parts) / 4, 2) if any(parts) else 0.0

    return JudgeResult(
        correctness=num("correctness"),
        completeness=num("completeness"),
        quality=num("quality"),
        clarity=num("clarity"),
        overall=overall,
        raw=text or "",
        parsed=True,
    )


async def judge_response(
    task: str,
    response: str,
    complete: Callable[[str], Awaitable[str]],
) -> JudgeResult:
    """Judge a single response. `complete` runs the judge model on a prompt."""
    prompt = JUDGE_PROMPT.format(task=task, response=response)
    try:
        out = await complete(prompt)
    except Exception as e:
        logger.error("judge failed: %s", e)
        return JudgeResult(raw=str(e), parsed=False)
    return parse_judge_output(out)


async def judge_batch(
    cases: list[dict],
    complete: Callable[[str], Awaitable[str]],
) -> dict:
    """Judge a batch of {task, response} cases. Returns results + averages."""
    results: list[JudgeResult] = []
    for c in cases:
        results.append(await judge_response(c.get("task", ""), c.get("response", ""), complete))

    scored = [r for r in results if r.parsed]
    avg = round(sum(r.overall for r in scored) / len(scored), 2) if scored else 0.0
    return {
        "count": len(results),
        "scored": len(scored),
        "avg_overall": avg,
        "results": [r.to_dict() for r in results],
    }

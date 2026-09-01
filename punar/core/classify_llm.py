"""Optional LLM variant of the reason classifier in core/classify.py.

Design
------
What gets an LLM variant: only the reason-classification step (the
rule-based `classify_detailed` in core/classify.py). Nothing downstream of
that changes -- guardrails, retry budgets and the taxonomy stay rule-based
no matter which classifier diagnosed the decline.

Features in: the same signals the rule-based classifier already reads --
decline error_code, the high-level error_description/message, and payment
method. No amount, no customer identifiers, no free-text customer messages
are sent to the model.

Output: (reason_key, confidence) where reason_key is one of the 11
punar.core.taxonomy.REASONS keys and confidence is a float in [0, 1].

This module is never imported by core/classify.py and has zero import-time
side effects. The `groq` client is imported lazily, inside the function
that needs it, so the package (and its default classifier) import fine with
`groq` absent from the environment. The real path only ever runs when BOTH
`USE_LLM_DIAGNOSIS=1` and `GROQ_API_KEY` are set; anything short of that
raises `LLMUnavailable` rather than touching the network, and callers that
want a result regardless should use `classify_llm_or_mock`, which falls
back to the deterministic mock below.
"""
import hashlib
import os
from typing import Any

from punar.core.classify import _fields, classify_detailed
from punar.core.taxonomy import reason_labels

_VALID_REASONS = set(reason_labels())


class LLMUnavailable(Exception):
    """Raised when the real LLM path is requested but cannot run.

    Covers: USE_LLM_DIAGNOSIS unset, GROQ_API_KEY unset, the `groq` package
    missing, or the API call/response failing. Callers (the eval script,
    tests) catch this and fall back rather than crash.
    """


def _prompt_fields(failure: dict[str, Any]) -> tuple[str, str, str]:
    """Reuse the rule-based classifier's field extraction (same inputs, no
    second parser to keep in sync)."""
    error_code, blob, method, _reason, _notes, _src = _fields(failure)
    return error_code, blob, method


def _build_prompt(error_code: str, description: str, method: str) -> str:
    reasons = ", ".join(reason_labels())
    return (
        "Classify this failed payment into exactly one of these reason codes: "
        f"{reasons}.\n"
        f"error_code: {error_code}\ndescription: {description}\nmethod: {method}\n"
        'Reply with strict JSON: {"reason": "<one of the codes above>", '
        '"confidence": <0..1 float>}. No other text.'
    )


def _parse_llm_response(text: str) -> tuple[str, float]:
    import json

    try:
        data = json.loads(text)
        reason = str(data["reason"])
        confidence = float(data["confidence"])
    except Exception as exc:  # malformed JSON, missing keys, bad types
        raise LLMUnavailable(f"could not parse LLM response: {text!r}") from exc
    if reason not in _VALID_REASONS:
        # ponytail: clamp an out-of-taxonomy answer to the low-confidence
        # catch-all rather than raising -- an LLM hallucinating a plausible
        # but wrong label is expected, not exceptional. Revisit if the real
        # path ever ships: log the raw label so drift is visible.
        return "bank_decline_general", 0.0
    return reason, max(0.0, min(1.0, confidence))


def classify_llm(failure: dict[str, Any], *, model: str = "llama-3.1-8b-instant") -> tuple[str, float]:
    """Real LLM classification via Groq. Raises LLMUnavailable on any problem
    (missing gate env vars, missing package, network/API failure, bad
    response) -- never raises a raw groq/network exception to the caller.
    """
    if os.environ.get("USE_LLM_DIAGNOSIS") != "1":
        raise LLMUnavailable("USE_LLM_DIAGNOSIS is not set to 1")
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise LLMUnavailable("GROQ_API_KEY is not set")

    try:
        from groq import Groq  # lazy: keeps `groq` out of the default import graph
    except ImportError as exc:
        raise LLMUnavailable(f"groq package not installed: {exc}") from exc

    error_code, description, method = _prompt_fields(failure)
    prompt = _build_prompt(error_code, description, method)
    try:
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=60,
        )
        text = resp.choices[0].message.content or ""
    except Exception as exc:  # groq.APIError subclasses, timeouts, etc.
        raise LLMUnavailable(f"groq call failed: {exc}") from exc
    return _parse_llm_response(text)


def mock_classify_llm(failure: dict[str, Any], error_rate: float = 0.08) -> tuple[str, float]:
    """Deterministic stand-in for an LLM call. NOT a model -- a clearly
    labelled imitation so `scripts/llm_ablation.py` runs end to end with no
    API key. Seeded per-case so results are reproducible.

    # ponytail: real semantic reasoning replaced with "perturb the
    # rule-based answer at a fixed, seeded error rate". This is a shortcut
    # for exercising the eval plumbing, not a model of real LLM accuracy.
    # Upgrade path: point classify_llm_or_mock at a real GROQ_API_KEY, or
    # replace this stub with a small local model if offline eval numbers
    # ever need to mean something.
    """
    true_reason, _meta, _matched = classify_detailed(failure)
    salt = f"{failure.get('case_id', '')}|{true_reason}|mock"
    digest = int(hashlib.sha1(salt.encode()).hexdigest(), 16)
    roll = (digest % 10_000) / 10_000.0
    if roll >= error_rate:
        return true_reason, 0.75 + (digest % 21) / 100.0  # confident and right
    others = [r for r in reason_labels() if r != true_reason]
    wrong = others[digest % len(others)]
    return wrong, 0.40 + (digest % 21) / 100.0  # unconfident and wrong


def classify_llm_or_mock(failure: dict[str, Any]) -> tuple[str, float, bool]:
    """Return (reason, confidence, was_real). Tries the real path; falls
    back to the mock when it is unavailable. This is what
    scripts/llm_ablation.py calls -- it is why that script works with zero
    setup and clearly reports which path actually ran.
    """
    try:
        reason, confidence = classify_llm(failure)
        return reason, confidence, True
    except LLMUnavailable:
        reason, confidence = mock_classify_llm(failure)
        return reason, confidence, False


if __name__ == "__main__":
    # ponytail: assert-based self-check instead of a tests/test_*.py file --
    # this module has one branch worth checking (mock output shape) and the
    # repo's own style prefers the smaller option for a single-branch check.
    case = {"case_id": "case-demo-0001", "error": {"code": "INSUFFICIENT_FUNDS",
                                                    "description": "insufficient balance"}}
    reason, confidence = mock_classify_llm(case)
    assert reason in _VALID_REASONS, reason
    assert 0.0 <= confidence <= 1.0, confidence

    reason2, confidence2, was_real = classify_llm_or_mock(case)
    assert reason2 in _VALID_REASONS
    assert was_real is False, "no GROQ_API_KEY/USE_LLM_DIAGNOSIS set in this environment"

    try:
        classify_llm(case)
        raise AssertionError("classify_llm should have raised LLMUnavailable")
    except LLMUnavailable:
        pass

    print("classify_llm.py self-check OK")

"""Command-line interface for the text-to-SQL agent.

Usage:
    python -m app.cli "What is our Average Revenue per Membership?"
    semantic-agent "What was total revenue by region in Q2 2024?"

A thin wrapper around app.agents.text_to_sql_agent.answer_question -- no
business logic lives here, only argument parsing and result formatting.
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from app.agents.llm_client import LLMClient
from app.agents.text_to_sql_agent import answer_question

MAX_PRINTED_ROWS = 20


def main(argv: Optional[List[str]] = None, llm_client: Optional[LLMClient] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="semantic-agent",
        description="Ask a natural-language business question and get back validated, executed SQL.",
    )
    parser.add_argument("question", help="A natural-language question about revenue or subscribers.")
    parser.add_argument("--top-k", type=int, default=None, help="Number of context documents to retrieve.")
    parser.add_argument(
        "--max-attempts", type=int, default=None, help="Max LLM generation attempts before giving up."
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Also generate a one-sentence natural-language summary of the result (extra LLM call).",
    )
    args = parser.parse_args(argv)

    kwargs = {"llm_client": llm_client, "summarize": args.summarize}
    if args.top_k is not None:
        kwargs["top_k"] = args.top_k
    if args.max_attempts is not None:
        kwargs["max_attempts"] = args.max_attempts

    print(f"Question: {args.question}\n")

    try:
        result = answer_question(args.question, **kwargs)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print("See the Setup section in README.md.", file=sys.stderr)
        return 1

    attempt_word = "attempt" if result.attempts == 1 else "attempts"
    print(f"({result.attempts} {attempt_word})\n")

    if not result.succeeded:
        print(f"Could not answer this question: {result.error}")
        return 1

    print("SQL:")
    print(result.sql)
    print()

    df = result.result
    if len(df) > MAX_PRINTED_ROWS:
        print(f"Result ({len(df)} rows, showing first {MAX_PRINTED_ROWS}):")
        print(df.head(MAX_PRINTED_ROWS).to_string(index=False))
    else:
        print(f"Result ({len(df)} rows):")
        print(df.to_string(index=False))

    ci = result.confidence_interval
    if ci is not None:
        pct = int(ci.confidence_level * 100)
        print(
            f"\n{pct}% CI: [{ci.lower:.4f}, {ci.upper:.4f}] "
            f"(estimate={ci.estimate:.4f}, se={ci.standard_error:.4f}, n={ci.n_units} periods)"
        )

    if result.summary is not None:
        print(f"\nSummary: {result.summary}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

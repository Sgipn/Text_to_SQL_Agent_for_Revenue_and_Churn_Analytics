import unittest

import pandas as pd

from app.agents.result_summarization import MAX_SUMMARIZED_ROWS, summarize_result


class FakeLLMClient:
    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.calls: list = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self._responses.pop(0)


class RaisingLLMClient:
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise RuntimeError("boom")


class SummarizeResultTests(unittest.TestCase):
    def test_returns_stripped_summary_on_success(self) -> None:
        df = pd.DataFrame({"region_id": ["US", "EMEA"], "total_net_revenue": [100.0, 200.0]})
        client = FakeLLMClient(["  EMEA had higher revenue than the US.  "])

        summary = summarize_result("revenue by region?", df, client)

        self.assertEqual(summary, "EMEA had higher revenue than the US.")

    def test_includes_question_and_data_in_the_prompt(self) -> None:
        df = pd.DataFrame({"region_id": ["US"], "total_net_revenue": [100.0]})
        client = FakeLLMClient(["summary"])

        summarize_result("What was revenue in the US?", df, client)

        system_prompt, user_prompt = client.calls[0]
        self.assertIn("What was revenue in the US?", user_prompt)
        self.assertIn("100", user_prompt)
        self.assertIn("only", system_prompt.lower())  # grounding instruction present

    def test_returns_none_for_empty_result_without_calling_llm(self) -> None:
        df = pd.DataFrame({"region_id": [], "total_net_revenue": []})
        client = FakeLLMClient(["should not be used"])

        summary = summarize_result("anything?", df, client)

        self.assertIsNone(summary)
        self.assertEqual(client.calls, [])

    def test_returns_none_rather_than_raising_on_llm_failure(self) -> None:
        df = pd.DataFrame({"region_id": ["US"], "total_net_revenue": [100.0]})

        summary = summarize_result("anything?", df, RaisingLLMClient())

        self.assertIsNone(summary)

    def test_notes_truncation_for_large_result_sets(self) -> None:
        df = pd.DataFrame({"x": range(MAX_SUMMARIZED_ROWS + 5)})
        client = FakeLLMClient(["summary"])

        summarize_result("anything?", df, client)

        _, user_prompt = client.calls[0]
        self.assertIn(f"first {MAX_SUMMARIZED_ROWS}", user_prompt)
        self.assertIn(str(MAX_SUMMARIZED_ROWS + 5), user_prompt)


if __name__ == "__main__":
    unittest.main()

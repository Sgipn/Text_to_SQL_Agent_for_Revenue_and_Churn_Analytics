import unittest

from app.agents.sql_extraction import extract_sql, parse_llm_response


class SqlExtractionTests(unittest.TestCase):
    def test_extracts_from_fenced_block(self) -> None:
        text = "Here you go:\n```sql\nSELECT 1\n```\nThanks"
        self.assertEqual(extract_sql(text), "SELECT 1")

    def test_falls_back_to_raw_text_when_unfenced(self) -> None:
        text = "  SELECT 1  "
        self.assertEqual(extract_sql(text), "SELECT 1")

    def test_extracts_last_fenced_block_when_multiple_present(self) -> None:
        text = "```sql\nSELECT 1\n```\nOn second thought:\n```sql\nSELECT 2\n```"
        self.assertEqual(extract_sql(text), "SELECT 2")


class ParseLlmResponseTests(unittest.TestCase):
    def test_parses_sql_response(self) -> None:
        sql, decline_reason = parse_llm_response("```sql\nSELECT 1\n```")
        self.assertEqual(sql, "SELECT 1")
        self.assertIsNone(decline_reason)

    def test_parses_no_query_decline(self) -> None:
        sql, decline_reason = parse_llm_response(
            "NO_QUERY: Net Promoter Score is not a metric defined in the available semantic view."
        )
        self.assertIsNone(sql)
        self.assertEqual(
            decline_reason, "Net Promoter Score is not a metric defined in the available semantic view."
        )


if __name__ == "__main__":
    unittest.main()

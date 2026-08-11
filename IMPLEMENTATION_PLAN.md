# Project 1 Implementation Plan

## 1. Foundation and setup
- Create the project folder structure.
- Add initial configuration files for Python and dbt.
- Define the repository conventions and naming standards.

## 2. Data model and semantic layer
- Define the core business entities: subscriptions, revenue, plans, regions, and time.
- Create staging models for source data preparation.
- Create mart models for monthly revenue and subscriber activity.
- Define semantic metric specifications for ARM and related metrics.

## 3. Query safety and validation
- Add SQL parsing and AST validation.
- Restrict generated queries to approved semantic views.
- Block destructive operations and enforce safe execution patterns.

## 4. Retrieval and grounding layer
- Prepare schema, metric, and column metadata for retrieval.
- Build a vector store for semantic search over business context.
- Retrieve relevant context for natural-language prompts.

## 5. Agent orchestration
- Implement the natural-language to SQL workflow.
- Connect prompt context, retrieved metadata, and SQL validation.
- Test the flow with example business questions.

## 6. Testing and refinement
- Add unit tests for validation logic.
- Verify query generation for representative prompts.
- Refine prompt instructions and safety behavior.

## 7. Documentation refresh
- Rewrite the README to describe the six-phase architecture that was actually built, not the original initial plan.
- Document local setup: conda environment, dependency install, dbt build, vector store indexing, and `.env` configuration.
- Add a sample question and a walkthrough of the pipeline's output (generated SQL, validation outcome, result).

## 8. Continuous integration
- Add a GitHub Actions workflow that installs dependencies and runs the offline test suite on every push.
- Exclude the live-integration tests from CI, since they require a paid API key.
- Document the CI status in the README once the workflow is green.

## 9. Command-line demo interface
- Add a CLI entry point that takes a natural-language question as an argument.
- Print the generated SQL, validation/attempt outcome, and result table.
- Keep it a thin wrapper around the existing agent orchestration -- no new business logic.

## 10. Minimal API layer
- Wrap the agent orchestration in a small FastAPI app with a single ask endpoint.
- Return SQL, result rows, attempt count, and error (if any) as JSON.
- Add basic request validation and a health-check endpoint; document how to run it locally.

## 11. Ratio-metric confidence intervals
- Implement the Delta Method variance formula for ratio metrics, validated against a Monte Carlo simulation (empirical coverage of the nominal interval) before trusting the numbers.
- Use Student's t distribution for the critical value, not a fixed normal z-score -- caught on review that a WHERE filter narrowing a slice to a few periods would otherwise understate uncertainty right when the estimate is least reliable; added `scipy` as a dependency for this.
- Expose a function that computes a confidence interval for a ratio metric over a given slice of data, plus a ratio-metric registry (`app/services/ratio_metric_registry.py`, with a manifest drift-check test) so it's not ARM-specific.
- Detect which questions are "about" a ratio metric from the validated query's SELECT-list columns, not retrieval rank -- an initial retrieval-based version proved unreliable in live testing.
- Surface the interval alongside the point estimate for ratio-type metric questions, in both the CLI and API outputs, as best-effort enrichment (NaN-guarded, degrades to `None` on any failure rather than breaking the primary answer).

## 12. Semantic layer expansion
- Add a second mart, subscriber growth/churn (`fct_monthly_subscriber_activity`), with its own metrics -- including a second ratio metric, `monthly_churn_rate` -- and semantic-view tag. Validate the churn/signup window-function logic against real data before trusting it.
- Resolve a MetricFlow entity-collision error surfaced by adding a second semantic model (both marts had used `region_id` as a "primary" entity; give each model its own synthetic primary entity instead).
- Rebuild the vector store and add automated tests confirming retrieval disambiguates between the two domains -- this is the first point at which retrieval was large enough to meaningfully fail at that.
- Confirm the existing semantic-view and ratio-metric registry drift tests already generalize to multiple views/metrics with no code changes (they were written generically from the start).
- Audit and update prompt instructions, tests, and README text that used "churn" as the go-to example of an undefined metric, now that it's a real one (swapped to Net Promoter Score).

## 13. Column-level query scope validation
- Extend SQL validation to check that only documented columns for each approved view are referenced.
- Add adversarial tests for column-level violations, e.g. a column that doesn't exist on the approved view.

## 14. Natural-language result summarization
- Add an optional step that summarizes a returned result set in a sentence or two, grounded only in the actual returned rows.
- Keep this fully separate from SQL generation so the safety-critical path is unaffected.

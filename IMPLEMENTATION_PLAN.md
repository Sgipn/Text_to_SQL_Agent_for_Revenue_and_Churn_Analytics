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
- Implement the Delta Method variance formula for the ARM ratio estimator.
- Expose a function that computes a confidence interval for a ratio metric over a given slice of data.
- Surface the interval alongside the point estimate for ratio-type metric questions.

## 12. Semantic layer expansion
- Add a second mart (e.g. subscriber growth or cohort activity) with its own metrics and semantic-view tag.
- Rebuild the vector store and confirm retrieval still surfaces the correct view/metric pair per question.
- Extend the registry drift test to cover multiple approved views.

## 13. Column-level query scope validation
- Extend SQL validation to check that only documented columns for each approved view are referenced.
- Add adversarial tests for column-level violations, e.g. a column that doesn't exist on the approved view.

## 14. Natural-language result summarization
- Add an optional step that summarizes a returned result set in a sentence or two, grounded only in the actual returned rows.
- Keep this fully separate from SQL generation so the safety-critical path is unaffected.

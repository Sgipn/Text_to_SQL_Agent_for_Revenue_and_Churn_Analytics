# Text-to-SQL Implementation Plan

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
- Extend the semantic-view registry to carry each approved view's documented column set (not just its schema), and extend SQL validation to reject any referenced column not on one of the tables actually named in the query -- checked empirically against sqlglot's AST before deciding the approach, not assumed.
- Exclude the query's own SELECT-list aliases from that check (recursing into subqueries too) -- `ORDER BY arm` after `SUM(...) AS arm` parses identically to a real column reference, and without this exclusion the project's own existing ARM queries would have started failing their own new validation.
- Add adversarial tests for column-level violations: a hallucinated column, one only valid in a WHERE clause, and (now that Phase 12 added a second mart) a column that's real on the *other* approved view but not the one referenced.
- Extend the registry drift test to check documented columns against the manifest, not just schema/tags. Live-test a batch of representative questions end to end to confirm the stricter check doesn't cause false-positive retries against real Claude-generated SQL.
- Caught on review: matching was case-sensitive (table/schema and column names), but sqlglot preserves whatever case a query is written in while DuckDB folds unquoted identifiers case-insensitively -- a semantically valid, differently-cased query (e.g. `Total_Net_Revenue`) would have been wrongly rejected. Fixed for both table/schema and column matching (the table-name gap predated this phase but lives in the same function). Added regression tests and re-verified end to end.

## 14. Natural-language result summarization
- Add `app/agents/result_summarization.py`: an optional, opt-in second LLM call (`summarize=True` / `--summarize` / `"summarize": true`) that describes a successful result in one or two sentences, using its own prompt with the actual returned rows embedded as text (truncated and noted past a row cap, to bound cost on large results).
- Keep it fully separate from SQL generation -- a distinct prompt and LLM call with no ability to generate or modify SQL -- and best-effort like the confidence interval: wrapped so any failure degrades to `None` rather than affecting whether the question is considered answered.
- Live-test grounding quality (a soft, prompt-based constraint, not something AST validation can enforce) across several representative questions before trusting it. Every number in every generated summary matched the underlying table exactly; on a truncated 24-row churn result, the model correctly limited its claim to the 20 months it was actually shown rather than describing the full history -- codified as a permanent live regression test.
- Surface it in `AgentResult.summary`, the CLI (`--summarize`), and the API (`summarize` request field / `summary` response field).

## 15. Deployment readiness: `/ask` abuse protection
- All 14 original phases complete; this phase came out of discussing how to actually deploy the API somewhere public, which surfaces a problem the local-only dev loop never had to face: every `/ask` request is a real, billed Claude API call, reachable by anyone with the URL.
- Add two free, dependency-free layers to `app/api.py`: an optional shared `ASK_API_KEY` (unset by default, so local dev/tests are unaffected; once set, `/ask` requires a matching `X-API-Key` header) and an in-memory sliding-window rate limiter per client (API key if configured, else caller IP) -- acceptable for a single-instance deployment (e.g. Render's free tier), not meant to survive multiple instances behind a load balancer.
- Caught on review: `ASK_API_KEY` would have been read before `.env` was ever loaded into the process -- the existing `load_dotenv()` call only fires lazily inside the *first* successful LLM call, well after the abuse-protection dependency runs, so a deployer relying on `.env` alone (rather than a true OS/host env var) would have had protection silently disabled on exactly the requests that matter most. Fixed by loading `.env` eagerly at module import time in `app/api.py`.
- Verified over real HTTP, not just the in-process test client: a live `uvicorn` server correctly returned 401/401/200 for no-key, wrong-key, and correct-key requests. Added regression tests for both layers, careful to reset the rate limiter's in-memory state between tests -- `TestClient` always reports the same fake IP, so without resetting, unrelated tests earlier in a run would silently count against later ones' limit.
- Checked current pricing/free-tier terms for Render and Fly.io before recommending either (pricing drifts and shouldn't be answered from stale training data) -- Render's genuinely-free tier (no card required) won over Fly.io's mandatory card + small monthly cost, given the priority was fastest/simplest to get live.
- Write a `Dockerfile` that bakes the DuckDB warehouse and Chroma vector index in at *build* time (both fully reproducible from the checked-in synthetic data via a fixed seed), so the container needs no setup step and no network access at startup, only for the Claude API at request time. Add `render.yaml` for one-click, reproducible Render deploys.
- Verified for real, not just written and assumed: built the image with Docker, ran the container, and hit it over real HTTP -- `/health` responded, an unauthenticated `/ask` correctly 401'd, and an authenticated `/ask` returned a real Claude-generated answer with correct data and confidence interval, all served from the image's baked-in DuckDB/Chroma data.
- Deployed for real on Render, walked through the actual browser UI/CLI friction of doing so (blueprint form fields, finding the service URL, a PowerShell `curl` alias silently mangling a JSON body with escaped quotes -- fixed with the `--%` stop-parsing token), not just the happy-path commands.
- Followed up with a minimal browser UI at `/` (plain HTML/CSS/vanilla JS inlined directly into `app/api.py`, no build step, no new dependency, no separate static-file packaging concerns) once curl proved unpleasant for interactive use -- a question box, summarize checkbox, and an API-key field persisted in the browser's `localStorage`. Verified over real HTTP: the page serves correctly, and the exact fetch() pattern the page's JS uses against `/ask` was tested against a live server and returned a correct answer.

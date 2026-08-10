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

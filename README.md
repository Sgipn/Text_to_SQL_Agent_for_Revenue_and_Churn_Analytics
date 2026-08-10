# Semantic Metric Repository & Self-Service Text-to-SQL Agent

## Project Overview

The goal of this project is to make business metrics consistent, trustworthy, and accessible to non-technical users. Instead of relying on ad-hoc SQL and inconsistent metric definitions, the system will provide:

- a semantic layer for business metrics E.g. Average Revenue per Membership (ARM)
- validated metric definitions that prevent incorrect aggregation logic
- a text-to-SQL workflow grounded in schema and metric context
- safe query generation with structural validation before execution

## Core Objectives

1. Standardize key business metrics across dashboards and reports.
2. Allow users to ask natural-language questions about revenue and subscriber activity.
3. Ensure generated SQL is safe, constrained, and aligned with approved semantic views.
4. Support analytics workflows with DuckDB, dbt, vector retrieval, and LLM-based orchestration.

## Proposed Technology Stack

- Python 3.11+
- DuckDB for in-process analytical querying
- dbt-duckdb for semantic modeling and metric definitions
- ChromaDB for vector-based schema and metric retrieval
- sqlglot for SQL parsing and safety validation
- Claude API or similar LLM provider for query generation

## Repository Scope

This repository will eventually include:

- metric definitions and semantic models
- SQL models and dbt transformations
- a retrieval-augmented generation pipeline for schema grounding
- a query orchestration layer for natural-language requests
- validation and safety checks for generated SQL

## Initial Execution Plan

- Define the semantic metric repository structure.
- Implement foundational dbt models for revenue and subscription metrics.
- Create the retrieval layer for schema and metric context.
- Build the text-to-SQL agent with safety validation.
- Test the workflow against example business questions.

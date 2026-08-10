{{ config(materialized='table') }}

-- Required by MetricFlow for ratio/cumulative metric time-window logic.
-- https://docs.getdbt.com/docs/build/metricflow-time-spine
select cast(unnest(generate_series(
    cast('2020-01-01' as date),
    cast('2030-01-01' as date),
    interval 1 day
)) as date) as date_day

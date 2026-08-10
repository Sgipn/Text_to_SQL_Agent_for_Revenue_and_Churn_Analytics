{{ config(schema='semantic_views', tags=['semantic_view'], materialized='table') }}

-- Note: ARM is intentionally omitted as a stored pre-computed ratio column
-- to prevent illegal downstream AVG() operations on non-additive metrics.
-- schema='semantic_views' (see macros/generate_schema_name.sql) makes this
-- the physical, query-safety-enforced boundary the text-to-SQL agent is
-- restricted to -- not just an app-level string check.
with base_subscriptions as (

    select
        user_id, region_id, plan_type,
        billing_period_start, billing_period_end,
        net_revenue_usd, is_paid_tier
    from {{ ref('stg_subscriptions') }}

)

select
    date_trunc('month', billing_period_start) as metric_month,
    region_id,
    plan_type,
    count(distinct case when is_paid_tier then user_id end) as active_paid_subscribers,
    sum(net_revenue_usd) as total_net_revenue

from base_subscriptions
group by 1, 2, 3

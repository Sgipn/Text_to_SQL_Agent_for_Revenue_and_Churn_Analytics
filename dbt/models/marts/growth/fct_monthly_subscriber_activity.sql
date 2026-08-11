{{ config(schema='semantic_views', tags=['semantic_view'], materialized='table') }}

-- Grain: one row per (metric_month, region_id). new_subscribers counts a
-- user in the month their first-ever billing period starts; churned_subscribers
-- counts a user in the month their last-ever billing period ends, EXCLUDING
-- the final month of the whole dataset -- a user still active at the data
-- cutoff is right-censored (we don't know if/when they'll actually churn),
-- not a churn event, and must not be counted as one.
--
-- Assumes each user has one continuous tenure (min/max active month, no
-- gaps) -- true of app.utils.generate_synthetic_data today, which never
-- models a churned user re-subscribing. If win-back subscribers are ever
-- added to the source data, this logic will silently miss the gap and
-- undercount both events for that user; it would need streak-detection
-- (e.g. a running gap counter) instead of a plain min/max per user.
with subscriber_months as (

    select
        user_id,
        region_id,
        date_trunc('month', billing_period_start) as metric_month
    from {{ ref('stg_subscriptions') }}
    group by 1, 2, 3

),

subscriber_lifecycle as (

    select
        user_id,
        region_id,
        metric_month,
        min(metric_month) over (partition by user_id) as first_active_month,
        max(metric_month) over (partition by user_id) as last_active_month
    from subscriber_months

),

max_month_in_data as (

    select max(metric_month) as max_month from subscriber_months

)

select
    l.metric_month,
    l.region_id,
    count(distinct l.user_id) as active_subscribers,
    count(distinct case when l.metric_month = l.first_active_month then l.user_id end) as new_subscribers,
    count(distinct case
        when l.metric_month = l.last_active_month and l.metric_month < m.max_month
        then l.user_id
    end) as churned_subscribers

from subscriber_lifecycle l
cross join max_month_in_data m
group by 1, 2

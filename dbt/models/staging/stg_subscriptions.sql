with source as (

    select * from {{ source('raw', 'subscriptions') }}

)

select
    user_id,
    region_id,
    plan_type,
    cast(billing_period_start as date) as billing_period_start,
    cast(billing_period_end as date) as billing_period_end,
    net_revenue_usd,
    is_paid_tier

from source

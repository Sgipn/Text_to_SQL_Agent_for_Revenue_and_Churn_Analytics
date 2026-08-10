{% macro generate_schema_name(custom_schema_name, node) -%}
    {#- Marts set +schema explicitly (e.g. 'semantic_views') to become that
       schema exactly, rather than dbt's default '<target_schema>_<custom>'
       concatenation, so the physical DuckDB schema matches the namespace
       the query validator and generated SQL both expect. -#}
    {%- if custom_schema_name is none -%}
        {{ target.schema }}
    {%- else -%}
        {{ custom_schema_name | trim }}
    {%- endif -%}
{%- endmacro %}

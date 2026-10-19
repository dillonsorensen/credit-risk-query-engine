"""
Northbridge Bank - Credit Risk Query Engine (Streamlit front-end)

Natural-language questions -> verified SQL template OR generated read-only SQL -> five-check validation gate
-> (one retry) -> execution on a read-only SQLite connection -> narrative, chart, confidence and audit log.
"""
import os
import re
import json
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # headless backend for the server
import matplotlib.pyplot as plt
import sqlparse
import streamlit as st
from langchain_openai import ChatOpenAI

st.set_page_config(page_title="Credit Risk Query Engine", page_icon="🏦", layout="wide")

# ---------------------------------------------------------------- credentials & models
def _secret(name, default=None):
    """Read from Streamlit secrets first, then environment variables."""
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name, default)

OPENAI_API_KEY = _secret("OPENAI_API_KEY")
OPENAI_API_BASE = _secret("OPENAI_API_BASE")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not configured. Add it under App settings -> Secrets (see README).")
    st.stop()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if OPENAI_API_BASE:
    os.environ["OPENAI_BASE_URL"] = OPENAI_API_BASE

GENERATION_MODEL = _secret("GENERATION_MODEL", "gpt-4o")
EVALUATION_MODEL = _secret("EVALUATION_MODEL", "gpt-4o-mini")
llm = ChatOpenAI(model=GENERATION_MODEL, temperature=0)
evaluator_llm = ChatOpenAI(model=EVALUATION_MODEL, temperature=0)

# ---------------------------------------------------------------- database (read-only)
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "credit_risk_portfolio.db")
conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)     # new connection per script run (thread-safe)


# ---------------------------------------------------------------- schema, templates, library
database_schema = """
sector_master:
  sector_code (TEXT, PK): internal sector identifier (e.g., SEC_RE, SEC_INFRA)
  sector_name (TEXT): human-readable sector name (e.g., Real Estate, Infrastructure)
  naics_code (TEXT): NAICS industry classification code
  naics_description (TEXT): NAICS code description
  is_sensitive_sector (INTEGER): 1 if sensitive sector, 0 otherwise

loan_master:
  loan_account_number (TEXT, PK): unique loan identifier
  borrower_id (TEXT): borrower identifier (joins to borrower_rating.borrower_id)
  borrower_name (TEXT): registered legal name of the borrower
  borrower_type (TEXT): entity type (C-Corporation, S-Corporation, LLC, LP, Partnership, Sole Proprietorship)
  group_name (TEXT): business group affiliation, NULL if standalone
  state (TEXT): state of registered office
  product_type (TEXT): Term Loan, Working Capital, Cash Credit, Overdraft, Bill Discounting, Letter of Credit
  loan_category (TEXT): Corporate, Mid-Corporate, SME
  sector_code (TEXT, FK): joins to sector_master.sector_code
  sanctioned_amount (REAL): original approved loan amount in USD
  disbursed_amount (REAL): total amount disbursed in USD
  outstanding_principal (REAL): current principal outstanding in USD
  outstanding_interest (REAL): accrued interest outstanding in USD
  total_outstanding (REAL): outstanding_principal + outstanding_interest in USD
  interest_rate (REAL): current interest rate as percentage
  rate_type (TEXT): Fixed, Floating, MCLR-linked, Repo-linked
  sanction_date (DATE): date of original sanction
  maturity_date (DATE): contractual maturity date
  repayment_frequency (TEXT): Monthly, Quarterly, Bullet
  branch_code (TEXT): originating branch identifier
  branch_name (TEXT): originating branch name
  relationship_manager (TEXT): assigned relationship manager name
  is_consortium (INTEGER): 1 if consortium loan, 0 otherwise
  is_restructured (INTEGER): 1 if restructured, 0 otherwise
  restructuring_date (DATE): date of last restructuring, NULL if not restructured
  is_secured (INTEGER): 1 if secured, 0 if unsecured
  days_past_due (INTEGER): current maximum days past due for the loan
  asset_classification (TEXT): Pass, Special Mention, Substandard, Doubtful, Loss
  classification_date (DATE): date current classification was assigned

borrower_rating:
  rating_id (INTEGER, PK): auto-increment identifier
  borrower_id (TEXT, FK): joins to loan_master.borrower_id
  rating_date (DATE): date of rating assessment
  internal_rating (TEXT): bank's internal rating grade (AAA through D, 18-grade scale)
  previous_rating (TEXT): rating grade from prior assessment
  rating_direction (TEXT): Upgraded, Downgraded, Maintained
  external_rating_agency (TEXT): S&P, Moody's, Fitch, DBRS Morningstar, Kroll, or NULL
  external_rating (TEXT): external agency rating
  pd_estimate (REAL): probability of default (decimal, e.g., 0.02 for 2%)
  rating_model_version (TEXT): internal rating model version

provisioning:
  provision_id (INTEGER, PK): auto-increment identifier
  loan_account_number (TEXT, FK): joins to loan_master.loan_account_number
  reporting_date (DATE): quarter-end reporting date
  ifrs9_stage (INTEGER): IFRS 9 stage (1, 2, or 3)
  stage_rationale (TEXT): reason for stage assignment
  pd_12_month (REAL): 12-month probability of default
  pd_lifetime (REAL): lifetime probability of default
  lgd_estimate (REAL): loss given default (decimal)
  ead_amount (REAL): exposure at default in USD
  ecl_amount (REAL): expected credit loss in USD
  provision_held (REAL): provision amount held in USD
  provision_coverage_ratio (REAL): provision_held / total_outstanding * 100
  is_individually_assessed (INTEGER): 1 if individually assessed, 0 if modeled

Available reporting_date values in provisioning: 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Available rating_date values in borrower_rating: 2024-09-30, 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Latest reporting_date: 2025-09-30
Latest rating_date: 2025-09-30
NPA definition: asset_classification IN ('Substandard', 'Doubtful', 'Loss')
"""


sql_1 = """
SELECT
    s.sector_name AS sector_name,
    ROUND(SUM(l.total_outstanding) / 1000000.0, 2) AS total_outstanding_mn,
    ROUND(SUM(CASE WHEN l.asset_classification IN ('Substandard', 'Doubtful', 'Loss')
                   THEN l.total_outstanding ELSE 0 END) / 1000000.0, 2) AS npa_outstanding_mn
FROM loan_master l
JOIN sector_master s ON l.sector_code = s.sector_code
GROUP BY s.sector_name
ORDER BY total_outstanding_mn DESC
"""

sql_2 = """
SELECT
    loan_category,
    ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_mn,
    COUNT(*) AS loan_count
FROM loan_master
GROUP BY loan_category
ORDER BY total_outstanding_mn DESC
"""

sql_3 = """
SELECT
    ifrs9_stage,
    COUNT(*) AS loan_count,
    ROUND(SUM(ead_amount) / 1000000.0, 2) AS total_ead_mn,
    ROUND(SUM(ecl_amount) / 1000000.0, 2) AS total_ecl_mn
FROM provisioning
WHERE reporting_date = '2025-09-30'
GROUP BY ifrs9_stage
ORDER BY ifrs9_stage
"""

sql_4 = """
SELECT
    s.sector_name AS sector_name,
    ROUND(AVG(p.provision_coverage_ratio), 2) AS avg_coverage_ratio_pct
FROM provisioning p
JOIN loan_master l ON p.loan_account_number = l.loan_account_number
JOIN sector_master s ON l.sector_code = s.sector_code
WHERE p.reporting_date = '2025-09-30'
GROUP BY s.sector_name
ORDER BY avg_coverage_ratio_pct DESC
"""

sql_5 = """
SELECT
    l.borrower_name AS borrower_name,
    s.sector_name AS sector_name,
    ROUND(l.total_outstanding / 1000000.0, 2) AS total_outstanding_mn,
    l.asset_classification AS asset_classification
FROM loan_master l
JOIN sector_master s ON l.sector_code = s.sector_code
ORDER BY l.total_outstanding DESC
LIMIT 10
"""

sql_6 = """
SELECT
    group_name,
    COUNT(*) AS loan_count,
    ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_mn
FROM loan_master
WHERE group_name IS NOT NULL AND TRIM(group_name) <> ''
GROUP BY group_name
ORDER BY total_outstanding_mn DESC
LIMIT 5
"""

sql_7 = """
SELECT
    l.loan_account_number AS loan_account_number,
    l.borrower_name AS borrower_name,
    s.sector_name AS sector_name,
    ROUND(l.total_outstanding / 1000000.0, 2) AS total_outstanding_mn,
    l.days_past_due AS days_past_due,
    l.asset_classification AS asset_classification
FROM loan_master l
JOIN sector_master s ON l.sector_code = s.sector_code
WHERE l.days_past_due > 0
ORDER BY l.days_past_due DESC
"""

sql_8 = """
SELECT
    CASE
        WHEN days_past_due = 0 THEN '0 (Current)'
        WHEN days_past_due BETWEEN 1 AND 30 THEN '1-30'
        WHEN days_past_due BETWEEN 31 AND 60 THEN '31-60'
        WHEN days_past_due BETWEEN 61 AND 90 THEN '61-90'
        ELSE '90+'
    END AS dpd_bucket,
    COUNT(*) AS loan_count,
    ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_mn
FROM loan_master
GROUP BY dpd_bucket
ORDER BY MIN(days_past_due)
"""

sql_9 = """
SELECT
    borrower_id,
    previous_rating,
    internal_rating AS current_rating,
    pd_estimate
FROM borrower_rating
WHERE rating_date = '2025-09-30'
  AND rating_direction = 'Downgraded'
ORDER BY pd_estimate DESC
"""

sql_10 = """
SELECT
    reporting_date,
    ROUND(SUM(ecl_amount) / 1000000.0, 2) AS total_ecl_mn
FROM provisioning
GROUP BY reporting_date
ORDER BY reporting_date
"""


verified_query_library = {
    'VQ1': {
        'description': 'Sector-wise total outstanding and NPA amount breakdown across all sectors',
        'sql': sql_1
    },


    'VQ2': {
        'description': 'Total portfolio outstanding broken down by loan category (Corporate, Mid-Corporate, SME)',
        'sql': sql_2
    },


    'VQ3': {
        'description': 'IFRS 9 stage-wise summary showing loan count, exposure at default, and expected credit loss for the latest quarter',
        'sql': sql_3
    },


    'VQ4': {
        'description': 'Average provision coverage ratio by sector for the latest reporting quarter',
        'sql': sql_4
    },


    'VQ5': {
        'description': 'Top 10 largest loan exposures by outstanding amount at the borrower level',
        'sql': sql_5
    },


    'VQ6': {
        'description': 'Top 5 largest exposures aggregated at the business group level',
        'sql': sql_6
    },


    'VQ7': {
        'description': 'All overdue loan accounts with their days past due and asset classification',
        'sql': sql_7
    },


    'VQ8': {
        'description': 'Distribution of loans across days-past-due buckets showing aging profile of the portfolio',
        'sql': sql_8
    },


    'VQ9': {
        'description': 'Borrowers whose internal rating was downgraded in the latest rating cycle',
        'sql': sql_9
    },


    'VQ10': {
        'description': 'Expected credit loss trend across all reporting quarters showing provisioning movement over time',
        'sql': sql_10
    }
}


# ---------------------------------------------------------------- engine tools
def classify_intent(user_question, query_library):
    """
    Classifies the user question and decides which route to take.

    Parameters:
    - user_question (str): The natural language question from the user.
    - query_library (dict): The verified query template library.

    Returns:
    - dict: Contains 'route' (verified or generated),
                     'query_id' (template ID or None),
                     'match_reason' (short explanation of the decision).
    """

    library_descriptions = '\n'.join(
        [f"{qid}: {entry['description']}" for qid, entry in query_library.items()]
    )

    classification_prompt = f"""
You are the intent router of Northbridge Bank's credit-risk query engine. Decide whether a business
user's question can be answered by ONE pre-approved (verified) SQL template, or whether new SQL has to be generated.

### ROUTING RULES
1. Choose "verified" ONLY when a single template already returns the metric, the dimension (grouping) and the
   time scope that the question needs.
2. A template may return more rows than the question asks for (for example every sector when the user asks about
   one sector). That is acceptable: a later step highlights the relevant rows. Simple arithmetic on the returned
   columns (a ratio or share) is also acceptable.
3. Choose "generated" when the question needs a metric, filter, grouping or time window that no template returns:
   a different measure, a population defined by a loan flag, a trend restricted to one segment, a list with
   extra computed flags, or a combination of several templates.
4. Do NOT force a close-but-wrong match. A template that is merely on the same topic is not a match. If in doubt,
   choose "generated".
5. Judge fit from the template description, not from the template number.

### USER QUESTION
{user_question}

### VERIFIED TEMPLATE LIBRARY
{library_descriptions}

### OUTPUT

Return ONLY a valid JSON dictionary with these exact keys:
{{
  "route": "verified" or "generated",
  "query_id": "VQ1" or "VQ2" ... "VQ10" or null,
  "match_reason": "one short sentence explaining the decision"
}}
Do not include any other text.
"""

    response = llm.invoke(classification_prompt).content.strip()
    # Extract JSON from potential markdown blocks
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            # Safety net: an unknown template ID can never be executed, so fall back to generation
            if parsed.get('route') == 'verified' and parsed.get('query_id') in query_library:
                return parsed
            if parsed.get('route') == 'generated':
                parsed['query_id'] = None
                return parsed
        except json.JSONDecodeError:
            pass
    return {"route": "generated", "query_id": None, "match_reason": "Could not parse classification"}


def generate_query(user_question, schema_context):
    """
    Generates a candidate SQL query for a novel question using the database schema.

    Parameters:
    - user_question (str): The natural language question.
    - schema_context (str): Full database schema description.

    Returns:
    - str: Candidate SQL query as a string.
    """

    generation_prompt = f"""
You are a senior credit-risk data analyst writing SQL for a SQLite database that holds a commercial lending portfolio.
Write ONE read-only SQLite query that answers the user's question.

### HARD RULES
- Output a single SELECT statement (a WITH ... SELECT is allowed). Never write INSERT, UPDATE, DELETE, DROP, ALTER,
  CREATE, PRAGMA or ATTACH, and never write more than one statement.
- Use ONLY the tables and columns listed in the schema below. Do not invent columns.
- Use SQLite syntax (window functions are available; no ILIKE, no TOP, no DATEDIFF).
- Return the SQL only: no explanation, no comments, no markdown fences.

### BUSINESS RULES
- NPA / non-performing / impaired means asset_classification IN ('Substandard', 'Doubtful', 'Loss').
- Amounts are in USD. Show amounts in millions with ROUND(x / 1000000.0, 2) and end the alias with _mn.
- Exposure / outstanding means loan_master.total_outstanding.
- Provisioning (IFRS 9 stage, ECL, EAD, coverage) has one row per loan per quarter in `provisioning`.
  Use reporting_date = '2025-09-30' (latest) for a point-in-time question, and group by reporting_date when the
  user asks about a trend, movement or "last N quarters" (there are four reporting dates).
- Ratings live in `borrower_rating` and join to loans through borrower_id. Latest rating_date is '2025-09-30'.
- Join sector_master on sector_code whenever a sector name is needed, and return sector_name rather than sector_code.
- Do not truncate results with LIMIT unless the user asks for a top-N.
- When the user asks for a list AND an overall summary figure (a total, a count or a percentage), return the list
  rows with an extra column holding the overall figure (use a window function such as SUM(...) OVER ()), so one
  statement returns everything.
- When the user asks to flag or classify rows, return the flag as an explicit column (1/0 or a text label).
- Respect the requested sort order. Use readable column aliases.

### DATABASE SCHEMA
{schema_context}

### USER QUESTION
{user_question}

### SQL
"""

    sql = llm.invoke(generation_prompt).content.strip()
    # Strip markdown fences if present
    sql = re.sub(r'^```sql\s*|\s*```$', '', sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    sql = re.sub(r'^```\s*|\s*```$', '', sql, flags=re.MULTILINE).strip()
    return sql


def validate_query(user_question, candidate_sql, db_connection, query_library, query_id=None):
    """
    Validates a candidate SQL query through five checks before execution.

    Parameters:
    - user_question (str): The original user question.
    - candidate_sql (str): The SQL query to validate.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query library (for integrity check).
    - query_id (str, optional): Template ID if from verified track.

    Returns:
    - dict: Contains 'passed' (bool), 'failed_check' (str or None), 'details' (str),
            and 'relevance_confidence' (float, 0-1).
    """

    result = {
        'passed': False,
        'failed_check': None,
        'details': '',
        'relevance_confidence': None
    }

    # Work on a copy without string literals, so values such as 'Loss' are never mistaken for identifiers/keywords
    sql_no_literals = re.sub(r"'[^']*'", "''", candidate_sql)

    # ---------------- Check 1: Read-only shape check ----------------
    sql_upper = sql_no_literals.upper().strip()
    forbidden_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE', 'REPLACE', 'ATTACH',
                          'CREATE', 'PRAGMA', 'DETACH', 'VACUUM']
    if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Query must start with SELECT or WITH'
        return result
    for kw in forbidden_keywords:
        if re.search(r'\b' + kw + r'\b', sql_upper):
            result['failed_check'] = 'read_only_shape'
            result['details'] = f'Forbidden keyword detected: {kw}'
            return result
    n_statements = len([s for s in sqlparse.parse(candidate_sql) if str(s).strip().strip(';').strip()])
    if n_statements > 1 or ';' in candidate_sql.strip().rstrip(';'):
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Multiple statements are not allowed'
        return result

    # ---------------- Check 2: Schema conformance check ----------------
    cur = db_connection.cursor()
    real_tables = {r[0].lower() for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    real_columns = set()
    for t in real_tables:
        for col_info in cur.execute(f"PRAGMA table_info({t})").fetchall():
            real_columns.add(col_info[1].lower())

    cte_names = {m.lower() for m in re.findall(r'\b([A-Za-z_]\w*)\s+AS\s*\(', sql_no_literals, flags=re.IGNORECASE)}
    output_aliases = {m.lower() for m in re.findall(r'\bAS\s+([A-Za-z_]\w*)', sql_no_literals, flags=re.IGNORECASE)}

    # every table named after FROM / JOIN must exist (or be a CTE defined in the query)
    referenced_tables = re.findall(r'\b(?:FROM|JOIN)\s+([A-Za-z_]\w*)', sql_no_literals, flags=re.IGNORECASE)
    unknown_tables = sorted({t for t in referenced_tables if t.lower() not in real_tables and t.lower() not in cte_names})
    # every qualified reference alias.column must be a real column (or a column alias defined in the query)
    qualified_cols = re.findall(r'\b[A-Za-z_]\w*\.([A-Za-z_]\w*)\b', sql_no_literals)
    unknown_cols = sorted({c for c in qualified_cols if c.lower() not in real_columns and c.lower() not in output_aliases})
    if unknown_tables or unknown_cols:
        result['failed_check'] = 'schema_conformance'
        result['details'] = f'Unknown tables: {unknown_tables}; unknown columns: {unknown_cols}'
        return result

    # ---------------- Check 3: Parse-and-plan dry run using EXPLAIN ----------------
    try:
        cur.execute(f"EXPLAIN {candidate_sql.strip().rstrip(';')}")
        cur.fetchall()
    except sqlite3.Error as e:
        result['failed_check'] = 'parse_plan_dry_run'
        result['details'] = f'SQL failed to parse or plan: {str(e)}'
        return result

    # ---------------- Check 4: LLM relevance check (independent evaluator model) ----------------
    is_verified_track = query_id is not None and query_id in query_library
    if is_verified_track:
        review_task = """This SQL is a PRE-APPROVED VERIFIED TEMPLATE that the bank has already tested.
Templates deliberately return the FULL breakdown (every sector, every stage, every bucket, every quarter)
and are NOT filtered to what the user asked. A later step picks out the relevant rows for the user.

Therefore do NOT judge whether the SQL is filtered or scoped to the question, and never fail it for lacking a
WHERE clause, a sector filter or a LIMIT.

Answer only this question: does the RESULT of this SQL contain the metric(s) and the grouping dimension needed to
answer the user's question, so that the answer can be read from its rows (simple arithmetic such as a share or
percentage is fine)? If yes, the verdict is "yes"."""
    else:
        review_task = """This SQL was freshly generated for this specific question, so it must answer it directly.
Review against these points:
1. Tables and joins: are the right tables used and joined on the right keys?
2. Metric: is the right column used (e.g. total_outstanding for exposure, ecl_amount for ECL, ead_amount for EAD)?
3. Aggregation and grouping: do the GROUP BY / aggregates match what is being asked?
4. Date logic: the latest snapshot (2025-09-30) for point-in-time questions, all reporting dates for trends.
5. NPA / impaired logic: asset_classification IN ('Substandard', 'Doubtful', 'Loss') whenever non-performing,
   NPA or impaired is mentioned.
6. Completeness: no rows dropped by an unrequested filter or LIMIT; requested flags / summary figures are present."""

    relevance_prompt = f"""
You are an independent SQL reviewer at a bank's credit-risk analytics team. You did not write this SQL.

### YOUR TASK
{review_task}

### DATABASE SCHEMA
{database_schema}

### USER QUESTION
{user_question}

### SQL UNDER REVIEW
{candidate_sql}

### OUTPUT
Return ONLY a JSON dictionary:
{{
  "verdict": "yes" or "no",
  "confidence": 0.0 to 1.0,
  "reason": "one short sentence"
}}
"verdict" is "yes" only if the review task above is satisfied. "confidence" is how sure you are in that verdict.
"""
    try:
        relevance_response = evaluator_llm.invoke(relevance_prompt).content.strip()
        json_match = re.search(r'\{.*\}', relevance_response, re.DOTALL)
        relevance_json = json.loads(json_match.group()) if json_match else None
    except Exception as e:                       # API or parsing problem -> fail closed, never silently pass
        relevance_json = None
        relevance_response = str(e)

    if relevance_json is None:
        result['failed_check'] = 'llm_relevance'
        result['details'] = f'Relevance check could not be completed: {relevance_response[:120]}'
        return result

    result['relevance_confidence'] = float(relevance_json.get('confidence', 0.0))
    if relevance_json.get('verdict') == 'no' or result['relevance_confidence'] < 0.6:
        result['failed_check'] = 'llm_relevance'
        result['details'] = f"Relevance check failed: {relevance_json.get('reason', 'unknown')}"
        return result

    # ---------------- Check 5: Verified template integrity check (verified track only) ----------------
    if is_verified_track:
        expected_sql = query_library[query_id]['sql'].strip().rstrip(';')
        try:
            # wrapping in a sub-select keeps this valid even when the template has its own LIMIT
            expected_cols = [d[0] for d in cur.execute(f"SELECT * FROM ({expected_sql}) LIMIT 0").description]
            actual_cols = [d[0] for d in cur.execute(f"SELECT * FROM ({candidate_sql.strip().rstrip(';')}) LIMIT 0").description]
            if expected_cols != actual_cols:
                result['failed_check'] = 'template_integrity'
                result['details'] = f'Expected columns {expected_cols}, got {actual_cols}'
                return result
        except sqlite3.Error as e:
            result['failed_check'] = 'template_integrity'
            result['details'] = f'Template integrity check failed: {str(e)}'
            return result

    result['passed'] = True
    result['details'] = 'All validation checks passed'
    return result


def retry_generation(user_question, failed_sql, error_message, schema_context):
    """
    Regenerates SQL after a validation failure, feeding the error back to the LLM.

    Parameters:
    - user_question (str): The original user question.
    - failed_sql (str): The SQL that failed validation.
    - error_message (str): The specific failure reason.
    - schema_context (str): Database schema description.

    Returns:
    - str: Revised SQL as a string.
    """

    retry_prompt = f"""
You are a senior credit-risk data analyst. A SQLite query you wrote failed the bank's automated validation gate.
Fix it so that it passes the gate AND still answers the original question exactly.

### RULES
- Fix the specific problem named in the validation error; do not change the intent of the question.
- Output ONE read-only SQLite SELECT (or WITH ... SELECT) statement. No INSERT, UPDATE, DELETE, DROP, ALTER,
  CREATE, PRAGMA or ATTACH, and no second statement.
- Use only tables and columns from the schema. NPA / impaired = asset_classification IN ('Substandard', 'Doubtful', 'Loss').
- Amounts in USD millions: ROUND(x / 1000000.0, 2) with an alias ending in _mn.
- Latest snapshot is 2025-09-30 for point-in-time questions; use all reporting dates for trends.
- Return the corrected SQL only: no explanation, no comments, no markdown fences.

User Question:
{user_question}

Failed SQL:
{failed_sql}

Validation Error:
{error_message}

Database Schema:
{schema_context}

Corrected SQL:
"""

    revised_sql = llm.invoke(retry_prompt).content.strip()
    revised_sql = re.sub(r'^```sql\s*|\s*```$', '', revised_sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    revised_sql = re.sub(r'^```\s*|\s*```$', '', revised_sql, flags=re.MULTILINE).strip()
    return revised_sql


def execute_query(validated_sql, db_connection):
    """
    Executes a gate-passed SQL query and returns the result as a DataFrame.

    Parameters:
    - validated_sql (str): SQL query that has passed all validation checks.
    - db_connection: Read-only SQLite connection object.

    Returns:
    - dict: Contains 'dataframe' (pandas DataFrame), 'reasonable' (bool),
            and 'warnings' (list of warning strings).
    """

    result = {
        'dataframe': None,
        'reasonable': True,
        'warnings': []
    }

    df = pd.read_sql_query(validated_sql.strip().rstrip(';'), db_connection)
    result['dataframe'] = df

    # Reasonableness checks
    if df.empty:
        result['warnings'].append('Query returned an empty result')

    for col in df.select_dtypes(include='number').columns:
        if (df[col] < 0).any() and 'deviation' not in col.lower() and 'change' not in col.lower():
            result['warnings'].append(f'Column {col} contains negative values')
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            if null_count > len(df) * 0.5:
                result['warnings'].append(f'Column {col} has {null_count} null values')

    if len(result['warnings']) > 2:
        result['reasonable'] = False

    return result


def generate_response(user_question, dataframe, route, query_id=None):
    """
    Generates a focused natural language response from the query result.

    Parameters:
    - user_question (str): The original user question.
    - dataframe (pd.DataFrame): The full query result.
    - route (str): 'verified' or 'generated'.
    - query_id (str, optional): Template ID if from verified track.

    Returns:
    - str: Natural language response focused on what the user asked.
    """

    max_rows = 100                                    # protect the context window on very long result sets
    data_text = dataframe.head(max_rows).to_string(index=False)
    row_note = (f"(first {max_rows} of {len(dataframe)} rows shown)" if len(dataframe) > max_rows
                else f"({len(dataframe)} rows, complete result)")

    response_prompt = f"""
You are a credit-risk analyst answering a business user (CRO office, Credit Committee) at Northbridge Bank.
Write the answer to the question using ONLY the query result below.

### RULES
- Answer exactly what was asked. If the table covers more than the question (for example all sectors when the user
  asked about one), lead with the relevant row(s) and add at most one sentence of context.
- Quote exact figures from the table. Columns ending in _mn are USD millions; write them as $X.XM or $X,XXX.XM.
- You may compute simple derived figures (a share, a percentage, a count of flagged rows, a change between periods)
  from the table. Show the numbers they come from. Never invent or estimate a number that is not in, or derivable
  from, the table.
- If the user asked for a list with many rows, do not reproduce every row: give the count, the key totals and two or
  three notable examples.
- For a trend, state the direction and size of the change between the first and last period.
- Be concise: 3 to 6 sentences or short bullets, plain business language. Do not mention SQL, tables or columns.
- End with one short "So what" sentence if the data supports a clear takeaway for a risk manager.

### USER QUESTION
{user_question}

### QUERY RESULT {row_note}
Source: {route} route{(' (' + query_id + ')') if query_id else ''}
{data_text}

### ANSWER
"""

    narrative = llm.invoke(response_prompt).content.strip()
    return narrative


def create_chart(dataframe, question=''):
    """
    Builds a chart when the result is a small summary table (one label/date column plus numeric measures).
    Returns a matplotlib Figure, or None when a chart would not add value (e.g. long loan-level lists).
    """
    df = dataframe
    if df is None or df.empty or len(df) < 2 or len(df) > 25:
        return None

    # x-axis: first text/date column; otherwise a small coded column such as ifrs9_stage
    x_col = next((c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])), None)
    if x_col is None:
        x_col = next((c for c in df.columns if any(k in c.lower() for k in ('stage', 'bucket', 'year', 'quarter'))), None)
    if x_col is None:
        return None

    # measures: numeric columns that are not identifiers; at most three panels
    y_cols = [c for c in df.columns if c != x_col and pd.api.types.is_numeric_dtype(df[c])
              and not c.lower().endswith('_id') and 'days_past_due' not in c.lower()][:3]
    if not y_cols:
        return None

    is_time = any(k in x_col.lower() for k in ('date', 'quarter', 'year'))
    labels = df[x_col].astype(str)
    horizontal = (not is_time) and (len(df) > 6 or labels.str.len().max() > 14)

    def pretty(col):
        name = (col[:-3] + ' (USD M)') if col.endswith('_mn') else col
        name = name.replace('_', ' ').title()
        for k, v in {'Ead': 'EAD', 'Ecl': 'ECL', 'Npa': 'NPA', 'Dpd': 'DPD', 'Pct': '%', 'Usd': 'USD', ' M)': ' M)'}.items():
            name = name.replace(k, v)
        return name

    fig, axes = plt.subplots(1, len(y_cols), figsize=(5.6 * len(y_cols), max(3.6, 0.38 * len(df)) if horizontal else 4.2))
    axes = np.atleast_1d(axes)
    for ax, col in zip(axes, y_cols):
        fmt = '%d' if pd.api.types.is_integer_dtype(df[col]) else '%.1f'
        if is_time:
            ax.plot(labels, df[col], marker='o', color='#1f5fa8', linewidth=2)
            for xi, yi in zip(labels, df[col]):
                ax.annotate((fmt % yi) if fmt == '%d' else f"{yi:,.1f}", (xi, yi), textcoords='offset points', xytext=(0, 7), ha='center', fontsize=8)
            ax.tick_params(axis='x', rotation=20)
        elif horizontal:
            bars = ax.barh(labels, df[col], color='#1f5fa8')
            ax.invert_yaxis()                              # keep the query's sort order top-to-bottom
            ax.bar_label(bars, fmt=fmt, padding=3, fontsize=8)
        else:
            bars = ax.bar(labels, df[col], color='#1f5fa8')
            ax.bar_label(bars, fmt=fmt, padding=3, fontsize=8)
        ax.set_title(pretty(col), fontsize=10)
        ax.margins(y=0.15, x=0.08)
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    plt.close(fig)                                         # returned explicitly, so avoid double display
    return fig


AUDIT_LOG_PATH = 'audit_log.jsonl'

def write_audit_log(log, path=AUDIT_LOG_PATH):
    """
    Appends one JSON line per analytical output (append-only audit trail).
    The database stays read-only, so the trail is stored in a separate file.
    """
    record = {'timestamp_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'), **log}
    try:
        with open(path, 'a') as f:
            f.write(json.dumps(record, default=str) + '\n')
    except OSError:
        pass                                                # a read-only host must never break an answer
    return record


# Tool registry: every stage of the engine is an explicit, individually testable tool
TOOLS = {
    'classify_intent':   classify_intent,
    'generate_query':    generate_query,
    'validate_query':    validate_query,
    'retry_generation':  retry_generation,
    'execute_query':     execute_query,
    'generate_response': generate_response,
    'create_chart':      create_chart,
    'write_audit_log':   write_audit_log,
}


def run_pipeline(user_question, db_connection, query_library, schema_context, verbose=True):
    """
    Runs the complete query engine pipeline for a single user question.

    Parameters:
    - user_question (str): The natural language question.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query template library.
    - schema_context (str): Database schema description.
    - verbose (bool): If True, prints intermediate pipeline stages.

    Returns:
    - dict: Complete pipeline output including narrative, SQL, data, and log.
    """

    log = {
        'user_question': user_question,
        'route': None,
        'query_id': None,
        'match_reason': None,
        'candidate_sql': None,
        'gate_result': None,
        'retry_used': False,
        'escalated': False,
        'executed_sql': None,
        'row_count': None,
        'warnings': [],
        'confidence': None,
        'narrative': None
    }

    # Step 1: Intent classification
    classification = TOOLS['classify_intent'](user_question, query_library)
    log['route'] = classification['route']
    log['query_id'] = classification.get('query_id')
    log['match_reason'] = classification.get('match_reason')

    if verbose:
        print(f"[1] Intent Classification: route={log['route']}, query_id={log['query_id']}")
        print(f"    Reason: {log['match_reason']}")


    # Step 2: Query construction
    if log['route'] == 'verified' and log['query_id'] in query_library:
        candidate_sql = query_library[log['query_id']]['sql']
    else:
        candidate_sql = TOOLS['generate_query'](user_question, schema_context)
    log['candidate_sql'] = candidate_sql

    if verbose:
        print(f"[2] Query Construction: {'loaded from library' if log['route']=='verified' else 'generated fresh SQL'}")


    # Step 3: Validation gate
    gate = TOOLS['validate_query'](user_question, candidate_sql, db_connection, query_library, log['query_id'])
    log['gate_result'] = gate

    if verbose:
        print(f"[3] Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
        if not gate['passed']:
            print(f"    Failed check: {gate.get('failed_check')}")
            print(f"    Details: {gate.get('details')}")


    # Step 4: Retry once on generated track if validation fails
    if not gate['passed'] and log['route'] == 'generated':
        if verbose:
            print(f"    Retrying: {gate['details']}")
        candidate_sql = TOOLS['retry_generation'](user_question, candidate_sql, gate['details'], schema_context)
        log['candidate_sql'] = candidate_sql
        log['retry_used'] = True
        gate = TOOLS['validate_query'](user_question, candidate_sql, db_connection, query_library, None)
        log['gate_result'] = gate

        if verbose:
            print(f"    Retry Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
            if not gate['passed']:
                print(f"    Retry failed check: {gate.get('failed_check')}")
                print(f"    Retry details: {gate.get('details')}")


    # Step 5: Escalate if still failing
    if not gate['passed']:
        log['escalated'] = True
        log['narrative'] = f"Query could not be reliably resolved. Escalated to human analyst. Failure: {gate['details']}"
        log['confidence'] = 'ESCALATED'
        TOOLS['write_audit_log'](log)
        if verbose:
            print(f"[!] Escalated to human: {gate['details']}")
        return {'log': log, 'dataframe': None, 'figure': None, **log}


    # Step 6: Execute
    log['executed_sql'] = candidate_sql
    try:
        exec_result = TOOLS['execute_query'](candidate_sql, db_connection)
    except Exception as e:                                   # a runtime failure is escalated, never hidden
        log['escalated'] = True
        log['narrative'] = f"Query failed at execution and was escalated to a human analyst. Error: {e}"
        log['confidence'] = 'ESCALATED'
        TOOLS['write_audit_log'](log)
        if verbose:
            print(f"[!] Execution error, escalated: {e}")
        return {'log': log, 'dataframe': None, 'figure': None, **log}

    df = exec_result['dataframe']
    log['row_count'] = len(df)
    log['warnings'] = exec_result['warnings']

    if verbose:
        print(f"[4] Execute: {len(df)} rows returned")
        if exec_result['warnings']:
            print(f"    Warnings: {exec_result['warnings']}")


    # Step 7: Visualisation (only when the result is a small summary table)
    figure = TOOLS['create_chart'](df, user_question)
    if verbose:
        print(f"[5] Visualisation: {'chart created' if figure is not None else 'no chart (table is better for this result)'}")


    # Step 8: Response generation
    narrative = TOOLS['generate_response'](user_question, df, log['route'], log['query_id'])
    log['narrative'] = narrative


    # Confidence: carried directly from the validation gate's relevance check (0-1)
    log['confidence'] = gate.get('relevance_confidence')

    # Step 9: Audit trail
    TOOLS['write_audit_log'](log)

    if verbose:
        print(f"[6] Response Generation: confidence={log['confidence']}")
        print(f"[7] Audit trail: appended to {AUDIT_LOG_PATH}")

    return {'log': log, 'dataframe': df, 'figure': figure, **log}

# ---------------------------------------------------------------- Streamlit UI
SAMPLE_QUESTIONS = {
    "TC-01 Real estate exposure and NPA": "How much of our book is in real estate, and how much of that is non-performing?",
    "TC-02 DPD aging profile": "Show me how our overdue book is distributed across DPD buckets.",
    "TC-03 Restructured loans": "List every restructured loan, flag which are impaired, and show the overall impaired percentage.",
    "TC-04 Average interest rate by sector": "Show me the average interest rate by sector, sorted from highest to lowest.",
    "TC-05 Stage 3 trend": "How has the Stage 3 book size and its expected credit loss moved over the last four quarters?",
}

st.title("🏦 Credit Risk Query Engine")
st.caption("Northbridge Bank | commercial lending portfolio | read-only | every answer shows its SQL, data, confidence and audit entry")

with st.sidebar:
    st.header("Sample questions")
    choice = st.radio("Pick a test case", ["(type my own)"] + list(SAMPLE_QUESTIONS), index=0)
    st.markdown("---")
    st.markdown("**How it works**\n\n1. Route to a verified template or generate SQL\n2. Five-check validation gate\n3. One retry, then escalate to an analyst\n4. Execute read-only\n5. Answer + chart + confidence + audit log")

default_q = SAMPLE_QUESTIONS.get(choice, "")
question = st.text_area("Ask a portfolio question in plain English", value=default_q, height=90, key=f"q_{choice}")

if st.button("Run query", type="primary") and question.strip():
    with st.spinner("Running the query engine..."):
        res = run_pipeline(question.strip(), conn, verified_query_library, database_schema, verbose=False)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Route", res["route"])
    c2.metric("Template", res["query_id"] or "generated SQL")
    conf = res["confidence"]
    c3.metric("Confidence", f"{conf:.2f}" if isinstance(conf, (int, float)) else str(conf))
    c4.metric("Retry used", "yes" if res["retry_used"] else "no")

    if res["escalated"]:
        st.warning(res["narrative"])
    else:
        st.subheader("Answer")
        st.markdown(res["narrative"].replace("$", r"\$"))   # escape $ so Streamlit does not render $...$ as LaTeX math
        if res["figure"] is not None:
            st.pyplot(res["figure"])
        st.subheader("Raw data")
        st.dataframe(res["dataframe"])

    with st.expander("SQL used"):
        st.code(sqlparse.format(res["executed_sql"] or res["candidate_sql"], reindent=True, keyword_case="upper"), language="sql")
    with st.expander("Routing & validation trace"):
        st.write(f"**Match reason:** {res['match_reason']}")
        st.json(res["gate_result"])
    with st.expander("Audit log entry"):
        st.json({k: v for k, v in res["log"].items()})

from snowflake.snowpark import Session
from snowflake.snowpark.functions import col
from snowflake.snowpark import DataFrame

def build_transaction_features(session: Session) -> DataFrame:
    """
    Build transaction-level AML features in Snowpark/Snowflake.
    One row = one transaction.
    """

    # --- 1. Basic cleaning and validation (as SQL CTEs) ---

    feature_sql = """
    WITH
    -- 1) Base transactions, cast types and basic filters
    -- Add a stable per-transaction id to prevent accidental row multiplication in later joins.
    base_tx AS (
        SELECT
            -- Cast timestamp (explicit format: 'YYYY/MM/DD HH24:MI')
            TO_TIMESTAMP_NTZ(TIMESTAMP, 'YYYY/MM/DD HH24:MI') AS TS,
            FROM_BANK,
            FROM_ACCOUNT,
            TO_BANK,
            TO_ACCOUNT,
            -- Cast amounts, enforce > 0 by filtering later
            TRY_TO_DOUBLE(AMOUNT_PAID)       AS AMOUNT_PAID,
            TRY_TO_DOUBLE(AMOUNT_RECEIVED)   AS AMOUNT_RECEIVED,
            RECEIVING_CURRENCY,
            PAYMENT_CURRENCY,
            PAYMENT_FORMAT,
            -- Ensure laundering is 0/1
            CASE
                WHEN IS_LAUNDERING IS NULL THEN 0
                WHEN IS_LAUNDERING IN (0, 1) THEN IS_LAUNDERING
                WHEN TRY_TO_NUMBER(IS_LAUNDERING) IN (0, 1)
                    THEN TRY_TO_NUMBER(IS_LAUNDERING)
                ELSE 0
            END AS IS_LAUNDERING
        FROM AML_PROJECT.RAW.RAW_TRANSACTIONS
        WHERE
            TRY_TO_DOUBLE(AMOUNT_PAID) > 0
            AND TRY_TO_DOUBLE(AMOUNT_RECEIVED) > 0
            AND FROM_ACCOUNT IS NOT NULL
            AND TO_ACCOUNT IS NOT NULL
    ),

    -- 2) Join account/entity info for FROM and TO sides
    from_acct AS (
        SELECT
            BANK_ID         AS FROM_BANK,
            ACCOUNT_NUMBER  AS FROM_ACCOUNT,
            ENTITY_ID       AS FROM_ENTITY_ID,
            ENTITY_NAME     AS FROM_ENTITY_NAME,
            BANK_NAME       AS FROM_BANK_NAME
        FROM AML_PROJECT.RAW.RAW_ACCOUNTS
    ),

    to_acct AS (
        SELECT
            BANK_ID         AS TO_BANK,
            ACCOUNT_NUMBER  AS TO_ACCOUNT,
            ENTITY_ID       AS TO_ENTITY_ID,
            ENTITY_NAME     AS TO_ENTITY_NAME,
            BANK_NAME       AS TO_BANK_NAME
        FROM AML_PROJECT.RAW.RAW_ACCOUNTS
    ),

    tx_enriched AS (
        SELECT
            t.*,
            fa.FROM_ENTITY_ID,
            fa.FROM_ENTITY_NAME,
            fa.FROM_BANK_NAME,
            ta.TO_ENTITY_ID,
            ta.TO_ENTITY_NAME,
            ta.TO_BANK_NAME
        FROM base_tx t
        LEFT JOIN from_acct fa
          ON t.FROM_BANK = fa.FROM_BANK
         AND t.FROM_ACCOUNT = fa.FROM_ACCOUNT
        LEFT JOIN to_acct ta
          ON t.TO_BANK = ta.TO_BANK
         AND t.TO_ACCOUNT = ta.TO_ACCOUNT
    ),

    -- 3) Entity-level account counts (how many accounts per entity)
    entity_accounts AS (
        SELECT
            ENTITY_ID,
            COUNT(DISTINCT ACCOUNT_NUMBER) AS NUM_ACCOUNTS_PER_ENTITY
        FROM AML_PROJECT.RAW.RAW_ACCOUNTS
        GROUP BY ENTITY_ID
    ),

    tx_with_entity AS (
        SELECT
            e.*,
            ea_from.NUM_ACCOUNTS_PER_ENTITY AS FROM_NUM_ACCOUNTS_PER_ENTITY,
            ea_to.NUM_ACCOUNTS_PER_ENTITY   AS TO_NUM_ACCOUNTS_PER_ENTITY
        FROM tx_enriched e
        LEFT JOIN entity_accounts ea_from
          ON e.FROM_ENTITY_ID = ea_from.ENTITY_ID
        LEFT JOIN entity_accounts ea_to
          ON e.TO_ENTITY_ID = ea_to.ENTITY_ID
    ),

    -- 4) Add simple time features
    tx_time AS (
        SELECT
            *,
            DATE(TS)                                  AS TX_DATE,
            EXTRACT(HOUR      FROM TS)               AS HOUR_OF_DAY,
            EXTRACT(DAYOFWEEK FROM TS)               AS DAY_OF_WEEK,   -- 0=Sun
            CASE WHEN EXTRACT(DAYOFWEEK FROM TS) IN (0, 6) THEN 1 ELSE 0 END AS IS_WEEKEND,
            CASE WHEN FROM_BANK = TO_BANK THEN 1 ELSE 0 END AS SAME_BANK
        FROM tx_with_entity
    ),

    -- 5) Window definitions (Snowflake supports RANGE INTERVAL)
    --    We'll compute FROM_* features and TO_* features separately.

    -- FROM-side rolling windows
    from_features AS (
        SELECT
            t.*,

            -- Time since last txn FROM this account
            DATEDIFF('second',
                LAG(TS) OVER (
                    PARTITION BY FROM_BANK, FROM_ACCOUNT
                    ORDER BY TS
                ),
                TS
            ) AS TIME_SINCE_LAST_TXN_FROM,

            -- 1h / 24h / 7d windows for FROM side
            COUNT(*) OVER (
                PARTITION BY FROM_BANK, FROM_ACCOUNT
                ORDER BY TS
                RANGE BETWEEN INTERVAL '1 hour' PRECEDING AND CURRENT ROW
            ) AS TXN_COUNT_1H_FROM,

            COUNT(*) OVER (
                PARTITION BY FROM_BANK, FROM_ACCOUNT
                ORDER BY TS
                RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW
            ) AS TXN_COUNT_24H_FROM,

            COUNT(*) OVER (
                PARTITION BY FROM_BANK, FROM_ACCOUNT
                ORDER BY TS
                RANGE BETWEEN INTERVAL '7 days' PRECEDING AND CURRENT ROW
            ) AS TXN_COUNT_7D_FROM,

            SUM(AMOUNT_PAID) OVER (
                PARTITION BY FROM_BANK, FROM_ACCOUNT
                ORDER BY TS
                RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW
            ) AS TXN_AMOUNT_SUM_24H_FROM,

            AVG(AMOUNT_PAID) OVER (
                PARTITION BY FROM_BANK, FROM_ACCOUNT
                ORDER BY TS
                RANGE BETWEEN INTERVAL '7 days' PRECEDING AND CURRENT ROW
            ) AS AVG_TXN_AMOUNT_7D_FROM,

            MAX(AMOUNT_PAID) OVER (
                PARTITION BY FROM_BANK, FROM_ACCOUNT
                ORDER BY TS
                RANGE BETWEEN INTERVAL '7 days' PRECEDING AND CURRENT ROW
            ) AS MAX_TXN_AMOUNT_7D_FROM,

            -- Fan-out: lifetime unique receivers
            COUNT(DISTINCT TO_ACCOUNT) OVER (
                PARTITION BY FROM_BANK, FROM_ACCOUNT
            ) AS NUM_UNIQUE_RECEIVERS,

            -- Std dev of amounts over 7d
            STDDEV(AMOUNT_PAID) OVER (
                PARTITION BY FROM_BANK, FROM_ACCOUNT
                ORDER BY TS
                RANGE BETWEEN INTERVAL '7 days' PRECEDING AND CURRENT ROW
            ) AS TXN_AMOUNT_STD_7D_FROM,

            -- Distinct banks this FROM account has ever used (no frame/order to avoid Snowflake limitations)
            COUNT(DISTINCT TO_BANK) OVER (
                PARTITION BY FROM_BANK, FROM_ACCOUNT
            ) AS NUM_UNIQUE_BANKS,

            -- Cross-bank ratio over last 7d: inter-bank tx count / total tx count
            SUM(CASE WHEN FROM_BANK <> TO_BANK THEN 1 ELSE 0 END) OVER (
                PARTITION BY FROM_BANK, FROM_ACCOUNT
                ORDER BY TS
                RANGE BETWEEN INTERVAL '7 days' PRECEDING AND CURRENT ROW
            )::FLOAT
            /
            NULLIF(
                COUNT(*) OVER (
                    PARTITION BY FROM_BANK, FROM_ACCOUNT
                    ORDER BY TS
                    RANGE BETWEEN INTERVAL '7 days' PRECEDING AND CURRENT ROW
                ),
                0
            ) AS CROSS_BANK_RATIO_7D,

            -- First txn of calendar day for this account (sequence pattern)
            ROW_NUMBER() OVER (
                PARTITION BY FROM_BANK, FROM_ACCOUNT, TX_DATE
                ORDER BY TS
            ) AS _RN_DAY

        FROM tx_time t
    ),

    -- TO-side rolling windows
    full_features AS (
        SELECT
            f.*,

            -- Time since last txn TO this account
            DATEDIFF('second',
                LAG(TS) OVER (
                    PARTITION BY TO_BANK, TO_ACCOUNT
                    ORDER BY TS
                ),
                TS
            ) AS TIME_SINCE_LAST_TXN_TO,

            -- Fan-in: lifetime unique senders
            COUNT(DISTINCT FROM_ACCOUNT) OVER (
                PARTITION BY TO_BANK, TO_ACCOUNT
            ) AS NUM_UNIQUE_SENDERS

        FROM from_features f
    ),

    -- 6) Gap std dev in 7d (irregular timing)
    gaps AS (
        SELECT
            *,
            -- gap between successive FROM-side txns
            DATEDIFF('second',
                LAG(TS) OVER (
                    PARTITION BY FROM_BANK, FROM_ACCOUNT
                    ORDER BY TS
                ),
                TS
            ) AS FROM_GAP_SEC
        FROM full_features
    ),

    final AS (
        SELECT
            *,
            STDDEV(FROM_GAP_SEC) OVER (
                PARTITION BY FROM_BANK, FROM_ACCOUNT
                ORDER BY TS
                RANGE BETWEEN INTERVAL '7 days' PRECEDING AND CURRENT ROW
            ) AS TXN_GAP_STD_7D
        FROM gaps
    ),

    -- 9) Counterparty novelty / repeat counts
    counterparty_feats AS (
        SELECT
            f.*,

            -- Is this a new counterparty for this FROM account overall?
            CASE
                WHEN ROW_NUMBER() OVER (
                        PARTITION BY FROM_BANK, FROM_ACCOUNT, TO_BANK, TO_ACCOUNT
                        ORDER BY TS
                     ) = 1
                THEN 1 ELSE 0
            END AS IS_NEW_COUNTERPARTY,

            -- Repeat counterparty count (how many times seen before this txn)
            (ROW_NUMBER() OVER (
                 PARTITION BY FROM_BANK, FROM_ACCOUNT, TO_BANK, TO_ACCOUNT
                 ORDER BY TS
             ) - 1) AS REPEAT_COUNTERPARTY_COUNT

        FROM final f
    ),

    -- 10) Similar amount transactions in last 24h for FROM side
    -- Step 1: compute previous amount via LAG (no nesting of window functions)
    prev_amount AS (
        SELECT
            c.*,
            LAG(AMOUNT_PAID) OVER (
                PARTITION BY FROM_BANK, FROM_ACCOUNT
                ORDER BY TS
            ) AS PREV_AMOUNT_PAID
        FROM counterparty_feats c
    ),

    -- Step 2: count similar-amount txns in 24h window using PREV_AMOUNT_PAID
    similar_amounts AS (
        SELECT
            p.*,
            SUM(
                CASE
                    WHEN PREV_AMOUNT_PAID IS NOT NULL
                         AND ABS(AMOUNT_PAID - PREV_AMOUNT_PAID) <= 1e-6
                    THEN 1 ELSE 0
                END
            ) OVER (
                PARTITION BY FROM_BANK, FROM_ACCOUNT
                ORDER BY TS
                RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW
            ) AS NUM_SIMILAR_AMOUNT_TXNS_24H
        FROM prev_amount p
    ),

    -- Temporal/velocity/sequence features (all backward-looking)
    temporal_feats AS (
        SELECT
            *,
            CASE WHEN _RN_DAY = 1 THEN 1 ELSE 0 END AS FIRST_TXN_OF_DAY_FROM,
            AMOUNT_PAID / NULLIF(AVG_TXN_AMOUNT_7D_FROM, 0) AS AMOUNT_VS_7D_AVG_RATIO,
            (AMOUNT_PAID - AVG_TXN_AMOUNT_7D_FROM) / NULLIF(TXN_AMOUNT_STD_7D_FROM, 0) AS AMOUNT_ZSCORE_7D_FROM,
            (AMOUNT_PAID - PREV_AMOUNT_PAID) / NULLIF(PREV_AMOUNT_PAID, 0) AS PCT_CHANGE_VS_PREV,
            TXN_COUNT_7D_FROM / 7.0 AS TXN_VELOCITY_7D_FROM
        FROM similar_amounts
    )

    SELECT
        -- Keys
        TS AS EVENT_TIMESTAMP,
        FROM_BANK,
        FROM_ACCOUNT,
        TO_BANK,
        TO_ACCOUNT,

        -- Labels
        IS_LAUNDERING,

        -- Raw amounts
        AMOUNT_PAID,
        AMOUNT_RECEIVED,
        LN(AMOUNT_PAID) AS LOG_AMOUNT,

        -- Large txn flag (simple rule-of-thumb; you can replace with percentile-based later)
        CASE WHEN AMOUNT_PAID >= 100000 THEN 1 ELSE 0 END AS IS_LARGE_TXN,

        -- Rounded amount (multiples of 1000)
        CASE WHEN MOD(AMOUNT_PAID, 1000) = 0 THEN 1 ELSE 0 END AS IS_ROUNDED_AMOUNT,

        -- Currencies & formats
        RECEIVING_CURRENCY,
        PAYMENT_CURRENCY,
        CASE WHEN RECEIVING_CURRENCY <> PAYMENT_CURRENCY THEN 1 ELSE 0 END AS CURRENCIES_DIFFERENT,
        PAYMENT_FORMAT,
        SAME_BANK,
        HOUR_OF_DAY,
        DAY_OF_WEEK,
        IS_WEEKEND,

        -- FROM-side features
        TIME_SINCE_LAST_TXN_FROM,
        TXN_COUNT_1H_FROM,
        TXN_COUNT_24H_FROM,
        TXN_COUNT_7D_FROM,
        TXN_AMOUNT_SUM_24H_FROM,
        AVG_TXN_AMOUNT_7D_FROM,
        MAX_TXN_AMOUNT_7D_FROM,
        NUM_UNIQUE_RECEIVERS,
        TXN_AMOUNT_STD_7D_FROM,
        NUM_UNIQUE_BANKS,
        CROSS_BANK_RATIO_7D,
        FIRST_TXN_OF_DAY_FROM,
        AMOUNT_VS_7D_AVG_RATIO,
        AMOUNT_ZSCORE_7D_FROM,
        PCT_CHANGE_VS_PREV,
        TXN_VELOCITY_7D_FROM,

        -- TO-side features
        TIME_SINCE_LAST_TXN_TO,
        NUM_UNIQUE_SENDERS,

        -- Counterparty features
        IS_NEW_COUNTERPARTY,
        REPEAT_COUNTERPARTY_COUNT,
        NUM_SIMILAR_AMOUNT_TXNS_24H,

        -- Gap irregularity
        TXN_GAP_STD_7D,

        -- Entity features
        FROM_ENTITY_ID,
        FROM_ENTITY_NAME,
        TO_ENTITY_ID,
        TO_ENTITY_NAME,
        FROM_BANK_NAME,
        TO_BANK_NAME,
        FROM_NUM_ACCOUNTS_PER_ENTITY,
        TO_NUM_ACCOUNTS_PER_ENTITY

    FROM temporal_feats
    """

    df_features = session.sql(feature_sql)
    return df_features


session = Session.builder.configs(connection_parameters).create()
df_features = build_transaction_features(session)
df_features.write.save_as_table("AML_PROJECT.FEATURES.TRANSACTION_FEATURES", mode="overwrite")

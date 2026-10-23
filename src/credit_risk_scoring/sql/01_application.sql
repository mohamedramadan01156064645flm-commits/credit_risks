-- Credit application: cleaning and business ratios (one row per client).
-- Corrections derived from exploratory data analysis (notebooks/01_exploration.ipynb).
SELECT
    * EXCLUDE (CODE_GENDER, DAYS_EMPLOYED),   -- CODE_GENDER: sensitive feature, excluded from the model

    -- 365243 = "no declared employment" code (mostly pensioners), not an actual duration
    CASE WHEN DAYS_EMPLOYED = 365243 THEN NULL ELSE DAYS_EMPLOYED END AS DAYS_EMPLOYED,
    (DAYS_EMPLOYED = 365243)::INTEGER                          AS DAYS_EMPLOYED_ANOM,

    -DAYS_BIRTH / 365.25                                       AS AGE_YEARS,
    AMT_CREDIT  / NULLIF(AMT_INCOME_TOTAL, 0)                  AS CREDIT_INCOME_RATIO,
    AMT_ANNUITY / NULLIF(AMT_INCOME_TOTAL, 0)                  AS ANNUITY_INCOME_RATIO,   -- debt-to-income ratio
    AMT_ANNUITY / NULLIF(AMT_CREDIT, 0)                        AS CREDIT_TERM,            -- inverse of term length
    AMT_CREDIT  / NULLIF(AMT_GOODS_PRICE, 0)                   AS CREDIT_GOODS_RATIO,     -- financed share
    AMT_INCOME_TOTAL / NULLIF(CNT_FAM_MEMBERS, 0)              AS INCOME_PER_PERSON,
    -- share of adult life spent in current employment; note: original column still contains 365243
    NULLIF(DAYS_EMPLOYED, 365243) / NULLIF(DAYS_BIRTH, 0)     AS EMPLOYED_AGE_RATIO
FROM application_train

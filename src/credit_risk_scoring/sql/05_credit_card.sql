-- Monthly credit card balances at Home Credit. SK_DPD = days past due.
SELECT
    SK_ID_CURR,
    COUNT(DISTINCT SK_ID_PREV)                                     AS CC_N_CARDS,
    AVG(AMT_BALANCE / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0))          AS CC_MEAN_UTILIZATION,   -- credit limit utilization share
    MAX(AMT_BALANCE / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0))          AS CC_MAX_UTILIZATION,
    AVG(AMT_DRAWINGS_ATM_CURRENT)                                  AS CC_MEAN_ATM_DRAWINGS,  -- ATM cash withdrawals
    AVG((SK_DPD > 0)::INTEGER)                                     AS CC_LATE_MONTHS_RATIO,
    MAX(SK_DPD)                                                    AS CC_MAX_DPD
FROM credit_card_balance
GROUP BY SK_ID_CURR

-- Monthly balances for point-of-sale (POS) consumer loans and personal loans at Home Credit.
WITH by_loan AS (
    -- one loan = one row; keep the remaining unpaid share as of its most recent statement
    SELECT
        SK_ID_CURR,
        SK_ID_PREV,
        ARG_MAX(CNT_INSTALMENT_FUTURE / NULLIF(CNT_INSTALMENT, 0), MONTHS_BALANCE) AS remaining_ratio,
        AVG((SK_DPD > 0)::INTEGER)                                                 AS late_months_ratio,
        COUNT(*)                                                                   AS n_months,
        MAX(SK_DPD)                                                                AS max_dpd
    FROM POS_CASH_balance
    GROUP BY SK_ID_CURR, SK_ID_PREV
)
SELECT
    SK_ID_CURR,
    COUNT(*)                                                           AS POS_N_LOANS,
    SUM(late_months_ratio * n_months) / SUM(n_months)                  AS POS_LATE_MONTHS_RATIO,
    MAX(max_dpd)                                                       AS POS_MAX_DPD,
    AVG(remaining_ratio)                                               AS POS_REMAINING_RATIO
FROM by_loan
GROUP BY SK_ID_CURR

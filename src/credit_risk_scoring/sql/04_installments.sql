-- Repayment history for previous Home Credit loans.
-- An installment may be paid across multiple payments: group payments by installment first.
WITH by_instalment AS (
    SELECT
        SK_ID_CURR,
        SK_ID_PREV,
        NUM_INSTALMENT_NUMBER,
        NUM_INSTALMENT_VERSION,
        MAX(DAYS_INSTALMENT)                       AS due_day,
        MAX(DAYS_ENTRY_PAYMENT)                    AS last_payment_day,
        MAX(AMT_INSTALMENT)                        AS amount_due,
        SUM(AMT_PAYMENT)                           AS amount_paid
    FROM installments_payments
    GROUP BY ALL
),
enriched AS (
    SELECT
        *,
        GREATEST(last_payment_day - due_day, 0)    AS days_late,
        -- round to the cent: avoids counting 0.000000001 as unpaid due to floating-point imprecision
        GREATEST(ROUND(amount_due - amount_paid, 2), 0) AS amount_unpaid
    FROM by_instalment
)
SELECT
    SK_ID_CURR,
    COUNT(*)                                       AS INST_N_INSTALMENTS,
    AVG((days_late > 0)::INTEGER)                  AS INST_LATE_RATIO,
    AVG((days_late > 30)::INTEGER)                 AS INST_LATE_30_RATIO,
    AVG(days_late)                                 AS INST_MEAN_DAYS_LATE,
    MAX(days_late)                                 AS INST_MAX_DAYS_LATE,
    AVG((amount_unpaid > 0)::INTEGER)              AS INST_UNDERPAID_RATIO,
    SUM(amount_unpaid)                             AS INST_TOTAL_UNPAID,
    -- recent behavior: installments in the last year
    AVG((days_late > 0)::INTEGER) FILTER (WHERE due_day >= -365) AS INST_LATE_RATIO_LAST_YEAR
FROM enriched
GROUP BY SK_ID_CURR

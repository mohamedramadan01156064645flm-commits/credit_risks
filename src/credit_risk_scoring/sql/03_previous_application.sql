-- Client's previous applications with Home Credit.
SELECT
    SK_ID_CURR,
    COUNT(*)                                                       AS PREV_N_APPLICATIONS,
    AVG((NAME_CONTRACT_STATUS = 'Approved')::INTEGER)              AS PREV_APPROVED_RATIO,
    AVG((NAME_CONTRACT_STATUS = 'Refused')::INTEGER)               AS PREV_REFUSED_RATIO,
    AVG((NAME_CONTRACT_STATUS = 'Canceled')::INTEGER)              AS PREV_CANCELED_RATIO,
    -- credit amount approved / amount applied for: < 1 if the lender granted less than requested
    AVG(AMT_CREDIT / NULLIF(AMT_APPLICATION, 0))                   AS PREV_CREDIT_APPLICATION_RATIO,
    AVG(AMT_CREDIT)                                                AS PREV_MEAN_CREDIT,
    AVG(CNT_PAYMENT)                                               AS PREV_MEAN_N_PAYMENTS,
    MAX(DAYS_DECISION)                                             AS PREV_LAST_DECISION_DAYS,
    MAX(DAYS_DECISION) FILTER (WHERE NAME_CONTRACT_STATUS = 'Refused') AS PREV_LAST_REFUSAL_DAYS
FROM previous_application
GROUP BY SK_ID_CURR

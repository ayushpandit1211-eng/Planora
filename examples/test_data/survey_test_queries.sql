-- Test queries for annual-enterprise-survey-2025.csv
-- Upload the CSV with dataset name: survey   (the table is then called "survey")
-- Paste ONE query at a time into "...or paste a slow SQL query" and click "Run optimisation pipeline".

-- Q1. Read everything (expected: "already efficient", no index needed)
SELECT year, industry_name_anzsic, rme_size_grp, variable, value
FROM survey;

-- Q2. Total business count per industry in 2025 (expected: index on year, about -89% cost)
SELECT industry_name_anzsic, sum(value) AS total
FROM survey
WHERE variable = 'Activity unit' AND year = 2025
GROUP BY 1
ORDER BY 2 DESC;

-- Q3. Employees per year and size group (expected: index on variable, about -24% cost)
SELECT year, rme_size_grp, sum(value) AS employees
FROM survey
WHERE variable = 'Rolling mean employees'
GROUP BY 1, 2
ORDER BY 1, 2;

-- Q4. All Manufacturing rows for 2025 (expected: index on (industry_code_anzsic, year), about -72% cost)
SELECT *
FROM survey
WHERE industry_code_anzsic = 'C' AND year = 2025;

-- Q5. Income of mid-size firms, largest first (expected: 3-column index, about -96% cost)
SELECT industry_name_anzsic, year, value
FROM survey
WHERE variable = 'Total income' AND rme_size_grp = 'f_50-99'
ORDER BY value DESC;

-- Q6. INEFFICIENT on purpose: rows above that year's average, using a subquery
--     that re-scans the table for every row (expected: two indexes, about -99% cost)
SELECT s.industry_name_anzsic, s.year, s.value
FROM survey s
WHERE s.variable = 'Activity unit'
  AND s.value > (SELECT avg(x.value)
                 FROM survey x
                 WHERE x.variable = 'Activity unit' AND x.year = s.year)
ORDER BY s.year;

-- Allowed values you can put in the filters:
-- variable     : 'Activity unit', 'Fixed tangible assets', 'Operating profit before tax',
--                'Rolling mean employees', 'Salaries and wages paid',
--                'Sales, government funding, grants and subsidies', 'Total assets',
--                'Total expenditure', 'Total income'
-- rme_size_grp : 'a_0', 'b_1-5', 'c_6-9', 'd_10-19', 'e_20-49', 'f_50-99',
--                'g_100-199', 'h_200+', 'i_Industry_Total', 'j_Grand_Total'
-- year         : 2011 to 2025

# Data Dictionary: Online Retail II

**Source:** UCI Machine Learning Repository,
[Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) (Chen, 2019)

**What it is:** every transaction of a UK-based online retailer that mainly sells
giftware, many of its customers being wholesalers.

**How to get it:**

```bash
python -m econocausal.data.download   # downloads and extracts the Excel file (~46 MB)
python -m econocausal.data.load       # converts both sheets to data/raw/online_retail_ii.parquet
```

## Overview

| Property | Value |
|----------|-------|
| Rows | 1,067,371 (one row per product line on an invoice) |
| Period | 2009-12-01 to 2011-12-09 |
| Rows per year | 2009: 45,228 · 2010: 522,714 · 2011: 499,429 |
| Invoices | 53,628 |
| Stock codes | 5,131 |
| Customers | 5,942 |
| Countries | 43 (United Kingdom is 91.9% of rows) |
| Original file | Excel, two sheets: "Year 2009-2010" and "Year 2010-2011" |

## Columns

Column names are converted to snake_case in `econocausal.data.load.standardize`.

| Column | Original name | Type | Description | Missing |
|--------|---------------|------|-------------|---------|
| `invoice` | Invoice | string | Invoice number, 6 digits. Starts with **C** for a cancellation and **A** for an accounting adjustment. | 0 |
| `stock_code` | StockCode | string | Product code, usually 5 digits plus an optional letter for variants (e.g. `85123A`). Some codes are not products (see below). | 0 |
| `description` | Description | string | Product name. | 4,382 |
| `quantity` | Quantity | int | Units of the product on this invoice line. Negative for cancellations and stock adjustments. | 0 |
| `invoice_date` | InvoiceDate | datetime | Date and time of the invoice. | 0 |
| `price` | Price | float | Unit price in GBP (£). | 0 |
| `customer_id` | Customer ID | nullable int | 5-digit customer number. | 243,007 (22.8%) |
| `country` | Country | string | Country where the customer lives. | 0 |

## How these columns map to the causal model

| Role in the model | Built from |
|-------------------|------------|
| Treatment: log price | `price`, aggregated per product per week |
| Outcome: log quantity | `quantity`, summed per product per week |
| Unit of analysis | `stock_code` x week of `invoice_date` |
| Confounders | week, month, holidays (from `invoice_date`); product and category (from `stock_code`, `description`); customer mix (`country`, `customer_id`); past sales and prices |

## Data quality issues found

Handled in `econocausal.data.clean`. The "Plan" column describes the rule applied.

| Issue | Size | Plan |
|-------|------|------|
| Cancelled invoices (`C...`) | 19,494 rows, all with negative quantity | Remove from sales. Possibly keep a cancellation rate as a feature. |
| Other negative quantities | 3,457 rows, all with price 0 (stock write-offs, damages) | Remove |
| Price ≤ 0 | 6,207 rows | Remove; a zero price is not a real sale |
| Accounting adjustments (`A...`, stock code `B`, "Adjust bad debt") | 6 rows with large negative prices | Remove |
| Non-product stock codes | 6,093 rows over 61 codes: `POST` (postage), `DOT` (dotcom postage), `M` (manual), `C2` (carriage), `D` (discount), `S` (samples), `BANK CHARGES`, `ADJUST`, `AMAZONFEE`, gift vouchers, etc. | Remove; they are fees and adjustments, not products |
| Exact duplicate rows | 34,335 | Investigate: some may be real repeat lines on one invoice. Likely drop. |
| Missing `customer_id` | 22.8% of rows | Keep for sales totals (the sale still happened); only needed for customer-level features |
| Missing `description` | 4,382 rows | Fill from other rows with the same `stock_code` |
| Same stock code, several descriptions | up to 9 per code | Use the most frequent description per code |
| Extreme quantities | min −80,995, max 80,995 (one bulk order that was cancelled); 99th percentile is 100 | Clip or remove extreme outliers after removing cancellations |
| Extreme prices | max £38,970 (manual/adjustment lines); 99th percentile is £18 | Mostly removed with the non-product codes; check what remains |

## Cleaning results

Output of `python -m econocausal.data.build`:

| Step | Rows removed | Rows left |
|------|-------------:|----------:|
| Raw data | | 1,067,371 |
| Cancellations and adjustments (`C`, `A` invoices) | 19,500 | 1,047,871 |
| Quantity ≤ 0 or price ≤ 0 | 6,202 | 1,041,669 |
| Non-product stock codes | 4,792 | 1,036,877 |
| Exact duplicate lines | 33,663 | 1,003,214 |
| Price more than 10x above/below the product's median | 882 | 1,002,332 |
| Bulk lines above the 99.99th quantity percentile | 101 | **1,002,231** |

Missing descriptions are then filled with the most frequent description of each
stock code, so every product has exactly one name.

## Weekly panel (`weekly_sales` table)

Built in `econocausal.data.panel`. One row per product per week (weeks start on Monday).

| Column | Description |
|--------|-------------|
| `stock_code`, `description`, `category` | product, its name and a keyword-based category |
| `week_start` | Monday of the week |
| `price` | **median** unit price that week (treatment, before the log) |
| `quantity` | total units sold that week (outcome, before the log) |
| `revenue` | total revenue that week |
| `n_invoices`, `n_customers` | number of orders and distinct customers |
| `uk_share` | share of the week's order lines from UK customers |
| `guest_share` | share of the week's order lines with no customer ID. These one-off buyers pay a higher retail price (about 2x the product's median), so this column is a key confounder |
| `log_price`, `log_quantity` | model variables |
| `year`, `month`, `week_of_year` | calendar |
| `weeks_to_christmas` | weeks until the next 25 December |
| `is_peak_season` | September to November, when wholesale customers stock up for Christmas |
| `has_bank_holiday` | week contains a UK bank holiday |

Filters and result:

- The last week (starting 2011-12-05) is dropped because the data ends on a Friday.
- Products need at least **30 weeks** of sales and at least **5 different weekly prices**.
- Result: **126,672 rows, 1,877 products, weeks from 2009-11-30 to 2011-11-28**, no missing values.
- Weeks with no sales are not in the panel, because the price is not observed in those weeks.
- About 37% of panel rows fall into category `other`; the keyword rules can be refined later.

## Notes for the elasticity analysis

- **Price variation:** of 4,745 products with valid sales, **2,160 have at least 5
  different unit prices**. That is enough variation to estimate elasticities for a good
  number of products.
- **Bulk pricing:** the same product is often sold at a lower unit price in large
  quantities. Price per invoice line can therefore be partly *caused by* quantity. The
  weekly aggregation will use a typical (median) price per product per week to reduce
  this effect (see `docs/problem_statement.md`, section 4).
- **2009 is only one month** (December), so yearly comparisons should start in 2010.

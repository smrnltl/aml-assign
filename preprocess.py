"""
Preprocess NEPSE commercial-bank financial disclosures.

Pipeline:
  raw long CSV -> annual wide CSV -> ratios CSV (with forward NPL labels)

Row grain:        one row per (Bank, FiscalYear)
Stock fields:     Q4 value (Balance Sheet, NRB Ratios are point-in-time)
Flow fields:      Q4 cumulative (Income Statement reports YTD; Q4 = full year)
Duplicate cols:   coalesced (older NRB schema vs newer NFRS schema; zero overlap)
Forward label:    NPL(t+1 year) - NPL(t), regression target
                  binary deterioration class = 1 if delta > 0.5pp else 0

Outputs:
  dataset/processed/financial_data_wide.csv   - all source fields, pivoted
  dataset/processed/financial_ratios.csv      - 8 features + 2 labels
"""

import csv
import os
from collections import defaultdict

INPUT_PATH = os.path.join("dataset", "financial_data_all_normalized.csv")
OUT_DIR = os.path.join("dataset", "processed")
WIDE_PATH = os.path.join(OUT_DIR, "financial_data_wide.csv")
RATIOS_PATH = os.path.join(OUT_DIR, "financial_ratios.csv")

# Pairs found to have zero row-overlap. First entry = older schema, second = newer NFRS schema.
# Coalesce policy: take first non-null across the pair.
DUPLICATE_PAIRS = [
    ("TOTAL_ASSETS", "Total_Assets"),
    ("LIABILITIES", "Total_Liabilities"),
    ("SHAREHOLDERS_EQUITY", "Total_Equity"),
    ("TOTAL_LIABILITIES_AND_EQUITY", "Total_Liabilities_And_Equity"),
    ("PAID_UP_CAPITAL", "Share_Capital"),
    ("Profit_Loss_for_the_period", "Profit_(Loss)_For_The_Period"),
    ("Net_Fees_and_Commission_Income", "Net_Fees_And_Commission_Income"),
    ("Fees_and_Commission_Income", "Fees_And_Commission_Income"),
    ("Fees_and_Commission_Expense", "Fees_And_Commission_Expense"),
    ("Net_Interest_Fee_and_Commission_Income", "Net_Interest,_Fees_And_Commission_Income"),
    ("Impairment_Charge", "Impairment_Charge_(Reversal)_For_Loans_And_Other_Losses"),
    ("Staff_Expenses", "Personnel_Expenses"),
    ("Depreciation_and_Amortization", "Depreciation_&_Amortization"),
    ("Profit_before_Tax", "Profit_Before_Income_Tax"),
    ("Income_Tax", "Income_Tax_Expense"),
    ("CASH_AND_BANK_BALANCE", "Cash_And_Cash_Equivalent"),
    ("Due_from_NRB", "Due_From_Nepal_Rastra_Bank"),
    ("Placement_with_BFIs", "Placement_With_Bank_And_Financial_Institutions"),
    ("Loans_and_Advances_to_BFIs", "Loan_And_Advances_To_B_Fis"),
    ("Loans_and_Advances_to_Customers", "Loans_And_Advances_To_Customers"),
    ("Investment_in_Subsidiaries", "Investment_In_Subsidiaries"),
    ("Investment_in_Associates", "Investment_In_Associates"),
    ("PROPERTY_AND_EQUIPMENT", "Property_And_Equipment"),
    ("Goodwill_and_Intangible_Assets", "Goodwill_And_Intangible_Assets"),
    ("Due_to_BFIs", "Due_To_Bank_And_Financial_Institutions"),
    ("Due_to_NRB", "Due_To_Nepal_Rastra_Bank"),
    ("DEPOSITS", "Deposits_From_Customers"),
]


def to_float(v):
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() in ("none", "nan", "null"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def non_null(v):
    return to_float(v) is not None


def non_null_nonzero(v):
    f = to_float(v)
    return f is not None and f != 0.0


def coalesce(*vals):
    for v in vals:
        f = to_float(v)
        if f is not None:
            return f
    return None


def coalesce_nonzero(*vals):
    """Like coalesce but skips literal zeros. Used for Net Profit fallback where
    `0.0` in the older-schema column actually means 'not reported under this schema'."""
    for v in vals:
        f = to_float(v)
        if f is not None and f != 0.0:
            return f
    return None


def load_raw(path):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader]
    return header, rows


def build_wide(header, rows):
    """
    Pivot from (bank, year, quarter, report_type) -> one row per (bank, year).

    For each bank-year we keep:
      - All Balance Sheet fields at Q4 (point-in-time stock at year end).
      - All Income Statement fields at Q4 (cumulative YTD = full-year flow).
      - All NRB Ratios fields at Q4 (regulatory ratios at year end).
      - All Distributable Profit fields at Q4.
    If Q4 is missing for a report type, fall back to the latest available quarter.
    """
    idx = {h: i for i, h in enumerate(header)}

    META_COLS = {"Company", "FiscalYear", "FiscalYearAD", "StockSymbol",
                 "ReportType", "Quarter", "Timeline", "Precision", "DataId", "Status"}
    data_cols = [h for h in header if h not in META_COLS]

    # group rows by (bank, fiscal_year, quarter, report_type)
    cells = defaultdict(dict)  # (bank, fy) -> {col: value}
    fy_ad_lookup = {}           # (bank, fy) -> FiscalYearAD

    # First pass: build per-bank-year, picking best quarter per report type per column.
    QUARTER_PRIORITY = ["Q4", "Q3", "Q2", "Q1"]  # prefer Q4; fall back if missing

    by_group = defaultdict(list)  # (bank, fy, report_type) -> list of rows
    for row in rows:
        bank, fy, _fy_ad, _sym, rtype, qtr = row[0], row[1], row[2], row[3], row[4], row[5]
        by_group[(bank, fy, rtype)].append((qtr, row))
        fy_ad_lookup[(bank, fy)] = row[idx["FiscalYearAD"]]

    for (bank, fy, rtype), rlist in by_group.items():
        # Pick the row with the highest-priority quarter that has any value.
        # Two-pass: first prefer non-zero values (zeros under one schema often mean
        # "not reported in this disclosure format"); only fall back to zero if every
        # quarter genuinely has zero.
        by_q = {r[0]: r[1] for r in rlist}
        for col in data_cols:
            ci = idx[col]
            picked = None
            # pass 1: best non-zero value across quarters in priority order
            for q in QUARTER_PRIORITY:
                if q in by_q:
                    v = by_q[q][ci]
                    if non_null_nonzero(v):
                        picked = to_float(v)
                        break
            # pass 2: accept zero if non-zero never found
            if picked is None:
                for q in QUARTER_PRIORITY:
                    if q in by_q:
                        v = by_q[q][ci]
                        if non_null(v):
                            picked = to_float(v)
                            break
            if picked is not None:
                key = (bank, fy)
                # don't overwrite if another report type already wrote this col
                if col not in cells[key]:
                    cells[key][col] = picked

    # Coalesce duplicate column pairs.
    coalesced_cols = []
    drop_cols = set()
    for old, new in DUPLICATE_PAIRS:
        if old in idx and new in idx:
            # canonical name = old (shorter, original NRB-style)
            for key in cells:
                v = coalesce(cells[key].get(old), cells[key].get(new))
                if v is not None:
                    cells[key][old] = v
                if new in cells[key]:
                    del cells[key][new]
            drop_cols.add(new)

    # Final column list for wide CSV
    final_cols = ["Company", "FiscalYear", "FiscalYearAD"] + [
        c for c in data_cols if c not in drop_cols
    ]
    return cells, final_cols, fy_ad_lookup


def write_wide(cells, final_cols, fy_ad_lookup, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    keys = sorted(cells.keys(), key=lambda k: (k[0], k[1]))
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(final_cols)
        for (bank, fy) in keys:
            row_out = [bank, fy, fy_ad_lookup.get((bank, fy), "")]
            for col in final_cols[3:]:
                v = cells[(bank, fy)].get(col)
                row_out.append("" if v is None else v)
            w.writerow(row_out)
    return len(keys)


def safe_div(a, b):
    if a is None or b is None:
        return None
    if b == 0:
        return None
    return a / b


def compute_ratios(cells, fy_ad_lookup):
    """
    Produce one row per (bank, year) with input features and forward labels.

    Base features (8):
      1. CAR              <- Capital_Fund_to_RWA
      2. NPL_Ratio        <- Non-Performing_Loan_(NPL)_to_Total_Loan
      3. CD_Ratio         <- Credit_to_Deposit_Ratio
      4. Cost_of_Funds    <- Cost_of_Funds
      5. Base_Rate        <- Base_Rate
      6. Interest_Spread  <- Interest_Rate_Spread
      7. ROE_derived      <- Net_Profit / SHAREHOLDERS_EQUITY (annualised)
      8. Loan_Growth_YoY  <- YoY change in Loans_and_Advances_to_Customers

    Lagged features (5, momentum signal, no look-ahead):
      9. NPL_Ratio_lag1
     10. CAR_lag1
     11. CD_Ratio_lag1
     12. Delta_NPL_lag1   <- NPL(t) - NPL(t-1)  (last year's actual change)
     13. Delta_CAR_lag1   <- CAR(t) - CAR(t-1)

    Bank-relative deviations (3, cross-sectional position within year):
     14. NPL_dev          <- NPL_Ratio - median across 10 banks that year
     15. CAR_dev          <- CAR - median across 10 banks that year
     16. ROE_dev          <- ROE_derived - median across 10 banks that year

    Labels (computed after sorting per bank by year):
      NPL_next_year             - next-year NPL ratio (raw, for inspection)
      Delta_NPL_next_year       - NPL(t+1) - NPL(t)         [regression target]
      Deteriorate_next_year     - 1 if Delta_NPL > 0.5pp    [classification target]
    """
    # Build per-bank ordered timelines first so we can compute YoY and forward labels.
    keys = sorted(cells.keys(), key=lambda k: (k[0], k[1]))

    # Pre-extract working columns per (bank, fy)
    work = {}
    for key in keys:
        row = cells[key]
        npl = row.get("Non-Performing_Loan_(NPL)_to_Total_Loan")
        car = row.get("Capital_Fund_to_RWA")
        cd = row.get("Credit_to_Deposit_Ratio")
        cof = row.get("Cost_of_Funds")
        br = row.get("Base_Rate")
        spread = row.get("Interest_Rate_Spread")

        equity = row.get("SHAREHOLDERS_EQUITY")
        # Net profit: pick the first non-zero across the three possible schema names.
        # A literal 0.0 in any of these columns means "not reported in that schema",
        # not "the bank made zero profit".
        net_profit = coalesce_nonzero(
            row.get("Net_Profit_Loss_as_per_profit_or_loss"),
            row.get("Profit_Loss_for_the_period"),
            row.get("Net_Profit_or_Loss_as_per_statement_of_Profit_or_Loss"),
        )

        loans = row.get("Loans_and_Advances_to_Customers")
        if loans is None:
            loans = row.get("LOANS")

        work[key] = {
            "CAR": car,
            "NPL_Ratio": npl,
            "CD_Ratio": cd,
            "Cost_of_Funds": cof,
            "Base_Rate": br,
            "Interest_Spread": spread,
            "ROE_derived": (safe_div(net_profit, equity) * 100) if (net_profit is not None and equity not in (None, 0, 0.0)) else None,
            "_loans": loans,
        }

    # Group by bank, ordered by FY string (FY codes sort lexically because format is "YYYY/YYYY").
    by_bank = defaultdict(list)
    for key in keys:
        by_bank[key[0]].append(key)

    # Build interim rows first (base + lagged + forward labels). Bank-relative
    # deviations require a second pass because they depend on cross-bank medians per year.
    interim = []
    for bank, bank_keys in by_bank.items():
        # bank_keys already sorted by fy
        for i, key in enumerate(bank_keys):
            cur = work[key]
            # Loan growth YoY: this year's loans vs previous year's
            loan_growth = None
            if i > 0:
                prev = work[bank_keys[i - 1]]
                if cur["_loans"] is not None and prev["_loans"] not in (None, 0):
                    loan_growth = (cur["_loans"] - prev["_loans"]) / prev["_loans"] * 100

            # Lagged features (look one year back, never forward)
            npl_lag1 = None
            car_lag1 = None
            cd_lag1 = None
            delta_npl_lag1 = None
            delta_car_lag1 = None
            if i > 0:
                prev = work[bank_keys[i - 1]]
                npl_lag1 = prev["NPL_Ratio"]
                car_lag1 = prev["CAR"]
                cd_lag1 = prev["CD_Ratio"]
                if cur["NPL_Ratio"] is not None and prev["NPL_Ratio"] is not None:
                    delta_npl_lag1 = cur["NPL_Ratio"] - prev["NPL_Ratio"]
                if cur["CAR"] is not None and prev["CAR"] is not None:
                    delta_car_lag1 = cur["CAR"] - prev["CAR"]

            # Forward label: NPL(t+1) - NPL(t)
            npl_next = None
            delta_npl = None
            deteriorate = None
            if i + 1 < len(bank_keys):
                nxt = work[bank_keys[i + 1]]
                npl_next = nxt["NPL_Ratio"]
                if cur["NPL_Ratio"] is not None and npl_next is not None:
                    delta_npl = npl_next - cur["NPL_Ratio"]
                    deteriorate = 1 if delta_npl > 0.5 else 0

            interim.append({
                "Company": bank,
                "FiscalYear": key[1],
                "FiscalYearAD": fy_ad_lookup.get(key, ""),
                "CAR": cur["CAR"],
                "NPL_Ratio": cur["NPL_Ratio"],
                "CD_Ratio": cur["CD_Ratio"],
                "Cost_of_Funds": cur["Cost_of_Funds"],
                "Base_Rate": cur["Base_Rate"],
                "Interest_Spread": cur["Interest_Spread"],
                "ROE_derived": cur["ROE_derived"],
                "Loan_Growth_YoY": loan_growth,
                "NPL_Ratio_lag1": npl_lag1,
                "CAR_lag1": car_lag1,
                "CD_Ratio_lag1": cd_lag1,
                "Delta_NPL_lag1": delta_npl_lag1,
                "Delta_CAR_lag1": delta_car_lag1,
                "NPL_next_year": npl_next,
                "Delta_NPL_next_year": delta_npl,
                "Deteriorate_next_year": deteriorate,
            })

    # Second pass: bank-relative deviations (cross-sectional median per FY)
    by_fy = defaultdict(list)
    for r in interim:
        by_fy[r["FiscalYear"]].append(r)

    def median(values):
        vs = sorted(v for v in values if v is not None)
        if not vs:
            return None
        n = len(vs)
        return vs[n // 2] if n % 2 == 1 else (vs[n // 2 - 1] + vs[n // 2]) / 2

    for fy, fy_rows in by_fy.items():
        npl_median = median(r["NPL_Ratio"] for r in fy_rows)
        car_median = median(r["CAR"] for r in fy_rows)
        roe_median = median(r["ROE_derived"] for r in fy_rows)
        for r in fy_rows:
            r["NPL_dev"] = (r["NPL_Ratio"] - npl_median) if (r["NPL_Ratio"] is not None and npl_median is not None) else None
            r["CAR_dev"] = (r["CAR"] - car_median) if (r["CAR"] is not None and car_median is not None) else None
            r["ROE_dev"] = (r["ROE_derived"] - roe_median) if (r["ROE_derived"] is not None and roe_median is not None) else None

    return interim


def write_ratios(rows, out_path):
    cols = [
        "Company", "FiscalYear", "FiscalYearAD",
        # base features
        "CAR", "NPL_Ratio", "CD_Ratio", "Cost_of_Funds", "Base_Rate",
        "Interest_Spread", "ROE_derived", "Loan_Growth_YoY",
        # lagged features
        "NPL_Ratio_lag1", "CAR_lag1", "CD_Ratio_lag1",
        "Delta_NPL_lag1", "Delta_CAR_lag1",
        # bank-relative deviations
        "NPL_dev", "CAR_dev", "ROE_dev",
        # labels
        "NPL_next_year", "Delta_NPL_next_year", "Deteriorate_next_year",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow(["" if r[c] is None else r[c] for c in cols])
    return cols


def fill_summary(rows, cols):
    n = len(rows)
    print(f"\nRows: {n}")
    print("Fill rates:")
    for c in cols:
        if c in ("Company", "FiscalYear", "FiscalYearAD"):
            continue
        k = sum(1 for r in rows if r[c] is not None)
        print(f"  {c:24s} {k:3d}/{n}  ({k/n:.0%})")


def main():
    print(f"Reading {INPUT_PATH}")
    header, rows = load_raw(INPUT_PATH)
    print(f"  raw shape: {len(rows)} rows x {len(header)} cols")

    cells, final_cols, fy_ad_lookup = build_wide(header, rows)
    print(f"  pivoted to {len(cells)} (bank, year) rows x {len(final_cols)} cols")

    n_wide = write_wide(cells, final_cols, fy_ad_lookup, WIDE_PATH)
    print(f"Wrote {WIDE_PATH}  ({n_wide} rows)")

    ratio_rows = compute_ratios(cells, fy_ad_lookup)
    cols = write_ratios(ratio_rows, RATIOS_PATH)
    print(f"Wrote {RATIOS_PATH}  ({len(ratio_rows)} rows)")

    fill_summary(ratio_rows, cols)

    labeled = [r for r in ratio_rows if r["Delta_NPL_next_year"] is not None]
    print(f"\nUsable labeled rows (forward NPL label present): {len(labeled)}/{len(ratio_rows)}")
    if labeled:
        pos = sum(1 for r in labeled if r["Deteriorate_next_year"] == 1)
        print(f"  Deteriorate=1: {pos}  /  Deteriorate=0: {len(labeled)-pos}  "
              f"(base rate {pos/len(labeled):.0%})")


if __name__ == "__main__":
    main()

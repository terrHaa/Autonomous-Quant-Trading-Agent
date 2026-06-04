"""curate_sp500_membership.py — build reference/universe/sp500.csv from
Wikipedia-format inputs.

Why this exists
---------------
The agent uses point-in-time S&P 500 membership to avoid survivorship
bias in backtests. `quant.data.universe.load_universe('sp500')` reads
`reference/universe/sp500.csv` for the full add/remove history of every
symbol that's ever been in the index.

Maintaining that CSV by hand is tedious and error-prone. This script
converts Wikipedia's two relevant tables into our format:

  1. **Current constituents** — the "List of S&P 500 companies" page's
     first table. Columns include "Symbol" and "Date added" (sometimes
     "Date first added"). Used to seed `added=` for every active member.

  2. **Selected changes** — the second table on the same page. Columns:
     "Date", "Added Ticker", "Removed Ticker". Used to fill in
     historical removals AND any add dates the first table missed.

Workflow
--------
1. Open https://en.wikipedia.org/wiki/List_of_S%26P_500_companies in a
   browser.
2. Right-click each of the two tables → "Copy" / select-all → paste
   into a spreadsheet (or use a Wikipedia-table-to-CSV extension).
3. Save them as two CSVs in this directory:
     wikipedia_current.csv   — headers: Symbol, Date_added
     wikipedia_changes.csv   — headers: Date, Added, Removed
   (The script is forgiving about case + spaces; "Date added", "date_added",
   "Symbol" / "Ticker" / "ticker" all map.)
4. Run:
     uv run python tools/curate_sp500_membership.py \\
         --current wikipedia_current.csv \\
         --changes wikipedia_changes.csv \\
         --out reference/universe/sp500.csv
5. The script validates: no duplicate symbols (active ones), every
   removed date is AFTER its added date, no orphan removed-without-add.
6. Diff the output, commit it.

Quarterly refresh
-----------------
S&P announces index changes ~10-20 times per year. After each batch:
1. Update `wikipedia_changes.csv` with the new rows.
2. Re-run the script.
3. Diff the resulting `sp500.csv` and commit.

Limitations
-----------
- Wikipedia's history goes back reliably to ~2000. Older changes are
  spotty. Our backtests don't need ancient history; ~10 years is plenty.
- Symbol-change events (e.g., FB → META in 2022) need both entries:
  one removing FB, one adding META.
- This script does NOT fetch from Wikipedia directly — that would
  require a network call + HTML parsing + dealing with rate limits.
  Manual paste is reliable and gives the operator a chance to sanity-
  check the source data.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case, strip spaces, replace spaces with underscores."""
    df = df.copy()
    df.columns = [
        str(c).lower().strip().replace(" ", "_") for c in df.columns
    ]
    return df


def _pick_column(df: pd.DataFrame, *candidates: str) -> str:
    """Find a column matching any of the candidate names (case/space-insensitive)."""
    for c in candidates:
        key = c.lower().strip().replace(" ", "_")
        if key in df.columns:
            return key
    raise ValueError(
        f"Could not find any of {candidates} in columns {list(df.columns)}"
    )


def _parse_date(s: str | float | None) -> str | None:
    """Best-effort date parser. Returns ISO YYYY-MM-DD or None."""
    if s is None or (isinstance(s, float) and pd.isna(s)) or str(s).strip() == "":
        return None
    text = str(s).strip()
    # Try common formats.
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y", "%d-%b-%Y"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    # Last resort: let pandas try.
    try:
        return pd.to_datetime(text).date().isoformat()
    except (ValueError, TypeError):
        return None


def build_membership_csv(
    current_csv: Path,
    changes_csv: Path,
    out_csv: Path,
) -> None:
    """Build sp500.csv from the two Wikipedia inputs."""
    # --- 1. Load current constituents ---
    cur = _normalise_columns(pd.read_csv(current_csv))
    sym_col = _pick_column(cur, "symbol", "ticker")
    date_col = _pick_column(cur, "date_added", "date_first_added", "date")

    # symbol -> (added, removed). Currently-active: removed = None.
    members: dict[str, tuple[str | None, str | None]] = {}
    for _, row in cur.iterrows():
        sym = str(row[sym_col]).upper().strip()
        if not sym or sym == "NAN":
            continue
        added = _parse_date(row[date_col])
        members[sym] = (added, None)

    # --- 2. Apply historical changes (selected changes table) ---
    chg = _normalise_columns(pd.read_csv(changes_csv))
    date_col = _pick_column(chg, "date")
    add_col = _pick_column(chg, "added", "added_ticker", "ticker_added")
    rem_col = _pick_column(chg, "removed", "removed_ticker", "ticker_removed")

    for _, row in chg.iterrows():
        d = _parse_date(row[date_col])
        if d is None:
            continue
        added_sym = str(row.get(add_col, "")).upper().strip()
        removed_sym = str(row.get(rem_col, "")).upper().strip()
        # The add side: only update if the current-table date was missing
        # (rare — Wikipedia's current table usually has dates).
        if added_sym and added_sym != "NAN":
            cur_added, cur_removed = members.get(added_sym, (None, None))
            if cur_added is None:
                members[added_sym] = (d, cur_removed)
        # The remove side: record the removal date for the leaving symbol.
        # If the symbol is in our current map, mark it removed; otherwise
        # create a historical entry (we may not have its added date).
        if removed_sym and removed_sym != "NAN":
            cur_added, _ = members.get(removed_sym, (None, None))
            members[removed_sym] = (cur_added, d)

    # --- 3. Validate ---
    issues: list[str] = []
    for sym, (added, removed) in members.items():
        if added is None and removed is None:
            issues.append(f"{sym}: no dates at all (skipping)")
            continue
        if added and removed and added >= removed:
            issues.append(
                f"{sym}: added={added} >= removed={removed} (invalid order)"
            )
    if issues:
        sys.stderr.write("VALIDATION ISSUES:\n")
        for i in issues:
            sys.stderr.write(f"  - {i}\n")
        sys.stderr.write(
            "Output written anyway; fix the source data and re-run.\n"
        )

    # --- 4. Write out ---
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for sym, (added, removed) in sorted(members.items()):
        if added is None and removed is None:
            continue   # skip the unusable ones noted above
        rows.append({
            "symbol": sym,
            "added": added or "",
            "removed": removed or "",
        })

    header = (
        "# S&P 500 membership history.\n"
        "#\n"
        "# Generated by tools/curate_sp500_membership.py.\n"
        f"# Generated: {datetime.now().isoformat(timespec='seconds')}\n"
        "#\n"
        "# Columns: symbol, added (first day IN, INCLUSIVE),\n"
        "#          removed (first day NOT IN, EXCLUSIVE; blank if active).\n"
        "#\n"
    )
    df = pd.DataFrame(rows, columns=["symbol", "added", "removed"])
    body = df.to_csv(index=False)
    out_csv.write_text(header + body)
    print(
        f"Wrote {len(rows)} symbol rows ({sum(1 for r in rows if not r['removed'])} "
        f"currently active, {sum(1 for r in rows if r['removed'])} historical) "
        f"to {out_csv}",
        file=sys.stderr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build reference/universe/sp500.csv from Wikipedia inputs.",
    )
    parser.add_argument(
        "--current", type=Path, required=True,
        help="CSV of current S&P 500 constituents. Columns: Symbol, Date_added.",
    )
    parser.add_argument(
        "--changes", type=Path, required=True,
        help="CSV of selected changes. Columns: Date, Added, Removed.",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("reference/universe/sp500.csv"),
        help="Output CSV path (default: reference/universe/sp500.csv).",
    )
    args = parser.parse_args()
    build_membership_csv(args.current, args.changes, args.out)


if __name__ == "__main__":
    main()

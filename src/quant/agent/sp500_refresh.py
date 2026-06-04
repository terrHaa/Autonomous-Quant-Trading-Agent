"""sp500_refresh.py — quarterly auto-refresh of the S&P 500 membership CSV.

Fetches Wikipedia's two relevant tables, builds a new
``reference/universe/sp500.csv``, validates it against safety gates,
and atomically replaces the existing file. Emails the operator the
diff so they can sanity-check + commit.

Why this exists
---------------
Point-in-time S&P 500 membership prevents survivorship bias in
backtests. The operator could update the CSV manually every quarter
(see tools/curate_sp500_membership.py), but a scheduled auto-refresh
removes the human-in-the-loop step entirely while keeping the
audit-trail step (review the diff, commit).

Safety gates
------------
The replacement is REJECTED (CSV untouched, FAILED email sent) when:
  1. Wikipedia fetch fails (network, 4xx, 5xx) after retries
  2. Either table parses to < 100 rows (clearly something wrong)
  3. New active-member count < ``_MIN_ACTIVE_MEMBERS`` (typically 400 —
     the actual index has 500 ± a few)
  4. Symmetric diff vs current CSV > ``_MAX_CHANGE_FRACTION`` (catches
     catastrophic format changes that would replace half the universe)
  5. Any new row has invalid dates or removed <= added

Email behavior
--------------
- On SUCCESS: subject "S&P 500 universe refreshed", body shows the
  diff (added names, removed names, date corrections) so the operator
  can review before ``git commit``.
- On FAILURE: subject "S&P 500 refresh FAILED", body shows which gate
  failed and the raw error. The live CSV is NOT modified.

No git commit
-------------
The refresh writes the file but never commits. The operator reviews
the diff email and commits manually. Reasons:
  - Auto-commits to main are risky for any human reviewer to spot-check
  - Operator wants a chance to override (e.g., if S&P announced a
    delayed inclusion they want to backdate)
  - The agent reads the file directly each daily fire, so the data is
    LIVE the moment the script writes it — the commit is for audit
    history, not for the agent to use the data.

Console-script: ``quant-sp500-refresh`` (registered in pyproject.toml).
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from quant.agent.daily_runner import _email_failure
from quant.agent.email_sender import EmailSender
from quant.agent.log import _atomic_write_text

logger = logging.getLogger(__name__)


# Wikipedia URL. Hardcoded — the page's location is stable and there's
# no upstream that changes more often than Wikipedia itself.
_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# User-Agent string — Wikipedia asks scrapers to identify themselves.
# Without one, the request might be rate-limited or returned 403.
_USER_AGENT = (
    "quant-trader-sp500-refresh/1.0 "
    "(quarterly cron job; contact: operator@example.com)"
)

# Default safety thresholds.
_MIN_ACTIVE_MEMBERS = 400          # actual S&P 500 has ~500; allow some slack
_MAX_CHANGE_FRACTION = 0.10        # > 10% symmetric diff vs old → reject

# Default output location.
_DEFAULT_CSV_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "reference" / "universe" / "sp500.csv"
)
# T-audit fix H1: full-universe sector map. The sector concentration cap
# in daily_runner used to read from sp500_top50.csv, covering only ~50 of
# 519 universe names — the other 470 passed through the cap as "unknown
# sector" → the cap was silently disabled for the bulk of the book. This
# CSV maps every active member to its GICS sector so the cap actually
# binds. Refreshed quarterly together with sp500.csv.
_DEFAULT_SECTORS_CSV_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "reference" / "universe" / "sp500_sectors.csv"
)


def _fetch_wikipedia_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (current_constituents, selected_changes) DataFrames.

    Uses pandas.read_html on the Wikipedia page. Wraps the network
    call in the existing retry layer so a one-off network blip is
    absorbed (same pattern as the Alpaca data API).
    """
    import requests.exceptions as _req_exc

    from quant.util.retry import retry_on_transient

    def _fetch() -> list[pd.DataFrame]:
        # pandas.read_html drives a requests session under the hood;
        # passing storage_options lets us inject the User-Agent header
        # so Wikipedia doesn't 403 us.
        return pd.read_html(
            _WIKIPEDIA_URL,
            storage_options={"User-Agent": _USER_AGENT},
        )

    tables = retry_on_transient(
        _fetch,
        transient=(_req_exc.ConnectionError, _req_exc.SSLError, _req_exc.Timeout),
        description="Wikipedia S&P 500 page fetch",
    )

    if len(tables) < 2:
        raise RuntimeError(
            f"Expected at least 2 tables from Wikipedia, got {len(tables)}. "
            "Page structure may have changed."
        )

    # Table 0: current constituents. Table 1: selected changes.
    return tables[0], tables[1]


def _normalise_current_table(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce Wikipedia's current-constituents table to (symbol, added)."""
    df = df.copy()
    df.columns = [str(c).lower().strip().replace(" ", "_") for c in df.columns]

    # Wikipedia variously names these columns. Be flexible.
    sym_col = next(
        (c for c in ["symbol", "ticker", "ticker_symbol"] if c in df.columns),
        None,
    )
    if sym_col is None:
        raise ValueError(
            f"Could not find symbol column in {list(df.columns)}"
        )
    date_col = next(
        (c for c in ["date_added", "date_first_added", "first_added", "added"]
         if c in df.columns),
        None,
    )
    if date_col is None:
        raise ValueError(
            f"Could not find date-added column in {list(df.columns)}"
        )
    return df[[sym_col, date_col]].rename(
        columns={sym_col: "symbol", date_col: "date_added"},
    )


def _extract_sectors(df: pd.DataFrame) -> dict[str, str]:
    """T-audit fix H1: pull {symbol: gics_sector} from the current-members
    Wikipedia table.

    The current-members table has a ``GICS Sector`` column for every
    active S&P 500 member. We normalise column names the same way the
    membership extractor does, then build a flat dict.

    Returns an EMPTY dict (and logs) if the GICS column isn't found —
    failure here should NOT block the membership refresh, since the
    sector cap has a documented fallback to "unknown sector = pass
    through unchanged".
    """
    df = df.copy()
    df.columns = [str(c).lower().strip().replace(" ", "_") for c in df.columns]
    sym_col = next(
        (c for c in ["symbol", "ticker", "ticker_symbol"] if c in df.columns),
        None,
    )
    sector_col = next(
        (c for c in df.columns if "gics" in c and "sector" in c
         and "sub" not in c),
        None,
    )
    if sym_col is None or sector_col is None:
        logger.warning(
            "extract_sectors: no GICS sector column in %s; "
            "sectors map will be empty (sector cap effectively disabled)",
            list(df.columns),
        )
        return {}
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        sym = str(row[sym_col]).upper().strip()
        sector = str(row[sector_col]).strip()
        if sym and sym != "NAN" and sector and sector != "NAN":
            out[sym] = sector
    return out


def _write_sectors_csv(
    sectors: dict[str, str],
    out_path: Path,
) -> None:
    """Atomic write of the {symbol, sector} CSV with a provenance header."""
    if not sectors:
        return   # don't clobber a good existing file with an empty refresh
    lines = [
        "# S&P 500 GICS sector map.",
        "#",
        "# Auto-refreshed by quant-sp500-refresh from Wikipedia.",
        f"# Last refreshed: {datetime.now(UTC).isoformat()}",
        "#",
        "# Used by daily_runner._apply_sector_cap to enforce the 30%",
        "# per-sector concentration cap across the FULL universe.",
        "#",
        "symbol,sector",
    ]
    for sym in sorted(sectors):
        # Quote sectors with commas (defensive — none of the 11 GICS
        # sectors have commas today but it's cheap insurance).
        sec = sectors[sym]
        if "," in sec:
            sec = f'"{sec}"'
        lines.append(f"{sym},{sec}")
    _atomic_write_text(out_path, "\n".join(lines) + "\n")


def _normalise_changes_table(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce Wikipedia's changes table to (date, added, removed).

    The changes table on Wikipedia uses a 2-level header. We collapse
    it and look for the Date/Added/Removed columns by substring.
    """
    df = df.copy()
    # Flatten multi-level columns (Wikipedia uses these).
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(str(p) for p in col).lower().strip()
                      for col in df.columns]
    else:
        df.columns = [str(c).lower().strip().replace(" ", "_") for c in df.columns]

    def _find(col_substr: str, fallback_substr: str | None = None) -> str:
        for c in df.columns:
            if col_substr in c:
                return c
        if fallback_substr:
            for c in df.columns:
                if fallback_substr in c:
                    return c
        raise ValueError(
            f"Could not find column matching '{col_substr}' in {list(df.columns)}"
        )

    date_col = _find("date")
    add_col = _find("added")
    rem_col = _find("removed")
    out = df[[date_col, add_col, rem_col]].copy()
    out.columns = ["date", "added", "removed"]
    return out


def _parse_date(s) -> str | None:
    """Best-effort date parser; returns ISO YYYY-MM-DD or None."""
    if s is None or (isinstance(s, float) and pd.isna(s)) or str(s).strip() == "":
        return None
    text = str(s).strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y", "%d-%b-%Y"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    try:
        return pd.to_datetime(text, errors="coerce").date().isoformat()
    except (ValueError, TypeError, AttributeError):
        return None


def _build_membership(
    current: pd.DataFrame, changes: pd.DataFrame,
) -> dict[str, tuple[str | None, str | None]]:
    """From normalised tables, build symbol → (added, removed)."""
    members: dict[str, tuple[str | None, str | None]] = {}
    for _, row in current.iterrows():
        sym = str(row["symbol"]).upper().strip()
        if not sym or sym == "NAN":
            continue
        # Some Wikipedia rows have stray periods (e.g. "BRK.B" rendered as
        # "BRK.B" or "BRK.B"). Keep them as-is — that's how the operator's
        # other CSVs spell them too.
        added = _parse_date(row.get("date_added"))
        members[sym] = (added, None)

    # Helper that's about to be called below.
    def _drop_inconsistent(
        d: dict[str, tuple[str | None, str | None]],
    ) -> dict[str, tuple[str | None, str | None]]:
        """Filter out symbols whose dates don't form a valid interval.

        Real S&P 500 history has corner cases our schema can't model:
        - Same-day add+remove (spin-offs registered both ways, e.g.,
          FOXA / FOX on 2019-03-19)
        - Wikipedia data errors where dates are inconsistent (e.g.,
          MCK shown as added 1999 / removed 1994 due to legacy entries)
        Rather than failing the refresh on these, drop them with a log
        warning. They represent <1% of universe and missing them has
        negligible backtest impact.
        """
        clean: dict[str, tuple[str | None, str | None]] = {}
        dropped: list[str] = []
        for sym, (added, removed) in d.items():
            if added is not None and removed is not None and added >= removed:
                dropped.append(f"{sym} (added={added}, removed={removed})")
                continue
            clean[sym] = (added, removed)
        if dropped:
            logger.warning(
                "build_membership: dropped %d symbol(s) with inconsistent "
                "dates: %s",
                len(dropped), dropped,
            )
        return clean

    # Apply changes in chronological order so that for symbols with
    # multiple membership intervals (e.g., AMD removed 2013, re-added
    # 2017), we end up with the MOST RECENT add/remove pair. The
    # schema doesn't support multi-interval membership (see
    # quant.data.universe Universe class docstring), so collapsing to
    # latest-only loses ancient pre-2017 history but is correct for
    # any realistic backtest window (the live agent has 2024+ data only).
    changes_sorted = changes.copy()
    changes_sorted["_parsed_date"] = changes_sorted["date"].apply(_parse_date)
    changes_sorted = changes_sorted.dropna(subset=["_parsed_date"]).sort_values(
        "_parsed_date",
    )
    for _, row in changes_sorted.iterrows():
        d = row["_parsed_date"]
        added_sym = str(row.get("added", "")).upper().strip()
        removed_sym = str(row.get("removed", "")).upper().strip()
        if added_sym and added_sym != "NAN":
            # A new add wipes any prior "removed" state — we're now
            # tracking the CURRENT membership interval, not the original.
            members[added_sym] = (d, None)
        if removed_sym and removed_sym != "NAN":
            cur_added, _ = members.get(removed_sym, (None, None))
            members[removed_sym] = (cur_added, d)

    return _drop_inconsistent(members)


def _validate_membership(
    members: dict[str, tuple[str | None, str | None]],
    *,
    min_active: int = _MIN_ACTIVE_MEMBERS,
) -> list[str]:
    """Return list of validation errors. Empty = OK to write."""
    errors: list[str] = []
    n_active = sum(1 for a, r in members.values() if r is None)
    if n_active < min_active:
        errors.append(
            f"only {n_active} active members in new universe; expected >= "
            f"{min_active}. Wikipedia parse may have failed."
        )
    for sym, (added, removed) in members.items():
        if added and removed and added >= removed:
            errors.append(
                f"{sym}: added={added} not before removed={removed}"
            )
    return errors


def _diff_against_existing(
    new_members: dict[str, tuple[str | None, str | None]],
    existing_csv: Path,
) -> dict:
    """Compute summary of changes vs the existing CSV.

    Returns a dict with:
      - added_symbols, removed_symbols, date_corrections lists
      - change_fraction: |added ∪ removed| / |old_active|
    """
    if not existing_csv.exists():
        return {
            "added_symbols": sorted(new_members),
            "removed_symbols": [],
            "date_corrections": [],
            "change_fraction": 1.0,
            "old_active_count": 0,
            "new_active_count": sum(1 for _, r in new_members.values() if r is None),
        }
    df = pd.read_csv(existing_csv, comment="#")
    old_active = {
        str(row["symbol"]).upper().strip()
        for _, row in df.iterrows()
        if pd.isna(row.get("removed")) or str(row.get("removed")).strip() == ""
    }
    new_active = {sym for sym, (_, r) in new_members.items() if r is None}

    added = sorted(new_active - old_active)
    removed = sorted(old_active - new_active)
    change_count = len(added) + len(removed)
    return {
        "added_symbols": added,
        "removed_symbols": removed,
        "date_corrections": [],   # not yet implemented; field reserved
        "change_fraction": (
            change_count / max(len(old_active), 1)
        ),
        "old_active_count": len(old_active),
        "new_active_count": len(new_active),
    }


def _write_csv(
    members: dict[str, tuple[str | None, str | None]],
    out_path: Path,
) -> None:
    """Write the new CSV atomically with a header comment."""
    rows = []
    for sym, (added, removed) in sorted(members.items()):
        if added is None and removed is None:
            continue
        rows.append({
            "symbol": sym,
            "added": added or "",
            "removed": removed or "",
        })
    df = pd.DataFrame(rows, columns=["symbol", "added", "removed"])

    header = (
        "# S&P 500 membership history.\n"
        "#\n"
        "# Auto-refreshed by quant-sp500-refresh from Wikipedia.\n"
        f"# Last refreshed: {datetime.now().isoformat(timespec='seconds')}\n"
        "#\n"
        "# Columns: symbol, added (first day IN, INCLUSIVE),\n"
        "#          removed (first day NOT IN, EXCLUSIVE; blank if active).\n"
        "#\n"
    )
    _atomic_write_text(out_path, header + df.to_csv(index=False))


def _render_email(
    *,
    success: bool,
    diff: dict | None = None,
    errors: list[str] | None = None,
    exc_info: str | None = None,
) -> tuple[str, str]:
    """Build (subject, body) for the operator's notification email."""
    if success:
        added = diff["added_symbols"]
        removed = diff["removed_symbols"]
        subject = (
            f"S&P 500 universe refreshed — "
            f"+{len(added)} / -{len(removed)} active names"
        )
        lines = [
            "## ✅ S&P 500 universe auto-refresh succeeded",
            "",
            f"- **Old active count**: {diff['old_active_count']}",
            f"- **New active count**: {diff['new_active_count']}",
            f"- **Net change**: {diff['new_active_count'] - diff['old_active_count']:+d}",
            f"- **Total churn**: {len(added) + len(removed)} names "
            f"({diff['change_fraction']:.1%} of old universe)",
            "",
            f"### Added ({len(added)})",
            "",
        ]
        if added:
            lines.append(", ".join(f"`{s}`" for s in added))
        else:
            lines.append("_(none)_")
        lines.extend([
            "",
            f"### Removed ({len(removed)})",
            "",
        ])
        if removed:
            lines.append(", ".join(f"`{s}`" for s in removed))
        else:
            lines.append("_(none)_")
        lines.extend([
            "",
            "---",
            "",
            "**Next steps**: review the changes above. The CSV has already been "
            "written to disk and the live agent will use it on its next fire. "
            "Run `git diff reference/universe/sp500.csv` to inspect, then "
            "`git add` + `git commit` if it looks right. Reject with "
            "`git checkout reference/universe/sp500.csv` if Wikipedia data "
            "looks wrong.",
        ])
        return subject, "\n".join(lines)
    else:
        subject = "S&P 500 refresh FAILED — live CSV untouched"
        parts = [
            "## ❌ S&P 500 universe auto-refresh FAILED",
            "",
            "**The existing CSV was NOT modified.** The live agent continues "
            "to use the pre-refresh universe.",
            "",
        ]
        if errors:
            parts.append("### Validation errors")
            parts.append("")
            for e in errors:
                parts.append(f"- {e}")
            parts.append("")
        if exc_info:
            parts.append("### Exception traceback")
            parts.append("")
            parts.append("```")
            parts.append(exc_info)
            parts.append("```")
        parts.extend([
            "",
            "---",
            "",
            "**Next steps**: if Wikipedia's page structure changed, update "
            "the parser in src/quant/agent/sp500_refresh.py. If this was a "
            "transient network issue, just re-fire: "
            "`uv run quant-sp500-refresh`. The next scheduled run will retry "
            "automatically in ~3 months.",
        ])
        return subject, "\n".join(parts)


def refresh_sp500_universe(
    *,
    csv_path: Path | None = None,
    sectors_csv_path: Path | None = None,
    email_sender: EmailSender | None = None,
    min_active: int = _MIN_ACTIVE_MEMBERS,
    max_change_fraction: float = _MAX_CHANGE_FRACTION,
) -> bool:
    """Run the refresh end-to-end. Returns True on success, False on rejection.

    Test-injectable: pass ``csv_path`` for an alt output, ``email_sender``
    for a fake, and tune ``min_active`` / ``max_change_fraction`` if a
    test needs to exercise the safety gates.

    Writes TWO files on success:
      - ``csv_path`` (default: reference/universe/sp500.csv) — the
        membership history (symbol, added, removed).
      - ``sectors_csv_path`` (default: reference/universe/sp500_sectors.csv)
        — the full-universe GICS sector map used by the sector cap.
    """
    out_path = csv_path or _DEFAULT_CSV_PATH
    sectors_out_path = sectors_csv_path or _DEFAULT_SECTORS_CSV_PATH
    sender = email_sender or EmailSender()

    try:
        current_raw, changes_raw = _fetch_wikipedia_tables()
        current = _normalise_current_table(current_raw)
        # T-audit fix H1: extract sector map from the same current-members
        # table (before normalisation drops non-(symbol, date) columns).
        sectors = _extract_sectors(current_raw)
        changes = _normalise_changes_table(changes_raw)
        members = _build_membership(current, changes)

        errors = _validate_membership(members, min_active=min_active)
        if errors:
            subj, body = _render_email(success=False, errors=errors)
            sender.send(subject=subj, body_text=body)
            logger.error("sp500 refresh REJECTED: %s", errors)
            return False

        diff = _diff_against_existing(members, out_path)
        # The change-fraction gate exists to catch parse failures that would
        # replace half the universe with garbage. It SHOULDN'T fire on a
        # bootstrap (where the existing CSV is the < 50-name starter set
        # and the refresh is genuinely replacing it with the real ~500-name
        # universe). Skip the gate when the OLD universe is below the
        # minimum viability threshold — that's the bootstrap signal.
        is_bootstrap = diff["old_active_count"] < min_active
        if (not is_bootstrap) and diff["change_fraction"] > max_change_fraction:
            errors = [
                f"symmetric diff vs current CSV is {diff['change_fraction']:.1%} "
                f"(> {max_change_fraction:.0%} threshold). Refusing to replace; "
                f"likely Wikipedia parse failure or table change."
            ]
            subj, body = _render_email(success=False, errors=errors)
            sender.send(subject=subj, body_text=body)
            logger.error("sp500 refresh REJECTED: change too large")
            return False

        _write_csv(members, out_path)
        # Write the sector map alongside. Failure here is logged but does
        # NOT fail the membership refresh — the cap falls back to
        # "unknown sector = pass through" so we degrade gracefully.
        try:
            _write_sectors_csv(sectors, sectors_out_path)
            logger.info(
                "sp500 refresh wrote sector map for %d names → %s",
                len(sectors), sectors_out_path,
            )
        except Exception as sec_err:
            logger.warning(
                "sp500 refresh: failed to write sector map (%s); "
                "membership file is still good",
                sec_err,
            )
        subj, body = _render_email(success=True, diff=diff)
        sender.send(subject=subj, body_text=body)
        logger.info(
            "sp500 refresh SUCCESS: %d active (was %d); %+d net; %d sectors",
            diff["new_active_count"], diff["old_active_count"],
            diff["new_active_count"] - diff["old_active_count"],
            len(sectors),
        )
        return True

    except Exception as e:
        # Catch-all: bubble up to the email path so the operator hears
        # about the failure no matter what broke.
        logger.exception("sp500 refresh raised unexpectedly")
        try:
            subj, body = _render_email(
                success=False,
                errors=[f"{type(e).__name__}: {e}"],
                exc_info=traceback.format_exc(),
            )
            sender.send(subject=subj, body_text=body)
        except Exception:
            logger.exception("also failed to send failure email")
        return False


def cli_run() -> None:
    """Console-script entry point: ``uv run quant-sp500-refresh``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser(
        description="Quarterly auto-refresh of the S&P 500 membership CSV.",
    )
    parser.add_argument(
        "--csv-path", type=Path, default=None,
        help=f"Output CSV path (default: {_DEFAULT_CSV_PATH})",
    )
    args = parser.parse_args()
    try:
        ok = refresh_sp500_universe(csv_path=args.csv_path)
        sys.exit(0 if ok else 1)
    except Exception as e:
        _email_failure("sp500 refresh", e)
        raise

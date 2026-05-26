"""registry — strategy registry and promotion gates.

The registry is the audit trail of every strategy variant we've ever tested.
Two reasons this matters:

1. **Honest DSR.** The Deflated Sharpe Ratio correction needs the *true*
   number of trials. If we silently discard losing variants, we deceive
   ourselves about how lucky a winner is. The registry records every variant,
   passed or failed.
2. **Promotion gates.** A strategy moves through stages:
       research -> walk_forward -> paper -> live
   Each transition requires meeting specific criteria. The registry tracks
   the stage; ``check_promotion_gate`` enforces the DSR criterion at the
   critical walk_forward → paper transition.

Backing store: SQLite file (default ``data/registry.db``, gitignored).

Currently available:
- ``Registry`` — record, query, promote, gate-check.
- ``STAGES`` — the promotion ladder.
"""

from quant.registry.registry import STAGES, Registry, Stage

__all__ = ["STAGES", "Registry", "Stage"]

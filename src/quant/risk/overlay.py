"""overlay.py — hard risk caps and drawdown kill switch.

What this module is for
-----------------------
The platform's last line of defense. Strategies *propose*; allocators
*shape*; the risk overlay *disposes*. Configured limits in this module
are HARD — a strategy that wants 50% in AAPL can't override the 5% cap.

Two pieces:

1. ``clip_weights(weights, ...)`` — a pure function that takes proposed
   weights and applies the static caps: per-name, gross leverage, net
   exposure. Returns clipped weights plus a structured ``OverlayAudit``
   so the caller (and the reports module) can see what was clipped and
   why. Stateless: same inputs always produce the same outputs.

2. ``KillSwitch`` — a stateful peak-to-trough drawdown circuit breaker.
   Holds the running peak equity; trips when current equity falls more
   than the configured threshold below peak. Sticky by default —
   requires an explicit ``reset()`` so a strategy can't accidentally
   re-arm itself after a blowup.

Currently these are utilities, not yet wired into the engine. The engine
calls ``strategy.on_bar`` and trades the result directly; integrating the
overlay between allocator output and engine fill is a future architectural
step (a meta-strategy wrapper or an explicit allocator stage). For now use
these for post-hoc analysis or in live-trading code where you have access
to the broker's equity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from quant.config import Config

# ---------------------------------------------------------------------------
# OverlayAudit — what got clipped and why
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OverlayAudit:
    """Structured record of one overlay application.

    Frozen — part of the audit trail. Useful for assembling reports that
    can answer "the allocator wanted 8%, why did we trade 5%?".
    """

    initial_gross: float
    initial_net: float
    per_name_clipped: list[str] = field(default_factory=list)
    gross_scale: float = 1.0
    net_scale: float = 1.0
    final_gross: float = 0.0
    final_net: float = 0.0


# ---------------------------------------------------------------------------
# clip_weights — pure function: weights in, weights + audit out
# ---------------------------------------------------------------------------


def clip_weights(
    weights: pd.Series,
    *,
    max_position_weight: float,
    max_gross_leverage: float,
    max_net_exposure: float,
) -> tuple[pd.Series, OverlayAudit]:
    """Apply the three weight-side hard caps and return clipped weights.

    The caps are applied in this order:

    1. **Per-name cap**: each |w_i| is clipped to ``max_position_weight``.
       Sign is preserved (longs stay long, shorts stay short).
    2. **Gross leverage cap**: if ``sum(|w_i|)`` exceeds
       ``max_gross_leverage``, scale every weight by the same multiplier.
    3. **Net exposure cap**: if ``|sum(w_i)|`` exceeds ``max_net_exposure``,
       scale every weight by the same multiplier.

    Doing per-name first then gross matters: a single 50% weight in a
    universe with 5% caps gets clipped to 5% FIRST, which may then bring
    gross under the leverage cap with no further scaling needed.

    Parameters
    ----------
    weights
        Proposed weights, indexed by name.
    max_position_weight
        Per-name cap on |weight|. e.g., 0.05 = no single name above 5%.
    max_gross_leverage
        Cap on ``sum(|weights|)``. e.g., 1.5 = max 1.5x gross.
    max_net_exposure
        Cap on ``|sum(weights)|``. e.g., 1.0 = max fully-invested long or
        fully-invested short.

    Returns
    -------
    (clipped_weights, audit)
        ``clipped_weights`` has the same index as the input.
        ``audit`` records what changed and why.
    """
    if max_position_weight <= 0 or max_gross_leverage <= 0 or max_net_exposure < 0:
        raise ValueError(
            "all caps must be positive (max_net_exposure >= 0). Got "
            f"per_name={max_position_weight}, gross={max_gross_leverage}, "
            f"net={max_net_exposure}"
        )

    initial_gross = float(weights.abs().sum())
    initial_net = float(weights.sum())

    # 1. Per-name cap. clip(lower, upper) treats each value independently
    # and preserves sign.
    clipped = weights.clip(
        lower=-max_position_weight,
        upper=max_position_weight,
    )
    # Audit: which names actually moved? Use a small tolerance because
    # exact float equality is fragile.
    diff = (clipped - weights).abs()
    per_name_clipped = sorted(diff[diff > 1e-12].index.tolist())

    # 2. Gross leverage cap.
    gross = float(clipped.abs().sum())
    if gross > max_gross_leverage:
        gross_scale = max_gross_leverage / gross
        clipped = clipped * gross_scale
    else:
        gross_scale = 1.0

    # 3. Net exposure cap. We compare |net| to the cap so a market-neutral
    # book with high gross stays untouched, but a long-biased book with
    # too-high net gets scaled down.
    net = float(clipped.sum())
    if abs(net) > max_net_exposure:
        # Scale magnitude DOWN by max_net_exposure / |net|. The original
        # sign of `net` is preserved automatically since the multiplier
        # is positive.
        net_scale = max_net_exposure / abs(net)
        clipped = clipped * net_scale
    else:
        net_scale = 1.0

    audit = OverlayAudit(
        initial_gross=initial_gross,
        initial_net=initial_net,
        per_name_clipped=per_name_clipped,
        gross_scale=gross_scale,
        net_scale=net_scale,
        final_gross=float(clipped.abs().sum()),
        final_net=float(clipped.sum()),
    )
    return clipped, audit


def clip_weights_from_config(
    weights: pd.Series,
    *,
    config: Config,
) -> tuple[pd.Series, OverlayAudit]:
    """Convenience wrapper: apply ``clip_weights`` using thresholds from Config.

    Pulls ``config.risk.{max_position_weight, max_gross_leverage, max_net_exposure}``
    so the caller doesn't have to thread them by hand.
    """
    return clip_weights(
        weights,
        max_position_weight=config.risk.max_position_weight,
        max_gross_leverage=config.risk.max_gross_leverage,
        max_net_exposure=config.risk.max_net_exposure,
    )


# ---------------------------------------------------------------------------
# KillSwitch — stateful peak-to-trough drawdown circuit breaker
# ---------------------------------------------------------------------------


class KillSwitch:
    """Trips when equity falls more than ``max_drawdown`` below the running peak.

    Once tripped, stays tripped until ``reset()`` is called. The sticky
    behavior is deliberate: an automated re-arm would defeat the purpose
    (you'd just keep blowing up the same way). Operator review is required
    before resuming.

    Usage:

        ks = KillSwitch(max_drawdown=0.15)
        for equity in equity_stream:
            if ks.check(equity):
                flatten_everything()
                halt_trading()
                break

    Or post-hoc, by walking an equity curve and looking at ``ks.triggered``.

    Threshold convention: ``max_drawdown=0.15`` means trip at -15% (the
    config holds a positive number; we compare to a negative drawdown
    internally).
    """

    def __init__(self, max_drawdown: float) -> None:
        if not (0 < max_drawdown < 1):
            raise ValueError(
                f"max_drawdown must be in (0, 1); got {max_drawdown}. "
                "Pass a positive fraction like 0.15 for 15%."
            )
        self._threshold = max_drawdown
        # `-inf` so the first call always updates the peak. We compare
        # current vs peak, so an "unset" peak shouldn't trip the switch.
        self._peak: float = float("-inf")
        self._triggered: bool = False

    def check(self, current_equity: float) -> bool:
        """Update peak; return True if currently tripped.

        ``current_equity`` should be the mark-to-market equity at the
        end of the bar (matches the engine's ``equity_curve``).
        """
        if self._triggered:
            # Sticky — once tripped, always tripped until reset.
            return True

        if current_equity > self._peak:
            self._peak = current_equity

        # If we haven't seen anything yet or peak is non-positive, can't
        # compute a meaningful drawdown.
        if self._peak <= 0:
            return False

        drawdown = (current_equity - self._peak) / self._peak
        if drawdown < -self._threshold:
            self._triggered = True
        return self._triggered

    def reset(self) -> None:
        """Manual reset. Use after operator review of what caused the trip."""
        self._triggered = False
        # Note: we DON'T reset the peak. After a kill, the strategy should
        # re-prove itself against the prior high-water mark rather than
        # being given a fresh start.

    @property
    def triggered(self) -> bool:
        return self._triggered

    @property
    def peak(self) -> float:
        return self._peak

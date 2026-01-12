# backend/trading/paper_trader.py
"""
ChronosX Paper Trader with Live Regime Detection and Dynamic Bandit Weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional
from typing import List, Dict, Optional, Tuple

import pandas as pd

from backend.agents.signal_agents import (
    Candle,
    MomentumRSIAgent,
    MLClassifierAgent,
    OrderFlowAgent,
    SentimentAgent,
    EnsembleAgent,
    EnsembleDecision as AgentEnsembleDecision,
)
from backend.agents.portfolio_manager import ThompsonSamplingPortfolioManager
from backend.agents.regime_detector import RegimeDetector, MarketRegime
from backend.governance.rule_engine import (
    GovernanceEngine,
    TradingSignal,
    AccountState,
    GovernanceDecision,
)
from backend.config import TradingConfig


@dataclass
class TradeRecord:
    timestamp: datetime
    symbol: str
    side: str
    size: float
    entry_price: float
    exit_price: float
    pnl: float
    agent_id: str
    governance_reason: str
    risk_score: float
    regime: str = "unknown"
    contributing_agents: List[str] = None
    ensemble_confidence: float = 0.0

    def __post_init__(self):
        if self.contributing_agents is None:
            self.contributing_agents = []


class PaperTrader:
    def __init__(
        self,
        initial_balance: float = 10_000.0,
        symbol: str = "cmt_btcusdt",
        config: Optional[TradingConfig] = None,
    ):
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.config = config or TradingConfig()
        self.balance = initial_balance
        self.equity = initial_balance
        self.max_equity = initial_balance
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.volatility = 0.0
        self.recent_pnls: List[float] = []

        # Governance
        self.governance = GovernanceEngine()
        self.governance = GovernanceEngine(config=self.config)

        # Agents
        self.momentum_agent = MomentumRSIAgent()
        self.ml_agent = MLClassifierAgent()
        self.order_flow_agent = OrderFlowAgent()
        self.sentiment_agent = SentimentAgent()
        self.ensemble = EnsembleAgent()

        # Portfolio manager with Thompson Sampling
        self.portfolio_manager = ThompsonSamplingPortfolioManager(
            agent_ids=[
                "momentum_rsi",
                "ml_classifier",
                "order_flow",
                "sentiment",
            ]
        )

        # Regime detection
        self.regime_detector = RegimeDetector(lookback=20)
        self.current_regime = MarketRegime.UNKNOWN

        # Trade tracking
        self.trades: List[TradeRecord] = []
        self.open_position: Optional[TradeRecord] = None
        self.governance_trigger_log: List[dict] = []

        # Optional callback: will be set by API layer for monitor
        self.on_trade_closed = None

        # Track last ensemble decision for live trading loop
        self.last_ensemble_decision = None
        self.last_regime = None

    # ================================================================ #
    # Account state helpers
    # ================================================================ #

    def _update_equity(self, mark_price: float):
        if self.open_position:
            direction = 1 if self.open_position.side == "buy" else -1
            pnl_unrealized = (
                mark_price - self.open_position.entry_price
            ) * direction * self.open_position.size
        else:
            pnl_unrealized = 0.0

        self.equity = self.balance + pnl_unrealized
        self.max_equity = max(self.max_equity, self.equity)
        drawdown = (self.max_equity - self.equity) / self.max_equity
        self.max_drawdown = max(self.max_drawdown, drawdown)

    def _get_account_state(self) -> AccountState:
        return AccountState(
            balance=self.balance,
            equity=self.equity,
            daily_pnl=self.daily_pnl,
            total_pnl=self.total_pnl,
            max_drawdown=self.max_drawdown,
            open_positions=1 if self.open_position else 0,
            open_position_value=(
                self.open_position.size if self.open_position else 0.0
            ),
            daily_trades=len(self.trades),
            recent_win_rate=self._recent_win_rate(),
            volatility=self.volatility,
        )

    def _recent_win_rate(self) -> float:
        if not self.recent_pnls:
            return 0.5
        last = self.recent_pnls[-50:]
        wins = len([p for p in last if p > 0])
        return wins / len(last)

    def _should_exit(
        self, position: TradeRecord, candle: Candle, ensemble_decision: AgentEnsembleDecision
    ) -> Tuple[bool, Optional[str]]:
        """Check if a position should be closed based on multiple exit triggers."""

        # Exit Trigger 1: Opposing Signal from Ensemble
        if ensemble_decision:
            is_long = position.side == "buy"
            is_short = position.side == "sell"
            if (is_long and ensemble_decision.direction == -1) or (
                is_short and ensemble_decision.direction == 1
            ):
                return True, f"Opposing signal (conf: {ensemble_decision.confidence:.2f})"

        # Exit Trigger 2: Hard Stop Loss (from config)
        entry_price = position.entry_price
        current_price = candle.close
        direction = 1 if position.side == "buy" else -1
        pnl_pct = ((current_price - entry_price) / entry_price) * direction

        if pnl_pct < -self.config.HARDSTOP_PCT:
            return True, f"Hardstop loss hit: {pnl_pct:.2%}"

        # Exit Trigger 3: Max Hold Time (from config)
        hold_time_minutes = (candle.timestamp - position.timestamp).total_seconds() / 60
        if hold_time_minutes > self.config.MAX_HOLD_TIME_MINUTES:
            return True, f"Max hold time exceeded: {int(hold_time_minutes)}m"

        return False, None

    # ================================================================ #
    # Regime detection
    # ================================================================ #

    def _detect_regime(self, candle: Candle) -> MarketRegime:
        """Detect current market regime."""
        self.regime_detector.update(candle.close)
        regime_state = self.regime_detector.detect()
        self.current_regime = regime_state.current
        self.portfolio_manager.set_regime(self.current_regime)
        return self.current_regime

    # ================================================================ #
    # Main simulation APIs
    # ================================================================ #

    async def run_live_simulation(self, candle_stream, hours: int = 24):
        """Simulate N hours of trading on a candle async generator."""
        async for candle in candle_stream:
            await self.process_candle(candle)

    def update_candle(self, candle: Candle):
        """
        Synchronous candle update (wrapper for async process_candle).
        Called by WeexTradingLoop during live trading.
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Can't use run_until_complete in running loop, schedule as task
                asyncio.create_task(self.process_candle(candle))
            else:
                loop.run_until_complete(self.process_candle(candle))
        except RuntimeError:
            # No event loop, create one
            asyncio.run(self.process_candle(candle))

    async def process_candle(self, candle: Candle):

        """
        Process a single candle: detect regime, update agents, decide trade,
        apply governance, and simulate position.
        Process a single candle:
        1. Check for exits on open positions.
        2. If flat, check for new entries.
        3. Apply governance to all decisions.
        """
        # Detect regime
        regime = self._detect_regime(candle)
        print(f"[PaperTrader] Candle {candle.timestamp} close={candle.close}, regime={regime.value}")

        # Update agents
        # Update all agents with the new candle data
        self.momentum_agent.update(candle)
        self.sentiment_agent.update(candle)  # <-- ADD THIS
        self.order_flow_agent.reset_window()  # <-- AND THIS (or update with real flow data)
        self.sentiment_agent.update(candle)
        self.order_flow_agent.reset_window()  # Or update with real flow data
        self.ml_agent.update(candle)  # Also update ML agent
        self._update_equity(mark_price=candle.close)

        # Generate signals from all agents
        # --- Generate Ensemble Decision (needed for both exits and entries) ---
        signals = []
        sig_momentum = self.momentum_agent.generate()
        if sig_momentum: signals.append(sig_momentum)
        
        sig_orderflow = self.order_flow_agent.generate()
        if sig_orderflow: signals.append(sig_orderflow)

        sig1 = self.momentum_agent.generate()
        if sig1:
            signals.append(sig1)
        sig_sentiment = self.sentiment_agent.generate()
        if sig_sentiment: signals.append(sig_sentiment)

        sig3 = self.order_flow_agent.generate()
        if sig3:
            signals.append(sig3)
        # ML Agent signal generation (from Fix #3)
        sig_ml = self.ml_agent.generate()
        if sig_ml: signals.append(sig_ml)

        sig4 = self.sentiment_agent.generate()
        if sig4:
            signals.append(sig4)

        # ALPHA: Fallback trend signal if no directional signals
        # Fallback signal
        if not any(s.direction != 0 for s in signals):
            if len(self.momentum_agent.history) >= 20:
                closes = [c.close for c in self.momentum_agent.history[-20:]]
                sma20 = sum(closes) / len(closes)
                trend_dir = 1 if candle.close > sma20 else -1
                from backend.agents.signal_agents import TradingSignal
                trend_sig = TradingSignal(
                from backend.agents.signal_agents import TradingSignal as AgentTradingSignal
                trend_sig = AgentTradingSignal(
                    agent_id="trend_fallback",
                    direction=trend_dir,
                    confidence=0.12,
                    metadata={"sma20": sma20, "close": candle.close}
                )
                signals.append(trend_sig)
                print(f"[PaperTrader] ALPHA: Fallback trend signal dir={trend_dir}, conf=0.12")

        if not signals:
            print("[PaperTrader] No signals this candle")
            return
        # NEW: inspect which agents fired
        for s in signals:
            print(f"[PaperTrader] Signal from {s.agent_id}: dir={s.direction}, conf={s.confidence:.3f}")

        print(f"[PaperTrader] {len(signals)} raw signals")
        
        print(f"[PaperTrader] {len(signals)} raw signals")

        # Get dynamic weights from bandit
        # Get dynamic weights and combine signals
        weights = self.portfolio_manager.sample_weights_for_regime(regime)
        self.ensemble.weights = weights

        # Combine signals
        ensemble_decision = self.ensemble.combine(signals)
        print(
            f"[PaperTrader] Ensemble decision: dir={ensemble_decision.direction}, "
            f"conf={ensemble_decision.confidence:.3f}"
        )

        # Cache for external consumers (WeexTradingLoop)
        self.last_ensemble_decision = ensemble_decision
        self.last_regime = regime

        if ensemble_decision.direction == 0:
            print("[PaperTrader] Ensemble flat (direction=0) -> no trade")
            return
        # --- 1. EXIT LOGIC ---
        if self.open_position:
            should_exit, exit_reason = self._should_exit(self.open_position, candle, ensemble_decision)
            if should_exit:
                print(f"[PaperTrader] EXIT TRIGGERED: {exit_reason}")
                self._close_position(exit_price=candle.close, timestamp=candle.timestamp)
                # IMPORTANT: Return after closing to avoid immediate re-entry on the same candle
                return

        if ensemble_decision.confidence < 0.05:
            print("[PaperTrader] Confidence too low (<0.05) -> no trade")
            return
        # --- 2. ENTRY LOGIC (only if no open position) ---
        if not self.open_position:
            if ensemble_decision.direction == 0:
                print("[PaperTrader] Ensemble flat (direction=0) -> no trade")
                return

        side = "buy" if ensemble_decision.direction > 0 else "sell"
            if ensemble_decision.confidence < self.config.MIN_CONFIDENCE:
                print(f"[PaperTrader] Confidence {ensemble_decision.confidence:.2f} < {self.config.MIN_CONFIDENCE} -> no trade")
                return

        # Position sizing: simple risk-based (governance will further cap)
        risk_pct = 0.0005  # 0.05% of equity per trade during testing
        risk_amount = self.equity * risk_pct
        stop_loss_distance = candle.close * 0.005
        size = risk_amount / stop_loss_distance
            side = "buy" if ensemble_decision.direction > 0 else "sell"

        trading_signal = TradingSignal(
            symbol=self.symbol,
            side=side,
            size=size,
            confidence=ensemble_decision.confidence,
            stop_loss=-stop_loss_distance if side == "buy" else stop_loss_distance,
            take_profit=stop_loss_distance * 2,
            timestamp=candle.timestamp,
            agent_id="ensemble",
        )
            # Position sizing
            risk_pct = 0.0005
            risk_amount = self.equity * risk_pct
            stop_loss_distance = candle.close * 0.005
            size = risk_amount / stop_loss_distance

        # Governance check
        account_state = self._get_account_state()
        decision: GovernanceDecision = self.governance.evaluate(
            trading_signal, account_state
        )
        print(
            f"[PaperTrader] Governance: allow={decision.allow}, "
            f"adj_size={decision.adjusted_size:.6f}, "
            f"risk_score={decision.risk_score:.1f}, reason={decision.reason}"
        )
            trading_signal = TradingSignal(
                symbol=self.symbol,
                side=side,
                size=size,
                confidence=ensemble_decision.confidence,
                stop_loss=-stop_loss_distance if side == "buy" else stop_loss_distance,
                take_profit=stop_loss_distance * 2,
                timestamp=candle.timestamp,
                agent_id="ensemble",
            )

        # Log governance trigger
        self._log_governance_trigger(
            candle.timestamp, trading_signal, decision, regime
        )
            # Governance check
            account_state = self._get_account_state()
            decision: GovernanceDecision = self.governance.evaluate(
                trading_signal, account_state
            )
            print(
                f"[PaperTrader] Governance: allow={decision.allow}, "
                f"adj_size={decision.adjusted_size:.6f}, "
                f"risk_score={decision.risk_score:.1f}, reason={decision.reason}"
            )

        if not decision.allow or decision.adjusted_size <= 0:
            print("[PaperTrader] Trade blocked by governance or zero size")
            return
            self._log_governance_trigger(
                candle.timestamp, trading_signal, decision, regime
            )

        final_size = decision.adjusted_size
            if not decision.allow or decision.adjusted_size <= 0:
                print("[PaperTrader] Trade blocked by governance or zero size")
                return

        # Close open position if any
        if self.open_position:
            self._close_position(exit_price=candle.close, timestamp=candle.timestamp)
            final_size = decision.adjusted_size

        # Open new position
        self._open_position(
            side=side,
            size=final_size,
            price=candle.close,
            timestamp=candle.timestamp,
            governance_reason=decision.reason,
            risk_score=decision.risk_score,
            regime=regime.value,
            contributing_agents=[s.agent_id for s in ensemble_decision.agent_signals],
            ensemble_confidence=ensemble_decision.confidence,
        )
        print(
            f"[PaperTrader] Opened position side={side}, size={final_size:.6f}, "
            f"price={candle.close}"
        )
            # Open new position
            self._open_position(
                side=side,
                size=final_size,
                price=candle.close,
                timestamp=candle.timestamp,
                governance_reason=decision.reason,
                risk_score=decision.risk_score,
                regime=regime.value,
                contributing_agents=[s.agent_id for s in ensemble_decision.agent_signals],
                ensemble_confidence=ensemble_decision.confidence,
            )
            print(
                f"[PaperTrader] Opened position side={side}, size={final_size:.6f}, "
                f"price={candle.close}"
            )

    # ================================================================ #
    # Position management
    # ================================================================ #

    def _open_position(
        self,
        side: str,
        size: float,
        price: float,
        timestamp: datetime,
        governance_reason: str,
        risk_score: float,
        regime: str = "unknown",
        contributing_agents: List[str] = None,
        ensemble_confidence: float = 0.0,
    ):
        self.open_position = TradeRecord(
            timestamp=timestamp,
            symbol=self.symbol,
            side=side,
            size=size,
            entry_price=price,
            exit_price=price,
            pnl=0.0,
            agent_id="ensemble",
            governance_reason=governance_reason,
            risk_score=risk_score,
            regime=regime,
            contributing_agents=contributing_agents or [],
            ensemble_confidence=ensemble_confidence,
        )

    def _close_position(self, exit_price: float, timestamp: datetime):
        if not self.open_position:
            return

        direction = 1 if self.open_position.side == "buy" else -1
        pnl = (
            (exit_price - self.open_position.entry_price)
            * direction
            * self.open_position.size
        )

        trade = TradeRecord(
            timestamp=timestamp,
            symbol=self.open_position.symbol,
            side=self.open_position.side,
            size=self.open_position.size,
            entry_price=self.open_position.entry_price,
            exit_price=exit_price,
            pnl=pnl,
            agent_id=self.open_position.agent_id,
            governance_reason=self.open_position.governance_reason,
            risk_score=self.open_position.risk_score,
            regime=self.open_position.regime,
            contributing_agents=self.open_position.contributing_agents,
            ensemble_confidence=self.open_position.ensemble_confidence,
        )

        # NEW: hook external monitor if set
        if hasattr(self, "on_trade_closed") and callable(self.on_trade_closed):
            self.on_trade_closed(trade)

        self.trades.append(trade)
        self.balance += pnl
        self.total_pnl += pnl
        self.daily_pnl += pnl
        self.recent_pnls.append(pnl)

        # Record with risk-adjusted reward
        self.portfolio_manager.record_trade_result(
            agent_id=trade.agent_id,
            pnl=pnl,
            position_size=trade.size,
            entry_price=trade.entry_price,
        )

        print(
            f"[PaperTrader] Closed position side={trade.side}, size={trade.size:.6f}, "
            f"entry={trade.entry_price}, exit={exit_price}, pnl={pnl:.4f}"
        )

        self.open_position = None

    # ================================================================ #
    # Governance tracking
    # ================================================================ #

    def _log_governance_trigger(
        self,
        timestamp: datetime,
        signal: TradingSignal,
        decision: GovernanceDecision,
        regime: MarketRegime,
    ):
        """Log governance rule triggers for analytics."""
        self.governance_trigger_log.append(
            {
                "timestamp": timestamp.isoformat(),
                "side": signal.side,
                "size_before": signal.size,
                "size_after": decision.adjusted_size,
                "triggered_rules": decision.triggered_rules,
                "blocked": not decision.allow,
                "risk_score": decision.risk_score,
                "regime": regime.value,
            }
        )

    # ================================================================ #
    # Metrics / reporting
    # ================================================================ #

    def get_trades_df(self) -> pd.DataFrame:
        """Return trades as DataFrame."""
        if not self.trades:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "symbol",
                    "side",
                    "size",
                    "entry_price",
                    "exit_price",
                    "pnl",
                    "agent_id",
                    "governance_reason",
                    "risk_score",
                    "regime",
                    "contributing_agents",
                    "ensemble_confidence",
                ]
            )
        records = []
        for t in self.trades:
            r = t.__dict__.copy()
            r["contributing_agents"] = ",".join(r["contributing_agents"])
            records.append(r)
        return pd.DataFrame(records)

    def get_equity_curve(self) -> pd.Series:
        """Rebuild equity curve from trades."""
        balance = self.initial_balance
        curve = []
        times = []
        for t in self.trades:
            balance += t.pnl
            curve.append(balance)
            times.append(t.timestamp)
        if not curve:
            return pd.Series(dtype=float)
        return pd.Series(curve, index=times, name="equity")

    def get_summary_metrics(self) -> Dict:
        df = self.get_trades_df()
        if df.empty:
            return {
                "total_pnl": 0.0,
                "num_trades": 0,
                "win_rate": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
            }

        total_pnl = float(df["pnl"].sum())
        num_trades = len(df)
        wins = (df["pnl"] > 0).sum()
        win_rate = wins / num_trades if num_trades > 0 else 0.0

        returns = df["pnl"] / self.initial_balance
        sharpe = (
            returns.mean() / (returns.std() + 1e-8) * (len(returns) ** 0.5)
            if len(returns) > 0
            else 0.0
        )

        equity = self.get_equity_curve()
        if equity.empty:
            max_dd = 0.0
        else:
            roll_max = equity.cummax()
            dd = (roll_max - equity) / roll_max
            max_dd = float(dd.max())

        return {
            "total_pnl": float(total_pnl),
            "num_trades": int(num_trades),
            "win_rate": float(win_rate),
            "sharpe": float(sharpe),
            "max_drawdown": max_dd,
        }
    def get_ensemble_signal(self):
        """
        Return latest ensemble signal snapshot for external consumers.
        
        Structure:
        {
            "dir": int,
            "conf": float,
            "regime": str,
            "contributing_agents": List[str]
        }
        """
        if self.last_ensemble_decision is None:
            return None

        return {
            "dir": self.last_ensemble_decision.direction,
            "conf": self.last_ensemble_decision.confidence,
            "regime": self.last_regime.value if self.last_regime else "unknown",
            "contributing_agents": [
                s.agent_id for s in self.last_ensemble_decision.agent_signals
            ],
        }
# backend/trading/paper_trader.py
"""
ChronosX Paper Trader with Live Regime Detection and Dynamic Bandit Weights.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
import logging
import uuid

logger = logging.getLogger(__name__)

import pandas as pd

from backend.agents.signal_agents import (
    Candle,
    MomentumRSIAgent,
    MLClassifierAgent,
    OrderFlowAgent,
    SentimentAgent,
    EnsembleAgent,
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
    order_id: Optional[str] = None
    highest_pnl_pct: float = 0.0
    exit_reason: str = "unknown"

    def __post_init__(self):
        if self.contributing_agents is None:
            self.contributing_agents = []


class PaperTrader:
    def __init__(
        self,
        config: Optional[TradingConfig] = None,
        execution_client: Optional[Any] = None,
        **kwargs,
    ):
        if 'symbol' in kwargs:
            logger.warning("The 'symbol' argument for PaperTrader is deprecated and will be ignored. The trader now operates on multiple symbols.")

        self.config = config or TradingConfig()
        self.execution_client = execution_client
        initial_balance = self.config.ACCOUNT_EQUITY
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity = initial_balance
        self.max_equity = initial_balance
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.volatility = 0.0
        self.recent_pnls: List[float] = []

        # Governance
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
        self.open_positions: Dict[str, TradeRecord] = {}
        self.governance_trigger_log: List[dict] = []

        # Optional callback: will be set by API layer for monitor
        self.on_trade_closed = None

        # Track last ensemble decision for live trading loop
        self.last_ensemble_decision = None
        self.last_regime = None

    # ================================================================ #
    # Account state helpers
    # ================================================================ #

    def _update_equity(self, mark_price: float, symbol: str):
        pnl_unrealized = 0.0
        for pos in self.open_positions.values():
            # Use the current candle's price only for the symbol being updated
            current_price = mark_price if pos.symbol == symbol else pos.exit_price
            direction = 1 if pos.side == "buy" else -1
            pnl_unrealized += (current_price - pos.entry_price) * direction * pos.size
            pos.exit_price = current_price # Update exit price for unrealized PnL calc
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
            open_positions=len(self.open_positions),
            # Sum the value of all open positions
            open_position_value=sum(pos.size for pos in self.open_positions.values()),
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
        self, position: TradeRecord, candle: Candle, ensemble_decision: Any
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

        # Calculate PnL for Exits
        entry_price = position.entry_price
        current_price = candle.close
        direction = 1 if position.side == "buy" else -1
        pnl_pct = ((current_price - entry_price) / entry_price) * direction

        # Update High Water Mark (Peak PnL)
        if pnl_pct > position.highest_pnl_pct:
            position.highest_pnl_pct = pnl_pct

        # Exit Trigger 2: Hard Stop Loss (from config)
        if pnl_pct < -self.config.HARDSTOP_PCT:
            return True, f"Hardstop loss hit: {pnl_pct:.2%}"

        # Exit Trigger 3: Take Profit (2.0R Dynamic Target)
        if pnl_pct > self.config.HARDSTOP_PCT * 2.0:
            return True, f"Take Profit hit (2.0R): {pnl_pct:.2%}"

        # Exit Trigger 4: Trailing Stop (Activate after 0.5R profit)
        # If we are up > 0.5R, and give back 40% of peak profit, exit.
        activation_threshold = self.config.HARDSTOP_PCT * 0.5
        if position.highest_pnl_pct > activation_threshold:
            trail_floor = position.highest_pnl_pct * 0.6  # Secure 60% of peak
            if pnl_pct < trail_floor:
                return True, f"Trailing Stop hit: {pnl_pct:.2%} (Peak: {position.highest_pnl_pct:.2%})"

        # Exit Trigger 5: Breakeven Protection (Secure 1R)
        # If we reached 1R profit, ensure we don't lose money (stop at +0.05%)
        if position.highest_pnl_pct >= self.config.HARDSTOP_PCT:
            if pnl_pct <= 0.0005:
                return True, f"Breakeven protection hit (Peak: {position.highest_pnl_pct:.2%})"

        # Calculate hold time for time-based triggers
        hold_time_minutes = (candle.timestamp - position.timestamp).total_seconds() / 60

        # Exit Trigger 6: Stale Profit (Time Decay)
        # If held > 50% of max time and we have small profit (>0.2%), take it.
        if hold_time_minutes > (self.config.MAX_HOLD_TIME_MINUTES * 0.5) and pnl_pct > 0.002:
            return True, f"Stale profit exit: {pnl_pct:.2%} after {int(hold_time_minutes)}m"

        # Exit Trigger 3: Max Hold Time (from config)
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
        Process a single candle:
        1. Check for exits on open positions.
        2. If flat, check for new entries.
        3. Apply governance to all decisions.
        """
        # Detect regime
        regime = self._detect_regime(candle)
        print(f"[PaperTrader] Candle {candle.timestamp} close={candle.close}, regime={regime.value}")

        # Update all agents with the new candle data
        self.momentum_agent.update(candle)
        self.sentiment_agent.update(candle)
        self.order_flow_agent.reset_window()  # Or update with real flow data
        self.ml_agent.update(candle)  # Also update ML agent
        self._update_equity(mark_price=candle.close, symbol=candle.symbol)

        # --- Generate Ensemble Decision (needed for both exits and entries) ---
        signals = []
        sig_momentum = self.momentum_agent.generate()
        if sig_momentum: signals.append(sig_momentum)
        
        sig_orderflow = self.order_flow_agent.generate()
        if sig_orderflow: signals.append(sig_orderflow)

        sig_sentiment = self.sentiment_agent.generate()
        if sig_sentiment: signals.append(sig_sentiment)

        # ML Agent signal generation (from Fix #3)
        sig_ml = self.ml_agent.generate()
        if sig_ml: signals.append(sig_ml)

        # Fallback signal
        if not any(s.direction != 0 for s in signals):
            if len(self.momentum_agent.history) >= 20:
                closes = [c.close for c in self.momentum_agent.history[-20:]]
                sma20 = sum(closes) / len(closes)
                trend_dir = 1 if candle.close > sma20 else -1
                # Boost confidence to ensure we trade when primary agents are quiet
                fallback_conf = max(0.25, self.config.MIN_CONFIDENCE + 0.05)
                from backend.agents.signal_agents import TradingSignal as AgentTradingSignal
                trend_sig = AgentTradingSignal(
                    agent_id="trend_fallback",
                    direction=trend_dir,
                    confidence=fallback_conf,
                    metadata={"sma20": sma20, "close": candle.close}
                )
                signals.append(trend_sig)
                print(f"[PaperTrader] ALPHA: Fallback trend signal dir={trend_dir}, conf={fallback_conf:.2f}")

        # Get dynamic weights and combine signals
        weights = self.portfolio_manager.sample_weights_for_regime(regime)
        self.ensemble.weights = weights
        ensemble_decision = self.ensemble.combine(signals)

        # Cache for external consumers (WeexTradingLoop)
        self.last_ensemble_decision = ensemble_decision
        self.last_regime = regime

        # --- 1. EXIT LOGIC ---
        position_to_check = self.open_positions.get(candle.symbol)
        if position_to_check:
            should_exit, exit_reason = self._should_exit(position_to_check, candle, ensemble_decision)
            if should_exit:
                print(f"[PaperTrader] EXIT TRIGGERED: {exit_reason}")
                self._close_position(symbol=candle.symbol, exit_price=candle.close, timestamp=candle.timestamp, exit_reason=exit_reason)
                # IMPORTANT: Return after closing to avoid immediate re-entry on the same candle
                return

        # --- 2. ENTRY LOGIC (only if no open position) ---
        if candle.symbol not in self.open_positions:
            if ensemble_decision.direction == 0:
                print("[PaperTrader] Ensemble flat (direction=0) -> no trade")
                return

            if ensemble_decision.confidence < self.config.MIN_CONFIDENCE:
                print(f"[PaperTrader] Confidence {ensemble_decision.confidence:.2f} < {self.config.MIN_CONFIDENCE} -> no trade")
                return

            side = "buy" if ensemble_decision.direction > 0 else "sell"

            # DYNAMIC POSITION SIZING: Base size on a % of current equity, divided among potential symbols
            max_pos_usdt = self.equity * self.config.MAX_POSITION_AS_PCT_EQUITY
            base_size_usdt = max_pos_usdt * self.config.KELLY_FRACTION
            scaled_size_usdt = base_size_usdt * ensemble_decision.confidence
            size_in_btc = scaled_size_usdt / candle.close
            
            # QUANTIZATION FIX: Enforce WEEX contract step size (0.0001 BTC)
            step_size = 0.0001
            # Round to nearest step, ensure at least 1 step if we are trading
            size_in_btc = max(step_size, round(size_in_btc / step_size) * step_size)
            size_in_btc = round(size_in_btc, 4) # Prevent float precision errors

            trading_signal = TradingSignal(
                symbol=candle.symbol,
                side=side,
                size=size_in_btc,
                confidence=ensemble_decision.confidence,
                stop_loss=candle.close * self.config.HARDSTOP_PCT, # Stop loss distance
                take_profit=candle.close * self.config.HARDSTOP_PCT * 2, # Simple 2:1 R:R
                timestamp=candle.timestamp,
                agent_id="ensemble",
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

            self._log_governance_trigger(
                candle.timestamp, trading_signal, decision, regime
            )

            if not decision.allow or decision.adjusted_size <= 0:
                print("[PaperTrader] Trade blocked by governance or zero size")
                return

            final_size = decision.adjusted_size

            # Open new position
            self._open_position(
                symbol=candle.symbol,
                side=side,
                size=final_size,
                price=candle.close,
                timestamp=candle.timestamp,
                governance_reason=decision.reason,
                risk_score=decision.risk_score,
                regime=regime.value,
                contributing_agents=[s.agent_id for s in ensemble_decision.agent_signals],
                ensemble_confidence=ensemble_decision.confidence,
                stop_loss=trading_signal.stop_loss,
                exit_reason="open",  # Placeholder until closed
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
        symbol: str,
        side: str,
        size: float,
        price: float,
        timestamp: datetime,
        governance_reason: str,
        risk_score: float,
        regime: str = "unknown",
        contributing_agents: List[str] = None,
        ensemble_confidence: float = 0.0,
        stop_loss: float = 0.0,
        exit_reason: str = "open",
    ):
        new_position = TradeRecord(
            timestamp=timestamp,
            symbol=symbol,
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
            order_id=str(uuid.uuid4()),
            exit_reason=exit_reason,
        )
        
        self.open_positions[symbol] = new_position

        # EXECUTION: Send order to WEEX if client is connected
        if self.execution_client:
            try:
                # Map side to WEEX type: 1=Open Long, 2=Open Short
                order_type = "1" if side == "buy" else "2"
                print(f"[PaperTrader] EXECUTION: Placing {side} order for {size:.4f} BTC...")
                response = self.execution_client.place_order(
                    symbol=symbol,
                    size=str(size),
                    type_=order_type,
                    price=str(price),
                    client_order_id=new_position.order_id
                )
                print(f"[PaperTrader] EXECUTION SUCCESS: Order placed. Response: {response}")
            except Exception as e:
                print(f"[PaperTrader] EXECUTION ERROR: {e}")
                # CRITICAL: Rollback internal state if execution fails
                self.open_positions.pop(symbol, None)
                return

    def _close_position(self, symbol: str, exit_price: float, timestamp: datetime, exit_reason: str = "unknown"):
        position_to_close = self.open_positions.pop(symbol, None)
        if not position_to_close:
            logger.warning(f"Attempted to close a position for {symbol}, but none was open.")
            return

        direction = 1 if position_to_close.side == "buy" else -1
        pnl = ((exit_price - position_to_close.entry_price) * direction * position_to_close.size)

        trade = TradeRecord(
            timestamp=timestamp,
            symbol=position_to_close.symbol,
            side=position_to_close.side,
            size=position_to_close.size,
            entry_price=position_to_close.entry_price,
            exit_price=exit_price,
            pnl=pnl,
            agent_id=position_to_close.agent_id,
            governance_reason=position_to_close.governance_reason,
            risk_score=position_to_close.risk_score,
            regime=position_to_close.regime,
            contributing_agents=position_to_close.contributing_agents,
            ensemble_confidence=position_to_close.ensemble_confidence,
            order_id=position_to_close.order_id,
            highest_pnl_pct=position_to_close.highest_pnl_pct,
            exit_reason=exit_reason,
        )

        # NEW: hook external monitor if set (ensure dict is passed)
        if hasattr(self, "on_trade_closed") and callable(self.on_trade_closed):
            self.on_trade_closed(asdict(trade))

        # EXECUTION: Close position on WEEX
        if self.execution_client:
            try:
                # Map side to WEEX type: 3=Close Long, 4=Close Short
                # Note: If open was 'buy' (Long), we need to Close Long (3).
                close_type = "3" if position_to_close.side == "buy" else "4"
                print(f"[PaperTrader] EXECUTION: Closing {position_to_close.side} position for {symbol}...")
                response = self.execution_client.place_order(
                    symbol=symbol,
                    size=str(position_to_close.size),
                    type_=close_type,
                    price=str(exit_price),
                )
                print(f"[PaperTrader] EXECUTION SUCCESS: Position closed. Response: {response}")
            except Exception as e:
                print(f"[PaperTrader] EXECUTION ERROR (Close): {e}")

        self.trades.append(trade)
        self.balance += pnl
        self.total_pnl += pnl
        self.daily_pnl += pnl
        self.recent_pnls.append(pnl)

        # Distribute the result to all contributing agents for the bandit to learn
        for agent_id in trade.contributing_agents:
            self.portfolio_manager.record_trade_result(
                agent_id=agent_id,
                pnl=pnl,
                position_size=trade.size,
                entry_price=trade.entry_price,
            )

        print(
            f"[PaperTrader] Closed position side={trade.side}, size={trade.size:.6f}, "
            f"entry={trade.entry_price}, exit={exit_price}, pnl={pnl:.4f}, total_pnl={self.total_pnl:.4f}, reason='{exit_reason}'"
        )

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
                    "exit_reason",
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

    def reload_config(self, new_config: TradingConfig):
        """Reloads the configuration for the trader and its components."""
        self.config = new_config
        self.governance.reload_config(new_config)
        logger.warning("PaperTrader configuration reloaded.")
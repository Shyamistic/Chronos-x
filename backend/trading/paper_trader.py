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
import math

logger = logging.getLogger(__name__)

import pandas as pd

from backend.agents.signal_agents import (
    Candle,
    MomentumRSIAgent,
    MLClassifierAgent,
    OrderFlowAgent,
    SentimentAgent,
    TrendBiasAgent,
    LLMAnalysisAgent,
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


# WEEX Contract Step Sizes (Hardcoded for competition)
STEP_SIZES = {
    "cmt_btcusdt": 0.0001,
    "cmt_ethusdt": 0.001,
    "cmt_solusdt": 0.1,
}

def normalize_size(symbol: str, size: float) -> float:
    step_size = STEP_SIZES.get(symbol, 0.0001)
    if size < step_size:
        return 0.0
    # Floor to nearest step
    quantized_size = math.floor(size / step_size) * step_size
    # Fix floating point precision
    precision = int(abs(math.log10(step_size))) if step_size < 1 else 0
    return round(quantized_size, precision)

def get_precision(symbol: str) -> int:
    step_size = STEP_SIZES.get(symbol, 0.0001)
    return int(abs(math.log10(step_size))) if step_size < 1 else 0

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
    add_count: int = 0
    partial_taken: bool = False

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
        
        # --- COMPETITION OVERRIDES ---
        # Apply surgical tweaks to let trades run longer and risk slightly more
        if getattr(self.config, "COMPETITION_MODE", False):
            print("[PaperTrader] COMPETITION MODE DETECTED: Applying aggressive tuning overrides.")
            # 1. Let winners run: Target 3.0x ATR for swing moves ($50-100 zone)
            self.config.ATR_TAKE_PROFIT_MULTIPLIER = 3.0
            # 2. Trail later: Start at 1.5x ATR profit (was 2.0), keep stop at 1.0x ATR distance
            self.config.ATR_TRAILING_ACTIVATION_MULTIPLIER = 1.5
            self.config.ATR_TRAILING_FLOOR_MULTIPLIER = 1.0
            # 3. Stop Loss: 1.2x ATR (approx 0.2-0.3% risk)
            self.config.ATR_STOP_MULTIPLIER = 1.2
            # 4. Reduce risk to 1.5% (was 3%) to preserve capital during drawdowns
            self.config.MAX_RISK_PER_TRADE = 0.015
            # 5. Activate breakeven at 1.2x ATR to lock in fees before trailing starts
            self.config.ATR_BREAKEVEN_ACTIVATION_MULTIPLIER = 1.2
            # 6. Ensure breakeven exit covers fees (0.40% buffer)
            self.config.BREAKEVEN_PROFIT_PCT = 0.0040
            
        # Fee Configuration (WEEX Taker ~0.06%)
        self.taker_fee_pct = 0.0006 
        self.consecutive_losses = 0
        # -----------------------------

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

        # Agents (Per-Symbol Storage)
        self.agents_by_symbol: Dict[str, Dict] = {}
        
        # ATR for adaptive stops
        self.atr_history: Dict[str, List[float]] = {} # Store ATR per symbol
        self.atr_period = 14

        # Trade tracking
        self.trades: List[TradeRecord] = []
        self.open_positions: Dict[str, TradeRecord] = {}
        self.governance_trigger_log: List[dict] = []

        # Optional callback: will be set by API layer for monitor
        self.on_trade_closed = None

        # Track last ensemble decision for live trading loop
        self.last_ensemble_decision = None
        self.last_regime = None
        
        # Competition Metrics
        self.high_conviction_trades = 0
        self.processed_candles_count = 0 # Track processed candles for regime warm-up
        self.reconciliation_stable = False # Set by WeexTradingLoop

    def _get_agents(self, symbol: str) -> Dict:
        """Retrieve or initialize agents for a specific symbol."""
        if symbol not in self.agents_by_symbol:
            self.agents_by_symbol[symbol] = {
                "momentum": MomentumRSIAgent(),
                "ml": MLClassifierAgent(),
                "order_flow": OrderFlowAgent(),
                "sentiment": SentimentAgent(),
                "trend_bias": TrendBiasAgent(),
                "llm": LLMAnalysisAgent(update_interval_seconds=300), # 5 min heartbeat
                "ensemble": EnsembleAgent(),
                "portfolio": ThompsonSamplingPortfolioManager(
                    agent_ids=["momentum_rsi", "ml_classifier", "order_flow", "sentiment"]
                ),
                "regime_detector": RegimeDetector(lookback=20), # Always use sufficient lookback for Z-score
                "processed_candles": 0,
                "current_regime": MarketRegime.UNKNOWN
            }
        return self.agents_by_symbol[symbol]

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
        # Calculate PnL for Exits
        entry_price = position.entry_price
        current_price = candle.close
        direction = 1 if position.side == "buy" else -1
        pnl_pct = ((current_price - entry_price) / entry_price) * direction

        # Update High Water Mark (Peak PnL)
        if pnl_pct > position.highest_pnl_pct:
            position.highest_pnl_pct = pnl_pct

        # --- Adaptive Stop/Take-Profit based on ATR ---
        current_atr = self.atr_history.get(candle.symbol, [0])[-1] if self.atr_history.get(candle.symbol) else None
        
        if current_atr and entry_price > 0:
            # OPTIMIZATION: Enforce a minimum ATR floor (0.1% of price)
            # This prevents suffocation in low volatility regimes where ATR is tiny.
            min_atr = entry_price * 0.001
            effective_atr = max(current_atr, min_atr)

            # Convert ATR to a percentage of entry price for comparison
            atr_pct = (effective_atr / entry_price)
            
            # Adaptive Stop Loss: e.g., 2x ATR below entry
            adaptive_stop_loss_pct = self.config.ATR_STOP_MULTIPLIER * atr_pct
            if pnl_pct < -adaptive_stop_loss_pct:
                return True, f"Adaptive Hardstop loss hit: {pnl_pct:.2%} (ATR: {effective_atr:.4f})"

            # --- SOPHISTICATED: Dynamic Take Profit based on Trend Strength (ADX) ---
            # If ADX > 30 (Strong Trend), expand TP to let winners run further.
            tp_multiplier = self.config.ATR_TAKE_PROFIT_MULTIPLIER
            trend_agent = self._get_agents(candle.symbol).get("trend_bias")
            if trend_agent and hasattr(trend_agent, "adx") and trend_agent.adx > 30:
                tp_multiplier *= 1.5 # Expand target by 50% (e.g. 5.0 -> 7.5 ATR)
                # print(f"[PaperTrader] DYNAMIC TP: Strong Trend (ADX {trend_agent.adx:.1f}), expanding target to {tp_multiplier}x ATR")

            adaptive_take_profit_pct = tp_multiplier * atr_pct
            if pnl_pct > adaptive_take_profit_pct:
                return True, f"Adaptive Take Profit hit: {pnl_pct:.2%} (ATR: {effective_atr:.4f})"

            # Adaptive Trailing Stop (Activate after 1x ATR profit, trail at 0.5x ATR from peak)
            activation_threshold_atr = self.config.ATR_TRAILING_ACTIVATION_MULTIPLIER * atr_pct
            # FEE AWARENESS: Don't start trailing until we are at least 2x fees in profit
            fee_hurdle = self.taker_fee_pct * 3.0
            
            # WINNER'S GUARD: If partial profit taken, force trailing active immediately
            if position.partial_taken:
                activation_threshold_atr = 0.0
            
            if position.highest_pnl_pct > max(activation_threshold_atr, fee_hurdle):
                # Trail at a certain percentage below the highest PnL % achieved
                trail_floor_pct = position.highest_pnl_pct - (self.config.ATR_TRAILING_FLOOR_MULTIPLIER * atr_pct)
                if pnl_pct < trail_floor_pct:
                    return True, f"Adaptive Trailing Stop hit: {pnl_pct:.2%} (Peak: {position.highest_pnl_pct:.2%}, ATR: {effective_atr:.4f})"

            # Adaptive Breakeven Protection (Move stop to breakeven + small profit after 1x ATR profit)
            if position.highest_pnl_pct >= self.config.ATR_BREAKEVEN_ACTIVATION_MULTIPLIER * atr_pct:
                if pnl_pct <= self.config.BREAKEVEN_PROFIT_PCT: # Small profit to cover fees
                    return True, f"Adaptive Breakeven protection hit (Peak: {position.highest_pnl_pct:.2%}, ATR: {effective_atr:.4f})"
        else:
            # --- Fallback to Fixed Stop/Take-Profit if ATR not available ---
            if pnl_pct < -self.config.HARDSTOP_PCT:
                return True, f"Hardstop loss hit (fixed): {pnl_pct:.2%}"

            if pnl_pct > self.config.HARDSTOP_PCT * 2.0: # Simple 2:1 R:R
                return True, f"Take Profit hit (fixed 2.0R): {pnl_pct:.2%}"

            if position.highest_pnl_pct > self.config.HARDSTOP_PCT * 0.5:
                trail_floor = position.highest_pnl_pct * 0.6
                if pnl_pct < trail_floor:
                    return True, f"Trailing Stop hit (fixed): {pnl_pct:.2%} (Peak: {position.highest_pnl_pct:.2%})"

            if position.highest_pnl_pct >= self.config.HARDSTOP_PCT:
                if pnl_pct <= 0.0005:
                    return True, f"Breakeven protection hit (fixed): {pnl_pct:.2%}"

        # --- Time-based triggers (always active) ---
        hold_time_minutes = (candle.timestamp - position.timestamp).total_seconds() / 60
        
        # Check Trend Strength for Exit Override
        trend_agent = self._get_agents(candle.symbol).get("trend_bias")
        is_super_trend = trend_agent and hasattr(trend_agent, "adx") and trend_agent.adx > 40

        # Exit Trigger 6: Stale Profit (Time Decay) - Disabled in competition mode to let winners run
        if not self.config.COMPETITION_MODE and hold_time_minutes > (self.config.MAX_HOLD_TIME_MINUTES * 0.5) and pnl_pct > 0.002:
                if is_super_trend:
                    return False, None # Ignore stale profit in super trend
                return True, f"Stale profit exit: {pnl_pct:.2%} after {int(hold_time_minutes)}m"

        # Exit Trigger 3: Max Hold Time (from config)
        # STRATEGY UPDATE: Only enforce time limit if trade is stagnant.
        # If we are riding a winner (> 1% profit), ignore the clock and let the trailing stop work.
        if hold_time_minutes > self.config.MAX_HOLD_TIME_MINUTES:
            if is_super_trend:
                return False, None # Ignore time limit in super trend
            if pnl_pct < 0.01:
                return True, f"Max hold time exceeded with low profit: {int(hold_time_minutes)}m, PnL {pnl_pct:.2%}"
            # Else: Implicitly allow holding longer
            pass

        return False, None

    # ================================================================ #
    # Regime detection
    # ================================================================ #

    def _detect_regime(self, candle: Candle, agents: Dict) -> MarketRegime:
        """Detect current market regime."""
        detector = agents["regime_detector"]
        agents["processed_candles"] += 1
        detector.update(candle.close)
        regime_state = detector.detect()

        # AGGRESSIVE REGIME FORCING FOR COMPETITION: Never stay in UNKNOWN after warm-up
        if self.config.COMPETITION_MODE and regime_state.current == MarketRegime.UNKNOWN and agents["processed_candles"] > detector.lookback:
            z_score = detector.z_score if hasattr(detector, 'z_score') else 0.0
            # If z_score is significant, force a trend. Otherwise, assume REVERSAL/CHOP to enable trading.
            if abs(z_score) > 0.60: # Lowered to 0.60 to catch weak trends and enable trading
                forced_regime = MarketRegime.BULL_TREND if z_score > 0 else MarketRegime.BEAR_TREND
                print(f"[PaperTrader] [{candle.symbol}] COMPETITION: Forced regime from UNKNOWN to {forced_regime.value} due to strong z-score ({z_score:.2f}).")
                agents["current_regime"] = forced_regime
            else:
                print(f"[PaperTrader] [{candle.symbol}] COMPETITION: Forced regime from UNKNOWN to CHOP due to weak z-score ({z_score:.2f}).")
                agents["current_regime"] = MarketRegime("chop") # Use CHOP as a safe, tradable fallback
        else:
            # Use the detector's classification if it's not UNKNOWN, or if not in aggressive competition mode.
            agents["current_regime"] = regime_state.current

        agents["portfolio"].set_regime(agents["current_regime"])

        return agents["current_regime"]

    # ================================================================ #
    # Main simulation APIs
    # ================================================================ #

    async def prime_agents(self, symbol: str, candles: List[Candle]):
        """Warm up agents with historical data."""
        print(f"[PaperTrader] Priming agents for {symbol} with {len(candles)} candles...")
        agents = self._get_agents(symbol)
        
        # Prime agents that need history
        for candle in candles:
            agents["momentum"].update(candle)
            agents["ml"].update(candle)
            # The regime detector also benefits from this, even with a short lookback
            agents["regime_detector"].update(candle.close)
            
        # Set processed_candles to skip warm-up delay since we have historical data
        agents["processed_candles"] = len(candles)

        # Train the ML model once after priming
        if agents["ml"] and hasattr(agents["ml"], 'train'):
            df = pd.DataFrame([asdict(c) for c in candles])
            if not df.empty:
                try:
                    agents["ml"].train(df)
                except Exception as e:
                    print(f"[PaperTrader] ML Agent training failed during priming: {e}")

        # NEW: Detect initial regime so reconciliation knows the state
        if candles:
            self._detect_regime(candles[-1], agents)

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
        agents = self._get_agents(candle.symbol)
        
        # Detect regime
        regime = self._detect_regime(candle, agents)
        print(f"[PaperTrader] [{candle.symbol}] Candle {candle.timestamp} close={candle.close}, regime={regime.value}")

        # Update all agents with the new candle data
        agents["momentum"].update(candle)
        agents["sentiment"].update(candle)
        
        # --- NEW: Order Flow Estimation ---
        # Estimate buy/sell volume from candle data to make OrderFlowAgent functional
        try:
            total_range = candle.high - candle.low
            agents["order_flow"].reset_window() # Reset to capture only current candle's pressure
            if total_range > 0:
                buy_pressure = (candle.close - candle.low) / total_range
                # Dampen pressure to avoid 100% confidence on single candles (noise reduction)
                buy_pressure = max(0.1, min(0.9, buy_pressure))
                # Simple heuristic: volume is split by pressure
                buy_volume = candle.volume * buy_pressure
                sell_volume = candle.volume * (1 - buy_pressure)
                agents["order_flow"].update_volume(buy_volume, sell_volume)
        except Exception as e:
            print(f"[PaperTrader] Order flow estimation failed: {e}")
        # --- END Order Flow Estimation ---

        agents["ml"].update(candle)  # Also update ML agent
        self._update_equity(mark_price=candle.close, symbol=candle.symbol)
        
        # Calculate ATR for adaptive stops
        if candle.symbol not in self.atr_history:
            self.atr_history[candle.symbol] = []

        if len(agents["momentum"].history) >= self.atr_period:
            # Ensure history is for the current symbol if momentum_agent is global
            # Assuming momentum_agent.history is a list of candles for the current symbol, or a dict of lists
            highs = [c.high for c in agents["momentum"].history[-self.atr_period:]]
            lows = [c.low for c in agents["momentum"].history[-self.atr_period:]]
            closes = [c.close for c in agents["momentum"].history[-self.atr_period:]]
            
            tr_values = []
            for i in range(self.atr_period):
                tr = max(highs[i] - lows[i], abs(highs[i] - (closes[i-1] if i > 0 else closes[i])), abs(lows[i] - (closes[i-1] if i > 0 else closes[i])))
                tr_values.append(tr)
            current_atr = sum(tr_values) / self.atr_period
            self.atr_history[candle.symbol].append(current_atr)
        else:
            self.atr_history[candle.symbol].append(0.0) # Append 0 or handle as needed until enough data

        # --- FEE & VOLATILITY GATE ---
        # If the expected move (ATR) is smaller than 3x fees (approx 0.18%), the edge is eaten by costs.
        current_atr = self.atr_history[candle.symbol][-1]
        min_volatility_threshold = candle.close * (self.taker_fee_pct * 3.0)
        
        # --- TILT CONTROL ---
        # If we have taken 3 losses in a row, enter defensive mode.
        is_defensive = self.consecutive_losses >= 3
        if is_defensive:
            if self.consecutive_losses == 3: # Log once
                print(f"[PaperTrader] DEFENSIVE MODE: {self.consecutive_losses} consecutive losses. Tightening filters.")

        # --- Generate Ensemble Decision (needed for both exits and entries) ---
        signals = []
        sig_momentum = agents["momentum"].generate()
        if sig_momentum: signals.append(sig_momentum)
        
        sig_orderflow = agents["order_flow"].generate()
        if sig_orderflow: signals.append(sig_orderflow)

        sig_sentiment = agents["sentiment"].generate()
        # DISABLED: sentiment in recovery mode – it reacts to 1m noise
        # if sig_sentiment: signals.append(sig_sentiment)
        
        # LLM / Macro Agent
        agents["llm"].update(candle)
        sig_llm = agents["llm"].generate(recent_trades=self.trades[-10:])
        if sig_llm: signals.append(sig_llm)

        # Trend Bias (Fusion)
        agents["trend_bias"].update(candle) # Update ADX state
        rsi_val = sig_momentum.metadata.get("rsi", 50.0) if sig_momentum else 50.0
        buy_ratio = sig_orderflow.metadata.get("buy_ratio", 0.5) if sig_orderflow else 0.5
        sell_ratio = sig_orderflow.metadata.get("sell_ratio", 0.5) if sig_orderflow else 0.5
        
        sig_trend = agents["trend_bias"].generate(regime.value, rsi_val, buy_ratio, sell_ratio)
        if sig_trend.direction != 0:
            signals.append(sig_trend)

        # ML Agent signal generation (from Fix #3)
        sig_ml = agents["ml"].generate()
        if sig_ml: signals.append(sig_ml)

        # Fallback signal
        # Fallback trend signal DISABLED for recovery mode – only trade when real agents agree
        if False and not any(s.direction != 0 for s in signals):
            if len(agents["momentum"].history) >= 20:
                closes = [c.close for c in agents["momentum"].history[-20:]]
                sma20 = sum(closes) / len(closes)
                trend_dir = 1 if candle.close > sma20 else -1
                # Set a fixed, reasonable confidence for the fallback signal
                fallback_conf = 0.55
                from backend.agents.signal_agents import TradingSignal as AgentTradingSignal
                trend_sig = AgentTradingSignal(
                    agent_id="trend_fallback",
                    direction=trend_dir,
                    confidence=fallback_conf,
                    metadata={"sma20": sma20, "close": candle.close}
                )
                signals.append(trend_sig)

        # Get dynamic weights and combine signals
        weights = agents["portfolio"].sample_weights_for_regime(regime) # Pass equity for bandit
        agents["ensemble"].weights = weights
        ensemble_decision = agents["ensemble"].combine(signals)

        # COMPETITION OVERRIDE: If ensemble is flat OR weak, let the strongest single agent take over
        # DISABLED: overriding ensemble with single noiser agent causes over-trading
        if False and self.config.COMPETITION_MODE and signals:
            is_flat = ensemble_decision.direction == 0
            is_weak = ensemble_decision.confidence < self.config.MIN_CONFIDENCE
            
            if is_flat or is_weak:
                # Find the signal with the highest confidence
                strongest_signal = max(signals, key=lambda s: s.confidence)
                
                # If we have a strong signal, use it
                if strongest_signal.confidence >= 0.6: # Increased from 0.5 to filter noise
                    # NEW: Enforce Regime Alignment for Overrides
                    regime_dir = 0
                    if regime == MarketRegime.BULL_TREND: regime_dir = 1
                    elif regime == MarketRegime.BEAR_TREND: regime_dir = -1
                    
                    # Only allow override if it aligns with trend or if we are in chop (and taking a scalp)
                    # For Swing Strategy: Strictly enforce trend alignment for overrides in trend regimes
                    is_aligned = (regime_dir == 0) or (strongest_signal.direction == regime_dir)
                    
                    if is_aligned:
                        print(f"[PaperTrader] COMPETITION: Ensemble {'flat' if is_flat else 'weak'} (conf={ensemble_decision.confidence:.2f}), overriding with strongest agent '{strongest_signal.agent_id}' (conf: {strongest_signal.confidence:.2f}).")
                        from backend.agents.signal_agents import TradingDecision
                        # Create a new ensemble decision based on this single agent
                        ensemble_decision = TradingDecision(
                            direction=strongest_signal.direction,
                            confidence=strongest_signal.confidence,
                            agent_signals=[strongest_signal] # Log that this was an override
                        )
                    else:
                        print(f"[PaperTrader] COMPETITION: Strongest signal '{strongest_signal.agent_id}' ({strongest_signal.direction}) opposes regime {regime.value}. Ignoring override.")

        # Cache for external consumers (WeexTradingLoop)
        self.last_ensemble_decision = ensemble_decision
        self.last_regime = regime

        # --- POSITION MANAGEMENT LOGIC ---
        current_position = self.open_positions.get(candle.symbol)
        new_direction = ensemble_decision.direction

        is_pyramiding = False

        # Update equity at the start of processing each candle
        self._update_equity(mark_price=candle.close, symbol=candle.symbol)

        # 1. Handle Signal Flips & Pyramiding
        if current_position and new_direction != 0:
            is_long = current_position.side == 'buy'
            
            # Signal Flip: Close opposing
            if (is_long and new_direction == -1) or (not is_long and new_direction == 1):
                # Before closing, check if the flip is due to low confidence.
                # If confidence is very low, it might be noise, so don't flip aggressively.
                
                # ANTI-CHURN LOGIC: Calculate hold time and PnL
                hold_time_minutes = (candle.timestamp - current_position.timestamp).total_seconds() / 60
                pnl_pct = ((candle.close - current_position.entry_price) / current_position.entry_price) * (1 if is_long else -1)

                # NOISE FILTER: If trade hasn't hit stop or target, ignore flip unless confidence is extreme
                if -0.01 < pnl_pct < 0.015 and ensemble_decision.confidence < 0.85:
                     print(f"[PaperTrader] NOISE FLIP IGNORED: PnL {pnl_pct:.2%} within noise band. Holding {current_position.side}.")
                     return

                # Base flip threshold
                flip_threshold = self.config.MIN_CONFIDENCE + 0.2
                if regime.value == 'chop':
                    flip_threshold += 0.15
                
                # Cap threshold at 0.95 to ensure a flip is mathematically possible with max conviction
                flip_threshold = min(flip_threshold, 0.95)
                
                # HYSTERESIS: If trade is young (< 15 mins) and not in danger (PnL > -0.5%), require overwhelming evidence (0.85+) to flip
                if hold_time_minutes < 15 and pnl_pct > -0.005:
                    flip_threshold = max(flip_threshold, 0.85)

                if ensemble_decision.confidence < flip_threshold:
                    print(f"[PaperTrader] SIGNAL FLIP: Suppressing due to low confidence/young trade (Conf: {ensemble_decision.confidence:.2f} < Threshold: {flip_threshold:.2f}, Hold: {int(hold_time_minutes)}m).")
                    return
                print(f"[PaperTrader] SIGNAL FLIP: Closing {current_position.side} position for {candle.symbol}.")
                self._close_position(symbol=candle.symbol, exit_price=candle.close, timestamp=candle.timestamp, exit_reason="signal_flip")
                current_position = None # Position is now closed
            
            # Add-on Logic (Pyramiding on Dips)
            # Pyramiding DISABLED in recovery mode – one slot per symbol only
            elif False and self.config.COMPETITION_MODE and self.reconciliation_stable:
                # Check for "Add on Dips" condition
                # 1. Strong Regime
                # 2. Same direction signal
                # 3. Price dipped 0.4-0.6% from entry (averaging in on trend pullback) OR breakout
                # 4. Max 1 add-on
                if current_position.add_count < 1 and regime.value in ("bull_trend", "bear_trend"):
                    dip_threshold = 0.004 # 0.4% dip
                    if is_long and new_direction == 1 and candle.close < current_position.entry_price * (1 - dip_threshold):
                         is_pyramiding = True
                    elif not is_long and new_direction == -1 and candle.close > current_position.entry_price * (1 + dip_threshold):
                         is_pyramiding = True

        # 2. Handle Exits (SL, TP, etc.) if a position is still open.
        # Re-check self.open_positions as it might have been closed by the flip logic.
        if candle.symbol in self.open_positions:
            position_to_check = self.open_positions[candle.symbol]
            
            # Check Partial Take Profit
            self._check_partial_exit(position_to_check, candle)
            
            should_exit, exit_reason = self._should_exit(position_to_check, candle, ensemble_decision)
            if should_exit:
                print(f"[PaperTrader] EXIT TRIGGERED: {exit_reason}")
                self._close_position(symbol=candle.symbol, exit_price=candle.close, timestamp=candle.timestamp, exit_reason=exit_reason)
                return # Stop processing for this candle after a SL/TP exit.

        # 3. Handle Entries
        
        # Block new entries until reconciliation has run successfully
        if not self.reconciliation_stable:
            print(f"[PaperTrader] WARNING: Reconciliation not stable. Proceeding with INTERNAL LEDGER ONLY (no exchange sync).")
            # Do NOT return here – allow entries.

        # Allow entry if no position OR if pyramiding is active and we want to add to the same-direction position
        allow_entry = False
        if candle.symbol not in self.open_positions:
            allow_entry = True
        elif is_pyramiding and current_position and current_position.side == ("buy" if new_direction > 0 else "sell"):
            # If pyramiding is enabled and the new signal is in the same direction as the existing position
            allow_entry = True
            print(f"[PaperTrader] PYRAMIDING: Adding to winning {current_position.side} position (Conf: {ensemble_decision.confidence:.2f})")


        if allow_entry:
            # Handle UNKNOWN regime during warm-up or if detector is truly stuck
            if regime == MarketRegime.UNKNOWN:
                if agents["processed_candles"] < agents["regime_detector"].lookback:
                    print(f"[PaperTrader] WARM-UP: Trading in UNKNOWN regime with conservative sizing (candle {agents['processed_candles']}/{agents['regime_detector'].lookback}).")
                    # Allow to proceed, but sizing will be minimal due to default Kelly and low confidence
                elif ensemble_decision.confidence > 0.8:
                    print(f"[PaperTrader] COMPETITION: OVERRIDE UNKNOWN regime due to high confidence ({ensemble_decision.confidence:.2f})")
                else:
                    print("[PaperTrader] COMPETITION RULE: Trade blocked in persistent UNKNOWN regime after warm-up.")
                    return

            if new_direction == 0:
                print("[PaperTrader] Ensemble flat (direction=0) -> no trade")
                return

            if ensemble_decision.confidence < self.config.MIN_CONFIDENCE:
                print(f"[PaperTrader] Confidence {ensemble_decision.confidence:.2f} < {self.config.MIN_CONFIDENCE} -> no trade")
            # --- SOPHISTICATED: Smart Recovery (Dynamic Confidence) ---
            # If we are down for the day, require higher confidence to enter new trades.
            # "Play tight when you're losing."
            dynamic_min_conf = self.config.MIN_CONFIDENCE
            if self.daily_pnl < 0:
                dynamic_min_conf += 0.05 # Raise bar by 5%

            if ensemble_decision.confidence < dynamic_min_conf:
                print(f"[PaperTrader] Confidence {ensemble_decision.confidence:.2f} < {dynamic_min_conf:.2f} (Dynamic) -> no trade")
                return
            
            # RR-AWARE ENTRY VETO
            atr_pct = current_atr / candle.close
            expected_move_pct = atr_pct * self.config.ATR_TAKE_PROFIT_MULTIPLIER
            # Require at least 0.05% potential move (lowered from 0.1%) to justify fees and risk
            if expected_move_pct < 0.0005:
                print(f"[PaperTrader] ENTRY VETO: Expected move {expected_move_pct:.2%} < 0.05% (ATR too low for swing).")
                return
            
            # Stricter entry in CHOP regime to prevent fee burn
            if regime.value == 'chop' and ensemble_decision.confidence < (self.config.MIN_CONFIDENCE + 0.1):
                print(f"[PaperTrader] CHOP REGIME: Blocking entry due to insufficient confidence ({ensemble_decision.confidence:.2f} < {self.config.MIN_CONFIDENCE + 0.1:.2f})")
                return
            
            # Regime-based trading restriction (Concentrate Risk)
            if self.config.COMPETITION_MODE and not self.config.TRADE_IN_CHOPPY_REGIME and regime.value == 'chop':
                print("[PaperTrader] COMPETITION MODE: Blocking new entry in CHOP regime.")
                return
            
            # DEFENSIVE MODE CHECKS
            if is_defensive:
                # 1. Must align with trend if in trend regime
                if regime.value in ('bull_trend', 'bear_trend'):
                    trend_dir = 1 if regime.value == 'bull_trend' else -1
                    if new_direction != trend_dir:
                        print(f"[PaperTrader] DEFENSIVE: Ignoring counter-trend signal (Regime: {regime.value}).")
                        return
                # 2. If in chop, require higher confidence
                elif ensemble_decision.confidence < 0.8:
                    print(f"[PaperTrader] DEFENSIVE: Ignoring low confidence signal in chop ({ensemble_decision.confidence:.2f} < 0.8).")
                    return
            
            side = "buy" if new_direction > 0 else "sell"
            
            # --- HIGH-CONVICTION SIZING LOGIC ---
            # Determine sizing mode based on regime, confidence, and session safety
            # Boosted Size: Strong Trend + Signal Agreement (TrendBias active) + High Confidence
            is_boosted = False
            if regime.value in ("bull_trend", "bear_trend") and not is_defensive:
                # Check if TrendBias signal aligns
                if sig_trend.direction == new_direction and ensemble_decision.confidence >= 0.75:
                     # Check expected move
                     if expected_move_pct >= 0.006: # > 0.6% expected move
                         is_boosted = True
            
            leverage = 20.0
            if is_boosted:
                # Boosted: 0.4x Equity Notional
                target_margin_pct = 0.60 
                print(f"[PaperTrader] SIZING: BOOSTED Trend Setup. Using {target_margin_pct:.0%} equity.")
            else:
                # Base: 0.15x Equity Notional
                target_margin_pct = 0.30
                print(f"[PaperTrader] SIZING: Base Trend Setup. Using {target_margin_pct:.0%} equity.")

            target_margin_usdt = self.equity * target_margin_pct
            scaled_size_usdt = target_margin_usdt * leverage
            
            # Cap by max risk per trade (e.g. 1.5% of equity loss)
            # Stop distance is ~1.2 ATR.
            stop_loss_pct = atr_pct * self.config.ATR_STOP_MULTIPLIER
            if stop_loss_pct > 0:
                risk_per_share = stop_loss_pct * candle.close
                max_loss_usdt = self.equity * self.config.MAX_RISK_PER_TRADE
                max_size_by_risk = (max_loss_usdt / risk_per_share) * candle.close
                
                if max_size_by_risk < scaled_size_usdt:
                    print(f"[PaperTrader] SIZING: Capped by risk. Target Notional: {scaled_size_usdt:.2f} -> Risk Cap: {max_size_by_risk:.2f}")
                    scaled_size_usdt = max_size_by_risk

            # For pyramiding, we are adding to an existing position.
            # The size should be an *additional* amount, not the total.
            if is_pyramiding:
                # Add 50-70% of base size
                scaled_size_usdt = (self.equity * 0.15 * leverage) * 0.6
                
            size = scaled_size_usdt / candle.close
            
            trading_signal = TradingSignal(
                symbol=candle.symbol,
                side=side,
                size=size,
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
                candle.timestamp, trading_signal, decision, regime, is_pyramiding
            )
            if not decision.allow or decision.adjusted_size <= 0:
                print("[PaperTrader] Trade blocked by governance or zero size")
                return
            
            if ensemble_decision.confidence > 0.7:
                self.high_conviction_trades += 1
                
            final_size = decision.adjusted_size
            # Open new position
            success = self._open_position(
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
            if success:
                print( # Changed message for pyramiding
                    f"[PaperTrader] {'Pyramided' if is_pyramiding else 'Opened'} position side={side}, size={final_size:.6f}, "
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
    ) -> bool:
        # Generate order ID first
        order_id = str(uuid.uuid4())

        # Quantize size to ensure WEEX acceptance
        final_size = normalize_size(symbol, size)
        if final_size <= 0:
            print(f"[PaperTrader] Size {size} too small for {symbol} (step {STEP_SIZES.get(symbol)}). Skipping.")
            return False

        # EXECUTION: Send order to WEEX if client is connected
        # CRITICAL FIX: Only record position internally if execution succeeds
        if self.execution_client:
            try:
                # Map side to WEEX type: 1=Open Long, 2=Open Short
                order_type = "1" if side == "buy" else "2"
                
                # If position exists (pyramiding), we are just adding size. 
                # WEEX API handles this naturally for same-side orders in Hedge Mode? 
                # Actually, in Hedge Mode, multiple opens just add to the position or create new ones depending on exchange.
                # Assuming standard behavior: Open Long adds to Long position.
                print(f"[PaperTrader] EXECUTION: Placing SMART {side} order for {final_size} {symbol}...")
                response = self.execution_client.place_smart_order(symbol, f"{final_size:.{get_precision(symbol)}f}", side
                )
                if isinstance(response, dict) and response.get("code") and response["code"] != "00000":
                    print(f"[PaperTrader] EXECUTION API ERROR: {response}")
                    return False
                print(f"[PaperTrader] EXECUTION SUCCESS: Order placed. Response: {response}")
                
                # AI LOG UPLOAD (Compliance)
                if hasattr(self.execution_client, 'upload_ai_log'):
                    try:
                        ai_log = {
                            "symbol": symbol,
                            "side": "1" if side == "buy" else "2",
                            "size": str(final_size),
                            "price": str(price),
                            "timestamp": int(timestamp.timestamp() * 1000),
                            # REQUIRED FIELDS FOR COMPETITION COMPLIANCE
                            "model": "ChronosX-Ensemble",
                            "input": f"Regime: {regime}, Confidence: {ensemble_confidence:.2f}",
                            "output": f"Open {side} {final_size}",
                            "explanation": governance_reason or "High conviction ensemble signal",
                            "stage": "production",
                            # Legacy/Optional
                            "confidence": str(ensemble_confidence),
                        }
                        self.execution_client.upload_ai_log(ai_log)
                    except Exception as e:
                        print(f"[PaperTrader] AI Log Upload Failed: {e}")
            except Exception as e:
                print(f"[PaperTrader] EXECUTION ERROR: {e}")
                return False

        # Update internal state
        if symbol in self.open_positions:
            # Pyramiding: Update existing position
            pos = self.open_positions[symbol]
            total_size = pos.size + final_size
            avg_price = ((pos.size * pos.entry_price) + (final_size * price)) / total_size
            pos.size = total_size
            pos.entry_price = avg_price
            # Reset PnL tracking for new avg price? Or keep relative? 
            # Keeping simple: update entry price, PnL will recalculate next tick.
            pos.add_count += 1
            return True

        new_position = TradeRecord(
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            size=final_size,
            entry_price=price,
            exit_price=price,
            pnl=0.0,
            agent_id="ensemble",
            governance_reason=governance_reason,
            risk_score=risk_score,
            regime=regime,
            contributing_agents=contributing_agents or [],
            ensemble_confidence=ensemble_confidence,
            order_id=order_id,
            exit_reason=exit_reason,
        )
        
        self.open_positions[symbol] = new_position
        return True

    def inject_manual_position(self, symbol: str, side: str, size: float, entry_price: float):
        """Manually inject a position to sync with exchange when API fails."""
        print(f"[PaperTrader] MANUAL SYNC: Injecting {side} {size} {symbol} @ {entry_price}")
        self.open_positions[symbol] = TradeRecord(
            timestamp=datetime.now(),
            symbol=symbol,
            side=side,
            size=size,
            entry_price=entry_price,
            exit_price=entry_price,
            pnl=0.0,
            agent_id="manual_sync",
            governance_reason="Manual Dashboard Sync",
            risk_score=0.0,
            regime="unknown",
            exit_reason="open"
        )

    def _close_position(self, symbol: str, exit_price: float, timestamp: datetime, exit_reason: str = "unknown"):
        # Peek at the position first; do not remove until execution is confirmed
        position_to_close = self.open_positions.get(symbol)
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
                print(f"[PaperTrader] EXECUTION: Closing {position_to_close.side} position for {symbol} via MARKET order...")
                response = self.execution_client.place_order(
                    symbol=symbol,
                    size=f"{position_to_close.size:.{get_precision(symbol)}f}",
                    type_=close_type,
                    price="0", # Price is ignored for market order, set to 0
                    match_price="1" # CRITICAL: '1' for market order to ensure execution
                )
                
                # CRITICAL FIX: If API fails, do NOT close internal position. Retry next tick.
                if isinstance(response, dict) and response.get("code") and response["code"] != "00000":
                    print(f"[PaperTrader] EXECUTION API ERROR (Close): {response}")
                    return

                print(f"[PaperTrader] EXECUTION SUCCESS: Position closed. Response: {response}")
                
                # AI LOG UPLOAD (Compliance for Close)
                if hasattr(self.execution_client, 'upload_ai_log'):
                    try:
                        ai_log = {
                            "symbol": symbol,
                            "side": close_type, # 3 or 4
                            "size": f"{position_to_close.size:.{get_precision(symbol)}f}",
                            "price": str(exit_price),
                            "timestamp": int(timestamp.timestamp() * 1000),
                            # REQUIRED FIELDS FOR COMPETITION COMPLIANCE
                            "model": "ChronosX-Ensemble",
                            "input": f"Regime: {position_to_close.regime}, PnL: {pnl:.4f}",
                            "output": f"Close {position_to_close.side}",
                            "explanation": exit_reason,
                            "stage": "production",
                        }
                        self.execution_client.upload_ai_log(ai_log)
                    except Exception as e:
                        print(f"[PaperTrader] AI Log Upload Failed (Close): {e}")
            except Exception as e:
                print(f"[PaperTrader] EXECUTION ERROR (Close): {e}")
                return

        # Execution successful (or not required), now safe to remove from ledger
        self.open_positions.pop(symbol, None)
        self.trades.append(trade)
        self.balance += pnl
        self.total_pnl += pnl
        self.daily_pnl += pnl
        self.recent_pnls.append(pnl)
        
        # Update Tilt State
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # Distribute the result to all contributing agents for the bandit to learn
        agents = self._get_agents(trade.symbol)
        for agent_id in trade.contributing_agents:
            agents["portfolio"].record_trade_result(
                agent_id=agent_id,
                pnl=pnl,
                position_size=trade.size,
                entry_price=trade.entry_price,
            )

        print(
            f"[PaperTrader] Closed position side={trade.side}, size={trade.size:.6f}, "
            f"entry={trade.entry_price}, exit={exit_price}, pnl={pnl:.4f}, total_pnl={self.total_pnl:.4f}, reason='{exit_reason}'"
        )

    def _check_partial_exit(self, position: TradeRecord, candle: Candle):
        """Check and execute partial take profit."""
        if position.partial_taken:
            return

        current_price = candle.close
        direction = 1 if position.side == "buy" else -1
        pnl_pct = ((current_price - position.entry_price) / position.entry_price) * direction

        # Target: Configurable Partial TP (default 1.5%)
        if pnl_pct >= self.config.PARTIAL_TP_PCT:
            print(f"[PaperTrader] PARTIAL TP: PnL {pnl_pct:.2%} hit target. Closing {self.config.PARTIAL_TP_SIZE:.0%}.")
            
            close_size = position.size * self.config.PARTIAL_TP_SIZE
            final_size = normalize_size(position.symbol, close_size)
            
            if final_size > 0:
                self._execute_partial_close(position, final_size, current_price, "partial_tp")
                position.partial_taken = True
                # Move stop to breakeven logic is handled by ATR_BREAKEVEN_ACTIVATION_MULTIPLIER

    def _execute_partial_close(self, position: TradeRecord, size: float, price: float, reason: str):
        """Execute a partial close without removing the position record."""
        if self.execution_client:
            try:
                close_type = "3" if position.side == "buy" else "4"
                response = self.execution_client.place_order(
                    symbol=position.symbol,
                    size=f"{size:.{get_precision(position.symbol)}f}",
                    type_=close_type,
                    price="0",
                    match_price="1"
                )
                print(f"[PaperTrader] PARTIAL EXECUTION SUCCESS: {response}")
                
                # Update internal state
                position.size -= size
                
                # Log PnL for the closed portion
                direction = 1 if position.side == "buy" else -1
                pnl = ((price - position.entry_price) * direction * size)
                self.balance += pnl
                self.total_pnl += pnl
                self.daily_pnl += pnl
                self.recent_pnls.append(pnl)
                
            except Exception as e:
                print(f"[PaperTrader] PARTIAL EXECUTION FAILED: {e}")


    def reconcile_positions_from_response(self, resp: Dict[str, Any]):
        """Parse WEEX API response and reconcile positions."""
        positions = []
        if isinstance(resp, dict) and "data" in resp:
            # Handle WEEX response format (often data is a list or data['lists'])
            data = resp["data"]
            if isinstance(data, list): positions = data
            elif isinstance(data, dict) and "lists" in data: positions = data["lists"]
        
        if positions:
            self.reconcile_positions(positions)
            # Message moved to WeexTradingLoop to centralize state reporting
        else:
            print("[PaperTrader] No open positions found on exchange during reconciliation.")

    def reconcile_positions(self, external_positions: List[Dict[str, Any]]):
        """
        Sync internal state with actual exchange positions on startup.
        """
        print(f"[PaperTrader] Reconciling {len(external_positions)} external positions...")
        for pos in external_positions:
            symbol = pos.get("symbol")
            if not symbol: continue
            
            # WEEX specific field mapping
            # Try 'holdAmount' (common) or 'size'
            size = float(pos.get("holdAmount", 0) or pos.get("size", 0))
            if size <= 0: continue
            
            # Determine side: 1=long, 2=short usually
            side_raw = str(pos.get("side", "")).lower()
            if "long" in side_raw or side_raw == "1":
                side = "buy"
            elif "short" in side_raw or side_raw == "2":
                side = "sell"
            else:
                continue # Unknown side

            entry_price = float(pos.get("averageOpenPrice", 0) or pos.get("openPrice", 0))
            
            # --- CHECK INTERNAL STATE FIRST ---
            # If we manually injected a position, respect it if it matches.
            if symbol in self.open_positions:
                existing = self.open_positions[symbol]
                
                # Case 1: Perfect Match (or close enough) -> Update details
                if existing.side == side:
                    print(f"[PaperTrader] RECONCILIATION: Confirmed existing {side} position for {symbol}. Updating details.")
                    existing.size = size
                    existing.entry_price = entry_price
                    continue # Done with this position

                # Case 2: Side Mismatch -> Hedged State (Handled below)
                # We fall through to the existing logic for conflict resolution.
                pass 

            # --- ORPHAN HANDLING (No internal record) ---
            
            # --- NEW: Legacy/Misaligned Position Cleanup ---
            agents = self._get_agents(symbol)
            current_regime = agents.get("current_regime", MarketRegime.UNKNOWN)
            # Only apply cleanup if this is truly an orphan (not in open_positions)
            if symbol not in self.open_positions and self.config.COMPETITION_MODE and current_regime in (MarketRegime.BULL_TREND, MarketRegime.BEAR_TREND):
                is_misaligned = False
                if current_regime == MarketRegime.BULL_TREND and side == "sell":
                    is_misaligned = True
                elif current_regime == MarketRegime.BEAR_TREND and side == "buy":
                    is_misaligned = True
                
                if is_misaligned:
                    print(f"[PaperTrader] RECONCILIATION: Found misaligned {side} position for {symbol} in {current_regime.value}. Auto-closing legacy position.")
                    if self.execution_client:
                        try:
                            # 3=Close Long, 4=Close Short
                            close_type = "3" if side == "buy" else "4"
                            self.execution_client.place_order(
                                symbol=symbol,
                                size=f"{size:.{get_precision(symbol)}f}",
                                type_=close_type,
                                price="0",
                                match_price="1"
                            )
                            # Log compliance
                            if hasattr(self.execution_client, 'upload_ai_log'):
                                self.execution_client.upload_ai_log({
                                    "symbol": symbol,
                                    "side": close_type,
                                    "size": str(size),
                                    "price": str(entry_price),
                                    "timestamp": int(datetime.now().timestamp() * 1000),
                                    "model": "ChronosX-Governance",
                                    "input": f"Regime: {current_regime.value}",
                                    "output": "Close Legacy",
                                    "explanation": "Startup cleanup of misaligned legacy position",
                                    "stage": "production"
                                })
                        except Exception as e:
                            print(f"[PaperTrader] Failed to auto-close legacy position: {e}")
                    continue # Do not adopt this position
            # -----------------------------------------------

            if symbol in self.open_positions:
                existing = self.open_positions[symbol]
                print(f"[PaperTrader] CRITICAL WARNING: HEDGED STATE DETECTED for {symbol}. Found {side} while holding {existing.side}.")
                
                # COMPETITION LOGIC: Auto-resolve hedge by closing the newly found position to maintain single-slot state
                if self.execution_client:
                    print(f"[PaperTrader] RECONCILIATION: Auto-closing conflicting {side} position for {symbol} to enforce single-mode.")
                    try:
                        # 3=Close Long, 4=Close Short
                        close_type = "3" if side == "buy" else "4"
                        self.execution_client.place_order(
                            symbol=symbol,
                            size=f"{size:.{get_precision(symbol)}f}",
                            type_=close_type,
                            price="0",
                            match_price="1"
                        )
                    except Exception as e:
                        print(f"[PaperTrader] Failed to auto-close hedged position: {e}")
                continue
                
            print(f"[PaperTrader] Adopting orphan position: {symbol} {side} {size} @ {entry_price}")
            self.open_positions[symbol] = TradeRecord(
                timestamp=datetime.now(), # Unknown original time
                symbol=symbol,
                side=side,
                size=size,
                entry_price=entry_price,
                exit_price=entry_price,
                pnl=0.0,
                agent_id="manual_recovery",
                governance_reason="Startup Reconciliation",
                risk_score=0.0,
                regime="unknown",
                exit_reason="open"
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
        is_pyramiding: bool,
    ):
        """Log governance rule triggers for analytics."""
        # STEP 5: FIX THE NARRATIVE
        decision_type = "STANDARD_APPROVAL"
        if not decision.allow:
            decision_type = "BLOCKED"
        elif is_pyramiding:
            decision_type = "PYRAMID"
        elif signal.confidence >= 0.85 and regime.value in ("bull_trend", "bear_trend"):
            decision_type = "RISK_ESCALATION"

        self.governance_trigger_log.append(
            {
                "timestamp": timestamp.isoformat(),
                "decision_type": decision_type,
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
                "high_conviction_trades": self.high_conviction_trades
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
            "high_conviction_trades": self.high_conviction_trades
        }

    def reload_config(self, new_config: TradingConfig):
        """Reloads the configuration for the trader and its components."""
        self.config = new_config
        self.governance.reload_config(new_config)
        logger.warning("PaperTrader configuration reloaded.")
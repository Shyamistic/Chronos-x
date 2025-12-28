# ChronosX Frontend Build Prompt

Build a complete, production-ready HTML frontend for the ChronosX AI trading system. Create a single `index.html` file that includes HTML, CSS, and JavaScript to provide a comprehensive dashboard for monitoring and controlling the trading system.

## Backend API Information

### Base URL
- **Development**: `http://localhost:8000`
- **API Documentation**: Available at `/docs` (FastAPI auto-generated)

### Core API Endpoints

#### System Health & Control
- `GET /system/health` - System status, uptime, trading state, database connectivity
- `GET /health` - Simple health check
- `POST /trading/live` - Start/stop trading (body: `{"action": "start"}` or `{"action": "stop"}`)
- `GET /trading/live-status` - Current trading loop status

#### Analytics & Performance
- `GET /analytics/metrics` - Real-time performance metrics (PnL, Sharpe, win rate, drawdown)
- `GET /analytics/trades?limit=100` - Trade history from database or memory
- `GET /analytics/export.csv` - Export trades as CSV file
- `GET /agents/performance` - Per-agent signal quality metrics

#### Governance & Configuration
- `GET /governance/rules` - Current governance configuration and risk rules
- `GET /config` - Trading configuration parameters
- `GET /system/alpha-disclaimer` - Alpha mode disclaimer and status

### Data Structures

#### System Health Response
```json
{
  "status": "healthy",
  "trading_enabled": true,
  "last_trade_at": "2024-01-15T10:30:00",
  "symbols_active": ["cmt_btcusdt"],
  "governance_mode": "alpha",
  "uptime_seconds": 3600,
  "database_connected": true,
  "total_trades": 25
}
```

#### Trading Status Response
```json
{
  "running": true,
  "trades": 25,
  "open_positions": 1,
  "mode": "ALPHA (force_execute=true)",
  "current_pnl": 150.75
}
```

#### Analytics Metrics Response
```json
{
  "total_pnl": 150.75,
  "num_trades": 25,
  "win_rate": 0.64,
  "sharpe_ratio": 1.23,
  "max_drawdown": -0.08,
  "profit_factor": 1.45,
  "recovery_factor": 2.1,
  "equity_curve": [{"timestamp": "2024-01-15T10:00:00", "equity": 50000}, ...],
  "recent_trades": [...]
}
```

#### Trade Object Structure
```json
{
  "timestamp": "2024-01-15T10:30:00",
  "order_id": "order_123",
  "symbol": "cmt_btcusdt",
  "side": "buy",
  "size": 0.001,
  "entry_price": 42500.0,
  "exit_price": 42650.0,
  "pnl": 0.15,
  "slippage": 0.001,
  "execution_latency_ms": 250,
  "status": "filled"
}
```

#### Governance Rules Response
```json
{
  "mode": "ALPHA (force_execute=true)",
  "force_execute_mode": true,
  "mpc_threshold": 2,
  "mpc_nodes": 3,
  "min_confidence": 0.05,
  "max_position_size": 0.01,
  "kelly_fraction": 0.25,
  "circuit_breaker": {
    "max_daily_loss": "-10%",
    "max_weekly_loss": "-25%",
    "max_leverage": "10x",
    "max_drawdown": "-20%"
  }
}
```

#### Agent Performance Response
```json
{
  "status": "running",
  "agents": {
    "sentiment": {
      "signal_count": 150,
      "avg_confidence": 0.294,
      "last_signal": "2024-01-15T10:29:00",
      "direction": 1
    },
    "momentum_rsi": {
      "signal_count": 145,
      "avg_confidence": 0.456,
      "last_signal": "2024-01-15T10:29:00",
      "direction": -1
    },
    "ml_classifier": {
      "status": "active",
      "trained": true
    },
    "order_flow": {
      "status": "active",
      "buy_volume": 1250.5,
      "sell_volume": 980.2
    }
  }
}
```

## System Architecture Context

### Trading System Components
1. **Signal Agents**: 4 independent AI agents generating trading signals
   - **SentimentAgent**: Price momentum sentiment (0.22-0.9 confidence range)
   - **MomentumRSIAgent**: RSI + SMA mean-reversion/momentum hybrid
   - **MLClassifierAgent**: XGBoost classifier predicting price direction
   - **OrderFlowAgent**: Buy/sell volume ratio analysis

2. **Ensemble Decision Making**: Confidence-weighted soft voting
3. **Portfolio Management**: Thompson Sampling for agent weight optimization
4. **Governance Engine**: 12-rule risk management system with MPC approval
5. **Risk Management**: 6-layer circuit breaker system with Kelly Criterion sizing
6. **Execution**: Smart execution with quality gates and slippage protection

### Current System Status
- **Mode**: ALPHA testing with loosened risk parameters (5x normal thresholds)
- **Governance**: MPC bypassed with `force_execute=true` for throughput testing
- **Symbol**: Trading CMT/BTCUSDT pair
- **Account**: $50,000 starting equity
- **Database**: PostgreSQL for trade persistence and analytics

## Frontend Requirements

### Core Features Required

#### 1. System Overview Dashboard
- **System Status**: Live/Offline indicator with color coding
- **Trading Mode**: ALPHA/PRODUCTION mode display
- **Uptime Counter**: System uptime in hours/minutes
- **Connection Status**: API connectivity indicator
- **Current Time**: UTC clock display

#### 2. Performance Metrics Panel
- **Total PnL**: Current profit/loss with color coding (green/red)
- **Trade Count**: Total number of executed trades
- **Win Rate**: Percentage of profitable trades
- **Sharpe Ratio**: Risk-adjusted return metric
- **Max Drawdown**: Largest peak-to-trough decline
- **Profit Factor**: Ratio of gross profit to gross loss

#### 3. Live Trading Control
- **Start/Stop Trading**: Buttons to control trading loop
- **Trading Status**: Current state (Running/Stopped/Error)
- **Open Positions**: Number of active positions
- **Current PnL**: Real-time profit/loss of open positions

#### 4. AI Agents Status
- **Agent Grid**: 2x2 grid showing all 4 agents
- **Signal Indicators**: Current direction (↑/↓/→) and confidence level
- **Agent Health**: Active/Inactive status for each agent
- **Last Signal Time**: Timestamp of most recent signal
- **Signal Counts**: Total signals generated per agent

#### 5. Governance & Risk Monitoring
- **Governance Mode**: ALPHA/PRODUCTION indicator
- **Risk Rules Status**: Table showing key risk parameters
  - Min Confidence Threshold: 5%
  - Max Position Size: 0.01 BTC
  - Max Daily Loss: -10%
  - Max Leverage: 10x
  - Max Drawdown: -20%
- **Circuit Breaker Status**: Current risk levels vs limits

#### 6. Trade History Table
- **Recent Trades**: Last 10-20 trades in tabular format
- **Columns**: Timestamp, Symbol, Side, Size, Entry Price, PnL
- **Color Coding**: Green for profitable trades, red for losses
- **Pagination**: Load more trades on demand

#### 7. Analytics & Charts
- **Equity Curve**: Line chart showing account balance over time
- **PnL Distribution**: Histogram of trade profits/losses
- **Agent Performance**: Bar chart comparing agent success rates
- **Export Functionality**: Download trades as CSV

#### 8. System Events Log
- **Event Stream**: Real-time log of system events
- **Event Types**: Trades, signals, errors, status changes
- **Timestamps**: All events with precise timing
- **Filtering**: Ability to filter by event type

### Technical Requirements

#### Design & UX
- **Modern Glass Morphism**: Translucent panels with blur effects
- **Dark Theme**: Professional dark color scheme suitable for trading
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Real-time Updates**: Auto-refresh every 5 seconds
- **Loading States**: Proper loading indicators for API calls
- **Error Handling**: Graceful error messages and retry mechanisms

#### Performance
- **Efficient API Calls**: Batch requests where possible
- **Caching**: Cache static data to reduce API load
- **Debouncing**: Prevent excessive API calls during rapid updates
- **Memory Management**: Clean up old data to prevent memory leaks

#### Accessibility
- **Color Blind Friendly**: Use patterns/icons in addition to colors
- **Keyboard Navigation**: Full keyboard accessibility
- **Screen Reader Support**: Proper ARIA labels and semantic HTML
- **High Contrast**: Ensure sufficient color contrast ratios

### Configuration
```javascript
const CONFIG = {
    API_BASE_URL: 'http://localhost:8000',
    REFRESH_INTERVAL: 5000, // 5 seconds
    TIMEOUT: 10000, // 10 seconds
    MAX_TRADES_DISPLAY: 20,
    MAX_EVENTS_DISPLAY: 50
};
```

### Error Handling Strategy
- **Connection Loss**: Show offline indicator, queue failed requests
- **API Errors**: Display user-friendly error messages
- **Data Validation**: Validate all API responses before rendering
- **Fallback States**: Show placeholder data when APIs are unavailable

### Security Considerations
- **CORS**: Backend already configured for cross-origin requests
- **Input Validation**: Sanitize all user inputs
- **XSS Prevention**: Use proper HTML escaping
- **Rate Limiting**: Respect API rate limits

## Implementation Notes

### Key JavaScript Functions Needed
- `fetchAPI(endpoint)` - Centralized API calling with error handling
- `updateSystemHealth(data)` - Update system status indicators
- `updateMetrics(data)` - Update performance metrics display
- `updateTrades(data)` - Refresh trade history table
- `updateAgents(data)` - Update AI agent status grid
- `updateGovernance(data)` - Update governance rules display
- `startTrading()` / `stopTrading()` - Control trading loop
- `exportTrades()` - Download CSV export
- `showToast(message, type)` - Show notification messages

### CSS Classes Structure
- `.dashboard-grid` - Main grid layout
- `.panel` - Individual dashboard panels
- `.metric-card` - Performance metric displays
- `.agent-card` - AI agent status cards
- `.trade-row` - Trade history table rows
- `.status-indicator` - System status badges
- `.btn-primary` / `.btn-danger` - Action buttons
- `.loading` - Loading state indicators

### Data Flow
1. **Initialization**: Load initial data from all endpoints
2. **Real-time Updates**: Poll key endpoints every 5 seconds
3. **User Actions**: Handle start/stop trading, export data
4. **Error Recovery**: Retry failed requests, show connection status
5. **State Management**: Maintain application state in JavaScript

Build a complete, professional trading dashboard that showcases the full capabilities of the ChronosX AI trading system. The frontend should be production-ready, visually impressive, and provide comprehensive monitoring and control capabilities for the sophisticated backend architecture.
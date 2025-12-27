# backend/database/schema.sql
CREATE TABLE trades (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    order_id VARCHAR(50) UNIQUE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    size DECIMAL(18, 8) NOT NULL,
    entry_price DECIMAL(18, 2) NOT NULL,
    exit_price DECIMAL(18, 2),
    pnl DECIMAL(18, 2),
    slippage DECIMAL(10, 6),
    execution_latency_ms INTEGER,
    agent_signals JSONB,
    governance_approval JSONB,
    status VARCHAR(20) NOT NULL
);

CREATE TABLE metrics_snapshots (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    total_pnl DECIMAL(18, 2),
    num_trades INTEGER,
    win_rate DECIMAL(5, 4),
    sharpe_ratio DECIMAL(10, 4),
    max_drawdown DECIMAL(5, 4),
    snapshot_data JSONB
);

CREATE TABLE system_events (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    message TEXT,
    metadata JSONB
);

CREATE INDEX idx_trades_timestamp ON trades(timestamp DESC);
CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_system_events_timestamp ON system_events(timestamp DESC);
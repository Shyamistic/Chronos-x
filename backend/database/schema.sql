# Create the database directory structure
mkdir -p backend/database

# Create schema file
cat > backend/database/schema.sql << 'EOF'
-- ChronosX Trading Database Schema

-- Trades table
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    order_id VARCHAR(50) UNIQUE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    size DECIMAL(18, 8) NOT NULL,
    entry_price DECIMAL(18, 2) NOT NULL,
    exit_price DECIMAL(18, 2),
    pnl DECIMAL(18, 2) DEFAULT 0,
    slippage DECIMAL(10, 6),
    execution_latency_ms INTEGER,
    agent_signals JSONB,
    governance_approval JSONB,
    status VARCHAR(20) NOT NULL
);

-- Metrics snapshots table
CREATE TABLE IF NOT EXISTS metrics_snapshots (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    total_pnl DECIMAL(18, 2),
    num_trades INTEGER,
    win_rate DECIMAL(5, 4),
    sharpe_ratio DECIMAL(10, 4),
    max_drawdown DECIMAL(5, 4),
    snapshot_data JSONB
);

-- System events table
CREATE TABLE IF NOT EXISTS system_events (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    message TEXT,
    metadata JSONB
);

-- Positions table (placeholder - no logic)
CREATE TABLE IF NOT EXISTS positions (
    symbol TEXT PRIMARY KEY,
    size REAL,
    avg_entry REAL,
    unrealized_pnl REAL,
    updated_at TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_order_id ON trades(order_id);
CREATE INDEX IF NOT EXISTS idx_system_events_timestamp ON system_events(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics_snapshots(timestamp DESC);

-- Views for quick analytics
CREATE OR REPLACE VIEW daily_performance AS
SELECT 
    DATE(timestamp) as trade_date,
    COUNT(*) as num_trades,
    SUM(pnl) as daily_pnl,
    AVG(CASE WHEN pnl > 0 THEN 1.0 ELSE 0.0 END) as win_rate,
    AVG(slippage) as avg_slippage,
    AVG(execution_latency_ms) as avg_latency_ms
FROM trades
GROUP BY DATE(timestamp)
ORDER BY trade_date DESC;
EOF

# Apply schema
PGPASSWORD='ChronosX2025!SecurePass' psql -U chronosx -d chronosx -h localhost -f backend/database/schema.sql
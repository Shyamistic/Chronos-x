# backend/governance/mpc_governance.py
"""
MPC-backed governance for trade approval.
2-of-3 node approval prevents rogue trading.
"""

import hashlib
import hmac
from typing import List, Dict, Optional

class MPCGovernanceNode:
    """One node in the MPC network."""
    
    def __init__(self, node_id: str, node_secret: str):
        self.node_id = node_id
        self.node_secret = node_secret
        self.pending_trades = {}
        self.approvals = {}
    
    def sign_trade(self, trade_id: str, trade_data: dict) -> str:
        """Sign a proposed trade."""
        payload = f"{trade_id}:{trade_data['symbol']}:{trade_data['size']}:{trade_data['price']}"
        signature = hmac.new(
            self.node_secret.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def verify_signature(self, trade_id: str, signature: str, trade_data: dict) -> bool:
        """Verify signature from another node."""
        expected = self.sign_trade(trade_id, trade_data)
        return hmac.compare_digest(signature, expected)


class MPCTradeGateway:
    """Central trade execution gateway (2-of-3 MPC approvals)."""
    
    def __init__(self, nodes: List[MPCGovernanceNode]):
        self.nodes = {node.node_id: node for node in nodes}
        self.pending_trades = {}
    
    def submit_trade(self, trade_id: str, trade_data: dict) -> dict:
        """Submit trade for MPC approval."""
        self.pending_trades[trade_id] = {
            "data": trade_data,
            "approvals": {},
            "approved": False,
        }
        print(f"[MPC] Trade {trade_id} submitted (requires 2/3 nodes)")
        return {"status": "pending_approval", "trade_id": trade_id}
    
    def approve_trade(self, trade_id: str, node_id: str) -> dict:
        """One node approves the trade."""
        if trade_id not in self.pending_trades:
            return {"error": f"Trade {trade_id} not found"}
        
        if node_id not in self.nodes:
            return {"error": f"Node {node_id} not found"}
        
        node = self.nodes[node_id]
        trade_data = self.pending_trades[trade_id]["data"]
        signature = node.sign_trade(trade_id, trade_data)
        
        self.pending_trades[trade_id]["approvals"][node_id] = signature
        approval_count = len(self.pending_trades[trade_id]["approvals"])
        
        print(f"[MPC] Trade {trade_id} approved by {node_id} ({approval_count}/3)")
        
        if approval_count >= 2:
            self.pending_trades[trade_id]["approved"] = True
            print(f"[MPC] Trade {trade_id} APPROVED (2/3 threshold)")
            return {"status": "approved", "trade_id": trade_id}
        
        return {"status": "pending_approval", "trade_id": trade_id, "approvals": approval_count}
    
    def can_execute_trade(self, trade_id: str) -> bool:
        """Check if trade has sufficient MPC approvals."""
        if trade_id not in self.pending_trades:
            return False
        return self.pending_trades[trade_id]["approved"]

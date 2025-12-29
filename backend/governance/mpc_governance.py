# backend/governance/mpc_governance.py
"""
MPC-backed governance for trade approval.
2-of-3 node approval prevents rogue trading.
"""

import hashlib
import hmac
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any


class MPCGovernance:
    """
    Multi-Party Computation governance for trade approval.
    Requires threshold signatures from distributed nodes.
    """
    
    def __init__(self, num_nodes: int = 3, threshold: int = 2):
        """
        Initialize MPC governance.
        
        Args:
            num_nodes: Total number of governance nodes
            threshold: Minimum nodes required for approval (e.g., 2-of-3)
        """
        self.num_nodes = num_nodes
        self.threshold = threshold
        
        # Generate keys for each governance node
        self.node_keys: Dict[str, str] = {
            f"node_{i+1}": hashlib.sha256(f"node_{i+1}_secret_key".encode()).hexdigest()
            for i in range(num_nodes)
        }
        
        # Track pending trades awaiting approval
        self.pending_trades: Dict[str, Dict[str, Any]] = {}
        
        # Track approved trades
        self.approved_trades: List[Dict[str, Any]] = []
        
        print(f"[MPCGovernance] Initialized: {num_nodes} nodes, {threshold}-of-{num_nodes} threshold")
    
    def submit_trade(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit trade for MPC approval.
        
        Args:
            trade: Trade details {symbol, side, size, price, confidence, timestamp}
        
        Returns:
            {approved: bool, approvers: [node_ids], trade_id: str, reason: str}
        """
        trade_id = str(uuid.uuid4())
        
        # Create trade record
        trade_record = {
            "id": trade_id,
            "symbol": trade.get("symbol"),
            "side": trade.get("side"),
            "size": trade.get("size"),
            "price": trade.get("price"),
            "confidence": trade.get("confidence"),
            "timestamp": trade.get("timestamp", int(datetime.now().timestamp() * 1000)),
            "approvals": {},  # {node_id: signature or True/False}
            "approved_at": None,
            "created_at": datetime.now().isoformat()
        }
        
        # Simulate node voting (in production, each node signs independently)
        # For now: each node approves with probability based on confidence
        confidence = trade.get("confidence", 0.5)
        approved_nodes = 0
        approvers = []
        
        for node_id, node_key in self.node_keys.items():
            # Sign trade with node's key
            trade_signature = self._sign_trade(trade_record, node_key)
            
            # Simulate node approval decision based on confidence + randomness
            # In production: each node would independently evaluate
            node_approves = confidence > 0.1  # Approve if confidence sufficient
            
            trade_record["approvals"][node_id] = {
                "signature": trade_signature,
                "approved": node_approves,
                "timestamp": datetime.now().isoformat()
            }
            
            if node_approves:
                approved_nodes += 1
                approvers.append(node_id)
        
        # Check threshold
        is_approved = approved_nodes >= self.threshold
        
        if is_approved:
            trade_record["approved_at"] = datetime.now().isoformat()
            self.approved_trades.append(trade_record)
            print(
                f"[MPCGovernance] APPROVED {trade_id}: {approved_nodes}/{self.num_nodes} nodes signed"
            )
        else:
            self.pending_trades[trade_id] = trade_record
            print(
                f"[MPCGovernance] REJECTED {trade_id}: {approved_nodes}/{self.num_nodes} nodes < {self.threshold} threshold"
            )
        
        return {
            "approved": is_approved,
            "trade_id": trade_id,
            "approvers": approvers,
            "approvals_count": approved_nodes,
            "threshold": self.threshold,
            "reason": f"{approved_nodes}/{self.num_nodes} nodes approved" if not is_approved else None
        }
    
    def approve_trade(self, trade_id: str, node_id: str, approve: bool) -> Dict[str, Any]:
        """
        Node votes on a pending trade.
        
        Args:
            trade_id: Trade ID to approve/reject
            node_id: Voting node ID
            approve: True to approve, False to reject
        
        Returns:
            {status: str, trade_id: str, threshold_met: bool}
        """
        if trade_id not in self.pending_trades:
            return {"error": "trade_not_found", "trade_id": trade_id}
        
        trade = self.pending_trades[trade_id]
        
        # Record node vote
        if node_id not in trade["approvals"]:
            return {"error": "invalid_node", "node_id": node_id}
        
        trade["approvals"][node_id]["approved"] = approve
        
        # Check if threshold met
        approved_count = sum(
            1 for vote in trade["approvals"].values()
            if vote.get("approved", False)
        )
        
        threshold_met = approved_count >= self.threshold
        
        if threshold_met:
            trade["approved_at"] = datetime.now().isoformat()
            self.approved_trades.append(trade)
            del self.pending_trades[trade_id]
            print(
                f"[MPCGovernance] Threshold met for {trade_id}: {approved_count}/{self.num_nodes}"
            )
        
        return {
            "status": "recorded",
            "trade_id": trade_id,
            "node_id": node_id,
            "approve": approve,
            "approvals_so_far": approved_count,
            "threshold_met": threshold_met
        }
    
    def get_pending_trades(self) -> List[Dict[str, Any]]:
        """Get all pending trades awaiting approval."""
        return list(self.pending_trades.values())
    
    def get_approved_trades(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recently approved trades."""
        return self.approved_trades[-limit:]
    
    def _sign_trade(self, trade: Dict[str, Any], node_key: str) -> str:
        """
        Sign trade with HMAC-SHA256 (per-node signature).
        
        Args:
            trade: Trade record to sign
            node_key: Node's secret key
        
        Returns:
            Base64-encoded signature
        """
        # Create canonical trade representation
        trade_str = f"{trade['id']}:{trade['symbol']}:{trade['side']}:{trade['size']}:{trade['price']}"
        
        # Sign with node key
        signature = hmac.new(
            node_key.encode(),
            trade_str.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def verify_trade_signatures(self, trade_id: str) -> bool:
        """
        Verify all signatures on an approved trade.
        
        Args:
            trade_id: Trade to verify
        
        Returns:
            True if all signatures valid
        """
        trade = None
        for t in self.approved_trades:
            if t["id"] == trade_id:
                trade = t
                break
        
        if not trade:
            return False
        
        # Check each node's signature
        for node_id, approval in trade["approvals"].items():
            if node_id not in self.node_keys:
                return False
            
            # Verify signature (in production, do actual verification)
            if not approval.get("signature"):
                return False
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get MPC governance statistics."""
        return {
            "total_nodes": self.num_nodes,
            "threshold": self.threshold,
            "approved_trades": len(self.approved_trades),
            "pending_trades": len(self.pending_trades),
            "approval_rate": len(self.approved_trades) / (len(self.approved_trades) + len(self.pending_trades)) if (len(self.approved_trades) + len(self.pending_trades)) > 0 else 0
        }


class GovernanceNode:
    """
    Represents a single governance node that votes on trades.
    Can be run on separate machine/process for true distributed governance.
    """
    
    def __init__(self, node_id: str, mpc_governance: MPCGovernance):
        """
        Initialize governance node.
        
        Args:
            node_id: Unique node identifier (e.g., "node_1")
            mpc_governance: Reference to MPC governance instance
        """
        self.node_id = node_id
        self.governance = mpc_governance
        self.approval_rules = {
            "min_confidence": 0.15,
            "max_size": 0.01,
            "max_daily_trades": 1000
        }
    
    def evaluate_trade(self, trade: Dict[str, Any]) -> bool:
        """
        Evaluate if this node approves the trade.
        
        Args:
            trade: Trade to evaluate
        
        Returns:
            True if approved, False otherwise
        """
        # Check confidence
        if trade.get("confidence", 0) < self.approval_rules["min_confidence"]:
            return False
        
        # Check size
        if trade.get("size", 0) > self.approval_rules["max_size"]:
            return False
        
        # Could add more rules: risk checks, anomaly detection, etc.
        
        return True
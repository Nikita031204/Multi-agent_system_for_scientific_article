"""
Semantic Supply Chain Environment for testing multi-agent coordination.

This environment simulates a logistics scenario where agents must:
1. Navigate to warehouses
2. Allocate resources (pickup orders)
3. Negotiate prices at delivery zones
4. Complete deliveries

Used to measure coordination metrics in a controlled setting.
"""

import copy
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class CustomerOrder:
    """Represents a customer order in the supply chain."""
    id: int
    customer_text: str
    warehouse_zone: Tuple[int, int]
    delivery_zone: Tuple[int, int]
    status: str = "pending"  # pending -> allocated -> negotiating -> delivered
    assigned_to: Optional[str] = None
    priority: str = "standard"


@dataclass
class EnvAgentState:
    """State of an agent in the environment."""
    id: str
    pos: Tuple[int, int]
    allocated_order: Optional[int] = None
    has_negotiated: bool = False


@dataclass
class EnvironmentState:
    """Complete state of the environment."""
    step_count: int = 0
    grid_size: int = 3
    agents: Dict[str, EnvAgentState] = field(default_factory=dict)
    orders: Dict[int, CustomerOrder] = field(default_factory=dict)
    total_reward: float = 0.0
    is_done: bool = False


# Sample order texts for demonstration
ORDER_TEXTS = [
    "Urgent: 500 units of electronic components needed by downtown warehouse.",
    "Standard order: 200kg of raw materials to industrial zone.",
    "High-priority: Medical supplies to hospital district.",
]


class SemanticSupplyChainEnv:
    """
    Semantic Supply Chain (SSC) environment for multi-agent coordination.
    
    Actions:
        - Check_Inventory(direction): Move in direction (up/down/left/right)
        - Allocate_Resource(order_id): Pick up order at warehouse (+0.5 reward)
        - Negotiate_Price(order_id): Negotiate at delivery zone (+0.5 reward)
        - Plan_Route(order_id): Complete delivery (+2.0 reward)
    
    Workflow for each order:
        warehouse -> Allocate -> delivery zone -> Negotiate -> Plan_Route
    
    Attributes:
        grid_size: Size of the grid (default 3x3)
        max_steps: Maximum steps before episode ends
        seed: Random seed for reproducibility
    """
    
    def __init__(self, grid_size: int = 3, max_steps: int = 20, seed: int = 42):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.seed = seed
        self.rng = random.Random(seed)
        self.state = EnvironmentState(grid_size=grid_size)
        self._round_actions = 0
    
    def reset(self, agent_ids: List[str]) -> str:
        """
        Reset environment with new agent configuration.
        
        Args:
            agent_ids: List of agent identifiers
        
        Returns:
            Initial observation string
        """
        self.state = EnvironmentState(grid_size=self.grid_size)
        self.state.step_count = 0
        self._round_actions = 0
        self.rng = random.Random(self.seed)
        
        # Place agents
        for i, agent_id in enumerate(agent_ids):
            start_pos = (0, i * (self.grid_size - 1) // max(1, len(agent_ids) - 1))
            self.state.agents[agent_id] = EnvAgentState(id=agent_id, pos=start_pos)
        
        # Create orders
        priorities = ["urgent", "standard", "flexible"]
        used_zones = set()
        
        for i in range(min(3, len(ORDER_TEXTS))):
            # Warehouse zone
            wh = (self.rng.randint(0, self.grid_size-1), 
                  self.rng.randint(0, self.grid_size-1))
            while wh in used_zones:
                wh = (self.rng.randint(0, self.grid_size-1), 
                      self.rng.randint(0, self.grid_size-1))
            used_zones.add(wh)
            
            # Delivery zone
            dl = (self.rng.randint(0, self.grid_size-1), 
                  self.rng.randint(0, self.grid_size-1))
            while dl == wh or dl in used_zones:
                dl = (self.rng.randint(0, self.grid_size-1), 
                      self.rng.randint(0, self.grid_size-1))
            used_zones.add(dl)
            
            self.state.orders[i] = CustomerOrder(
                id=i,
                customer_text=ORDER_TEXTS[i],
                warehouse_zone=wh,
                delivery_zone=dl,
                priority=priorities[i]
            )
        
        return self._get_observation_text()
    
    def step(self, agent_id: str, action: Dict) -> Tuple[str, float, bool, bool]:
        """
        Execute action for an agent.
        
        Args:
            agent_id: Agent performing action
            action: Dict with "type" and action-specific params
        
        Returns:
            Tuple of (observation, reward, is_done, is_productive)
        """
        if self.state.is_done:
            return "Finished.", 0.0, True, False
        
        if self._round_actions == 0:
            self.state.step_count += 1
        self._round_actions += 1
        
        reward, is_productive = 0.0, False
        error_msg = ""
        agent = self.state.agents.get(agent_id)
        if not agent:
            return "Error: agent not found", -1.0, False, False
        
        action_type = action.get("type", "wait")
        
        if action_type == "Check_Inventory":
            d = action.get("direction", "")
            if not d:
                return self._get_observation_text(), -0.1, False, False
            x, y = agent.pos
            if d == "up": y = max(0, y - 1)
            elif d == "down": y = min(self.grid_size - 1, y + 1)
            elif d == "left": x = max(0, x - 1)
            elif d == "right": x = min(self.grid_size - 1, x + 1)
            else:
                return self._get_observation_text(), -0.1, False, False
            if agent.pos != (x, y):
                is_productive = True
            agent.pos = (x, y)
        
        elif action_type == "Allocate_Resource":
            if agent.allocated_order is None:
                order_id = action.get("order_id")
                if order_id is not None and order_id in self.state.orders:
                    order = self.state.orders[order_id]
                    if order.status == "pending" and order.warehouse_zone == agent.pos:
                        order.status = "allocated"
                        order.assigned_to = agent_id
                        agent.allocated_order = order_id
                        agent.has_negotiated = False
                        reward, is_productive = 0.5, True
                    elif order.status == "pending":
                        reward = -0.3
                        error_msg = f"Agent at {agent.pos}, warehouse at {order.warehouse_zone}"
                    else:
                        error_msg = f"Order#{order_id} already {order.status}"
                else:
                    reward = -0.3
                    error_msg = f"Invalid order_id={order_id}"
            else:
                error_msg = f"Agent already has Order#{agent.allocated_order}"
        
        elif action_type == "Negotiate_Price":
            order_id = action.get("order_id")
            if order_id is not None and agent.allocated_order == order_id:
                order = self.state.orders[order_id]
                if order.status == "allocated" and order.delivery_zone == agent.pos:
                    order.status = "negotiating"
                    agent.has_negotiated = True
                    reward, is_productive = 0.5, True
                elif order.status == "allocated":
                    d = action.get("direction", "")
                    if d:
                        x, y = agent.pos
                        if d == "up": y = max(0, y - 1)
                        elif d == "down": y = min(self.grid_size - 1, y + 1)
                        elif d == "left": x = max(0, x - 1)
                        elif d == "right": x = min(self.grid_size - 1, x + 1)
                        if agent.pos != (x, y):
                            is_productive = True
                        agent.pos = (x, y)
        
        elif action_type == "Plan_Route":
            if agent.allocated_order is not None:
                order = self.state.orders[agent.allocated_order]
                if order.delivery_zone == agent.pos and order.status == "negotiating":
                    order.status = "delivered"
                    reward, is_productive = 2.0, True
                elif order.delivery_zone == agent.pos and order.status == "allocated":
                    order.status = "delivered"
                    reward, is_productive = 1.5, True  # Penalty for skipping negotiation
                else:
                    reward = -0.5
                agent.allocated_order = None
                agent.has_negotiated = False
        
        # Check termination
        if self._check_all_delivered():
            self.state.is_done = True
            reward += 1.0
        elif self.state.step_count >= self.max_steps:
            self.state.is_done = True
        
        self.state.total_reward += reward
        obs = self._get_observation_text()
        if error_msg:
            obs = f"FAILED: {error_msg}\n\n{obs}"
        
        return obs, reward, self.state.is_done, is_productive
    
    def end_round(self):
        """Called after all agents have acted in a round."""
        self._round_actions = 0
        self.state.total_reward -= 0.05
    
    def snapshot(self) -> dict:
        """Save environment state."""
        return {
            'state': copy.deepcopy(self.state),
            '_round_actions': self._round_actions,
        }
    
    def restore(self, snap: dict):
        """Restore environment state."""
        self.state = copy.deepcopy(snap['state'])
        self._round_actions = snap['_round_actions']
    
    def _get_observation_text(self) -> str:
        """Generate observation string."""
        agents_block = "\n  ".join(
            [f"{a.id}: pos={a.pos}" + 
             (f" allocated order #{a.allocated_order} (negotiated={a.has_negotiated})" 
              if a.allocated_order is not None else " [free]")
             for a in self.state.agents.values()]
        )
        
        orders_block = []
        for oid, o in self.state.orders.items():
            if o.status == "pending":
                orders_block.append(
                    f"  Order#{oid}: [{o.priority}] warehouse={o.warehouse_zone} "
                    f"-> delivery={o.delivery_zone} [pending]"
                )
            elif o.status in ("allocated", "negotiating"):
                orders_block.append(
                    f"  Order#{oid}: allocated to {o.assigned_to}, "
                    f"delivery={o.delivery_zone} [{o.status}]"
                )
            elif o.status == "delivered":
                orders_block.append(f"  Order#{oid}: [DELIVERED ✓]")
        
        return (f"Step: {self.state.step_count}/{self.max_steps}\n"
                f"Agents:\n  {agents_block}\n"
                f"Orders:\n" + "\n".join(orders_block) + "\n")
    
    def _check_all_delivered(self) -> bool:
        """Check if all orders are delivered."""
        return all(o.status == "delivered" for o in self.state.orders.values())

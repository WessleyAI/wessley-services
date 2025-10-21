"""
A* pathfinding with physics constraints for optimal wire routing in 3D space.
Considers electrical properties, mechanical constraints, and manufacturing feasibility.
"""

import numpy as np
import heapq
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class WireType(Enum):
    POWER = "power"
    SIGNAL = "signal"
    DATA = "data"
    GROUND = "ground"

@dataclass
class WireSpec:
    """Wire specification with electrical and physical properties."""
    id: str
    wire_type: WireType
    gauge: float  # mm²
    max_current: float  # amperes
    voltage_rating: float  # volts
    min_bend_radius: float  # mm
    weight_per_meter: float  # kg/m
    cost_per_meter: float  # currency/m
    shielded: bool = False
    temperature_rating: float = 85.0  # °C

@dataclass
class RouteConstraint:
    """Constraint for wire routing."""
    constraint_type: str  # "avoid_zone", "must_pass", "max_distance", etc.
    zone: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None
    max_distance: Optional[float] = None
    priority: float = 1.0

@dataclass
class RoutePoint:
    """3D point in routing space with associated costs."""
    x: float
    y: float
    z: float
    g_cost: float = 0.0  # Cost from start
    h_cost: float = 0.0  # Heuristic cost to goal
    f_cost: float = 0.0  # Total cost
    parent: Optional['RoutePoint'] = None
    wire_conflicts: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        self.f_cost = self.g_cost + self.h_cost
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost

@dataclass
class WireRoute:
    """Complete wire route with path and properties."""
    wire_id: str
    start_point: Tuple[float, float, float]
    end_point: Tuple[float, float, float]
    path_points: List[Tuple[float, float, float]]
    total_length: float
    total_cost: float
    bend_violations: List[str]
    clearance_violations: List[str]
    electromagnetic_issues: List[str]

class AStar3DRouting:
    """A* pathfinding algorithm adapted for 3D wire routing with physics constraints."""
    
    def __init__(self, 
                 workspace_bounds: Tuple[float, float, float],
                 grid_resolution: float = 5.0,  # mm
                 min_wire_clearance: float = 2.0):  # mm
        self.workspace_bounds = workspace_bounds
        self.grid_resolution = grid_resolution
        self.min_wire_clearance = min_wire_clearance
        
        # Discretize workspace into grid
        self.grid_dims = (
            int(workspace_bounds[0] / grid_resolution) + 1,
            int(workspace_bounds[1] / grid_resolution) + 1,
            int(workspace_bounds[2] / grid_resolution) + 1
        )
        
        # Obstacle and constraint maps
        self.obstacle_map = np.zeros(self.grid_dims, dtype=bool)
        self.heat_map = np.zeros(self.grid_dims, dtype=float)
        self.vibration_map = np.zeros(self.grid_dims, dtype=float)
        self.emi_map = np.zeros(self.grid_dims, dtype=float)
        
        # Existing wire routes
        self.existing_routes: Dict[str, WireRoute] = {}
        
    def add_obstacle(self, min_point: Tuple[float, float, float], 
                    max_point: Tuple[float, float, float]):
        """Add obstacle region to the routing space."""
        min_grid = self._world_to_grid(min_point)
        max_grid = self._world_to_grid(max_point)
        
        for x in range(min_grid[0], max_grid[0] + 1):
            for y in range(min_grid[1], max_grid[1] + 1):
                for z in range(min_grid[2], max_grid[2] + 1):
                    if self._is_valid_grid_pos(x, y, z):
                        self.obstacle_map[x, y, z] = True
    
    def add_heat_source(self, position: Tuple[float, float, float], 
                       intensity: float, radius: float):
        """Add heat source affecting wire routing costs."""
        center_grid = self._world_to_grid(position)
        radius_grid = int(radius / self.grid_resolution)
        
        for x in range(max(0, center_grid[0] - radius_grid), 
                      min(self.grid_dims[0], center_grid[0] + radius_grid + 1)):
            for y in range(max(0, center_grid[1] - radius_grid), 
                          min(self.grid_dims[1], center_grid[1] + radius_grid + 1)):
                for z in range(max(0, center_grid[2] - radius_grid), 
                              min(self.grid_dims[2], center_grid[2] + radius_grid + 1)):
                    distance = np.linalg.norm(np.array([x, y, z]) - np.array(center_grid))
                    if distance <= radius_grid:
                        heat_factor = intensity * (1 - distance / radius_grid)
                        self.heat_map[x, y, z] = max(self.heat_map[x, y, z], heat_factor)
    
    def route_wire(self, wire_spec: WireSpec, 
                   start: Tuple[float, float, float], 
                   end: Tuple[float, float, float],
                   constraints: List[RouteConstraint] = None) -> WireRoute:
        """Route a single wire using A* with physics constraints."""
        logger.info(f"Routing wire {wire_spec.id} from {start} to {end}")
        
        start_grid = self._world_to_grid(start)
        end_grid = self._world_to_grid(end)
        
        # Validate start and end points
        if not self._is_valid_route_point(start_grid, wire_spec):
            raise ValueError(f"Invalid start point for wire {wire_spec.id}")
        if not self._is_valid_route_point(end_grid, wire_spec):
            raise ValueError(f"Invalid end point for wire {wire_spec.id}")
        
        # A* algorithm
        open_set = []
        closed_set = set()
        
        start_point = RoutePoint(
            start_grid[0], start_grid[1], start_grid[2],
            g_cost=0.0,
            h_cost=self._heuristic_cost(start_grid, end_grid, wire_spec)
        )
        
        heapq.heappush(open_set, start_point)
        
        while open_set:
            current = heapq.heappop(open_set)
            current_pos = (current.x, current.y, current.z)
            
            if current_pos in closed_set:
                continue
                
            closed_set.add(current_pos)
            
            # Check if we reached the goal
            if current_pos == end_grid:
                return self._reconstruct_path(current, wire_spec, start, end)
            
            # Explore neighbors
            for neighbor_pos in self._get_neighbors(current_pos):
                if neighbor_pos in closed_set:
                    continue
                    
                if not self._is_valid_route_point(neighbor_pos, wire_spec):
                    continue
                
                # Calculate costs
                movement_cost = self._calculate_movement_cost(
                    current_pos, neighbor_pos, wire_spec)
                g_cost = current.g_cost + movement_cost
                h_cost = self._heuristic_cost(neighbor_pos, end_grid, wire_spec)
                
                neighbor = RoutePoint(
                    neighbor_pos[0], neighbor_pos[1], neighbor_pos[2],
                    g_cost=g_cost, h_cost=h_cost, parent=current
                )
                
                heapq.heappush(open_set, neighbor)
        
        # No path found
        raise RuntimeError(f"No valid route found for wire {wire_spec.id}")
    
    def route_multiple_wires(self, wire_specs: List[WireSpec],
                           start_points: List[Tuple[float, float, float]],
                           end_points: List[Tuple[float, float, float]],
                           constraints: Dict[str, List[RouteConstraint]] = None) -> Dict[str, WireRoute]:
        """Route multiple wires with interference consideration."""
        routes = {}
        constraints = constraints or {}
        
        # Sort wires by priority (power wires first, then by current rating)
        sorted_wires = sorted(wire_specs, key=lambda w: (
            0 if w.wire_type == WireType.POWER else 1,
            -w.max_current
        ))
        
        for i, wire_spec in enumerate(sorted_wires):
            wire_constraints = constraints.get(wire_spec.id, [])
            
            try:
                route = self.route_wire(
                    wire_spec, start_points[i], end_points[i], wire_constraints)
                routes[wire_spec.id] = route
                self.existing_routes[wire_spec.id] = route
                
                # Update grid with new wire for interference calculation
                self._add_wire_to_grid(route, wire_spec)
                
            except Exception as e:
                logger.error(f"Failed to route wire {wire_spec.id}: {e}")
                
        return routes
    
    def optimize_routes(self, routes: Dict[str, WireRoute], 
                       max_iterations: int = 100) -> Dict[str, WireRoute]:
        """Optimize existing routes using iterative improvement."""
        logger.info(f"Optimizing {len(routes)} wire routes")
        
        optimized_routes = routes.copy()
        
        for iteration in range(max_iterations):
            improved = False
            
            for wire_id, route in list(optimized_routes.items()):
                # Try to find better route by removing this wire and re-routing
                self._remove_wire_from_grid(route)
                
                try:
                    # Find corresponding wire spec
                    wire_spec = self._get_wire_spec_by_id(wire_id)
                    if wire_spec:
                        new_route = self.route_wire(
                            wire_spec, route.start_point, route.end_point)
                        
                        if new_route.total_cost < route.total_cost:
                            optimized_routes[wire_id] = new_route
                            self._add_wire_to_grid(new_route, wire_spec)
                            improved = True
                            logger.debug(f"Improved route for {wire_id}: "
                                       f"{route.total_cost:.2f} -> {new_route.total_cost:.2f}")
                        else:
                            self._add_wire_to_grid(route, wire_spec)
                            
                except Exception as e:
                    # Restore original route if optimization fails
                    self._add_wire_to_grid(route, wire_spec)
                    logger.warning(f"Failed to optimize route for {wire_id}: {e}")
            
            if not improved:
                logger.info(f"Optimization converged after {iteration + 1} iterations")
                break
                
        return optimized_routes
    
    def _world_to_grid(self, world_pos: Tuple[float, float, float]) -> Tuple[int, int, int]:
        """Convert world coordinates to grid coordinates."""
        return (
            int(world_pos[0] / self.grid_resolution),
            int(world_pos[1] / self.grid_resolution),
            int(world_pos[2] / self.grid_resolution)
        )
    
    def _grid_to_world(self, grid_pos: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Convert grid coordinates to world coordinates."""
        return (
            grid_pos[0] * self.grid_resolution,
            grid_pos[1] * self.grid_resolution,
            grid_pos[2] * self.grid_resolution
        )
    
    def _is_valid_grid_pos(self, x: int, y: int, z: int) -> bool:
        """Check if grid position is within bounds."""
        return (0 <= x < self.grid_dims[0] and 
                0 <= y < self.grid_dims[1] and 
                0 <= z < self.grid_dims[2])
    
    def _is_valid_route_point(self, grid_pos: Tuple[int, int, int], 
                            wire_spec: WireSpec) -> bool:
        """Check if position is valid for routing this wire."""
        x, y, z = grid_pos
        
        if not self._is_valid_grid_pos(x, y, z):
            return False
            
        if self.obstacle_map[x, y, z]:
            return False
        
        # Check temperature constraints
        if self.heat_map[x, y, z] > wire_spec.temperature_rating:
            return False
        
        # Check clearance from existing wires
        if not self._check_wire_clearance(grid_pos, wire_spec):
            return False
            
        return True
    
    def _check_wire_clearance(self, grid_pos: Tuple[int, int, int], 
                            wire_spec: WireSpec) -> bool:
        """Check if position maintains minimum clearance from existing wires."""
        clearance_grid = max(1, int(self.min_wire_clearance / self.grid_resolution))
        
        x, y, z = grid_pos
        for dx in range(-clearance_grid, clearance_grid + 1):
            for dy in range(-clearance_grid, clearance_grid + 1):
                for dz in range(-clearance_grid, clearance_grid + 1):
                    check_x, check_y, check_z = x + dx, y + dy, z + dz
                    
                    if self._is_valid_grid_pos(check_x, check_y, check_z):
                        # Check if any existing wire occupies this space
                        for route in self.existing_routes.values():
                            for path_point in route.path_points:
                                path_grid = self._world_to_grid(path_point)
                                if path_grid == (check_x, check_y, check_z):
                                    return False
        return True
    
    def _get_neighbors(self, pos: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """Get valid neighboring grid positions (26-connectivity)."""
        neighbors = []
        x, y, z = pos
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                        
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if self._is_valid_grid_pos(nx, ny, nz):
                        neighbors.append((nx, ny, nz))
        
        return neighbors
    
    def _heuristic_cost(self, pos1: Tuple[int, int, int], 
                       pos2: Tuple[int, int, int], wire_spec: WireSpec) -> float:
        """Calculate heuristic cost (3D Euclidean distance with wire properties)."""
        distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
        world_distance = distance * self.grid_resolution
        
        # Base cost is wire cost per meter
        base_cost = world_distance * wire_spec.cost_per_meter
        
        # Add penalty for high-current wires (they need straighter paths)
        current_penalty = wire_spec.max_current * 0.001
        
        return base_cost + current_penalty
    
    def _calculate_movement_cost(self, from_pos: Tuple[int, int, int], 
                               to_pos: Tuple[int, int, int], 
                               wire_spec: WireSpec) -> float:
        """Calculate cost of moving from one position to another."""
        # Base movement cost
        distance = np.linalg.norm(np.array(from_pos) - np.array(to_pos))
        world_distance = distance * self.grid_resolution
        base_cost = world_distance * wire_spec.cost_per_meter
        
        # Environmental penalties
        x, y, z = to_pos
        heat_penalty = self.heat_map[x, y, z] * 0.1
        vibration_penalty = self.vibration_map[x, y, z] * 0.05
        emi_penalty = self.emi_map[x, y, z] * 0.02
        
        # Bend penalty (prefer straight paths)
        bend_penalty = 0.0
        if hasattr(self, '_last_direction'):
            current_direction = np.array(to_pos) - np.array(from_pos)
            if not np.allclose(current_direction, self._last_direction):
                bend_penalty = wire_spec.min_bend_radius * 0.001
        
        # Height penalty (prefer lower routes for easier access)
        height_penalty = z * self.grid_resolution * 0.0001
        
        total_cost = (base_cost + heat_penalty + vibration_penalty + 
                     emi_penalty + bend_penalty + height_penalty)
        
        return total_cost
    
    def _reconstruct_path(self, end_point: RoutePoint, wire_spec: WireSpec,
                         start_world: Tuple[float, float, float],
                         end_world: Tuple[float, float, float]) -> WireRoute:
        """Reconstruct the complete path from A* result."""
        path_points = []
        current = end_point
        
        while current:
            world_pos = self._grid_to_world((current.x, current.y, current.z))
            path_points.append(world_pos)
            current = current.parent
        
        path_points.reverse()
        
        # Calculate total length
        total_length = 0.0
        for i in range(len(path_points) - 1):
            segment_length = np.linalg.norm(
                np.array(path_points[i+1]) - np.array(path_points[i]))
            total_length += segment_length
        
        # Calculate total cost
        total_cost = total_length * wire_spec.cost_per_meter
        
        # Analyze violations
        bend_violations = self._analyze_bend_violations(path_points, wire_spec)
        clearance_violations = self._analyze_clearance_violations(path_points, wire_spec)
        electromagnetic_issues = self._analyze_electromagnetic_issues(path_points, wire_spec)
        
        return WireRoute(
            wire_id=wire_spec.id,
            start_point=start_world,
            end_point=end_world,
            path_points=path_points,
            total_length=total_length,
            total_cost=total_cost,
            bend_violations=bend_violations,
            clearance_violations=clearance_violations,
            electromagnetic_issues=electromagnetic_issues
        )
    
    def _analyze_bend_violations(self, path_points: List[Tuple[float, float, float]], 
                               wire_spec: WireSpec) -> List[str]:
        """Analyze path for bend radius violations."""
        violations = []
        
        for i in range(1, len(path_points) - 1):
            # Calculate bend radius at this point
            v1 = np.array(path_points[i]) - np.array(path_points[i-1])
            v2 = np.array(path_points[i+1]) - np.array(path_points[i])
            
            # Skip if vectors are parallel
            if np.allclose(np.cross(v1, v2), 0):
                continue
            
            # Calculate radius of curvature
            angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
            if angle > 0:
                radius = np.linalg.norm(v1) / (2 * np.sin(angle / 2))
                
                if radius < wire_spec.min_bend_radius:
                    violations.append(f"Bend radius {radius:.1f}mm < minimum {wire_spec.min_bend_radius:.1f}mm at point {i}")
        
        return violations
    
    def _analyze_clearance_violations(self, path_points: List[Tuple[float, float, float]], 
                                    wire_spec: WireSpec) -> List[str]:
        """Analyze path for clearance violations."""
        violations = []
        # Implement clearance analysis logic
        return violations
    
    def _analyze_electromagnetic_issues(self, path_points: List[Tuple[float, float, float]], 
                                      wire_spec: WireSpec) -> List[str]:
        """Analyze path for electromagnetic interference issues."""
        issues = []
        # Implement EMI analysis logic
        return issues
    
    def _add_wire_to_grid(self, route: WireRoute, wire_spec: WireSpec):
        """Add wire route to grid for interference calculations."""
        # Mark grid cells as occupied by this wire
        pass
    
    def _remove_wire_from_grid(self, route: WireRoute):
        """Remove wire route from grid."""
        # Unmark grid cells
        pass
    
    def _get_wire_spec_by_id(self, wire_id: str) -> Optional[WireSpec]:
        """Get wire specification by ID."""
        # This would typically come from a database or configuration
        return None
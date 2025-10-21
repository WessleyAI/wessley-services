"""
gRPC service for real-time 3D optimization serving the 3D Model Service.
Provides high-performance optimization algorithms for electrical system layout.
"""

import grpc
from concurrent import futures
import threading
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
import json
from pathlib import Path

# gRPC generated modules (would be generated from .proto files)
# import optimization_pb2
# import optimization_pb2_grpc

# Import our optimization algorithms
from ..algorithms.spatial.placement_optimizer import GeneticPlacementOptimizer, Component, PlacementSolution
from ..algorithms.spatial.routing_optimizer import AStar3DRouting, WireSpec, WireRoute
from ..algorithms.spatial.layout_scorer import MultiObjectiveLayoutScorer, ComponentInfo, WireInfo
from ..models.reinforcement.placement_agent import PlacementAgent, PlacementEnvironment

logger = logging.getLogger(__name__)

@dataclass
class OptimizationRequest:
    """Request for optimization service."""
    request_id: str
    optimization_type: str  # "placement", "routing", "layout_scoring", "full_optimization"
    components: List[Dict[str, Any]]
    constraints: List[Dict[str, Any]]
    objectives: Dict[str, float]  # Objective weights
    workspace_bounds: Tuple[float, float, float]
    existing_layout: Optional[Dict[str, Any]] = None
    preferences: Optional[Dict[str, Any]] = None

@dataclass
class OptimizationResponse:
    """Response from optimization service."""
    request_id: str
    success: bool
    optimization_result: Optional[Dict[str, Any]]
    metrics: Dict[str, float]
    warnings: List[str]
    error_message: Optional[str] = None
    computation_time_ms: int = 0

class OptimizationService:
    """gRPC service for 3D electrical system optimization."""
    
    def __init__(self):
        # Initialize optimization algorithms
        self.placement_optimizer = None
        self.routing_optimizer = None
        self.layout_scorer = MultiObjectiveLayoutScorer()
        
        # RL agents (loaded models)
        self.placement_agent = None
        self.routing_agent = None
        
        # Service state
        self.active_requests: Dict[str, threading.Thread] = {}
        self.request_cache: Dict[str, OptimizationResponse] = {}
        self.cache_lock = threading.Lock()
        
        # Performance metrics
        self.request_count = 0
        self.total_computation_time = 0.0
        self.error_count = 0
        
        # Load pre-trained models
        self._load_pretrained_models()
        
        logger.info("Optimization service initialized")
    
    def _load_pretrained_models(self):
        """Load pre-trained RL models for optimization."""
        try:
            # Load placement agent
            model_path = Path("models/placement_agent.pth")
            if model_path.exists():
                self.placement_agent = PlacementAgent(state_dim=1000, action_dim=1000)
                self.placement_agent.load_model(str(model_path))
                logger.info("Loaded pre-trained placement agent")
            
            # Load routing agent (placeholder)
            # self.routing_agent = RoutingAgent(...)
            
        except Exception as e:
            logger.warning(f"Failed to load pre-trained models: {e}")
    
    def OptimizeComponentPlacement(self, request, context):
        """Optimize component placement using genetic algorithm and RL."""
        start_time = time.time()
        request_id = request.request_id
        
        try:
            logger.info(f"Starting component placement optimization: {request_id}")
            
            # Parse request
            components = self._parse_components(request.components)
            workspace_bounds = tuple(request.workspace_bounds)
            constraints = request.constraints
            
            # Initialize placement optimizer
            placement_optimizer = GeneticPlacementOptimizer(workspace_bounds)
            
            # Add components
            for component in components:
                placement_optimizer.add_component(component)
            
            # Run optimization
            if self.placement_agent and request.use_rl_agent:
                # Use RL agent for optimization
                placement_result = self._optimize_placement_with_rl(
                    components, workspace_bounds, constraints)
            else:
                # Use genetic algorithm
                placement_result = placement_optimizer.optimize()
            
            # Calculate metrics
            metrics = self._calculate_placement_metrics(placement_result, components)
            
            # Prepare response
            optimization_result = {
                "placement_solution": {
                    "positions": placement_result.positions,
                    "orientations": placement_result.orientations,
                    "fitness": placement_result.fitness,
                    "violations": placement_result.violations
                },
                "algorithm_used": "rl_agent" if self.placement_agent and request.use_rl_agent else "genetic_algorithm"
            }
            
            computation_time = int((time.time() - start_time) * 1000)
            
            response = OptimizationResponse(
                request_id=request_id,
                success=True,
                optimization_result=optimization_result,
                metrics=metrics,
                warnings=[],
                computation_time_ms=computation_time
            )
            
            self._update_service_metrics(computation_time, success=True)
            logger.info(f"Component placement optimization completed: {request_id} ({computation_time}ms)")
            
            return self._convert_to_grpc_response(response)
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"Component placement optimization failed: {request_id} - {error_message}")
            
            computation_time = int((time.time() - start_time) * 1000)
            
            response = OptimizationResponse(
                request_id=request_id,
                success=False,
                optimization_result=None,
                metrics={},
                warnings=[],
                error_message=error_message,
                computation_time_ms=computation_time
            )
            
            self._update_service_metrics(computation_time, success=False)
            return self._convert_to_grpc_response(response)
    
    def OptimizeWireRouting(self, request, context):
        """Optimize wire routing using A* algorithm."""
        start_time = time.time()
        request_id = request.request_id
        
        try:
            logger.info(f"Starting wire routing optimization: {request_id}")
            
            # Parse request
            wire_specs = self._parse_wire_specs(request.wire_specifications)
            start_points = [tuple(p) for p in request.start_points]
            end_points = [tuple(p) for p in request.end_points]
            workspace_bounds = tuple(request.workspace_bounds)
            
            # Initialize routing optimizer
            routing_optimizer = AStar3DRouting(workspace_bounds)
            
            # Add obstacles from existing components
            if hasattr(request, 'existing_components'):
                for component in request.existing_components:
                    min_point = tuple(component.position)
                    max_point = tuple(np.array(component.position) + np.array(component.dimensions))
                    routing_optimizer.add_obstacle(min_point, max_point)
            
            # Route multiple wires
            wire_routes = routing_optimizer.route_multiple_wires(
                wire_specs, start_points, end_points)
            
            # Optimize routes
            optimized_routes = routing_optimizer.optimize_routes(wire_routes)
            
            # Calculate metrics
            metrics = self._calculate_routing_metrics(optimized_routes)
            
            # Prepare response
            optimization_result = {
                "wire_routes": {
                    wire_id: {
                        "path_points": route.path_points,
                        "total_length": route.total_length,
                        "total_cost": route.total_cost,
                        "violations": route.bend_violations + route.clearance_violations
                    }
                    for wire_id, route in optimized_routes.items()
                },
                "routing_summary": {
                    "total_wire_length": sum(route.total_length for route in optimized_routes.values()),
                    "total_cost": sum(route.total_cost for route in optimized_routes.values()),
                    "violation_count": sum(len(route.bend_violations) + len(route.clearance_violations) 
                                         for route in optimized_routes.values())
                }
            }
            
            computation_time = int((time.time() - start_time) * 1000)
            
            response = OptimizationResponse(
                request_id=request_id,
                success=True,
                optimization_result=optimization_result,
                metrics=metrics,
                warnings=[],
                computation_time_ms=computation_time
            )
            
            self._update_service_metrics(computation_time, success=True)
            logger.info(f"Wire routing optimization completed: {request_id} ({computation_time}ms)")
            
            return self._convert_to_grpc_response(response)
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"Wire routing optimization failed: {request_id} - {error_message}")
            
            computation_time = int((time.time() - start_time) * 1000)
            
            response = OptimizationResponse(
                request_id=request_id,
                success=False,
                optimization_result=None,
                metrics={},
                warnings=[],
                error_message=error_message,
                computation_time_ms=computation_time
            )
            
            self._update_service_metrics(computation_time, success=False)
            return self._convert_to_grpc_response(response)
    
    def ScoreLayout(self, request, context):
        """Score a complete electrical system layout."""
        start_time = time.time()
        request_id = request.request_id
        
        try:
            logger.info(f"Starting layout scoring: {request_id}")
            
            # Parse request
            components = self._parse_component_info_for_scoring(request.components)
            wires = self._parse_wire_info_for_scoring(request.wires)
            
            # Score layout
            layout_score = self.layout_scorer.score_layout(components, wires)
            
            # Prepare detailed scoring breakdown
            scoring_breakdown = {
                "total_score": layout_score.total_score,
                "category_scores": {
                    category.value: score 
                    for category, score in layout_score.category_scores.items()
                },
                "detailed_metrics": layout_score.detailed_metrics,
                "violations": layout_score.violations,
                "recommendations": layout_score.recommendations,
                "confidence": layout_score.confidence
            }
            
            # Calculate additional metrics
            metrics = {
                "total_score": layout_score.total_score,
                "safety_score": layout_score.category_scores.get("safety", 0),
                "efficiency_score": layout_score.category_scores.get("efficiency", 0),
                "cost_score": layout_score.category_scores.get("cost", 0),
                "violation_count": len(layout_score.violations),
                "confidence": layout_score.confidence
            }
            
            computation_time = int((time.time() - start_time) * 1000)
            
            response = OptimizationResponse(
                request_id=request_id,
                success=True,
                optimization_result=scoring_breakdown,
                metrics=metrics,
                warnings=layout_score.violations,
                computation_time_ms=computation_time
            )
            
            self._update_service_metrics(computation_time, success=True)
            logger.info(f"Layout scoring completed: {request_id} ({computation_time}ms)")
            
            return self._convert_to_grpc_response(response)
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"Layout scoring failed: {request_id} - {error_message}")
            
            computation_time = int((time.time() - start_time) * 1000)
            
            response = OptimizationResponse(
                request_id=request_id,
                success=False,
                optimization_result=None,
                metrics={},
                warnings=[],
                error_message=error_message,
                computation_time_ms=computation_time
            )
            
            self._update_service_metrics(computation_time, success=False)
            return self._convert_to_grpc_response(response)
    
    def OptimizeFullLayout(self, request, context):
        """Perform full layout optimization (placement + routing + scoring)."""
        start_time = time.time()
        request_id = request.request_id
        
        try:
            logger.info(f"Starting full layout optimization: {request_id}")
            
            # Step 1: Component Placement
            placement_request = self._create_placement_request(request)
            placement_response = self.OptimizeComponentPlacement(placement_request, context)
            
            if not placement_response.success:
                return placement_response
            
            # Step 2: Wire Routing
            routing_request = self._create_routing_request(request, placement_response)
            routing_response = self.OptimizeWireRouting(routing_request, context)
            
            if not routing_response.success:
                return routing_response
            
            # Step 3: Layout Scoring
            scoring_request = self._create_scoring_request(request, placement_response, routing_response)
            scoring_response = self.ScoreLayout(scoring_request, context)
            
            # Combine results
            optimization_result = {
                "placement": placement_response.optimization_result,
                "routing": routing_response.optimization_result,
                "scoring": scoring_response.optimization_result,
                "optimization_summary": {
                    "total_computation_time_ms": (
                        placement_response.computation_time_ms +
                        routing_response.computation_time_ms +
                        scoring_response.computation_time_ms
                    ),
                    "final_score": scoring_response.metrics.get("total_score", 0),
                    "component_count": len(request.components),
                    "wire_count": len(getattr(request, 'wire_specifications', [])),
                    "violations": (placement_response.warnings + 
                                 routing_response.warnings + 
                                 scoring_response.warnings)
                }
            }
            
            # Combined metrics
            combined_metrics = {
                **placement_response.metrics,
                **routing_response.metrics,
                **scoring_response.metrics
            }
            
            computation_time = int((time.time() - start_time) * 1000)
            
            response = OptimizationResponse(
                request_id=request_id,
                success=True,
                optimization_result=optimization_result,
                metrics=combined_metrics,
                warnings=placement_response.warnings + routing_response.warnings + scoring_response.warnings,
                computation_time_ms=computation_time
            )
            
            self._update_service_metrics(computation_time, success=True)
            logger.info(f"Full layout optimization completed: {request_id} ({computation_time}ms)")
            
            return self._convert_to_grpc_response(response)
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"Full layout optimization failed: {request_id} - {error_message}")
            
            computation_time = int((time.time() - start_time) * 1000)
            
            response = OptimizationResponse(
                request_id=request_id,
                success=False,
                optimization_result=None,
                metrics={},
                warnings=[],
                error_message=error_message,
                computation_time_ms=computation_time
            )
            
            self._update_service_metrics(computation_time, success=False)
            return self._convert_to_grpc_response(response)
    
    def GetServiceHealth(self, request, context):
        """Get service health and performance metrics."""
        avg_computation_time = (self.total_computation_time / self.request_count 
                              if self.request_count > 0 else 0)
        
        health_info = {
            "status": "healthy",
            "request_count": self.request_count,
            "error_count": self.error_count,
            "success_rate": (self.request_count - self.error_count) / self.request_count if self.request_count > 0 else 1.0,
            "average_computation_time_ms": avg_computation_time,
            "active_requests": len(self.active_requests),
            "cache_size": len(self.request_cache),
            "models_loaded": {
                "placement_agent": self.placement_agent is not None,
                "routing_agent": self.routing_agent is not None
            }
        }
        
        # Return gRPC health response (would be defined in .proto)
        # return optimization_pb2.HealthResponse(**health_info)
        return health_info
    
    def _optimize_placement_with_rl(self, components: List[Component], 
                                  workspace_bounds: Tuple[float, float, float],
                                  constraints: List[Dict]) -> PlacementSolution:
        """Use RL agent for component placement optimization."""
        # Initialize environment
        environment = PlacementEnvironment(workspace_bounds)
        
        # Add components to environment
        component_dicts = []
        for comp in components:
            comp_dict = {
                'id': comp.id,
                'type': comp.type,
                'dimensions': comp.dimensions,
                'weight': comp.weight
            }
            component_dicts.append(comp_dict)
        
        environment.add_components(component_dicts)
        
        # Use trained agent to generate placement
        state = environment.reset()
        positions = {}
        orientations = {}
        
        while state.current_component is not None:
            action = self.placement_agent.select_action(state, training=False)
            next_state, reward, done, info = environment.step(action)
            
            if info['placement_successful']:
                positions[action.component_id] = action.position
                orientations[action.component_id] = action.orientation
            
            if done:
                break
            
            state = next_state
        
        # Calculate final fitness using layout scorer
        # (This would integrate with the layout scorer)
        fitness = 0.85  # Placeholder
        
        return PlacementSolution(
            positions=positions,
            orientations=orientations,
            fitness=fitness,
            violations=[]
        )
    
    def _parse_components(self, component_data: List[Dict]) -> List[Component]:
        """Parse component data from gRPC request."""
        components = []
        
        for comp_data in component_data:
            component = Component(
                id=comp_data['id'],
                type=comp_data['type'],
                dimensions=tuple(comp_data['dimensions']),
                weight=comp_data.get('weight', 1.0),
                heat_generation=comp_data.get('heat_generation', 0.0),
                accessibility_requirement=comp_data.get('accessibility_requirement', 0.5),
                vibration_sensitivity=comp_data.get('vibration_sensitivity', 0.5),
                electromagnetic_interference=comp_data.get('electromagnetic_interference', 0.1),
                connections=comp_data.get('connections', [])
            )
            components.append(component)
        
        return components
    
    def _parse_wire_specs(self, wire_data: List[Dict]) -> List[WireSpec]:
        """Parse wire specifications from gRPC request."""
        from ..algorithms.spatial.routing_optimizer import WireSpec, WireType
        
        wire_specs = []
        
        for wire_data_item in wire_data:
            wire_spec = WireSpec(
                id=wire_data_item['id'],
                wire_type=WireType(wire_data_item.get('type', 'signal')),
                gauge=wire_data_item.get('gauge', 1.0),
                max_current=wire_data_item.get('max_current', 1.0),
                voltage_rating=wire_data_item.get('voltage_rating', 12.0),
                min_bend_radius=wire_data_item.get('min_bend_radius', 5.0),
                weight_per_meter=wire_data_item.get('weight_per_meter', 0.01),
                cost_per_meter=wire_data_item.get('cost_per_meter', 0.1),
                shielded=wire_data_item.get('shielded', False),
                temperature_rating=wire_data_item.get('temperature_rating', 85.0)
            )
            wire_specs.append(wire_spec)
        
        return wire_specs
    
    def _parse_component_info_for_scoring(self, component_data: List[Dict]) -> List[ComponentInfo]:
        """Parse component info for layout scoring."""
        components = []
        
        for comp_data in component_data:
            component = ComponentInfo(
                id=comp_data['id'],
                type=comp_data['type'],
                position=tuple(comp_data['position']),
                dimensions=tuple(comp_data['dimensions']),
                electrical_properties=comp_data.get('electrical_properties', {}),
                mechanical_properties=comp_data.get('mechanical_properties', {}),
                cost=comp_data.get('cost', 100.0),
                reliability_rating=comp_data.get('reliability_rating', 0.95)
            )
            components.append(component)
        
        return components
    
    def _parse_wire_info_for_scoring(self, wire_data: List[Dict]) -> List[WireInfo]:
        """Parse wire info for layout scoring."""
        wires = []
        
        for wire_data_item in wire_data:
            wire = WireInfo(
                id=wire_data_item['id'],
                path=[tuple(p) for p in wire_data_item['path']],
                gauge=wire_data_item.get('gauge', 1.0),
                current_rating=wire_data_item.get('current_rating', 1.0),
                length=wire_data_item.get('length', 100.0),
                cost=wire_data_item.get('cost', 10.0),
                bend_count=wire_data_item.get('bend_count', 0),
                clearances=wire_data_item.get('clearances', [])
            )
            wires.append(wire)
        
        return wires
    
    def _calculate_placement_metrics(self, placement_result: PlacementSolution, 
                                   components: List[Component]) -> Dict[str, float]:
        """Calculate metrics for placement optimization."""
        return {
            "placement_fitness": placement_result.fitness,
            "component_count": len(components),
            "violation_count": len(placement_result.violations),
            "placement_success_rate": 1.0 if placement_result.violations == [] else 0.8,
            "space_utilization": min(len(placement_result.positions) / len(components), 1.0)
        }
    
    def _calculate_routing_metrics(self, wire_routes: Dict[str, WireRoute]) -> Dict[str, float]:
        """Calculate metrics for routing optimization."""
        total_length = sum(route.total_length for route in wire_routes.values())
        total_cost = sum(route.total_cost for route in wire_routes.values())
        total_violations = sum(
            len(route.bend_violations) + len(route.clearance_violations)
            for route in wire_routes.values()
        )
        
        return {
            "total_wire_length": total_length,
            "total_wire_cost": total_cost,
            "average_wire_length": total_length / len(wire_routes) if wire_routes else 0,
            "routing_violation_count": total_violations,
            "routing_success_rate": 1.0 if total_violations == 0 else 0.8
        }
    
    def _update_service_metrics(self, computation_time: int, success: bool):
        """Update service performance metrics."""
        self.request_count += 1
        self.total_computation_time += computation_time
        
        if not success:
            self.error_count += 1
    
    def _convert_to_grpc_response(self, response: OptimizationResponse):
        """Convert internal response to gRPC response format."""
        # This would convert to the actual gRPC response type
        # defined in the .proto file
        return response
    
    def _create_placement_request(self, full_request):
        """Create placement request from full optimization request."""
        # Extract placement-specific data from full request
        return full_request
    
    def _create_routing_request(self, full_request, placement_response):
        """Create routing request using placement results."""
        # Use placement results to set component positions for routing
        return full_request
    
    def _create_scoring_request(self, full_request, placement_response, routing_response):
        """Create scoring request using placement and routing results."""
        # Combine placement and routing results for scoring
        return full_request


def serve(port: int = 50051, max_workers: int = 10):
    """Start the gRPC optimization service."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    
    # Add service to server
    optimization_service = OptimizationService()
    # optimization_pb2_grpc.add_OptimizationServiceServicer_to_server(optimization_service, server)
    
    # Configure server
    listen_addr = f'[::]:{port}'
    server.add_insecure_port(listen_addr)
    
    # Start server
    server.start()
    logger.info(f"Optimization service started on port {port}")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down optimization service")
        server.stop(0)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    serve()
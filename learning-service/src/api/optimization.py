"""
FastAPI endpoints for optimization services.
Provides RESTful API access to ML optimization algorithms.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
import asyncio
import uuid
import logging
import time
from datetime import datetime
import numpy as np
import json
import io
import base64
from pathlib import Path

# Import optimization services
from ..services.optimization_service import OptimizationService, OptimizationRequest, OptimizationResponse
from ..algorithms.spatial.placement_optimizer import Component, PlacementSolution
from ..algorithms.spatial.routing_optimizer import WireSpec, WireRoute
from ..algorithms.feedback.user_behavior import UserBehaviorAnalyzer, UserInteraction, UserFeedback

logger = logging.getLogger(__name__)

# Pydantic models for API
class ComponentRequest(BaseModel):
    id: str
    type: str
    dimensions: List[float] = Field(..., min_items=3, max_items=3)
    weight: float = Field(default=1.0, ge=0)
    heat_generation: float = Field(default=0.0, ge=0)
    accessibility_requirement: float = Field(default=0.5, ge=0, le=1)
    vibration_sensitivity: float = Field(default=0.5, ge=0, le=1)
    electromagnetic_interference: float = Field(default=0.1, ge=0, le=1)
    connections: List[str] = Field(default_factory=list)
    electrical_properties: Dict[str, Any] = Field(default_factory=dict)
    mechanical_properties: Dict[str, Any] = Field(default_factory=dict)
    cost: float = Field(default=100.0, ge=0)

class WireSpecRequest(BaseModel):
    id: str
    type: str = Field(default="signal")
    gauge: float = Field(default=1.0, gt=0)
    max_current: float = Field(default=1.0, gt=0)
    voltage_rating: float = Field(default=12.0, gt=0)
    min_bend_radius: float = Field(default=5.0, gt=0)
    weight_per_meter: float = Field(default=0.01, gt=0)
    cost_per_meter: float = Field(default=0.1, gt=0)
    shielded: bool = False
    temperature_rating: float = Field(default=85.0, gt=0)

class ConstraintRequest(BaseModel):
    type: str
    parameters: Dict[str, Any]
    priority: float = Field(default=1.0, ge=0, le=1)

class PlacementOptimizationRequest(BaseModel):
    components: List[ComponentRequest]
    workspace_bounds: List[float] = Field(..., min_items=3, max_items=3)
    constraints: List[ConstraintRequest] = Field(default_factory=list)
    objectives: Dict[str, float] = Field(default_factory=dict)
    use_rl_agent: bool = True
    optimization_time_limit_seconds: int = Field(default=300, gt=0)

class RoutingOptimizationRequest(BaseModel):
    wire_specifications: List[WireSpecRequest]
    start_points: List[List[float]]
    end_points: List[List[float]]
    workspace_bounds: List[float] = Field(..., min_items=3, max_items=3)
    existing_components: List[ComponentRequest] = Field(default_factory=list)
    constraints: List[ConstraintRequest] = Field(default_factory=list)

class LayoutScoringRequest(BaseModel):
    components: List[ComponentRequest]
    wires: List[Dict[str, Any]]  # Wire layout information
    scoring_weights: Dict[str, float] = Field(default_factory=dict)

class FullOptimizationRequest(BaseModel):
    components: List[ComponentRequest]
    wire_specifications: List[WireSpecRequest]
    workspace_bounds: List[float] = Field(..., min_items=3, max_items=3)
    constraints: List[ConstraintRequest] = Field(default_factory=list)
    objectives: Dict[str, float] = Field(default_factory=dict)
    preferences: Dict[str, Any] = Field(default_factory=dict)
    use_rl_agents: bool = True

class UserInteractionRequest(BaseModel):
    session_id: str
    user_id: str
    interaction_type: str
    target_object_id: Optional[str] = None
    target_object_type: Optional[str] = None
    interaction_data: Dict[str, Any] = Field(default_factory=dict)
    duration_ms: int = Field(gt=0)
    success: bool = True

class UserFeedbackRequest(BaseModel):
    session_id: str
    user_id: str
    layout_id: str
    feedback_type: str
    rating: int = Field(ge=1, le=5)
    specific_issues: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)

# Response models
class OptimizationStatusResponse(BaseModel):
    request_id: str
    status: str  # "pending", "running", "completed", "failed"
    progress: float = Field(ge=0, le=1)
    estimated_completion_time: Optional[datetime] = None
    result_available: bool = False

class PlacementOptimizationResponse(BaseModel):
    request_id: str
    success: bool
    positions: Dict[str, List[float]]
    orientations: Dict[str, List[float]]
    fitness_score: float
    violations: List[str]
    metrics: Dict[str, float]
    computation_time_ms: int
    algorithm_used: str

class RoutingOptimizationResponse(BaseModel):
    request_id: str
    success: bool
    wire_routes: Dict[str, Dict[str, Any]]
    total_wire_length: float
    total_cost: float
    violation_count: int
    metrics: Dict[str, float]
    computation_time_ms: int

class LayoutScoreResponse(BaseModel):
    request_id: str
    success: bool
    total_score: float
    category_scores: Dict[str, float]
    detailed_metrics: Dict[str, float]
    violations: List[str]
    recommendations: List[str]
    confidence: float

# FastAPI app
app = FastAPI(
    title="Learning Service API",
    description="ML-powered optimization and learning services for electrical systems",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instances
optimization_service = OptimizationService()
user_behavior_analyzer = UserBehaviorAnalyzer()

# In-memory storage for async operations
pending_requests: Dict[str, Dict[str, Any]] = {}
completed_requests: Dict[str, OptimizationResponse] = {}

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting Learning Service API")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Learning Service API")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Check service health."""
    health_info = optimization_service.GetServiceHealth(None, None)
    return JSONResponse(content=health_info)

# Component Placement Optimization
@app.post("/api/v1/optimization/placement", response_model=OptimizationStatusResponse)
async def optimize_component_placement(
    request: PlacementOptimizationRequest,
    background_tasks: BackgroundTasks
):
    """Start component placement optimization."""
    request_id = str(uuid.uuid4())
    
    # Store request
    pending_requests[request_id] = {
        "type": "placement",
        "request": request,
        "status": "pending",
        "progress": 0.0,
        "start_time": datetime.now()
    }
    
    # Start optimization in background
    background_tasks.add_task(
        _run_placement_optimization, request_id, request
    )
    
    return OptimizationStatusResponse(
        request_id=request_id,
        status="pending",
        progress=0.0,
        result_available=False
    )

@app.get("/api/v1/optimization/placement/{request_id}/status", response_model=OptimizationStatusResponse)
async def get_placement_optimization_status(request_id: str):
    """Get placement optimization status."""
    if request_id in completed_requests:
        return OptimizationStatusResponse(
            request_id=request_id,
            status="completed",
            progress=1.0,
            result_available=True
        )
    elif request_id in pending_requests:
        req_info = pending_requests[request_id]
        return OptimizationStatusResponse(
            request_id=request_id,
            status=req_info["status"],
            progress=req_info["progress"],
            result_available=False
        )
    else:
        raise HTTPException(status_code=404, detail="Request not found")

@app.get("/api/v1/optimization/placement/{request_id}/result", response_model=PlacementOptimizationResponse)
async def get_placement_optimization_result(request_id: str):
    """Get placement optimization result."""
    if request_id not in completed_requests:
        raise HTTPException(status_code=404, detail="Result not available")
    
    result = completed_requests[request_id]
    
    if not result.success:
        raise HTTPException(status_code=500, detail=result.error_message)
    
    optimization_result = result.optimization_result["placement_solution"]
    
    return PlacementOptimizationResponse(
        request_id=request_id,
        success=True,
        positions=optimization_result["positions"],
        orientations=optimization_result["orientations"],
        fitness_score=optimization_result["fitness"],
        violations=optimization_result["violations"],
        metrics=result.metrics,
        computation_time_ms=result.computation_time_ms,
        algorithm_used=result.optimization_result["algorithm_used"]
    )

# Wire Routing Optimization
@app.post("/api/v1/optimization/routing", response_model=OptimizationStatusResponse)
async def optimize_wire_routing(
    request: RoutingOptimizationRequest,
    background_tasks: BackgroundTasks
):
    """Start wire routing optimization."""
    request_id = str(uuid.uuid4())
    
    # Validate start/end points match wire specs
    if len(request.start_points) != len(request.wire_specifications) or \
       len(request.end_points) != len(request.wire_specifications):
        raise HTTPException(
            status_code=400, 
            detail="Number of start/end points must match wire specifications"
        )
    
    pending_requests[request_id] = {
        "type": "routing",
        "request": request,
        "status": "pending",
        "progress": 0.0,
        "start_time": datetime.now()
    }
    
    background_tasks.add_task(
        _run_routing_optimization, request_id, request
    )
    
    return OptimizationStatusResponse(
        request_id=request_id,
        status="pending",
        progress=0.0,
        result_available=False
    )

@app.get("/api/v1/optimization/routing/{request_id}/result", response_model=RoutingOptimizationResponse)
async def get_routing_optimization_result(request_id: str):
    """Get routing optimization result."""
    if request_id not in completed_requests:
        raise HTTPException(status_code=404, detail="Result not available")
    
    result = completed_requests[request_id]
    
    if not result.success:
        raise HTTPException(status_code=500, detail=result.error_message)
    
    routing_result = result.optimization_result["routing_summary"]
    
    return RoutingOptimizationResponse(
        request_id=request_id,
        success=True,
        wire_routes=result.optimization_result["wire_routes"],
        total_wire_length=routing_result["total_wire_length"],
        total_cost=routing_result["total_cost"],
        violation_count=routing_result["violation_count"],
        metrics=result.metrics,
        computation_time_ms=result.computation_time_ms
    )

# Layout Scoring
@app.post("/api/v1/optimization/score", response_model=LayoutScoreResponse)
async def score_layout(request: LayoutScoringRequest):
    """Score a complete electrical system layout."""
    request_id = str(uuid.uuid4())
    
    try:
        # Create optimization request
        opt_request = OptimizationRequest(
            request_id=request_id,
            optimization_type="layout_scoring",
            components=[component.dict() for component in request.components],
            constraints=[],
            objectives=request.scoring_weights,
            workspace_bounds=(500, 500, 300),  # Default workspace
            existing_layout={"wires": request.wires}
        )
        
        # Run scoring
        result = optimization_service.ScoreLayout(opt_request, None)
        
        if not result.success:
            raise HTTPException(status_code=500, detail=result.error_message)
        
        scoring_result = result.optimization_result
        
        return LayoutScoreResponse(
            request_id=request_id,
            success=True,
            total_score=scoring_result["total_score"],
            category_scores=scoring_result["category_scores"],
            detailed_metrics=scoring_result["detailed_metrics"],
            violations=scoring_result["violations"],
            recommendations=scoring_result["recommendations"],
            confidence=scoring_result["confidence"]
        )
        
    except Exception as e:
        logger.error(f"Layout scoring failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Full Layout Optimization
@app.post("/api/v1/optimization/full", response_model=OptimizationStatusResponse)
async def optimize_full_layout(
    request: FullOptimizationRequest,
    background_tasks: BackgroundTasks
):
    """Start full layout optimization (placement + routing + scoring)."""
    request_id = str(uuid.uuid4())
    
    pending_requests[request_id] = {
        "type": "full",
        "request": request,
        "status": "pending",
        "progress": 0.0,
        "start_time": datetime.now()
    }
    
    background_tasks.add_task(
        _run_full_optimization, request_id, request
    )
    
    return OptimizationStatusResponse(
        request_id=request_id,
        status="pending",
        progress=0.0,
        result_available=False
    )

@app.get("/api/v1/optimization/full/{request_id}/result")
async def get_full_optimization_result(request_id: str):
    """Get full optimization result."""
    if request_id not in completed_requests:
        raise HTTPException(status_code=404, detail="Result not available")
    
    result = completed_requests[request_id]
    
    if not result.success:
        raise HTTPException(status_code=500, detail=result.error_message)
    
    return {
        "request_id": request_id,
        "success": True,
        "optimization_result": result.optimization_result,
        "metrics": result.metrics,
        "warnings": result.warnings,
        "computation_time_ms": result.computation_time_ms
    }

# User Behavior Analytics
@app.post("/api/v1/analytics/interaction")
async def record_user_interaction(request: UserInteractionRequest):
    """Record user interaction for analytics."""
    try:
        from ..algorithms.feedback.user_behavior import UserInteraction, InteractionType
        
        interaction = UserInteraction(
            session_id=request.session_id,
            user_id=request.user_id,
            timestamp=datetime.now(),
            interaction_type=InteractionType(request.interaction_type),
            target_object_id=request.target_object_id,
            target_object_type=request.target_object_type,
            before_state={},
            after_state={},
            interaction_data=request.interaction_data,
            duration_ms=request.duration_ms,
            success=request.success
        )
        
        user_behavior_analyzer.record_interaction(interaction)
        
        return {"status": "recorded", "interaction_id": str(uuid.uuid4())}
        
    except Exception as e:
        logger.error(f"Failed to record interaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/analytics/feedback")
async def record_user_feedback(request: UserFeedbackRequest):
    """Record user feedback for analytics."""
    try:
        from ..algorithms.feedback.user_behavior import UserFeedback, FeedbackType
        
        feedback = UserFeedback(
            session_id=request.session_id,
            user_id=request.user_id,
            timestamp=datetime.now(),
            layout_id=request.layout_id,
            feedback_type=FeedbackType(request.feedback_type),
            rating=request.rating,
            specific_issues=request.specific_issues,
            suggestions=request.suggestions,
            preferred_alternatives=[],
            context=request.context
        )
        
        user_behavior_analyzer.record_feedback(feedback)
        
        return {"status": "recorded", "feedback_id": str(uuid.uuid4())}
        
    except Exception as e:
        logger.error(f"Failed to record feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/analytics/user/{user_id}/patterns")
async def get_user_patterns(user_id: str):
    """Get user behavior patterns."""
    try:
        patterns = user_behavior_analyzer.identify_user_patterns(user_id)
        return patterns
    except Exception as e:
        logger.error(f"Failed to get user patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/analytics/user/{user_id}/recommendations")
async def get_user_recommendations(user_id: str):
    """Get personalized recommendations for user."""
    try:
        recommendations = user_behavior_analyzer.generate_personalized_recommendations(user_id)
        return recommendations
    except Exception as e:
        logger.error(f"Failed to get recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/analytics/usability-issues")
async def get_usability_issues():
    """Get system-wide usability issues."""
    try:
        issues = user_behavior_analyzer.detect_usability_issues()
        return {"issues": issues, "issue_count": len(issues)}
    except Exception as e:
        logger.error(f"Failed to get usability issues: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Model Training and Management
@app.post("/api/v1/training/start")
async def start_model_training(
    model_type: str,
    dataset_path: str,
    hyperparameters: Dict[str, Any] = None,
    background_tasks: BackgroundTasks = None
):
    """Start model training."""
    if model_type not in ["component_detector", "placement_agent", "wire_tracer"]:
        raise HTTPException(status_code=400, detail="Invalid model type")
    
    training_id = str(uuid.uuid4())
    
    # Store training request
    pending_requests[training_id] = {
        "type": "training",
        "model_type": model_type,
        "dataset_path": dataset_path,
        "hyperparameters": hyperparameters or {},
        "status": "pending",
        "progress": 0.0,
        "start_time": datetime.now()
    }
    
    if background_tasks:
        background_tasks.add_task(
            _run_model_training, training_id, model_type, dataset_path, hyperparameters
        )
    
    return {"training_id": training_id, "status": "started"}

@app.get("/api/v1/training/{training_id}/status")
async def get_training_status(training_id: str):
    """Get model training status."""
    if training_id in pending_requests:
        req_info = pending_requests[training_id]
        return {
            "training_id": training_id,
            "status": req_info["status"],
            "progress": req_info["progress"],
            "model_type": req_info.get("model_type"),
            "start_time": req_info["start_time"]
        }
    else:
        raise HTTPException(status_code=404, detail="Training request not found")

# Utility endpoints
@app.get("/api/v1/metrics/service")
async def get_service_metrics():
    """Get service performance metrics."""
    return {
        "optimization_service": optimization_service.GetServiceHealth(None, None),
        "pending_requests": len(pending_requests),
        "completed_requests": len(completed_requests),
        "user_profiles": len(user_behavior_analyzer.user_profiles)
    }

@app.delete("/api/v1/cache/clear")
async def clear_cache():
    """Clear request cache."""
    global pending_requests, completed_requests
    pending_requests.clear()
    completed_requests.clear()
    return {"status": "cache cleared"}

# Background task functions
async def _run_placement_optimization(request_id: str, request: PlacementOptimizationRequest):
    """Run placement optimization in background."""
    try:
        pending_requests[request_id]["status"] = "running"
        pending_requests[request_id]["progress"] = 0.1
        
        # Create optimization request
        opt_request = OptimizationRequest(
            request_id=request_id,
            optimization_type="placement",
            components=[component.dict() for component in request.components],
            constraints=[constraint.dict() for constraint in request.constraints],
            objectives=request.objectives,
            workspace_bounds=tuple(request.workspace_bounds),
            preferences={"use_rl_agent": request.use_rl_agent}
        )
        
        # Update progress
        pending_requests[request_id]["progress"] = 0.3
        
        # Run optimization
        result = optimization_service.OptimizeComponentPlacement(opt_request, None)
        
        # Store result
        completed_requests[request_id] = result
        
        # Remove from pending
        del pending_requests[request_id]
        
    except Exception as e:
        logger.error(f"Placement optimization failed: {e}")
        completed_requests[request_id] = OptimizationResponse(
            request_id=request_id,
            success=False,
            optimization_result=None,
            metrics={},
            warnings=[],
            error_message=str(e)
        )
        if request_id in pending_requests:
            del pending_requests[request_id]

async def _run_routing_optimization(request_id: str, request: RoutingOptimizationRequest):
    """Run routing optimization in background."""
    try:
        pending_requests[request_id]["status"] = "running"
        pending_requests[request_id]["progress"] = 0.1
        
        # Create optimization request (simplified)
        opt_request = OptimizationRequest(
            request_id=request_id,
            optimization_type="routing",
            components=[component.dict() for component in request.existing_components],
            constraints=[constraint.dict() for constraint in request.constraints],
            objectives={},
            workspace_bounds=tuple(request.workspace_bounds)
        )
        
        # Add wire specifications and points to request
        opt_request.wire_specifications = [spec.dict() for spec in request.wire_specifications]
        opt_request.start_points = request.start_points
        opt_request.end_points = request.end_points
        
        pending_requests[request_id]["progress"] = 0.3
        
        # Run optimization
        result = optimization_service.OptimizeWireRouting(opt_request, None)
        
        completed_requests[request_id] = result
        del pending_requests[request_id]
        
    except Exception as e:
        logger.error(f"Routing optimization failed: {e}")
        completed_requests[request_id] = OptimizationResponse(
            request_id=request_id,
            success=False,
            optimization_result=None,
            metrics={},
            warnings=[],
            error_message=str(e)
        )
        if request_id in pending_requests:
            del pending_requests[request_id]

async def _run_full_optimization(request_id: str, request: FullOptimizationRequest):
    """Run full optimization in background."""
    try:
        pending_requests[request_id]["status"] = "running"
        pending_requests[request_id]["progress"] = 0.1
        
        # Create optimization request
        opt_request = OptimizationRequest(
            request_id=request_id,
            optimization_type="full_optimization",
            components=[component.dict() for component in request.components],
            constraints=[constraint.dict() for constraint in request.constraints],
            objectives=request.objectives,
            workspace_bounds=tuple(request.workspace_bounds),
            preferences=request.preferences
        )
        
        pending_requests[request_id]["progress"] = 0.3
        
        # Run optimization
        result = optimization_service.OptimizeFullLayout(opt_request, None)
        
        completed_requests[request_id] = result
        del pending_requests[request_id]
        
    except Exception as e:
        logger.error(f"Full optimization failed: {e}")
        completed_requests[request_id] = OptimizationResponse(
            request_id=request_id,
            success=False,
            optimization_result=None,
            metrics={},
            warnings=[],
            error_message=str(e)
        )
        if request_id in pending_requests:
            del pending_requests[request_id]

async def _run_model_training(training_id: str, model_type: str, 
                            dataset_path: str, hyperparameters: Dict[str, Any]):
    """Run model training in background."""
    try:
        pending_requests[training_id]["status"] = "running"
        
        # Simulate training progress
        for progress in [0.1, 0.3, 0.5, 0.7, 0.9]:
            await asyncio.sleep(10)  # Simulate training time
            if training_id in pending_requests:
                pending_requests[training_id]["progress"] = progress
        
        # Mark as completed
        if training_id in pending_requests:
            pending_requests[training_id]["status"] = "completed"
            pending_requests[training_id]["progress"] = 1.0
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        if training_id in pending_requests:
            pending_requests[training_id]["status"] = "failed"
            pending_requests[training_id]["error"] = str(e)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
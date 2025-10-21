"""
PART_OF relationship model for hierarchical relationships between components and circuits
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class ComponentRole(str, Enum):
    """Enumeration of component roles in circuits"""
    SOURCE = "source"
    LOAD = "load"
    PROTECTION = "protection"
    CONTROL = "control"
    SWITCHING = "switching"
    SENSING = "sensing"
    DISTRIBUTION = "distribution"
    GROUNDING = "grounding"
    FILTERING = "filtering"
    REGULATION = "regulation"


class CriticalityLevel(str, Enum):
    """Enumeration of component criticality levels"""
    CRITICAL = "critical"
    IMPORTANT = "important"
    NORMAL = "normal"
    REDUNDANT = "redundant"
    OPTIONAL = "optional"


class FailureMode(str, Enum):
    """Enumeration of failure modes"""
    OPEN_CIRCUIT = "open_circuit"
    SHORT_CIRCUIT = "short_circuit"
    HIGH_RESISTANCE = "high_resistance"
    DEGRADED_PERFORMANCE = "degraded_performance"
    INTERMITTENT = "intermittent"
    STUCK_ON = "stuck_on"
    STUCK_OFF = "stuck_off"
    OUT_OF_TOLERANCE = "out_of_tolerance"


class PartOfRelationship(BaseModel):
    """
    PART_OF relationship model matching Neo4j schema
    
    Neo4j Schema:
    (:Component)-[:PART_OF {
      role: "string",
      is_critical: "boolean",
      created_at: "datetime"
    }]->(:Circuit)
    """
    
    # Component role in circuit
    role: ComponentRole = Field(
        default=ComponentRole.LOAD,
        description="Role in circuit: source, load, protection, control"
    )
    primary_function: str = Field(
        ...,
        description="Primary function description"
    )
    secondary_functions: Optional[List[str]] = Field(
        None,
        description="Secondary functions"
    )
    
    # Criticality and importance
    is_critical: bool = Field(
        default=False,
        description="Critical component flag"
    )
    criticality_level: CriticalityLevel = Field(
        default=CriticalityLevel.NORMAL,
        description="Criticality level"
    )
    failure_impact: str = Field(
        default="local",
        description="Failure impact scope: local, system, vehicle"
    )
    
    # Circuit behavior
    operational_state: Optional[str] = Field(
        None,
        description="Normal operational state"
    )
    duty_cycle: Optional[float] = Field(
        None,
        description="Duty cycle percentage"
    )
    operating_sequence: Optional[int] = Field(
        None,
        description="Order in operational sequence"
    )
    
    # Performance characteristics
    expected_load: Optional[float] = Field(None, description="Expected electrical load (A)")
    load_variation: Optional[Dict[str, float]] = Field(
        None,
        description="Load variation range {min: 0.5, max: 2.0}"
    )
    efficiency: Optional[float] = Field(None, description="Component efficiency percentage")
    power_factor: Optional[float] = Field(None, description="Power factor")
    
    # Timing characteristics
    activation_time: Optional[float] = Field(None, description="Activation time (ms)")
    deactivation_time: Optional[float] = Field(None, description="Deactivation time (ms)")
    response_time: Optional[float] = Field(None, description="Response time (ms)")
    settling_time: Optional[float] = Field(None, description="Settling time (ms)")
    
    # Reliability and maintenance
    mtbf: Optional[float] = Field(None, description="Mean Time Between Failures (hours)")
    mttr: Optional[float] = Field(None, description="Mean Time To Repair (hours)")
    failure_modes: Optional[List[FailureMode]] = Field(
        None,
        description="Known failure modes"
    )
    failure_indicators: Optional[List[str]] = Field(
        None,
        description="Failure indicators/symptoms"
    )
    
    # Dependencies
    depends_on: Optional[List[str]] = Field(
        None,
        description="Component IDs this component depends on"
    )
    enables: Optional[List[str]] = Field(
        None,
        description="Component IDs this component enables"
    )
    redundancy_group: Optional[str] = Field(
        None,
        description="Redundancy group identifier"
    )
    
    # Testing and diagnostics
    self_test_capable: bool = Field(default=False, description="Has self-test capability")
    diagnostic_capable: bool = Field(default=False, description="Has diagnostic capability")
    test_points_available: bool = Field(default=False, description="Has test points")
    test_procedures: Optional[List[str]] = Field(None, description="Test procedure references")
    
    # Circuit protection
    protection_provided: Optional[List[str]] = Field(
        None,
        description="Types of protection provided"
    )
    protection_required: Optional[List[str]] = Field(
        None,
        description="Types of protection required"
    )
    fuse_rating: Optional[str] = Field(None, description="Associated fuse rating")
    
    # Performance monitoring
    monitoring_required: bool = Field(default=False, description="Requires performance monitoring")
    monitoring_parameters: Optional[List[str]] = Field(
        None,
        description="Parameters to monitor"
    )
    alarm_thresholds: Optional[Dict[str, float]] = Field(
        None,
        description="Alarm thresholds for monitored parameters"
    )
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_verified: Optional[datetime] = Field(None, description="Last verification date")
    verification_method: Optional[str] = Field(None, description="Verification method used")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def to_neo4j_dict(self) -> Dict[str, Any]:
        """Convert to Neo4j relationship properties dictionary"""
        data = self.dict(exclude_none=True)
        
        # Convert enums to strings
        if 'role' in data:
            data['role'] = self.role.value
        if 'criticality_level' in data:
            data['criticality_level'] = self.criticality_level.value
        if 'failure_modes' in data and self.failure_modes:
            data['failure_modes'] = [mode.value for mode in self.failure_modes]
        
        # Convert datetime to string
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if self.last_verified:
            data['last_verified'] = self.last_verified.isoformat()
            
        return data
    
    @classmethod
    def from_neo4j_dict(cls, data: Dict[str, Any]) -> 'PartOfRelationship':
        """Create instance from Neo4j relationship dictionary"""
        # Convert datetime strings
        for field in ['created_at', 'last_verified']:
            if field in data and isinstance(data[field], str):
                data[field] = datetime.fromisoformat(data[field])
        
        # Convert string enums back to enum values
        if 'role' in data and isinstance(data['role'], str):
            data['role'] = ComponentRole(data['role'])
        if 'criticality_level' in data and isinstance(data['criticality_level'], str):
            data['criticality_level'] = CriticalityLevel(data['criticality_level'])
        if 'failure_modes' in data and isinstance(data['failure_modes'], list):
            data['failure_modes'] = [FailureMode(mode) for mode in data['failure_modes']]
        
        return cls(**data)
    
    def is_critical_component(self) -> bool:
        """Check if component is critical"""
        return (
            self.is_critical or 
            self.criticality_level == CriticalityLevel.CRITICAL or
            self.failure_impact == "vehicle"
        )
    
    def is_load_component(self) -> bool:
        """Check if component acts as a load"""
        return self.role in [ComponentRole.LOAD, ComponentRole.SENSING]
    
    def is_control_component(self) -> bool:
        """Check if component provides control functions"""
        return self.role in [ComponentRole.CONTROL, ComponentRole.SWITCHING]
    
    def is_protection_component(self) -> bool:
        """Check if component provides protection"""
        return self.role in [ComponentRole.PROTECTION, ComponentRole.FILTERING]
    
    def has_redundancy(self) -> bool:
        """Check if component has redundancy"""
        return self.redundancy_group is not None
    
    def calculate_reliability_score(self) -> float:
        """Calculate reliability score based on MTBF and criticality"""
        base_score = 0.5
        
        # MTBF contribution
        if self.mtbf:
            # Normalize MTBF (higher is better)
            mtbf_score = min(self.mtbf / 10000, 1.0)  # 10k hours = perfect
            base_score += mtbf_score * 0.3
        
        # Criticality penalty
        criticality_penalty = {
            CriticalityLevel.CRITICAL: 0.0,
            CriticalityLevel.IMPORTANT: 0.1,
            CriticalityLevel.NORMAL: 0.2,
            CriticalityLevel.REDUNDANT: 0.3,
            CriticalityLevel.OPTIONAL: 0.4
        }
        base_score += criticality_penalty.get(self.criticality_level, 0.0)
        
        # Diagnostic capability bonus
        if self.diagnostic_capable:
            base_score += 0.1
        if self.self_test_capable:
            base_score += 0.1
        
        return min(base_score, 1.0)
    
    def get_failure_risk_assessment(self) -> Dict[str, Any]:
        """Assess failure risk and impact"""
        risk_score = 1.0
        
        # Base risk from criticality
        criticality_multiplier = {
            CriticalityLevel.CRITICAL: 5.0,
            CriticalityLevel.IMPORTANT: 3.0,
            CriticalityLevel.NORMAL: 2.0,
            CriticalityLevel.REDUNDANT: 1.0,
            CriticalityLevel.OPTIONAL: 0.5
        }
        risk_score *= criticality_multiplier.get(self.criticality_level, 2.0)
        
        # MTBF adjustment
        if self.mtbf and self.mtbf < 1000:  # Less than 1000 hours
            risk_score *= 2.0
        elif self.mtbf and self.mtbf > 10000:  # More than 10000 hours
            risk_score *= 0.5
        
        # Failure impact adjustment
        impact_multiplier = {
            "local": 1.0,
            "system": 2.0,
            "vehicle": 4.0
        }
        risk_score *= impact_multiplier.get(self.failure_impact, 1.0)
        
        # Mitigation factors
        if self.has_redundancy():
            risk_score *= 0.3
        if self.diagnostic_capable:
            risk_score *= 0.8
        
        return {
            "risk_score": risk_score,
            "risk_level": "high" if risk_score > 8.0 else "medium" if risk_score > 3.0 else "low",
            "mitigation_available": self.has_redundancy() or self.diagnostic_capable
        }


class CircuitHierarchy(BaseModel):
    """Model for circuit hierarchy analysis"""
    circuit_id: str
    parent_circuits: List[str]
    child_circuits: List[str]
    component_count: int
    criticality_distribution: Dict[str, int]
    dependency_depth: int


class SystemDependency(BaseModel):
    """Model for system dependency analysis"""
    vehicle_signature: str
    dependency_graph: Dict[str, List[str]]
    critical_paths: List[List[str]]
    single_points_of_failure: List[str]
    redundancy_coverage: float
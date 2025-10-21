"""
LOCATED_IN relationship model for spatial relationships between components and zones
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class PositionType(str, Enum):
    """Enumeration of position types"""
    MOUNTED = "mounted"
    FLOATING = "floating"
    INTEGRATED = "integrated"
    EMBEDDED = "embedded"
    SUSPENDED = "suspended"
    BRACKET_MOUNTED = "bracket_mounted"
    PANEL_MOUNTED = "panel_mounted"


class AccessibilityLevel(str, Enum):
    """Enumeration of accessibility levels"""
    EASY = "easy"
    MODERATE = "moderate" 
    DIFFICULT = "difficult"
    VERY_DIFFICULT = "very_difficult"
    REQUIRES_DISASSEMBLY = "requires_disassembly"


class MountingMethod(str, Enum):
    """Enumeration of mounting methods"""
    BOLTED = "bolted"
    CLIPPED = "clipped"
    WELDED = "welded"
    ADHESIVE = "adhesive"
    PRESS_FIT = "press_fit"
    MAGNETIC = "magnetic"
    SLIDING = "sliding"
    THREADED = "threaded"


class LocatedInRelationship(BaseModel):
    """
    LOCATED_IN relationship model matching Neo4j schema
    
    Neo4j Schema:
    (:Component)-[:LOCATED_IN {
      position_type: "string",
      accessibility: "string", 
      created_at: "datetime"
    }]->(:Zone)
    """
    
    # Position characteristics
    position_type: PositionType = Field(
        default=PositionType.MOUNTED,
        description="Type of position: mounted, floating, integrated"
    )
    mounting_method: Optional[MountingMethod] = Field(
        None,
        description="How the component is mounted"
    )
    
    # Accessibility
    accessibility: AccessibilityLevel = Field(
        default=AccessibilityLevel.MODERATE,
        description="Accessibility level: easy, moderate, difficult"
    )
    removal_difficulty: Optional[AccessibilityLevel] = Field(
        None,
        description="Difficulty of removing the component"
    )
    
    # Spatial precision
    exact_position: Optional[Dict[str, float]] = Field(
        None,
        description="Exact position within zone {x, y, z}"
    )
    position_tolerance: Optional[float] = Field(
        None,
        description="Position tolerance in mm"
    )
    orientation: Optional[Dict[str, float]] = Field(
        None,
        description="Component orientation {roll, pitch, yaw} in degrees"
    )
    
    # Physical constraints
    clearance_required: Optional[Dict[str, float]] = Field(
        None,
        description="Required clearance {top, bottom, left, right, front, back} in mm"
    )
    mounting_points: Optional[int] = Field(None, description="Number of mounting points")
    weight_supported: Optional[float] = Field(None, description="Weight supported by mounting (kg)")
    
    # Environmental considerations
    vibration_isolation: bool = Field(default=False, description="Has vibration isolation")
    thermal_management: bool = Field(default=False, description="Requires thermal management")
    weatherproofing: Optional[str] = Field(None, description="Weather protection level (IP rating)")
    
    # Service access
    service_access_required: bool = Field(default=False, description="Requires regular service access")
    service_interval: Optional[str] = Field(None, description="Service interval (months)")
    special_tools_required: bool = Field(default=False, description="Requires special tools for access")
    
    # Installation considerations
    installation_order: Optional[int] = Field(None, description="Installation sequence order")
    installation_time: Optional[float] = Field(None, description="Installation time (minutes)")
    requires_calibration: bool = Field(default=False, description="Requires calibration after installation")
    
    # Safety considerations
    safety_critical_location: bool = Field(default=False, description="Safety-critical location")
    requires_safety_lockout: bool = Field(default=False, description="Requires safety lockout during service")
    hazardous_area: bool = Field(default=False, description="Located in hazardous area")
    
    # Documentation
    installation_notes: Optional[str] = Field(None, description="Installation notes")
    service_notes: Optional[str] = Field(None, description="Service access notes")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    verified_at: Optional[datetime] = Field(None, description="Last verification date")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def to_neo4j_dict(self) -> Dict[str, Any]:
        """Convert to Neo4j relationship properties dictionary"""
        data = self.dict(exclude_none=True)
        
        # Convert enums to strings
        if 'position_type' in data:
            data['position_type'] = self.position_type.value
        if 'accessibility' in data:
            data['accessibility'] = self.accessibility.value
        if 'mounting_method' in data:
            data['mounting_method'] = self.mounting_method.value
        if 'removal_difficulty' in data:
            data['removal_difficulty'] = self.removal_difficulty.value
        
        # Convert datetime to string
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if self.verified_at:
            data['verified_at'] = self.verified_at.isoformat()
            
        return data
    
    @classmethod
    def from_neo4j_dict(cls, data: Dict[str, Any]) -> 'LocatedInRelationship':
        """Create instance from Neo4j relationship dictionary"""
        # Convert datetime strings
        for field in ['created_at', 'verified_at']:
            if field in data and isinstance(data[field], str):
                data[field] = datetime.fromisoformat(data[field])
        
        # Convert string enums back to enum values
        enum_fields = {
            'position_type': PositionType,
            'accessibility': AccessibilityLevel,
            'mounting_method': MountingMethod,
            'removal_difficulty': AccessibilityLevel
        }
        
        for field, enum_class in enum_fields.items():
            if field in data and isinstance(data[field], str):
                data[field] = enum_class(data[field])
        
        return cls(**data)
    
    def is_easily_accessible(self) -> bool:
        """Check if component is easily accessible"""
        return self.accessibility in [AccessibilityLevel.EASY, AccessibilityLevel.MODERATE]
    
    def requires_disassembly(self) -> bool:
        """Check if accessing component requires disassembly"""
        return self.accessibility == AccessibilityLevel.REQUIRES_DISASSEMBLY
    
    def is_permanently_mounted(self) -> bool:
        """Check if component is permanently mounted"""
        permanent_methods = [MountingMethod.WELDED, MountingMethod.PRESS_FIT]
        return self.mounting_method in permanent_methods
    
    def is_removable(self) -> bool:
        """Check if component is easily removable"""
        removable_methods = [
            MountingMethod.BOLTED,
            MountingMethod.CLIPPED,
            MountingMethod.SLIDING,
            MountingMethod.THREADED
        ]
        return self.mounting_method in removable_methods
    
    def get_service_complexity(self) -> str:
        """Assess service complexity level"""
        if self.requires_disassembly() or self.special_tools_required:
            return "high"
        elif self.accessibility == AccessibilityLevel.DIFFICULT:
            return "medium"
        else:
            return "low"
    
    def calculate_installation_effort(self) -> Dict[str, Any]:
        """Calculate installation effort metrics"""
        effort_score = 1.0
        
        # Accessibility factor
        accessibility_multiplier = {
            AccessibilityLevel.EASY: 1.0,
            AccessibilityLevel.MODERATE: 1.5,
            AccessibilityLevel.DIFFICULT: 2.5,
            AccessibilityLevel.VERY_DIFFICULT: 4.0,
            AccessibilityLevel.REQUIRES_DISASSEMBLY: 6.0
        }
        effort_score *= accessibility_multiplier.get(self.accessibility, 2.0)
        
        # Mounting complexity
        if self.mounting_points and self.mounting_points > 4:
            effort_score *= 1.5
        
        # Special considerations
        if self.special_tools_required:
            effort_score *= 1.8
        if self.requires_calibration:
            effort_score *= 1.3
        if self.safety_critical_location:
            effort_score *= 1.5
        
        return {
            "effort_score": effort_score,
            "complexity": "high" if effort_score > 5.0 else "medium" if effort_score > 2.0 else "low",
            "estimated_time": (self.installation_time or 30) * effort_score
        }


class SpatialCluster(BaseModel):
    """Model for spatial clustering analysis"""
    zone_id: str
    component_count: int
    component_density: float
    accessibility_distribution: Dict[str, int]
    service_complexity: str


class ZoneUtilization(BaseModel):
    """Model for zone utilization analysis"""
    zone_id: str
    vehicle_signature: str
    total_volume: float
    occupied_volume: float
    utilization_percentage: float
    accessibility_score: float
    maintenance_complexity: str
    component_types: List[str]
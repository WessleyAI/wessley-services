"""
Circuit validation schemas for electrical circuit data integrity
"""

from pydantic import BaseModel, Field, validator, root_validator
from typing import Optional, Dict, Any, List
from datetime import datetime
import re
from enum import Enum


class CircuitType(str, Enum):
    """Standard circuit types"""
    POWER = "power"
    SIGNAL = "signal"
    GROUND = "ground"
    CAN_BUS = "can_bus"
    LIN_BUS = "lin_bus"
    ETHERNET = "ethernet"
    IGNITION = "ignition"
    CHARGING = "charging"
    LIGHTING = "lighting"
    ENGINE_CONTROL = "engine_control"
    TRANSMISSION = "transmission"
    BRAKE_SYSTEM = "brake_system"
    STEERING = "steering"
    HVAC = "hvac"
    INFOTAINMENT = "infotainment"
    SECURITY = "security"
    SAFETY = "safety"
    DIAGNOSTIC = "diagnostic"


class ProtectionType(str, Enum):
    """Circuit protection types"""
    FUSE = "fuse"
    CIRCUIT_BREAKER = "circuit_breaker"
    RELAY = "relay"
    PTC = "ptc"
    ELECTRONIC_FUSE = "electronic_fuse"
    CURRENT_LIMITER = "current_limiter"
    NONE = "none"


class CircuitValidationSchema(BaseModel):
    """Schema for validating circuit data before Neo4j insertion"""
    
    # Required fields
    id: str = Field(..., min_length=1, max_length=100, regex=r'^[a-zA-Z0-9_-]+$')
    vehicle_signature: str = Field(..., min_length=1, max_length=100)
    name: str = Field(..., min_length=1, max_length=200)
    circuit_type: CircuitType = Field(...)
    
    # Electrical characteristics
    voltage: float = Field(..., ge=0, le=1000, description="Operating voltage")
    max_current: float = Field(..., ge=0, le=1000, description="Maximum current capacity")
    nominal_current: Optional[float] = Field(None, ge=0, le=1000, description="Nominal operating current")
    
    # Protection
    protection: ProtectionType = Field(default=ProtectionType.FUSE)
    protection_rating: Optional[str] = Field(None, max_length=50, description="Protection device rating")
    
    # Circuit characteristics
    wire_gauge: Optional[str] = Field(None, max_length=20, description="Primary wire gauge")
    wire_color: Optional[str] = Field(None, max_length=50, description="Primary wire color")
    circuit_length: Optional[float] = Field(None, ge=0, le=10000, description="Total circuit length (mm)")
    
    # Performance characteristics
    voltage_drop_max: Optional[float] = Field(None, ge=0, le=100, description="Maximum allowable voltage drop (%)")
    load_factor: Optional[float] = Field(None, ge=0, le=2.0, description="Load factor (actual/rated)")
    duty_cycle: Optional[float] = Field(None, ge=0, le=100, description="Duty cycle percentage")
    
    # Environmental and operational
    operating_temperature_min: Optional[float] = Field(None, ge=-50, le=200, description="Min operating temp (°C)")
    operating_temperature_max: Optional[float] = Field(None, ge=-50, le=200, description="Max operating temp (°C)")
    environment: Optional[str] = Field(None, max_length=100, description="Operating environment")
    
    # Safety and criticality
    safety_critical: bool = Field(default=False, description="Safety-critical circuit")
    redundancy_available: bool = Field(default=False, description="Redundancy available")
    fail_safe_mode: Optional[str] = Field(None, max_length=50, description="Fail-safe behavior")
    
    # Documentation and maintenance
    schematic_reference: Optional[str] = Field(None, max_length=100, description="Schematic reference")
    test_procedures: Optional[List[str]] = Field(None, description="Test procedure references")
    maintenance_interval: Optional[str] = Field(None, max_length=50, description="Maintenance interval")
    
    # Additional specifications
    specifications: Optional[Dict[str, Any]] = Field(None)
    
    @validator('id')
    def validate_id(cls, v):
        """Validate circuit ID format"""
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Circuit ID must contain only alphanumeric characters, underscores, and hyphens')
        return v
    
    @validator('vehicle_signature')
    def validate_vehicle_signature(cls, v):
        """Validate vehicle signature format"""
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Vehicle signature must contain only alphanumeric characters, underscores, and hyphens')
        return v
    
    @validator('voltage')
    def validate_voltage_range(cls, v, values):
        """Validate voltage based on circuit type"""
        circuit_type = values.get('circuit_type')
        if circuit_type:
            voltage_ranges = {
                CircuitType.POWER: (6, 48),
                CircuitType.SIGNAL: (3.3, 24),
                CircuitType.CAN_BUS: (3.3, 5),
                CircuitType.LIN_BUS: (9, 18),
                CircuitType.IGNITION: (12, 48),
                CircuitType.LIGHTING: (12, 24)
            }
            
            if circuit_type in voltage_ranges:
                min_v, max_v = voltage_ranges[circuit_type]
                if not (min_v <= v <= max_v):
                    raise ValueError(f'{circuit_type.value} circuits must operate between {min_v}V and {max_v}V')
        
        return v
    
    @validator('nominal_current')
    def validate_nominal_current(cls, v, values):
        """Ensure nominal current <= max current"""
        max_current = values.get('max_current')
        if v is not None and max_current is not None and v > max_current:
            raise ValueError('Nominal current cannot exceed maximum current')
        return v
    
    @validator('protection_rating')
    def validate_protection_rating(cls, v, values):
        """Validate protection rating format and consistency"""
        if v is None:
            return v
        
        protection_type = values.get('protection')
        max_current = values.get('max_current')
        
        # Validate fuse rating format
        if protection_type == ProtectionType.FUSE:
            if not re.match(r'^\d+(\.\d+)?A$', v):
                raise ValueError('Fuse rating must be in format "10A" or "10.5A"')
            
            # Extract numerical value and compare with max_current
            rating_value = float(v.replace('A', ''))
            if max_current and rating_value < max_current * 0.8:
                raise ValueError('Fuse rating should be at least 80% of maximum current')
        
        return v
    
    @validator('wire_gauge')
    def validate_wire_gauge(cls, v, values):
        """Validate wire gauge format and adequacy"""
        if v is None:
            return v
        
        # Validate wire gauge format (AWG or metric)
        if not re.match(r'^(\d+(\.\d+)?(AWG|mm²)|#\d+)$', v):
            raise ValueError('Wire gauge must be in format "12AWG", "2.5mm²", or "#12"')
        
        # Check wire adequacy for current
        max_current = values.get('max_current')
        if max_current:
            # Simple wire current capacity check (conservative estimates)
            wire_capacity = {
                '18AWG': 10, '16AWG': 13, '14AWG': 15, '12AWG': 20, '10AWG': 30,
                '0.5mm²': 6, '0.75mm²': 8, '1.0mm²': 10, '1.5mm²': 15, '2.5mm²': 20,
                '4.0mm²': 27, '6.0mm²': 35, '10.0mm²': 50
            }
            
            capacity = wire_capacity.get(v)
            if capacity and max_current > capacity:
                raise ValueError(f'Wire gauge {v} insufficient for {max_current}A (max {capacity}A)')
        
        return v
    
    @root_validator
    def validate_temperature_range(cls, values):
        """Validate temperature range consistency"""
        temp_min = values.get('operating_temperature_min')
        temp_max = values.get('operating_temperature_max')
        
        if temp_min is not None and temp_max is not None:
            if temp_min >= temp_max:
                raise ValueError('Minimum operating temperature must be less than maximum')
        
        return values
    
    @root_validator
    def validate_current_consistency(cls, values):
        """Validate current-related parameters are consistent"""
        max_current = values.get('max_current')
        nominal_current = values.get('nominal_current')
        load_factor = values.get('load_factor')
        
        if nominal_current and load_factor:
            calculated_current = nominal_current * load_factor
            if calculated_current > max_current * 1.1:  # 10% tolerance
                raise ValueError('Load factor and nominal current result in current exceeding maximum')
        
        return values
    
    @root_validator
    def validate_safety_critical_requirements(cls, values):
        """Validate safety-critical circuit requirements"""
        safety_critical = values.get('safety_critical', False)
        
        if safety_critical:
            # Safety-critical circuits should have redundancy or fail-safe mode
            redundancy = values.get('redundancy_available', False)
            fail_safe = values.get('fail_safe_mode')
            
            if not redundancy and not fail_safe:
                raise ValueError('Safety-critical circuits must have redundancy or defined fail-safe mode')
        
        return values
    
    @validator('specifications')
    def validate_specifications(cls, v, values):
        """Validate circuit-specific specifications"""
        if v is None:
            return v
        
        if not isinstance(v, dict):
            raise ValueError('Specifications must be a dictionary')
        
        # Validate JSON serializability
        try:
            import json
            json.dumps(v)
        except (TypeError, ValueError):
            raise ValueError('All specification values must be JSON-serializable')
        
        # Circuit type-specific validations
        circuit_type = values.get('circuit_type')
        
        if circuit_type == CircuitType.CAN_BUS:
            required_specs = ['baud_rate', 'termination_resistance']
            missing = [spec for spec in required_specs if spec not in v]
            if missing:
                raise ValueError(f'CAN bus circuits must specify: {", ".join(missing)}')
        
        elif circuit_type == CircuitType.LIN_BUS:
            if 'baud_rate' not in v:
                raise ValueError('LIN bus circuits must specify baud_rate')
        
        elif circuit_type == CircuitType.POWER:
            if 'load_type' not in v:
                raise ValueError('Power circuits should specify load_type (resistive/inductive/capacitive)')
        
        return v


class CircuitConnectionSchema(BaseModel):
    """Schema for validating circuit-component connections"""
    
    circuit_id: str = Field(..., min_length=1, max_length=100)
    component_id: str = Field(..., min_length=1, max_length=100)
    vehicle_signature: str = Field(..., min_length=1, max_length=100)
    role: str = Field(..., description="Component role in circuit")
    is_critical: bool = Field(default=False)
    
    @validator('role')
    def validate_role(cls, v):
        """Validate component role in circuit"""
        valid_roles = ['source', 'load', 'protection', 'control', 'switching', 'sensing', 'distribution']
        if v not in valid_roles:
            raise ValueError(f'Role must be one of: {", ".join(valid_roles)}')
        return v


class CircuitBatchValidationSchema(BaseModel):
    """Schema for validating batch circuit operations"""
    
    vehicle_signature: str = Field(..., min_length=1, max_length=100)
    circuits: List[CircuitValidationSchema] = Field(..., min_items=1, max_items=500)
    
    @validator('circuits')
    def validate_circuit_uniqueness(cls, v, values):
        """Ensure circuit IDs are unique within batch"""
        vehicle_sig = values.get('vehicle_signature')
        if not vehicle_sig:
            return v
        
        circuit_ids = []
        for circuit in v:
            if circuit.vehicle_signature != vehicle_sig:
                raise ValueError('All circuits must have the same vehicle_signature as the batch')
            circuit_ids.append(circuit.id)
        
        if len(circuit_ids) != len(set(circuit_ids)):
            duplicates = [id for id in circuit_ids if circuit_ids.count(id) > 1]
            raise ValueError(f'Duplicate circuit IDs found: {", ".join(set(duplicates))}')
        
        return v
    
    @root_validator
    def validate_system_requirements(cls, values):
        """Validate system-level circuit requirements"""
        circuits = values.get('circuits', [])
        
        # Check for essential circuit types
        circuit_types = [c.circuit_type for c in circuits]
        
        if CircuitType.POWER not in circuit_types:
            raise ValueError('System must contain at least one power circuit')
        
        if CircuitType.GROUND not in circuit_types:
            raise ValueError('System must contain at least one ground circuit')
        
        # Check for voltage compatibility
        power_circuits = [c for c in circuits if c.circuit_type == CircuitType.POWER]
        voltages = set([c.voltage for c in power_circuits])
        
        if len(voltages) > 3:  # More than 3 different voltage levels
            raise ValueError('System should not have more than 3 different voltage levels')
        
        return values


class CircuitUpdateSchema(BaseModel):
    """Schema for validating circuit updates"""
    
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    voltage: Optional[float] = Field(None, ge=0, le=1000)
    max_current: Optional[float] = Field(None, ge=0, le=1000)
    nominal_current: Optional[float] = Field(None, ge=0, le=1000)
    protection_rating: Optional[str] = Field(None, max_length=50)
    wire_gauge: Optional[str] = Field(None, max_length=20)
    wire_color: Optional[str] = Field(None, max_length=50)
    specifications: Optional[Dict[str, Any]] = Field(None)
    
    @root_validator
    def validate_update_has_changes(cls, values):
        """Ensure update contains at least one field"""
        provided_fields = [k for k, v in values.items() if v is not None]
        if not provided_fields:
            raise ValueError('Update must specify at least one field to modify')
        return values


class CircuitQuerySchema(BaseModel):
    """Schema for validating circuit query parameters"""
    
    vehicle_signature: Optional[str] = Field(None, min_length=1, max_length=100)
    circuit_types: Optional[List[CircuitType]] = Field(None, max_items=10)
    voltage_min: Optional[float] = Field(None, ge=0, le=1000)
    voltage_max: Optional[float] = Field(None, ge=0, le=1000)
    current_min: Optional[float] = Field(None, ge=0, le=1000)
    current_max: Optional[float] = Field(None, ge=0, le=1000)
    safety_critical: Optional[bool] = None
    has_redundancy: Optional[bool] = None
    protection_type: Optional[ProtectionType] = None
    limit: Optional[int] = Field(None, ge=1, le=1000)
    offset: Optional[int] = Field(None, ge=0)
    
    @validator('voltage_max')
    def validate_voltage_range(cls, v, values):
        """Ensure voltage max >= voltage min"""
        voltage_min = values.get('voltage_min')
        if voltage_min is not None and v is not None and v < voltage_min:
            raise ValueError('voltage_max must be >= voltage_min')
        return v
    
    @validator('current_max')
    def validate_current_range(cls, v, values):
        """Ensure current max >= current min"""
        current_min = values.get('current_min')
        if current_min is not None and v is not None and v < current_min:
            raise ValueError('current_max must be >= current_min')
        return v
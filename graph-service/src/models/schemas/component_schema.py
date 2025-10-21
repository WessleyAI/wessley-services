"""
Component validation schemas for data integrity and consistency
"""

from pydantic import BaseModel, Field, validator, root_validator
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
import re
from enum import Enum


class ComponentType(str, Enum):
    """Standard component types"""
    BATTERY = "battery"
    ALTERNATOR = "alternator"
    STARTER = "starter"
    RELAY = "relay"
    FUSE = "fuse"
    SWITCH = "switch"
    SENSOR = "sensor"
    ECU = "ecu"
    ACTUATOR = "actuator"
    CONNECTOR = "connector"
    HARNESS = "harness"
    GROUND = "ground"
    JUNCTION = "junction"
    TERMINAL = "terminal"
    RESISTOR = "resistor"
    CAPACITOR = "capacitor"
    INDUCTOR = "inductor"
    DIODE = "diode"
    TRANSISTOR = "transistor"
    LIGHT = "light"
    MOTOR = "motor"
    SOLENOID = "solenoid"
    PUMP = "pump"
    FAN = "fan"
    HEATER = "heater"
    COOLER = "cooler"
    IGNITION = "ignition"
    FUEL_INJECTOR = "fuel_injector"


class ComponentValidationSchema(BaseModel):
    """Schema for validating component data before Neo4j insertion"""
    
    # Required fields
    id: str = Field(..., min_length=1, max_length=100, regex=r'^[a-zA-Z0-9_-]+$')
    vehicle_signature: str = Field(..., min_length=1, max_length=100)
    type: ComponentType = Field(...)
    name: str = Field(..., min_length=1, max_length=200)
    
    # Optional identification
    part_number: Optional[str] = Field(None, max_length=100)
    manufacturer: Optional[str] = Field(None, max_length=100)
    
    # Technical specifications
    voltage_rating: Optional[float] = Field(None, ge=0, le=1000)
    current_rating: Optional[float] = Field(None, ge=0, le=1000)
    power_rating: Optional[float] = Field(None, ge=0, le=100000)
    resistance: Optional[float] = Field(None, ge=0)
    
    # Spatial data
    position_x: Optional[float] = Field(None, ge=-10000, le=10000)
    position_y: Optional[float] = Field(None, ge=-10000, le=10000)
    position_z: Optional[float] = Field(None, ge=-10000, le=10000)
    zone: Optional[str] = Field(None, max_length=100)
    
    # Additional specifications
    specifications: Optional[Dict[str, Any]] = Field(None)
    
    @validator('id')
    def validate_id(cls, v):
        """Validate component ID format"""
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Component ID must contain only alphanumeric characters, underscores, and hyphens')
        return v
    
    @validator('vehicle_signature')
    def validate_vehicle_signature(cls, v):
        """Validate vehicle signature format"""
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Vehicle signature must contain only alphanumeric characters, underscores, and hyphens')
        return v
    
    @validator('voltage_rating')
    def validate_voltage(cls, v, values):
        """Validate voltage rating based on component type"""
        if v is None:
            return v
        
        component_type = values.get('type')
        if component_type:
            # Define voltage ranges for different component types
            voltage_ranges = {
                ComponentType.BATTERY: (6, 48),
                ComponentType.ALTERNATOR: (12, 48),
                ComponentType.ECU: (3.3, 24),
                ComponentType.SENSOR: (3.3, 24),
                ComponentType.RELAY: (5, 48),
                ComponentType.LIGHT: (12, 24),
                ComponentType.MOTOR: (12, 400)
            }
            
            if component_type in voltage_ranges:
                min_v, max_v = voltage_ranges[component_type]
                if not (min_v <= v <= max_v):
                    raise ValueError(f'{component_type.value} voltage rating must be between {min_v}V and {max_v}V')
        
        return v
    
    @validator('current_rating')
    def validate_current(cls, v, values):
        """Validate current rating reasonableness"""
        if v is None:
            return v
        
        component_type = values.get('type')
        if component_type:
            # Define current ranges for different component types
            current_ranges = {
                ComponentType.FUSE: (0.1, 200),
                ComponentType.RELAY: (0.01, 100),
                ComponentType.ECU: (0.1, 20),
                ComponentType.SENSOR: (0.001, 5),
                ComponentType.STARTER: (50, 500),
                ComponentType.LIGHT: (0.5, 20),
                ComponentType.MOTOR: (1, 200)
            }
            
            if component_type in current_ranges:
                min_i, max_i = current_ranges[component_type]
                if not (min_i <= v <= max_i):
                    raise ValueError(f'{component_type.value} current rating must be between {min_i}A and {max_i}A')
        
        return v
    
    @root_validator
    def validate_position_consistency(cls, values):
        """Validate that position coordinates are consistent"""
        x, y, z = values.get('position_x'), values.get('position_y'), values.get('position_z')
        
        # If any position coordinate is provided, all should be provided
        position_coords = [x, y, z]
        provided_coords = [coord for coord in position_coords if coord is not None]
        
        if len(provided_coords) > 0 and len(provided_coords) < 3:
            raise ValueError('If position is specified, all three coordinates (x, y, z) must be provided')
        
        return values
    
    @root_validator
    def validate_power_consistency(cls, values):
        """Validate power-related specifications are consistent"""
        voltage = values.get('voltage_rating')
        current = values.get('current_rating')
        power = values.get('power_rating')
        
        if voltage and current and power:
            calculated_power = voltage * current
            if abs(calculated_power - power) > (power * 0.1):  # 10% tolerance
                raise ValueError(f'Power rating ({power}W) inconsistent with voltage ({voltage}V) and current ({current}A)')
        
        return values
    
    @validator('specifications')
    def validate_specifications(cls, v, values):
        """Validate specifications dictionary"""
        if v is None:
            return v
        
        # Ensure specifications is a dictionary
        if not isinstance(v, dict):
            raise ValueError('Specifications must be a dictionary')
        
        # Validate that all values are JSON-serializable
        try:
            import json
            json.dumps(v)
        except (TypeError, ValueError):
            raise ValueError('All specification values must be JSON-serializable')
        
        # Component-specific specification validation
        component_type = values.get('type')
        if component_type == ComponentType.ECU:
            required_specs = ['processor', 'memory', 'can_channels']
            missing_specs = [spec for spec in required_specs if spec not in v]
            if missing_specs:
                raise ValueError(f'ECU components must specify: {", ".join(missing_specs)}')
        
        elif component_type == ComponentType.SENSOR:
            if 'signal_type' not in v:
                raise ValueError('Sensor components must specify signal_type')
        
        elif component_type == ComponentType.RELAY:
            if 'coil_voltage' not in v:
                raise ValueError('Relay components must specify coil_voltage')
        
        return v


class ComponentBatchValidationSchema(BaseModel):
    """Schema for validating batch component operations"""
    
    vehicle_signature: str = Field(..., min_length=1, max_length=100)
    components: List[ComponentValidationSchema] = Field(..., min_items=1, max_items=1000)
    
    @validator('components')
    def validate_component_uniqueness(cls, v, values):
        """Ensure component IDs are unique within the batch"""
        vehicle_sig = values.get('vehicle_signature')
        if not vehicle_sig:
            return v
        
        component_ids = []
        for component in v:
            # Check vehicle signature consistency
            if component.vehicle_signature != vehicle_sig:
                raise ValueError(f'All components must have the same vehicle_signature as the batch')
            
            component_ids.append(component.id)
        
        # Check for duplicates
        if len(component_ids) != len(set(component_ids)):
            duplicates = [id for id in component_ids if component_ids.count(id) > 1]
            raise ValueError(f'Duplicate component IDs found: {", ".join(set(duplicates))}')
        
        return v
    
    @root_validator
    def validate_batch_consistency(cls, values):
        """Validate batch-level consistency rules"""
        components = values.get('components', [])
        
        # Ensure there's at least one power source
        power_sources = [c for c in components if c.type in [ComponentType.BATTERY, ComponentType.ALTERNATOR]]
        if not power_sources:
            raise ValueError('Batch must contain at least one power source (battery or alternator)')
        
        # Ensure there's at least one ground
        grounds = [c for c in components if c.type == ComponentType.GROUND]
        if not grounds:
            raise ValueError('Batch must contain at least one ground component')
        
        return values


class ComponentUpdateSchema(BaseModel):
    """Schema for validating component updates"""
    
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    part_number: Optional[str] = Field(None, max_length=100)
    manufacturer: Optional[str] = Field(None, max_length=100)
    voltage_rating: Optional[float] = Field(None, ge=0, le=1000)
    current_rating: Optional[float] = Field(None, ge=0, le=1000)
    power_rating: Optional[float] = Field(None, ge=0, le=100000)
    position_x: Optional[float] = Field(None, ge=-10000, le=10000)
    position_y: Optional[float] = Field(None, ge=-10000, le=10000)
    position_z: Optional[float] = Field(None, ge=-10000, le=10000)
    zone: Optional[str] = Field(None, max_length=100)
    specifications: Optional[Dict[str, Any]] = Field(None)
    
    @root_validator
    def validate_update_has_changes(cls, values):
        """Ensure update contains at least one field to update"""
        provided_fields = [k for k, v in values.items() if v is not None]
        if not provided_fields:
            raise ValueError('Update must specify at least one field to modify')
        return values
    
    @root_validator  
    def validate_position_consistency(cls, values):
        """Validate position coordinates if updating position"""
        x, y, z = values.get('position_x'), values.get('position_y'), values.get('position_z')
        position_coords = [x, y, z]
        provided_coords = [coord for coord in position_coords if coord is not None]
        
        if len(provided_coords) > 0 and len(provided_coords) < 3:
            raise ValueError('If updating position, all three coordinates (x, y, z) must be provided')
        
        return values


class ComponentQuerySchema(BaseModel):
    """Schema for validating component query parameters"""
    
    vehicle_signature: Optional[str] = Field(None, min_length=1, max_length=100)
    component_types: Optional[List[ComponentType]] = Field(None, max_items=20)
    zone: Optional[str] = Field(None, max_length=100)
    manufacturer: Optional[str] = Field(None, max_length=100)
    has_position: Optional[bool] = None
    voltage_min: Optional[float] = Field(None, ge=0, le=1000)
    voltage_max: Optional[float] = Field(None, ge=0, le=1000)
    current_min: Optional[float] = Field(None, ge=0, le=1000)
    current_max: Optional[float] = Field(None, ge=0, le=1000)
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
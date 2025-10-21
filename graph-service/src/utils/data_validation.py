"""Data validation utilities for graph service"""

import re
from typing import Dict, List, Any, Optional, Set, Union
from datetime import datetime
from enum import Enum

class ValidationLevel(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

class ValidationResult:
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []
        self.is_valid: bool = True
    
    def add_error(self, message: str):
        """Add error message"""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str):
        """Add warning message"""
        self.warnings.append(message)
    
    def add_info(self, message: str):
        """Add info message"""
        self.info.append(message)
    
    def merge(self, other: 'ValidationResult'):
        """Merge another validation result"""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.info.extend(other.info)
        if not other.is_valid:
            self.is_valid = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings)
        }

class FieldValidator:
    """Field-level validation utilities"""
    
    # Regex patterns
    ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')
    VOLTAGE_PATTERN = re.compile(r'^\d+(\.\d+)?[vV]?$')
    CURRENT_PATTERN = re.compile(r'^\d+(\.\d+)?[aA]?$')
    WIRE_GAUGE_PATTERN = re.compile(r'^\d+(\.\d+)?(mm²?|awg)?$', re.IGNORECASE)
    COLOR_PATTERN = re.compile(r'^[a-zA-Z]+([/-][a-zA-Z]+)*$')
    
    # Valid enum values
    COMPONENT_TYPES = {
        'relay', 'fuse', 'connector', 'sensor', 'actuator', 'switch', 
        'ecu', 'harness', 'battery', 'alternator', 'starter', 'ground'
    }
    
    CIRCUIT_TYPES = {
        'power', 'ground', 'lighting', 'ignition', 'engine', 'body', 
        'comfort', 'safety', 'communication', 'auxiliary'
    }
    
    SIGNAL_TYPES = {
        'power', 'ground', 'digital', 'analog', 'can', 'lin', 'flexray',
        'ethernet', 'most', 'pwm', 'data'
    }
    
    @classmethod
    def validate_id(cls, value: str, field_name: str = "id") -> ValidationResult:
        """Validate ID format"""
        result = ValidationResult()
        
        if not value:
            result.add_error(f"{field_name} is required")
            return result
        
        if not isinstance(value, str):
            result.add_error(f"{field_name} must be a string")
            return result
        
        if len(value) > 100:
            result.add_error(f"{field_name} must be 100 characters or less")
        
        if not cls.ID_PATTERN.match(value):
            result.add_error(f"{field_name} contains invalid characters. Use only letters, numbers, hyphens, and underscores")
        
        return result
    
    @classmethod
    def validate_voltage(cls, value: Union[str, int, float], field_name: str = "voltage") -> ValidationResult:
        """Validate voltage format"""
        result = ValidationResult()
        
        if value is None:
            return result  # Optional field
        
        # Convert to string for pattern matching
        str_value = str(value)
        
        if not cls.VOLTAGE_PATTERN.match(str_value):
            result.add_warning(f"{field_name} has invalid format: {value}")
        
        # Try to extract numeric value
        try:
            numeric_value = float(re.sub(r'[vV]', '', str_value))
            if numeric_value < 0:
                result.add_error(f"{field_name} cannot be negative")
            elif numeric_value > 1000:
                result.add_warning(f"{field_name} seems unusually high: {numeric_value}V")
        except ValueError:
            result.add_error(f"{field_name} is not a valid number: {value}")
        
        return result
    
    @classmethod
    def validate_current(cls, value: Union[str, int, float], field_name: str = "current") -> ValidationResult:
        """Validate current format"""
        result = ValidationResult()
        
        if value is None:
            return result  # Optional field
        
        str_value = str(value)
        
        if not cls.CURRENT_PATTERN.match(str_value):
            result.add_warning(f"{field_name} has invalid format: {value}")
        
        try:
            numeric_value = float(re.sub(r'[aA]', '', str_value))
            if numeric_value < 0:
                result.add_error(f"{field_name} cannot be negative")
            elif numeric_value > 500:
                result.add_warning(f"{field_name} seems unusually high: {numeric_value}A")
        except ValueError:
            result.add_error(f"{field_name} is not a valid number: {value}")
        
        return result
    
    @classmethod
    def validate_wire_gauge(cls, value: str, field_name: str = "wire_gauge") -> ValidationResult:
        """Validate wire gauge format"""
        result = ValidationResult()
        
        if value is None:
            return result  # Optional field
        
        if not cls.WIRE_GAUGE_PATTERN.match(value):
            result.add_warning(f"{field_name} has invalid format: {value}")
        
        return result
    
    @classmethod
    def validate_color(cls, value: str, field_name: str = "color") -> ValidationResult:
        """Validate color format"""
        result = ValidationResult()
        
        if value is None:
            return result  # Optional field
        
        if not cls.COLOR_PATTERN.match(value):
            result.add_warning(f"{field_name} has invalid format: {value}")
        
        return result
    
    @classmethod
    def validate_position(cls, value: List[float], field_name: str = "position") -> ValidationResult:
        """Validate 3D position"""
        result = ValidationResult()
        
        if value is None:
            return result  # Optional field
        
        if not isinstance(value, list):
            result.add_error(f"{field_name} must be a list")
            return result
        
        if len(value) != 3:
            result.add_error(f"{field_name} must have exactly 3 coordinates [x, y, z]")
            return result
        
        for i, coord in enumerate(value):
            if not isinstance(coord, (int, float)):
                result.add_error(f"{field_name}[{i}] must be a number")
        
        return result
    
    @classmethod
    def validate_component_type(cls, value: str, field_name: str = "type") -> ValidationResult:
        """Validate component type"""
        result = ValidationResult()
        
        if value is None:
            result.add_error(f"{field_name} is required")
            return result
        
        if value.lower() not in cls.COMPONENT_TYPES:
            result.add_warning(f"Unknown {field_name}: {value}")
            result.add_info(f"Known types: {', '.join(sorted(cls.COMPONENT_TYPES))}")
        
        return result
    
    @classmethod
    def validate_circuit_type(cls, value: str, field_name: str = "circuit_type") -> ValidationResult:
        """Validate circuit type"""
        result = ValidationResult()
        
        if value and value.lower() not in cls.CIRCUIT_TYPES:
            result.add_warning(f"Unknown {field_name}: {value}")
            result.add_info(f"Known types: {', '.join(sorted(cls.CIRCUIT_TYPES))}")
        
        return result
    
    @classmethod
    def validate_signal_type(cls, value: str, field_name: str = "signal_type") -> ValidationResult:
        """Validate signal type"""
        result = ValidationResult()
        
        if value and value.lower() not in cls.SIGNAL_TYPES:
            result.add_warning(f"Unknown {field_name}: {value}")
            result.add_info(f"Known types: {', '.join(sorted(cls.SIGNAL_TYPES))}")
        
        return result

class DataValidator:
    """High-level data validation"""
    
    @staticmethod
    def validate_component(data: Dict[str, Any]) -> ValidationResult:
        """Validate component data"""
        result = ValidationResult()
        
        # Required fields
        result.merge(FieldValidator.validate_id(data.get('id'), 'id'))
        result.merge(FieldValidator.validate_id(data.get('vehicle_signature'), 'vehicle_signature'))
        result.merge(FieldValidator.validate_component_type(data.get('type'), 'type'))
        
        # Optional fields
        if 'voltage' in data:
            result.merge(FieldValidator.validate_voltage(data['voltage']))
        
        if 'current_rating' in data:
            result.merge(FieldValidator.validate_current(data['current_rating']))
        
        if 'position' in data:
            result.merge(FieldValidator.validate_position(data['position']))
        
        return result
    
    @staticmethod
    def validate_connection(data: Dict[str, Any]) -> ValidationResult:
        """Validate connection data"""
        result = ValidationResult()
        
        # Required fields
        from_id = data.get('from')
        to_id = data.get('to')
        
        if not from_id:
            result.add_error("'from' component ID is required")
        else:
            result.merge(FieldValidator.validate_id(from_id, 'from'))
        
        if not to_id:
            result.add_error("'to' component ID is required")
        else:
            result.merge(FieldValidator.validate_id(to_id, 'to'))
        
        # Check for self-connection
        if from_id and to_id and from_id == to_id:
            result.add_warning("Self-connection detected")
        
        result.merge(FieldValidator.validate_id(data.get('vehicle_signature'), 'vehicle_signature'))
        
        # Optional fields
        if 'wire_gauge' in data:
            result.merge(FieldValidator.validate_wire_gauge(data['wire_gauge']))
        
        if 'wire_color' in data:
            result.merge(FieldValidator.validate_color(data['wire_color']))
        
        if 'signal_type' in data:
            result.merge(FieldValidator.validate_signal_type(data['signal_type']))
        
        return result
    
    @staticmethod
    def validate_circuit(data: Dict[str, Any]) -> ValidationResult:
        """Validate circuit data"""
        result = ValidationResult()
        
        # Required fields
        result.merge(FieldValidator.validate_id(data.get('id'), 'id'))
        result.merge(FieldValidator.validate_id(data.get('vehicle_signature'), 'vehicle_signature'))
        
        # Optional fields
        if 'circuit_type' in data:
            result.merge(FieldValidator.validate_circuit_type(data['circuit_type']))
        
        if 'max_current' in data:
            result.merge(FieldValidator.validate_current(data['max_current'], 'max_current'))
        
        if 'voltage' in data:
            result.merge(FieldValidator.validate_voltage(data['voltage']))
        
        return result
    
    @staticmethod
    def validate_batch(data_list: List[Dict[str, Any]], entity_type: str) -> ValidationResult:
        """Validate a batch of entities"""
        result = ValidationResult()
        
        if not data_list:
            result.add_error("No data provided")
            return result
        
        # Check for duplicate IDs
        ids_seen = set()
        duplicates = set()
        
        for i, item in enumerate(data_list):
            item_id = item.get('id')
            vehicle_sig = item.get('vehicle_signature')
            
            if item_id and vehicle_sig:
                unique_key = (item_id, vehicle_sig)
                if unique_key in ids_seen:
                    duplicates.add(item_id)
                    result.add_error(f"Duplicate ID at index {i}: {item_id}")
                else:
                    ids_seen.add(unique_key)
            
            # Validate individual item
            if entity_type == "component":
                item_result = DataValidator.validate_component(item)
            elif entity_type == "connection":
                item_result = DataValidator.validate_connection(item)
            elif entity_type == "circuit":
                item_result = DataValidator.validate_circuit(item)
            else:
                result.add_error(f"Unknown entity type: {entity_type}")
                continue
            
            # Add item-specific errors with index
            for error in item_result.errors:
                result.add_error(f"Item {i}: {error}")
            for warning in item_result.warnings:
                result.add_warning(f"Item {i}: {warning}")
        
        if duplicates:
            result.add_info(f"Found {len(duplicates)} duplicate IDs")
        
        return result
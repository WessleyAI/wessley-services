"""Data validation service for import operations"""

from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import re
from ..utils.neo4j_utils import Neo4jClient

class ValidationService:
    def __init__(self, neo4j_client: Neo4jClient):
        self.neo4j_client = neo4j_client
        
        # Define validation rules
        self.component_required_fields = {'id', 'vehicle_signature'}
        self.connection_required_fields = {'from', 'to', 'vehicle_signature'}
        self.circuit_required_fields = {'id', 'vehicle_signature'}
        
        # Define field patterns
        self.id_pattern = re.compile(r'^[a-zA-Z0-9_-]+$')
        self.voltage_pattern = re.compile(r'^\d+(\.\d+)?[vV]?$')
        self.current_pattern = re.compile(r'^\d+(\.\d+)?[aA]?$')
    
    async def validate_components(self, components: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate component data"""
        errors = []
        warnings = []
        duplicates = set()
        
        # Check for duplicates by ID
        ids_seen = set()
        
        for i, component in enumerate(components):
            component_id = component.get('id')
            vehicle_sig = component.get('vehicle_signature')
            
            # Check required fields
            missing_fields = self.component_required_fields - set(component.keys())
            if missing_fields:
                errors.append(f"Component {i}: Missing required fields: {missing_fields}")
            
            # Check ID format
            if component_id and not self.id_pattern.match(component_id):
                errors.append(f"Component {i}: Invalid ID format: {component_id}")
            
            # Check for duplicates
            unique_key = (component_id, vehicle_sig)
            if unique_key in ids_seen:
                duplicates.add(component_id)
                errors.append(f"Component {i}: Duplicate ID: {component_id}")
            else:
                ids_seen.add(unique_key)
            
            # Validate specific fields
            self._validate_component_fields(component, i, errors, warnings)
        
        # Check existing components in database
        existing_conflicts = await self._check_existing_components(list(ids_seen))
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "duplicates": list(duplicates),
            "existing_conflicts": existing_conflicts,
            "total_components": len(components)
        }
    
    async def validate_connections(self, connections: List[Dict[str, Any]], 
                                 component_ids: Set[str] = None) -> Dict[str, Any]:
        """Validate connection data"""
        errors = []
        warnings = []
        orphaned = []
        
        for i, connection in enumerate(connections):
            # Check required fields
            missing_fields = self.connection_required_fields - set(connection.keys())
            if missing_fields:
                errors.append(f"Connection {i}: Missing required fields: {missing_fields}")
            
            from_id = connection.get('from')
            to_id = connection.get('to')
            
            # Check self-connections
            if from_id == to_id:
                warnings.append(f"Connection {i}: Self-connection detected: {from_id}")
            
            # Check if components exist (if component_ids provided)
            if component_ids:
                if from_id and from_id not in component_ids:
                    orphaned.append(f"Connection {i}: Source component not found: {from_id}")
                if to_id and to_id not in component_ids:
                    orphaned.append(f"Connection {i}: Target component not found: {to_id}")
            
            # Validate connection fields
            self._validate_connection_fields(connection, i, errors, warnings)
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "orphaned_connections": orphaned,
            "total_connections": len(connections)
        }
    
    async def validate_circuits(self, circuits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate circuit data"""
        errors = []
        warnings = []
        
        for i, circuit in enumerate(circuits):
            # Check required fields
            missing_fields = self.circuit_required_fields - set(circuit.keys())
            if missing_fields:
                errors.append(f"Circuit {i}: Missing required fields: {missing_fields}")
            
            # Validate circuit fields
            self._validate_circuit_fields(circuit, i, errors, warnings)
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "total_circuits": len(circuits)
        }
    
    def _validate_component_fields(self, component: Dict[str, Any], index: int, 
                                 errors: List[str], warnings: List[str]):
        """Validate specific component fields"""
        # Validate voltage
        if 'voltage' in component:
            voltage = str(component['voltage'])
            if not self.voltage_pattern.match(voltage):
                warnings.append(f"Component {index}: Invalid voltage format: {voltage}")
        
        # Validate current
        if 'current_rating' in component:
            current = str(component['current_rating'])
            if not self.current_pattern.match(current):
                warnings.append(f"Component {index}: Invalid current format: {current}")
        
        # Check position format
        if 'position' in component:
            position = component['position']
            if isinstance(position, list) and len(position) != 3:
                warnings.append(f"Component {index}: Position should be [x, y, z] format")
    
    def _validate_connection_fields(self, connection: Dict[str, Any], index: int,
                                  errors: List[str], warnings: List[str]):
        """Validate specific connection fields"""
        # Validate wire gauge
        if 'wire_gauge' in connection:
            gauge = str(connection['wire_gauge'])
            if not re.match(r'^\d+(\.\d+)?(mm²?|awg)?$', gauge, re.IGNORECASE):
                warnings.append(f"Connection {index}: Invalid wire gauge format: {gauge}")
        
        # Validate signal type
        valid_signal_types = {'power', 'ground', 'data', 'analog', 'digital', 'can', 'lin'}
        if 'signal_type' in connection:
            signal_type = connection['signal_type'].lower()
            if signal_type not in valid_signal_types:
                warnings.append(f"Connection {index}: Unknown signal type: {signal_type}")
    
    def _validate_circuit_fields(self, circuit: Dict[str, Any], index: int,
                               errors: List[str], warnings: List[str]):
        """Validate specific circuit fields"""
        # Validate max current
        if 'max_current' in circuit:
            max_current = circuit['max_current']
            if isinstance(max_current, (int, float)) and max_current <= 0:
                errors.append(f"Circuit {index}: Max current must be positive")
        
        # Validate circuit type
        valid_circuit_types = {'power', 'lighting', 'ignition', 'engine', 'body', 'comfort'}
        if 'circuit_type' in circuit:
            circuit_type = circuit['circuit_type'].lower()
            if circuit_type not in valid_circuit_types:
                warnings.append(f"Circuit {index}: Unknown circuit type: {circuit_type}")
    
    async def _check_existing_components(self, component_keys: List[tuple]) -> List[str]:
        """Check for existing components in database"""
        if not component_keys:
            return []
        
        query = """
        UNWIND $keys as key
        MATCH (c:Component {id: key[0], vehicle_signature: key[1]})
        RETURN c.id as existing_id
        """
        
        try:
            result = await self.neo4j_client.run(query, keys=component_keys)
            return [record['existing_id'] for record in result]
        except Exception:
            return []
    
    async def validate_vehicle_signature(self, vehicle_signature: str) -> Dict[str, Any]:
        """Validate vehicle signature format and existence"""
        # Check format
        if not re.match(r'^[a-zA-Z0-9_-]+$', vehicle_signature):
            return {
                "valid": False,
                "error": "Invalid vehicle signature format"
            }
        
        # Check if vehicle exists
        query = """
        MATCH (v:Vehicle {signature: $signature})
        RETURN count(v) as count
        """
        
        try:
            result = await self.neo4j_client.run(query, signature=vehicle_signature)
            exists = result[0]['count'] > 0 if result else False
            
            return {
                "valid": True,
                "exists": exists,
                "signature": vehicle_signature
            }
        except Exception as e:
            return {
                "valid": False,
                "error": f"Database validation failed: {str(e)}"
            }
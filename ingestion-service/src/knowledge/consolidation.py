"""
Knowledge consolidation and normalization for automotive electronics.

This module provides centralized knowledge management, normalization tables,
and consistency validation for automotive electrical systems.
"""
import json
import re
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from ..core.logging import StructuredLogger
from ..semantic.ontology import (
    AutomotiveComponent, ElectricalNet, AutomotiveConnector, 
    AutomotiveGround, AutomotiveFuseRelay, VehicleSignature,
    ComponentType, SystemType, ElectricalRole
)

logger = StructuredLogger(__name__)


@dataclass
class ComponentTypeInfo:
    """Normalized component type information."""
    id: str
    name: str
    category: str
    typical_pin_count: int
    pin_roles: List[str]
    typical_values: List[str]
    voltage_levels: List[str] = field(default_factory=list)
    packages: List[str] = field(default_factory=list)
    systems: List[SystemType] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "typical_pin_count": self.typical_pin_count,
            "pin_roles": self.pin_roles,
            "typical_values": self.typical_values,
            "voltage_levels": self.voltage_levels,
            "packages": self.packages,
            "systems": [s.value for s in self.systems]
        }


@dataclass
class FusePanelInfo:
    """Fuse panel information for specific vehicle."""
    vehicle_signature: str
    panel_name: str
    location: str
    slot_map: Dict[str, Dict[str, Any]]  # slot -> {rating, circuit, description}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "vehicle_signature": self.vehicle_signature,
            "panel_name": self.panel_name,
            "location": self.location,
            "slot_map": self.slot_map
        }


@dataclass
class GroundInfo:
    """Ground point information for specific vehicle."""
    vehicle_signature: str
    code: str
    location_hint: str
    systems: List[SystemType]
    resistance_spec: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "vehicle_signature": self.vehicle_signature,
            "code": self.code,
            "location_hint": self.location_hint,
            "systems": [s.value for s in self.systems],
            "resistance_spec": self.resistance_spec
        }


@dataclass
class WireInfo:
    """Wire specification information."""
    color_code: str
    gauge_awg: str
    gauge_metric: str
    current_rating: str
    typical_role: str
    voltage_rating: str = "12V"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "color_code": self.color_code,
            "gauge_awg": self.gauge_awg,
            "gauge_metric": self.gauge_metric,
            "current_rating": self.current_rating,
            "typical_role": self.typical_role,
            "voltage_rating": self.voltage_rating
        }


class KnowledgeBase:
    """Centralized automotive electronics knowledge base."""
    
    def __init__(self):
        """Initialize knowledge base with default data."""
        self.component_types: Dict[str, ComponentTypeInfo] = {}
        self.fuse_panels: Dict[str, List[FusePanelInfo]] = {}  # vehicle_sig -> panels
        self.grounds: Dict[str, List[GroundInfo]] = {}  # vehicle_sig -> grounds
        self.wires: Dict[str, WireInfo] = {}  # color_code -> info
        
        self._initialize_component_types()
        self._initialize_wire_specifications()
        self._initialize_common_grounds()
        self._initialize_fuse_panels()
        
        logger.info("KnowledgeBase initialized with automotive electronics data")
    
    def _initialize_component_types(self):
        """Initialize component type normalization table."""
        self.component_types = {
            "relay": ComponentTypeInfo(
                id="relay",
                name="Automotive Relay",
                category="switching",
                typical_pin_count=5,
                pin_roles=["coil_positive", "coil_negative", "common", "normally_open", "normally_closed"],
                typical_values=["30A", "40A", "60A"],
                voltage_levels=["12V"],
                packages=["mini", "micro", "standard"],
                systems=[SystemType.STARTING, SystemType.FUEL, SystemType.LIGHTING]
            ),
            "fuse": ComponentTypeInfo(
                id="fuse",
                name="Automotive Fuse",
                category="protection",
                typical_pin_count=2,
                pin_roles=["input", "output"],
                typical_values=["5A", "10A", "15A", "20A", "25A", "30A", "40A", "60A"],
                voltage_levels=["12V", "24V"],
                packages=["mini", "standard", "maxi"],
                systems=[SystemType.POWER_DISTRIBUTION]
            ),
            "resistor": ComponentTypeInfo(
                id="resistor",
                name="Resistor",
                category="passive",
                typical_pin_count=2,
                pin_roles=["terminal1", "terminal2"],
                typical_values=["1k", "10k", "100k", "1M"],
                voltage_levels=["5V", "12V"],
                packages=["0603", "0805", "1206", "through_hole"],
                systems=[SystemType.ENGINE_MANAGEMENT, SystemType.INSTRUMENT_CLUSTER]
            ),
            "capacitor": ComponentTypeInfo(
                id="capacitor",
                name="Capacitor",
                category="passive",
                typical_pin_count=2,
                pin_roles=["positive", "negative"],
                typical_values=["100nF", "1µF", "10µF", "100µF"],
                voltage_levels=["5V", "12V", "16V", "25V"],
                packages=["0603", "0805", "1206", "electrolytic"],
                systems=[SystemType.ENGINE_MANAGEMENT, SystemType.AUDIO]
            ),
            "ecu": ComponentTypeInfo(
                id="ecu",
                name="Electronic Control Unit",
                category="control",
                typical_pin_count=64,
                pin_roles=["power", "ground", "can_h", "can_l", "analog_input", "digital_output"],
                typical_values=["ECM", "PCM", "TCM", "BCM"],
                voltage_levels=["5V", "12V"],
                packages=["connector_64", "connector_80", "connector_104"],
                systems=[SystemType.ENGINE_MANAGEMENT, SystemType.TRANSMISSION, SystemType.BODY_CONTROL]
            ),
            "sensor": ComponentTypeInfo(
                id="sensor",
                name="Automotive Sensor",
                category="input",
                typical_pin_count=3,
                pin_roles=["power", "ground", "signal"],
                typical_values=["temperature", "pressure", "position", "speed"],
                voltage_levels=["5V", "12V"],
                packages=["3_pin", "4_pin", "integrated"],
                systems=[SystemType.ENGINE_MANAGEMENT, SystemType.TRANSMISSION]
            )
        }
    
    def _initialize_wire_specifications(self):
        """Initialize wire gauge and color specifications."""
        self.wires = {
            "black": WireInfo("black", "18", "0.75mm²", "10A", "ground", "12V"),
            "red": WireInfo("red", "16", "1.25mm²", "15A", "power", "12V"),
            "white": WireInfo("white", "18", "0.75mm²", "10A", "signal", "12V"),
            "green": WireInfo("green", "18", "0.75mm²", "10A", "signal", "12V"),
            "blue": WireInfo("blue", "18", "0.75mm²", "10A", "signal", "12V"),
            "yellow": WireInfo("yellow", "16", "1.25mm²", "15A", "power", "12V"),
            "brown": WireInfo("brown", "18", "0.75mm²", "10A", "signal", "12V"),
            "orange": WireInfo("orange", "16", "1.25mm²", "15A", "power", "12V"),
            "purple": WireInfo("purple", "18", "0.75mm²", "10A", "signal", "12V"),
            "pink": WireInfo("pink", "18", "0.75mm²", "10A", "signal", "12V"),
            "gray": WireInfo("gray", "18", "0.75mm²", "10A", "signal", "12V"),
            "red_black": WireInfo("red_black", "14", "2.0mm²", "20A", "power", "12V"),
            "white_black": WireInfo("white_black", "16", "1.25mm²", "15A", "signal", "12V"),
            "yellow_black": WireInfo("yellow_black", "14", "2.0mm²", "20A", "power", "12V"),
        }
    
    def _initialize_common_grounds(self):
        """Initialize common ground point information."""
        # Example for common automotive ground points
        common_grounds = [
            GroundInfo("generic", "G101", "Engine block", [SystemType.ENGINE_MANAGEMENT]),
            GroundInfo("generic", "G102", "Chassis rail", [SystemType.STARTING, SystemType.CHARGING]),
            GroundInfo("generic", "G103", "Fuel tank", [SystemType.FUEL]),
            GroundInfo("generic", "G104", "Dashboard", [SystemType.INSTRUMENT_CLUSTER]),
            GroundInfo("generic", "G105", "Door frame", [SystemType.BODY_CONTROL]),
            GroundInfo("generic", "E1", "ECM ground 1", [SystemType.ENGINE_MANAGEMENT]),
            GroundInfo("generic", "E2", "ECM ground 2", [SystemType.ENGINE_MANAGEMENT]),
        ]
        
        self.grounds["generic"] = common_grounds
    
    def _initialize_fuse_panels(self):
        """Initialize common fuse panel configurations."""
        # Example fuse panel for generic vehicle
        generic_panel = FusePanelInfo(
            vehicle_signature="generic",
            panel_name="Main Fuse Panel",
            location="Engine Bay",
            slot_map={
                "1": {"rating": "30A", "circuit": "MAIN", "description": "Main power"},
                "2": {"rating": "15A", "circuit": "IG1", "description": "Ignition 1"},
                "3": {"rating": "10A", "circuit": "IG2", "description": "Ignition 2"},
                "4": {"rating": "20A", "circuit": "ACC", "description": "Accessory"},
                "5": {"rating": "15A", "circuit": "FUEL", "description": "Fuel pump"},
                "6": {"rating": "10A", "circuit": "ECM", "description": "Engine control"},
                "7": {"rating": "15A", "circuit": "LIGHTS", "description": "Lighting"},
                "8": {"rating": "20A", "circuit": "FAN", "description": "Cooling fan"},
                "9": {"rating": "10A", "circuit": "HORN", "description": "Horn"},
                "10": {"rating": "15A", "circuit": "AUDIO", "description": "Audio system"}
            }
        )
        
        self.fuse_panels["generic"] = [generic_panel]
    
    def normalize_component_id(self, component_id: str, vehicle: Optional[VehicleSignature] = None) -> str:
        """
        Normalize component ID to standard format.
        
        Args:
            component_id: Original component ID
            vehicle: Vehicle context for normalization
            
        Returns:
            Normalized component ID
        """
        if not component_id:
            return component_id
        
        # Remove common prefixes/suffixes
        normalized = component_id.strip().upper()
        
        # Standardize common patterns
        patterns = [
            (r"^RELAY\s*([KR]?\d+)", r"K\1"),  # "RELAY K1" -> "K1"
            (r"^FUSE\s*([F]?\d+)", r"F\1"),    # "FUSE F10" -> "F10"
            (r"^([KFR])[-_](\d+)", r"\1\2"),   # "K-1" -> "K1"
            (r"^([A-Z]+)(\d+)([A-Z]*)$", r"\1\2\3"),  # Ensure consistent format
        ]
        
        for pattern, replacement in patterns:
            normalized = re.sub(pattern, replacement, normalized)
        
        return normalized
    
    def normalize_net_name(self, net_name: str, vehicle: Optional[VehicleSignature] = None) -> str:
        """
        Normalize electrical net name to standard format.
        
        Args:
            net_name: Original net name
            vehicle: Vehicle context for normalization
            
        Returns:
            Normalized net name
        """
        if not net_name:
            return net_name
        
        normalized = net_name.strip().upper()
        
        # Standardize common net names
        substitutions = {
            "BATTERY": "BATT",
            "GROUND": "GND",
            "IGNITION1": "IG1",
            "IGNITION2": "IG2",
            "ACCESSORY": "ACC",
            "POWER": "PWR",
            "SIGNAL": "SIG",
        }
        
        for old, new in substitutions.items():
            normalized = normalized.replace(old, new)
        
        # Remove common prefixes/suffixes
        normalized = re.sub(r"^(NET|SIGNAL|LINE)[-_]?", "", normalized)
        normalized = re.sub(r"[-_]?(NET|SIGNAL|LINE)$", "", normalized)
        
        return normalized
    
    def get_component_type_info(self, component_type: str) -> Optional[ComponentTypeInfo]:
        """Get normalized component type information."""
        return self.component_types.get(component_type.lower())
    
    def get_expected_pin_count(self, component_type: str) -> Optional[int]:
        """Get expected pin count for component type."""
        info = self.get_component_type_info(component_type)
        return info.typical_pin_count if info else None
    
    def get_expected_pin_roles(self, component_type: str) -> List[str]:
        """Get expected pin roles for component type."""
        info = self.get_component_type_info(component_type)
        return info.pin_roles if info else []
    
    def get_wire_info(self, color_code: str) -> Optional[WireInfo]:
        """Get wire specification by color code."""
        return self.wires.get(color_code.lower().replace(" ", "_"))
    
    def get_fuse_panels(self, vehicle: VehicleSignature) -> List[FusePanelInfo]:
        """Get fuse panel information for vehicle."""
        vehicle_sig = vehicle.to_string()
        
        # Try specific vehicle first, then generic
        if vehicle_sig in self.fuse_panels:
            return self.fuse_panels[vehicle_sig]
        elif "generic" in self.fuse_panels:
            return self.fuse_panels["generic"]
        else:
            return []
    
    def get_ground_points(self, vehicle: VehicleSignature) -> List[GroundInfo]:
        """Get ground point information for vehicle."""
        vehicle_sig = vehicle.to_string()
        
        # Try specific vehicle first, then generic
        if vehicle_sig in self.grounds:
            return self.grounds[vehicle_sig]
        elif "generic" in self.grounds:
            return self.grounds["generic"]
        else:
            return []
    
    def validate_component_consistency(self, component: AutomotiveComponent) -> List[str]:
        """
        Validate component against knowledge base.
        
        Args:
            component: Component to validate
            
        Returns:
            List of validation issues
        """
        issues = []
        
        # Get component type info
        type_info = self.get_component_type_info(component.type.value)
        if not type_info:
            issues.append(f"Unknown component type: {component.type.value}")
            return issues
        
        # Check pin count
        if component.pins:
            expected_pins = type_info.typical_pin_count
            actual_pins = len(component.pins)
            
            if expected_pins > 0 and actual_pins != expected_pins:
                # Allow some tolerance for components with variable pin counts
                tolerance = 0.2  # 20% tolerance
                min_pins = int(expected_pins * (1 - tolerance))
                max_pins = int(expected_pins * (1 + tolerance))
                
                if not (min_pins <= actual_pins <= max_pins):
                    issues.append(f"Pin count mismatch: expected ~{expected_pins}, got {actual_pins}")
        
        # Check pin roles
        if component.pins and type_info.pin_roles:
            expected_roles = set(type_info.pin_roles)
            actual_roles = set(pin.electrical_role.value for pin in component.pins 
                             if pin.electrical_role)
            
            # Check for missing critical roles
            critical_roles = {"power", "ground"}
            missing_critical = critical_roles & expected_roles - actual_roles
            if missing_critical:
                issues.append(f"Missing critical pin roles: {missing_critical}")
        
        # Check value format
        if component.value and type_info.typical_values:
            if not self._is_valid_component_value(component.value, type_info.typical_values):
                issues.append(f"Unusual component value: {component.value}")
        
        # Check system assignment
        if component.system and type_info.systems:
            if component.system not in type_info.systems:
                issues.append(f"Unusual system assignment: {component.type.value} in {component.system.value}")
        
        return issues
    
    def suggest_missing_connections(self, 
                                  components: List[AutomotiveComponent],
                                  nets: List[ElectricalNet],
                                  vehicle: Optional[VehicleSignature] = None) -> List[str]:
        """
        Suggest likely missing connections based on domain knowledge.
        
        Args:
            components: List of components
            nets: List of electrical nets
            vehicle: Vehicle context
            
        Returns:
            List of suggested connections
        """
        suggestions = []
        
        # Organize components by type
        components_by_type = {}
        for comp in components:
            comp_type = comp.type.value
            if comp_type not in components_by_type:
                components_by_type[comp_type] = []
            components_by_type[comp_type].append(comp)
        
        # Check for missing power distribution
        relays = components_by_type.get("relay", [])
        fuses = components_by_type.get("fuse", [])
        
        for relay in relays:
            # Relays should typically have fuse protection
            related_fuses = [f for f in fuses if f.system == relay.system]
            if not related_fuses:
                suggestions.append(f"Relay {relay.id} may need fuse protection")
        
        # Check for missing ECU connections
        ecus = components_by_type.get("ecu", [])
        for ecu in ecus:
            # ECUs should have power, ground, and communication
            power_pins = [p for p in ecu.pins if p.electrical_role == ElectricalRole.POWER]
            ground_pins = [p for p in ecu.pins if p.electrical_role == ElectricalRole.GROUND]
            can_pins = [p for p in ecu.pins if p.electrical_role in [ElectricalRole.CAN_H, ElectricalRole.CAN_L]]
            
            if not power_pins:
                suggestions.append(f"ECU {ecu.id} missing power connections")
            if not ground_pins:
                suggestions.append(f"ECU {ecu.id} missing ground connections")
            if not can_pins and ecu.system in [SystemType.ENGINE_MANAGEMENT, SystemType.TRANSMISSION]:
                suggestions.append(f"ECU {ecu.id} may need CAN bus connections")
        
        # Check for missing ground distribution
        power_nets = [n for n in nets if n.is_power_net]
        ground_nets = [n for n in nets if n.is_ground_net]
        
        if power_nets and not ground_nets:
            suggestions.append("Power nets found but no ground nets - check ground distribution")
        
        # Check for system-specific missing connections
        if vehicle:
            ground_points = self.get_ground_points(vehicle)
            ground_codes = {g.code for g in ground_points}
            
            # Suggest missing ground connections for high-current systems
            starting_components = [c for c in components if c.system == SystemType.STARTING]
            if starting_components:
                starter_grounds = {"G101", "G102"}
                missing_grounds = starter_grounds - ground_codes
                if missing_grounds:
                    suggestions.append(f"Starting system may need ground points: {missing_grounds}")
        
        return suggestions
    
    def infer_component_properties(self, component: AutomotiveComponent) -> AutomotiveComponent:
        """
        Infer missing component properties based on knowledge base.
        
        Args:
            component: Component to enhance
            
        Returns:
            Enhanced component with inferred properties
        """
        # Get type information
        type_info = self.get_component_type_info(component.type.value)
        if not type_info:
            return component
        
        # Infer system if not set
        if not component.system and type_info.systems:
            # Use most common system for this component type
            component.system = type_info.systems[0]
        
        # Infer pin roles if pins exist but roles are missing
        if component.pins and type_info.pin_roles:
            expected_roles = type_info.pin_roles
            
            for i, pin in enumerate(component.pins):
                if not pin.electrical_role and i < len(expected_roles):
                    # Map to expected role by position
                    role_name = expected_roles[i]
                    
                    # Map role names to ElectricalRole enum
                    role_mapping = {
                        "power": ElectricalRole.POWER,
                        "ground": ElectricalRole.GROUND,
                        "coil_positive": ElectricalRole.POWER,
                        "coil_negative": ElectricalRole.SIGNAL,
                        "common": ElectricalRole.POWER,
                        "normally_open": ElectricalRole.POWER,
                        "signal": ElectricalRole.SIGNAL,
                        "terminal1": ElectricalRole.SIGNAL,
                        "terminal2": ElectricalRole.SIGNAL,
                        "positive": ElectricalRole.POWER,
                        "negative": ElectricalRole.GROUND,
                    }
                    
                    if role_name in role_mapping:
                        pin.electrical_role = role_mapping[role_name]
        
        return component
    
    def _is_valid_component_value(self, value: str, typical_values: List[str]) -> bool:
        """Check if component value matches expected patterns."""
        value_lower = value.lower()
        
        # Check against typical values
        for typical in typical_values:
            if typical.lower() in value_lower or value_lower in typical.lower():
                return True
        
        # Check patterns for different component types
        patterns = [
            r"\d+[kKmM]?[Ω\u03A9]?",  # Resistance values
            r"\d+[\.,]?\d*[uµnpmk]?[FH]",  # Capacitance/Inductance
            r"\d+[Aa]",  # Current ratings
            r"\d+[Vv]",  # Voltage ratings
        ]
        
        for pattern in patterns:
            if re.search(pattern, value):
                return True
        
        return False
    
    def add_vehicle_specific_data(self, vehicle: VehicleSignature, data: Dict[str, Any]):
        """
        Add vehicle-specific knowledge data.
        
        Args:
            vehicle: Vehicle signature
            data: Knowledge data (fuse_panels, grounds, etc.)
        """
        vehicle_sig = vehicle.to_string()
        
        # Add fuse panels
        if "fuse_panels" in data:
            if vehicle_sig not in self.fuse_panels:
                self.fuse_panels[vehicle_sig] = []
            
            for panel_data in data["fuse_panels"]:
                panel = FusePanelInfo(
                    vehicle_signature=vehicle_sig,
                    panel_name=panel_data["panel_name"],
                    location=panel_data["location"],
                    slot_map=panel_data["slot_map"]
                )
                self.fuse_panels[vehicle_sig].append(panel)
        
        # Add ground points
        if "grounds" in data:
            if vehicle_sig not in self.grounds:
                self.grounds[vehicle_sig] = []
            
            for ground_data in data["grounds"]:
                ground = GroundInfo(
                    vehicle_signature=vehicle_sig,
                    code=ground_data["code"],
                    location_hint=ground_data["location_hint"],
                    systems=[SystemType(s) for s in ground_data.get("systems", [])],
                    resistance_spec=ground_data.get("resistance_spec")
                )
                self.grounds[vehicle_sig].append(ground)
        
        logger.info(f"Added vehicle-specific data for {vehicle_sig}")
    
    def export_knowledge_base(self) -> Dict[str, Any]:
        """Export entire knowledge base to dictionary."""
        return {
            "component_types": {k: v.to_dict() for k, v in self.component_types.items()},
            "fuse_panels": {
                k: [panel.to_dict() for panel in panels] 
                for k, panels in self.fuse_panels.items()
            },
            "grounds": {
                k: [ground.to_dict() for ground in grounds]
                for k, grounds in self.grounds.items()
            },
            "wires": {k: v.to_dict() for k, v in self.wires.items()},
            "export_timestamp": datetime.utcnow().isoformat()
        }
    
    def import_knowledge_base(self, data: Dict[str, Any]):
        """Import knowledge base from dictionary."""
        # Import component types
        if "component_types" in data:
            for type_id, type_data in data["component_types"].items():
                self.component_types[type_id] = ComponentTypeInfo(
                    id=type_data["id"],
                    name=type_data["name"],
                    category=type_data["category"],
                    typical_pin_count=type_data["typical_pin_count"],
                    pin_roles=type_data["pin_roles"],
                    typical_values=type_data["typical_values"],
                    voltage_levels=type_data.get("voltage_levels", []),
                    packages=type_data.get("packages", []),
                    systems=[SystemType(s) for s in type_data.get("systems", [])]
                )
        
        # Import other data...
        logger.info("Imported knowledge base data")


# Global knowledge base instance
_global_knowledge_base: Optional[KnowledgeBase] = None

def get_knowledge_base() -> KnowledgeBase:
    """Get global knowledge base."""
    global _global_knowledge_base
    if _global_knowledge_base is None:
        _global_knowledge_base = KnowledgeBase()
    return _global_knowledge_base
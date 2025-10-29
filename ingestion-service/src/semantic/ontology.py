"""
Automotive electronics domain ontology and object model.

This module defines the core automotive electronics entities, relationships,
and domain-specific knowledge for structured data extraction and semantic search.
"""
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime

# Core Enums for Automotive Electronics Domain

class ComponentType(str, Enum):
    """Automotive electronic component types."""
    RESISTOR = "resistor"
    CAPACITOR = "capacitor" 
    POLARIZED_CAP = "polarized_cap"
    INDUCTOR = "inductor"
    DIODE = "diode"
    ZENER = "zener"
    BJT_NPN = "bjt_npn"
    BJT_PNP = "bjt_pnp"
    MOSFET_N = "mosfet_n"
    MOSFET_P = "mosfet_p"
    OPAMP = "opamp"
    GROUND = "ground"
    POWER_FLAG = "power_flag"
    CONNECTOR = "connector"
    IC = "ic"
    FUSE = "fuse"
    RELAY = "relay"
    LAMP = "lamp"
    SWITCH = "switch"
    NET_LABEL = "net_label"
    JUNCTION = "junction"
    ARROW = "arrow"
    ECU = "ecu"
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    MOTOR = "motor"
    SOLENOID = "solenoid"
    TRANSFORMER = "transformer"
    CRYSTAL = "crystal"
    BATTERY = "battery"


class ElectricalRole(str, Enum):
    """Pin electrical roles in automotive circuits."""
    POWER = "power"
    GROUND = "ground"
    SIGNAL = "signal"
    CAN_H = "can_h"
    CAN_L = "can_l"
    LIN = "lin"
    K_LINE = "k_line"
    ANALOG_INPUT = "analog_input"
    DIGITAL_INPUT = "digital_input"
    ANALOG_OUTPUT = "analog_output"
    DIGITAL_OUTPUT = "digital_output"
    PWM = "pwm"
    REFERENCE = "reference"
    SHIELD = "shield"
    NC = "nc"  # No Connection


class BusType(str, Enum):
    """Automotive communication bus types."""
    CAN = "can"
    LIN = "lin"
    K_LINE = "k_line"
    MOST = "most"
    FLEXRAY = "flexray"
    ETHERNET = "ethernet"
    SPI = "spi"
    I2C = "i2c"
    UART = "uart"


class SystemType(str, Enum):
    """Automotive electrical systems."""
    STARTING = "starting"
    CHARGING = "charging"
    IGNITION = "ignition"
    FUEL = "fuel"
    ENGINE_MANAGEMENT = "engine_management"
    TRANSMISSION = "transmission"
    ABS = "abs"
    AIRBAG = "airbag"
    LIGHTING = "lighting"
    CLIMATE = "climate"
    AUDIO = "audio"
    NAVIGATION = "navigation"
    SECURITY = "security"
    BODY_CONTROL = "body_control"
    INSTRUMENT_CLUSTER = "instrument_cluster"
    POWER_DISTRIBUTION = "power_distribution"
    GROUND_DISTRIBUTION = "ground_distribution"


class WireColor(str, Enum):
    """Standard automotive wire colors."""
    BLACK = "black"
    RED = "red"
    WHITE = "white"
    GREEN = "green"
    BLUE = "blue"
    YELLOW = "yellow"
    BROWN = "brown"
    ORANGE = "orange"
    PURPLE = "purple"
    PINK = "pink"
    GRAY = "gray"
    VIOLET = "violet"
    TAN = "tan"
    # Color combinations
    RED_BLACK = "red_black"
    WHITE_BLACK = "white_black"
    GREEN_BLACK = "green_black"
    BLUE_BLACK = "blue_black"
    YELLOW_BLACK = "yellow_black"


# Core Domain Entities

@dataclass
class VehicleSignature:
    """Vehicle identification for context."""
    make: str
    model: str
    year: int
    market: Optional[str] = None
    engine: Optional[str] = None
    transmission: Optional[str] = None
    
    def to_string(self) -> str:
        """Create normalized string representation."""
        base = f"{self.make}_{self.model}_{self.year}"
        if self.market:
            base += f"_{self.market}"
        return base.lower().replace(" ", "_")


@dataclass
class ElectricalPin:
    """Pin on an automotive component."""
    name: str
    index: Optional[int] = None
    electrical_role: Optional[ElectricalRole] = None
    voltage_level: Optional[str] = None
    current_rating: Optional[str] = None
    description: Optional[str] = None
    
    def __post_init__(self):
        if self.electrical_role is None:
            # Infer role from common pin names
            name_lower = self.name.lower()
            if name_lower in ['vcc', 'vdd', 'vbat', 'power', '+12v', '+5v', '+3.3v']:
                self.electrical_role = ElectricalRole.POWER
            elif name_lower in ['gnd', 'ground', 'vss', 'earth']:
                self.electrical_role = ElectricalRole.GROUND
            elif 'can_h' in name_lower or 'canh' in name_lower:
                self.electrical_role = ElectricalRole.CAN_H
            elif 'can_l' in name_lower or 'canl' in name_lower:
                self.electrical_role = ElectricalRole.CAN_L
            elif 'lin' in name_lower:
                self.electrical_role = ElectricalRole.LIN
            else:
                self.electrical_role = ElectricalRole.SIGNAL


@dataclass
class ElectricalNet:
    """Electrical net/signal in automotive schematic."""
    name: str
    voltage_hint: Optional[str] = None
    bus_type: Optional[BusType] = None
    system: Optional[SystemType] = None
    description: Optional[str] = None
    is_power_net: bool = False
    is_ground_net: bool = False
    
    def __post_init__(self):
        # Infer properties from net name
        name_lower = self.name.lower()
        
        # Power nets
        if any(power in name_lower for power in ['vcc', 'vdd', 'vbat', 'ig1', 'ig2', 'acc', '+12v', '+5v']):
            self.is_power_net = True
            
        # Ground nets
        if any(gnd in name_lower for gnd in ['gnd', 'ground', 'earth', 'chassis']):
            self.is_ground_net = True
            
        # Bus types
        if 'can' in name_lower:
            self.bus_type = BusType.CAN
        elif 'lin' in name_lower:
            self.bus_type = BusType.LIN
        elif 'k-line' in name_lower or 'kline' in name_lower:
            self.bus_type = BusType.K_LINE


@dataclass
class AutomotiveComponent:
    """Automotive electronic component with domain knowledge."""
    id: str
    type: ComponentType
    value: Optional[str] = None
    rating: Optional[str] = None
    location_hint: Optional[str] = None
    system: Optional[SystemType] = None
    pins: List[ElectricalPin] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None
    
    def __post_init__(self):
        # Infer system from component ID patterns
        if not self.system:
            id_upper = self.id.upper()
            if any(prefix in id_upper for prefix in ['ECM', 'PCM', 'ECU']):
                self.system = SystemType.ENGINE_MANAGEMENT
            elif any(prefix in id_upper for prefix in ['K', 'RELAY']):
                if 'START' in id_upper:
                    self.system = SystemType.STARTING
                elif 'FUEL' in id_upper:
                    self.system = SystemType.FUEL
            elif any(prefix in id_upper for prefix in ['F', 'FUSE']):
                self.system = SystemType.POWER_DISTRIBUTION
            elif any(prefix in id_upper for prefix in ['G', 'GND']):
                self.system = SystemType.GROUND_DISTRIBUTION
    
    def get_expected_pins(self) -> List[ElectricalPin]:
        """Get expected pins based on component type."""
        if self.type == ComponentType.RELAY:
            return [
                ElectricalPin("85", electrical_role=ElectricalRole.SIGNAL),  # Coil -
                ElectricalPin("86", electrical_role=ElectricalRole.POWER),   # Coil +
                ElectricalPin("30", electrical_role=ElectricalRole.POWER),   # Common
                ElectricalPin("87", electrical_role=ElectricalRole.POWER),   # NO
                ElectricalPin("87a", electrical_role=ElectricalRole.POWER),  # NC (optional)
            ]
        elif self.type == ComponentType.OPAMP:
            return [
                ElectricalPin("1", electrical_role=ElectricalRole.SIGNAL),   # Output
                ElectricalPin("2", electrical_role=ElectricalRole.SIGNAL),   # - Input
                ElectricalPin("3", electrical_role=ElectricalRole.SIGNAL),   # + Input
                ElectricalPin("4", electrical_role=ElectricalRole.GROUND),   # V-
                ElectricalPin("7", electrical_role=ElectricalRole.POWER),    # V+
            ]
        elif self.type in [ComponentType.RESISTOR, ComponentType.CAPACITOR, ComponentType.INDUCTOR]:
            return [
                ElectricalPin("1", electrical_role=ElectricalRole.SIGNAL),
                ElectricalPin("2", electrical_role=ElectricalRole.SIGNAL),
            ]
        elif self.type == ComponentType.DIODE:
            return [
                ElectricalPin("A", electrical_role=ElectricalRole.SIGNAL),   # Anode
                ElectricalPin("K", electrical_role=ElectricalRole.SIGNAL),   # Cathode
            ]
        else:
            return []


@dataclass  
class AutomotiveConnector:
    """Automotive electrical connector."""
    code: str
    cavity_count: int
    location_hint: Optional[str] = None
    pins: List[ElectricalPin] = field(default_factory=list)
    mating_connector: Optional[str] = None
    color: Optional[str] = None
    
    def get_pin_by_cavity(self, cavity: int) -> Optional[ElectricalPin]:
        """Get pin by cavity number."""
        for pin in self.pins:
            if pin.index == cavity:
                return pin
        return None


@dataclass
class AutomotiveGround:
    """Automotive ground point."""
    code: str
    chassis_location: Optional[str] = None
    resistance_spec: Optional[str] = None
    connected_systems: Set[SystemType] = field(default_factory=set)


@dataclass
class AutomotiveFuseRelay:
    """Automotive fuse or relay."""
    id: str
    type: str  # "fuse" or "relay"
    rating: str
    slot: Optional[str] = None
    panel: Optional[str] = None
    protected_circuit: Optional[str] = None
    system: Optional[SystemType] = None


@dataclass
class AutomotiveWire:
    """Automotive wire specification."""
    gauge: str  # AWG or mm²
    color_code: WireColor
    route_hint: Optional[str] = None
    length_hint: Optional[str] = None
    connector_from: Optional[str] = None
    connector_to: Optional[str] = None
    pin_from: Optional[str] = None
    pin_to: Optional[str] = None


# Domain Knowledge Tables

class AutomotiveOntology:
    """Central automotive electronics ontology and knowledge base."""
    
    def __init__(self):
        self.component_types = self._init_component_types()
        self.system_relationships = self._init_system_relationships()
        self.voltage_levels = self._init_voltage_levels()
        self.wire_gauges = self._init_wire_gauges()
        
    def _init_component_types(self) -> Dict[ComponentType, Dict[str, Any]]:
        """Initialize component type knowledge."""
        return {
            ComponentType.RELAY: {
                "typical_pins": 5,
                "pin_roles": ["coil_positive", "coil_negative", "common", "normally_open", "normally_closed"],
                "systems": [SystemType.STARTING, SystemType.FUEL, SystemType.LIGHTING],
                "ratings": ["30A", "40A", "60A"],
            },
            ComponentType.FUSE: {
                "typical_pins": 2,
                "systems": [SystemType.POWER_DISTRIBUTION],
                "ratings": ["5A", "10A", "15A", "20A", "25A", "30A", "40A"],
            },
            ComponentType.ECU: {
                "typical_pins": [24, 32, 48, 64, 80, 104],
                "systems": [SystemType.ENGINE_MANAGEMENT, SystemType.TRANSMISSION, SystemType.ABS],
                "voltage_levels": ["5V", "12V"],
            },
            ComponentType.SENSOR: {
                "voltage_levels": ["5V", "12V"],
                "signal_types": ["analog", "digital", "frequency"],
                "systems": [SystemType.ENGINE_MANAGEMENT, SystemType.TRANSMISSION],
            }
        }
    
    def _init_system_relationships(self) -> Dict[SystemType, Dict[str, Any]]:
        """Initialize system relationship knowledge."""
        return {
            SystemType.STARTING: {
                "typical_components": ["starter_relay", "ignition_switch", "clutch_switch"],
                "power_source": "BATT",
                "grounds": ["G101", "G102"],
                "fuses": ["MAIN", "IG1"],
            },
            SystemType.FUEL: {
                "typical_components": ["fuel_pump_relay", "fuel_pump", "fuel_injectors"],
                "power_source": "IG1",
                "grounds": ["G103"],
                "fuses": ["EFI", "FUEL"],
            },
            SystemType.ENGINE_MANAGEMENT: {
                "typical_components": ["ECM", "various_sensors", "ignition_coils"],
                "power_source": "IG1",
                "grounds": ["E1", "E2"],
                "communication": ["CAN"],
            }
        }
    
    def _init_voltage_levels(self) -> Dict[str, Dict[str, Any]]:
        """Initialize voltage level knowledge."""
        return {
            "12V": {"systems": ["power_distribution", "lighting", "motors"], "tolerance": "±10%"},
            "5V": {"systems": ["sensors", "ecu"], "tolerance": "±5%"},
            "3.3V": {"systems": ["digital_logic"], "tolerance": "±3%"},
            "CAN_H": {"nominal": "3.5V", "systems": ["communication"]},
            "CAN_L": {"nominal": "1.5V", "systems": ["communication"]},
        }
    
    def _init_wire_gauges(self) -> Dict[str, Dict[str, Any]]:
        """Initialize wire gauge knowledge."""
        return {
            "0.5mm²": {"awg": "20", "current": "7.5A", "typical_use": "signal"},
            "0.75mm²": {"awg": "18", "current": "10A", "typical_use": "signal"},
            "1.25mm²": {"awg": "16", "current": "15A", "typical_use": "low_power"},
            "2.0mm²": {"awg": "14", "current": "20A", "typical_use": "medium_power"},
            "3.0mm²": {"awg": "12", "current": "25A", "typical_use": "high_power"},
            "5.0mm²": {"awg": "10", "current": "40A", "typical_use": "starter"},
        }
    
    def get_component_context(self, component: AutomotiveComponent) -> str:
        """Generate symbolic context for a component."""
        context_parts = []
        
        # Component type and value
        context_parts.append(f"{component.type.value} {component.id}")
        if component.value:
            context_parts.append(f"value {component.value}")
        
        # System context
        if component.system:
            context_parts.append(f"system {component.system.value}")
        
        # Location context
        if component.location_hint:
            context_parts.append(f"location {component.location_hint}")
        
        # Pin context
        if component.pins:
            pin_roles = [f"pin {pin.name}" for pin in component.pins[:3]]  # Limit to first 3
            context_parts.extend(pin_roles)
        
        return " ".join(context_parts)
    
    def get_net_context(self, net: ElectricalNet) -> str:
        """Generate symbolic context for a net."""
        context_parts = [f"net {net.name}"]
        
        if net.voltage_hint:
            context_parts.append(f"voltage {net.voltage_hint}")
        
        if net.bus_type:
            context_parts.append(f"bus {net.bus_type.value}")
            
        if net.system:
            context_parts.append(f"system {net.system.value}")
            
        if net.is_power_net:
            context_parts.append("power")
        elif net.is_ground_net:
            context_parts.append("ground")
        
        return " ".join(context_parts)
    
    def validate_component_consistency(self, component: AutomotiveComponent) -> List[str]:
        """Validate component against domain knowledge."""
        issues = []
        
        # Check pin count vs expected
        expected_pins = component.get_expected_pins()
        if expected_pins and len(component.pins) != len(expected_pins):
            issues.append(f"Pin count mismatch: expected {len(expected_pins)}, got {len(component.pins)}")
        
        # Check pin roles
        if component.type == ComponentType.RELAY and len(component.pins) >= 4:
            power_pins = [p for p in component.pins if p.electrical_role == ElectricalRole.POWER]
            if len(power_pins) < 2:
                issues.append("Relay should have at least 2 power pins (86, 30, 87)")
        
        # Check system consistency
        if component.system and component.type in self.component_types:
            expected_systems = self.component_types[component.type].get("systems", [])
            if expected_systems and component.system not in expected_systems:
                issues.append(f"Unusual system assignment: {component.type.value} in {component.system.value}")
        
        return issues
    
    def suggest_missing_connections(self, components: List[AutomotiveComponent]) -> List[str]:
        """Suggest likely missing connections based on domain knowledge."""
        suggestions = []
        
        # Find components that typically connect
        relays = [c for c in components if c.type == ComponentType.RELAY]
        fuses = [c for c in components if c.type == ComponentType.FUSE]
        ecus = [c for c in components if c.type == ComponentType.ECU]
        
        # Relays should typically have fuse protection
        for relay in relays:
            relay_system = relay.system
            related_fuses = [f for f in fuses if f.system == relay_system]
            if not related_fuses:
                suggestions.append(f"Relay {relay.id} may need fuse protection")
        
        # ECUs should have power and ground connections
        for ecu in ecus:
            power_pins = [p for p in ecu.pins if p.electrical_role == ElectricalRole.POWER]
            ground_pins = [p for p in ecu.pins if p.electrical_role == ElectricalRole.GROUND]
            
            if not power_pins:
                suggestions.append(f"ECU {ecu.id} missing power connections")
            if not ground_pins:
                suggestions.append(f"ECU {ecu.id} missing ground connections")
        
        return suggestions


# Global ontology instance
automotive_ontology = AutomotiveOntology()
"""
CONTROLS relationship model for control relationships between components
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class ControlType(str, Enum):
    """Enumeration of control types"""
    SWITCH = "switch"
    RELAY = "relay"
    PWM = "pwm"
    DATA = "data"
    CAN_BUS = "can_bus"
    LIN_BUS = "lin_bus"
    ANALOG = "analog"
    DIGITAL = "digital"


class SignalType(str, Enum):
    """Enumeration of signal types"""
    ON_OFF = "on_off"
    VARIABLE = "variable"
    PWM = "pwm"
    ANALOG_VOLTAGE = "analog_voltage"
    ANALOG_CURRENT = "analog_current"
    DIGITAL_DATA = "digital_data"
    CAN_MESSAGE = "can_message"
    LIN_MESSAGE = "lin_message"


class ControlsRelationship(BaseModel):
    """
    CONTROLS relationship model matching Neo4j schema
    
    Neo4j Schema:
    (:Component)-[:CONTROLS {
      control_type: "string",
      signal_voltage: "float",
      response_time: "float",
      created_at: "datetime"
    }]->(:Component)
    """
    
    # Control characteristics
    control_type: ControlType = Field(
        default=ControlType.SWITCH,
        description="Type of control: switch, relay, pwm, data"
    )
    signal_type: SignalType = Field(
        default=SignalType.ON_OFF,
        description="Type of control signal"
    )
    signal_voltage: Optional[float] = Field(None, description="Control signal voltage")
    signal_current: Optional[float] = Field(None, description="Control signal current")
    
    # Timing characteristics
    response_time: Optional[float] = Field(None, description="Response time (ms)")
    activation_delay: Optional[float] = Field(None, description="Activation delay (ms)")
    deactivation_delay: Optional[float] = Field(None, description="Deactivation delay (ms)")
    switching_frequency: Optional[float] = Field(None, description="Maximum switching frequency (Hz)")
    
    # PWM specific
    pwm_frequency: Optional[float] = Field(None, description="PWM frequency (Hz)")
    duty_cycle_range: Optional[Dict[str, float]] = Field(None, description="Duty cycle range {min: 0, max: 100}")
    
    # Data communication specific
    communication_protocol: Optional[str] = Field(None, description="CAN, LIN, SPI, I2C, etc.")
    data_rate: Optional[float] = Field(None, description="Data rate (bps)")
    message_id: Optional[str] = Field(None, description="CAN/LIN message ID")
    data_bytes: Optional[int] = Field(None, description="Number of data bytes")
    
    # Analog signal specific
    signal_range: Optional[Dict[str, float]] = Field(None, description="Signal range {min: 0.0, max: 5.0}")
    resolution: Optional[int] = Field(None, description="Signal resolution (bits)")
    linearity: Optional[float] = Field(None, description="Signal linearity percentage")
    
    # Control logic
    inverted_logic: bool = Field(default=False, description="Inverted control logic")
    multi_state: bool = Field(default=False, description="Multi-state control")
    state_count: Optional[int] = Field(None, description="Number of states for multi-state")
    
    # Safety and reliability
    fail_safe_state: Optional[str] = Field(None, description="Fail-safe state: on, off, hold")
    diagnostics_available: bool = Field(default=False, description="Has diagnostic capabilities")
    fault_detection: bool = Field(default=False, description="Has fault detection")
    
    # Load characteristics
    load_type: Optional[str] = Field(None, description="resistive, inductive, capacitive")
    load_current: Optional[float] = Field(None, description="Controlled load current (A)")
    contact_rating: Optional[str] = Field(None, description="Contact rating for switches/relays")
    
    # Environmental
    operating_temperature_range: Optional[Dict[str, float]] = Field(
        None, 
        description="Operating temperature range {min: -40, max: 85}"
    )
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def to_neo4j_dict(self) -> Dict[str, Any]:
        """Convert to Neo4j relationship properties dictionary"""
        data = self.dict(exclude_none=True)
        
        # Convert enums to strings
        if 'control_type' in data:
            data['control_type'] = self.control_type.value
        if 'signal_type' in data:
            data['signal_type'] = self.signal_type.value
        
        # Convert datetime to string
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
            
        return data
    
    @classmethod
    def from_neo4j_dict(cls, data: Dict[str, Any]) -> 'ControlsRelationship':
        """Create instance from Neo4j relationship dictionary"""
        # Convert datetime strings
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        # Convert string enums back to enum values
        if 'control_type' in data and isinstance(data['control_type'], str):
            data['control_type'] = ControlType(data['control_type'])
        if 'signal_type' in data and isinstance(data['signal_type'], str):
            data['signal_type'] = SignalType(data['signal_type'])
        
        return cls(**data)
    
    def is_digital_control(self) -> bool:
        """Check if this is a digital control signal"""
        return self.signal_type in [
            SignalType.ON_OFF,
            SignalType.DIGITAL_DATA,
            SignalType.CAN_MESSAGE,
            SignalType.LIN_MESSAGE
        ]
    
    def is_analog_control(self) -> bool:
        """Check if this is an analog control signal"""
        return self.signal_type in [
            SignalType.ANALOG_VOLTAGE,
            SignalType.ANALOG_CURRENT,
            SignalType.VARIABLE
        ]
    
    def is_high_frequency(self) -> bool:
        """Check if this is a high-frequency control (>1kHz)"""
        return (
            self.switching_frequency and self.switching_frequency > 1000 or
            self.pwm_frequency and self.pwm_frequency > 1000
        )
    
    def calculate_power_consumption(self) -> Optional[float]:
        """Calculate control circuit power consumption"""
        if self.signal_voltage and self.signal_current:
            return self.signal_voltage * self.signal_current
        return None
    
    def get_control_complexity(self) -> str:
        """Assess control complexity level"""
        if self.control_type in [ControlType.CAN_BUS, ControlType.LIN_BUS]:
            return "high"
        elif self.control_type in [ControlType.PWM, ControlType.DATA]:
            return "medium"
        else:
            return "low"


class ControlChain(BaseModel):
    """Model for control chain analysis"""
    initiator_component: str
    control_sequence: List[Dict[str, Any]]
    total_response_time: float
    chain_complexity: str
    fault_tolerance: str


class ControlMatrix(BaseModel):
    """Model for control interaction matrix"""
    vehicle_signature: str
    control_relationships: List[Dict[str, Any]]
    interaction_count: int
    complexity_score: float
    potential_conflicts: List[str]
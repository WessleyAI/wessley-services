"""
Netlist generation and component catalog export functionality.
"""
import json
import xml.etree.ElementTree as ET
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import uuid

from ..core.schemas import Component, Pin, Net, NetConnection, Netlist, UnresolvedConnection
from .associate import ComponentWithText, PinAssociation
from .wires import WireNet, Junction, LineSegment


@dataclass
class ComponentCatalogEntry:
    """Entry in component catalog."""
    reference: str
    type: str
    value: Optional[str]
    footprint: Optional[str]
    pins: List[Dict[str, str]]  # List of pin definitions
    attributes: Dict[str, str]
    confidence: float
    page: int
    position: Tuple[float, float]


@dataclass
class NetlistExportResult:
    """Result of netlist export operation."""
    netlist: Netlist
    component_catalog: List[ComponentCatalogEntry]
    statistics: Dict[str, int]
    warnings: List[str]


class NetlistGenerator:
    """
    Generates netlists and component catalogs from schematic analysis results.
    """
    
    def __init__(self):
        """Initialize netlist generator."""
        self.component_counter = defaultdict(int)
        self.net_counter = 0
        self.warnings = []
    
    def generate_netlist(
        self, 
        components_with_text: List[ComponentWithText],
        wire_nets: List[WireNet],
        junctions: List[Junction],
        line_segments: List[LineSegment],
        page_number: int = 1
    ) -> NetlistExportResult:
        """
        Generate complete netlist and component catalog.
        
        Args:
            components_with_text: Components with associated text
            wire_nets: Detected wire networks
            junctions: Junction points
            line_segments: Wire line segments
            page_number: Page number for multi-page schematics
            
        Returns:
            Complete export result with netlist and catalog
        """
        self.warnings = []
        
        # Process components and build catalog
        component_catalog = self._build_component_catalog(components_with_text, page_number)
        
        # Build component-to-net mapping
        component_net_mapping = self._map_components_to_nets(
            components_with_text, wire_nets, line_segments
        )
        
        # Generate nets from wire networks
        nets = self._generate_nets_from_wires(
            wire_nets, component_net_mapping, components_with_text
        )
        
        # Find unresolved connections
        unresolved = self._find_unresolved_connections(
            component_catalog, component_net_mapping
        )
        
        # Create netlist
        netlist = Netlist(nets=nets, unresolved=unresolved)
        
        # Generate statistics
        statistics = self._calculate_statistics(netlist, component_catalog)
        
        return NetlistExportResult(
            netlist=netlist,
            component_catalog=component_catalog,
            statistics=statistics,
            warnings=self.warnings
        )
    
    def _build_component_catalog(
        self, 
        components_with_text: List[ComponentWithText], 
        page_number: int
    ) -> List[ComponentCatalogEntry]:
        """Build component catalog from components with text."""
        catalog = []
        
        for comp_with_text in components_with_text:
            component = comp_with_text.component
            
            # Generate reference designator
            reference = self._generate_reference_designator(comp_with_text)
            
            # Extract value
            value = self._extract_component_value(comp_with_text)
            
            # Build pin list
            pins = self._build_pin_list(comp_with_text)
            
            # Extract attributes
            attributes = self._extract_component_attributes(comp_with_text)
            
            # Calculate position
            position = (
                (component.bbox[0] + component.bbox[2]) / 2,
                (component.bbox[1] + component.bbox[3]) / 2
            )
            
            catalog_entry = ComponentCatalogEntry(
                reference=reference,
                type=component.label,
                value=value,
                footprint=None,  # Could be inferred from component analysis
                pins=pins,
                attributes=attributes,
                confidence=comp_with_text.confidence,
                page=page_number,
                position=position
            )
            
            catalog.append(catalog_entry)
        
        return catalog
    
    def _generate_reference_designator(self, comp_with_text: ComponentWithText) -> str:
        """Generate reference designator for component."""
        
        # Use associated reference label if available
        if comp_with_text.reference_label:
            return comp_with_text.reference_label.text.strip()
        
        # Generate based on component type
        component_type = comp_with_text.component.label.lower()
        
        # Mapping of component types to reference prefixes
        type_prefixes = {
            'resistor': 'R',
            'capacitor': 'C',
            'inductor': 'L',
            'ic': 'U',
            'transistor': 'Q',
            'diode': 'D',
            'led': 'D',
            'connector': 'J',
            'switch': 'SW',
            'fuse': 'F',
            'relay': 'K',
            'transformer': 'T',
            'crystal': 'Y',
            'test_point': 'TP'
        }
        
        prefix = type_prefixes.get(component_type, 'U')
        self.component_counter[prefix] += 1
        
        return f"{prefix}{self.component_counter[prefix]}"
    
    def _extract_component_value(self, comp_with_text: ComponentWithText) -> Optional[str]:
        """Extract component value from associated text."""
        if comp_with_text.value_label:
            return comp_with_text.value_label.text.strip()
        
        # Look in additional labels for value-like text
        for label in comp_with_text.additional_labels or []:
            text = label.text.strip()
            if self._looks_like_component_value(text):
                return text
        
        return None
    
    def _looks_like_component_value(self, text: str) -> bool:
        """Check if text looks like a component value."""
        import re
        
        value_patterns = [
            r'^\d+[kKmM]?[ΩΩohm]*$',      # Resistor values
            r'^\d+[pnumµμ]*[Ff]$',        # Capacitor values
            r'^\d+[pnumµμ]*[Hh]$',        # Inductor values
            r'^\d+(\.\d+)?[Vv]$',         # Voltage values
            r'^[A-Z]{2,}\d+[A-Z]*$',      # IC part numbers
        ]
        
        return any(re.match(pattern, text) for pattern in value_patterns)
    
    def _build_pin_list(self, comp_with_text: ComponentWithText) -> List[Dict[str, str]]:
        """Build pin list for component."""
        pins = []
        
        # Add pins from pin associations
        for pin_assoc in comp_with_text.pin_labels or []:
            pin_dict = {
                'number': pin_assoc.pin_number,
                'name': pin_assoc.pin_name or '',
                'type': 'unknown'  # Could be inferred from context
            }
            pins.append(pin_dict)
        
        # Add estimated pins from component detection
        if hasattr(comp_with_text.component, 'pins'):
            for i, pin in enumerate(comp_with_text.component.pins or []):
                pin_dict = {
                    'number': str(i + 1),
                    'name': pin.get('name', ''),
                    'type': pin.get('type', 'unknown')
                }
                pins.append(pin_dict)
        
        # If no pins found, estimate based on component type
        if not pins:
            pins = self._estimate_pins_by_type(comp_with_text.component.label)
        
        return pins
    
    def _estimate_pins_by_type(self, component_type: str) -> List[Dict[str, str]]:
        """Estimate pins based on component type."""
        type_lower = component_type.lower()
        
        if type_lower in ['resistor', 'capacitor', 'inductor', 'diode', 'led']:
            return [
                {'number': '1', 'name': '', 'type': 'passive'},
                {'number': '2', 'name': '', 'type': 'passive'}
            ]
        elif type_lower == 'transistor':
            return [
                {'number': '1', 'name': 'B', 'type': 'input'},  # Base
                {'number': '2', 'name': 'C', 'type': 'output'}, # Collector
                {'number': '3', 'name': 'E', 'type': 'output'}  # Emitter
            ]
        else:
            # Generic multi-pin component
            return [
                {'number': str(i), 'name': '', 'type': 'unknown'}
                for i in range(1, 5)  # Assume 4 pins
            ]
    
    def _extract_component_attributes(self, comp_with_text: ComponentWithText) -> Dict[str, str]:
        """Extract additional component attributes."""
        attributes = {}
        
        # Add confidence
        attributes['confidence'] = f"{comp_with_text.confidence:.3f}"
        
        # Add detection method
        attributes['detection_method'] = 'computer_vision'
        
        # Add any additional labels as attributes
        for i, label in enumerate(comp_with_text.additional_labels or []):
            attributes[f'label_{i}'] = label.text.strip()
        
        return attributes
    
    def _map_components_to_nets(
        self, 
        components_with_text: List[ComponentWithText],
        wire_nets: List[WireNet],
        line_segments: List[LineSegment]
    ) -> Dict[str, List[str]]:
        """Map components to wire nets they connect to."""
        mapping = defaultdict(list)
        
        for comp_with_text in components_with_text:
            component = comp_with_text.component
            reference = self._generate_reference_designator(comp_with_text)
            
            # Find wire nets that intersect with component
            connected_nets = self._find_connected_nets(
                component, wire_nets, line_segments
            )
            
            mapping[reference] = connected_nets
        
        return mapping
    
    def _find_connected_nets(
        self, 
        component, 
        wire_nets: List[WireNet], 
        line_segments: List[LineSegment],
        connection_tolerance: float = 15.0
    ) -> List[int]:
        """Find wire nets that connect to a component."""
        connected_nets = []
        
        # Component bounding box
        comp_bbox = component.bbox
        
        for net_idx, wire_net in enumerate(wire_nets):
            # Check if any segment endpoints are near the component
            for seg_idx in wire_net.segments:
                if seg_idx >= len(line_segments):
                    continue
                
                segment = line_segments[seg_idx]
                
                # Check both endpoints of the segment
                for endpoint in [segment.start, segment.end]:
                    if self._is_point_near_component(endpoint, comp_bbox, connection_tolerance):
                        connected_nets.append(net_idx)
                        break
                
                if net_idx in connected_nets:
                    break
        
        return connected_nets
    
    def _is_point_near_component(
        self, 
        point: Tuple[float, float], 
        component_bbox: Tuple[float, float, float, float], 
        tolerance: float
    ) -> bool:
        """Check if a point is near a component."""
        x, y = point
        x1, y1, x2, y2 = component_bbox
        
        # Expand component bbox by tolerance
        return (x1 - tolerance <= x <= x2 + tolerance and 
                y1 - tolerance <= y <= y2 + tolerance)
    
    def _generate_nets_from_wires(
        self, 
        wire_nets: List[WireNet],
        component_net_mapping: Dict[str, List[int]],
        components_with_text: List[ComponentWithText]
    ) -> List[Net]:
        """Generate Net objects from wire networks."""
        nets = []
        
        for net_idx, wire_net in enumerate(wire_nets):
            # Generate net name
            net_name = self._generate_net_name(wire_net, net_idx)
            
            # Find components connected to this net
            connections = []
            for comp_with_text in components_with_text:
                reference = self._generate_reference_designator(comp_with_text)
                
                if net_idx in component_net_mapping.get(reference, []):
                    # Add connections for each pin (simplified - assumes 2-pin components)
                    pin_count = len(comp_with_text.pin_labels or []) or 2
                    for pin_num in range(1, pin_count + 1):
                        connection = NetConnection(
                            component_id=reference,
                            pin=str(pin_num)
                        )
                        connections.append(connection)
            
            # Calculate confidence
            confidence = self._calculate_net_confidence(wire_net, connections)
            
            net = Net(
                name=net_name,
                connections=connections,
                page_spans=[1],  # Simplified - single page
                confidence=confidence
            )
            
            nets.append(net)
        
        return nets
    
    def _generate_net_name(self, wire_net: WireNet, net_idx: int) -> str:
        """Generate name for a net."""
        
        # Use propagated name if available
        if wire_net.propagated_name:
            return wire_net.propagated_name
        
        # Use first net label if available
        if wire_net.net_labels:
            return wire_net.net_labels[0].text
        
        # Generate generic name
        return f"Net_{net_idx + 1}"
    
    def _calculate_net_confidence(self, wire_net: WireNet, connections: List[NetConnection]) -> float:
        """Calculate confidence for a net."""
        
        # Base confidence from wire detection
        base_confidence = 0.8
        
        # Boost confidence if net has labels
        if wire_net.net_labels:
            label_confidence = sum(label.confidence for label in wire_net.net_labels) / len(wire_net.net_labels)
            base_confidence = (base_confidence + label_confidence) / 2
        
        # Reduce confidence if no connections
        if not connections:
            base_confidence *= 0.5
            self.warnings.append(f"Net {wire_net.propagated_name or 'unnamed'} has no component connections")
        
        return min(1.0, base_confidence)
    
    def _find_unresolved_connections(
        self, 
        component_catalog: List[ComponentCatalogEntry],
        component_net_mapping: Dict[str, List[int]]
    ) -> List[UnresolvedConnection]:
        """Find unresolved connections in the netlist."""
        unresolved = []
        
        for comp in component_catalog:
            connected_nets = component_net_mapping.get(comp.reference, [])
            
            # Check if component has expected number of connections
            expected_pins = len(comp.pins)
            actual_connections = len(connected_nets)
            
            if actual_connections < expected_pins:
                for pin_num in range(actual_connections + 1, expected_pins + 1):
                    unresolved.append(UnresolvedConnection(
                        reason="dangling_pin",
                        component_id=comp.reference,
                        pin=str(pin_num)
                    ))
            
            # Check for components with no connections at all
            if actual_connections == 0:
                unresolved.append(UnresolvedConnection(
                    reason="isolated_component",
                    component_id=comp.reference,
                    pin="all"
                ))
        
        return unresolved
    
    def _calculate_statistics(
        self, 
        netlist: Netlist, 
        component_catalog: List[ComponentCatalogEntry]
    ) -> Dict[str, int]:
        """Calculate statistics about the netlist."""
        
        # Count components by type
        component_types = defaultdict(int)
        for comp in component_catalog:
            component_types[comp.type] += 1
        
        # Calculate connections
        total_connections = sum(len(net.connections) for net in netlist.nets)
        
        statistics = {
            'total_components': len(component_catalog),
            'total_nets': len(netlist.nets),
            'total_connections': total_connections,
            'unresolved_connections': len(netlist.unresolved),
            'named_nets': sum(1 for net in netlist.nets if not net.name.startswith('Net_')),
            'average_connections_per_net': total_connections / len(netlist.nets) if netlist.nets else 0
        }
        
        # Add component type counts
        for comp_type, count in component_types.items():
            statistics[f'components_{comp_type}'] = count
        
        return statistics


class NetlistExporter:
    """
    Export netlists to various formats (JSON, GraphML, NDJSON).
    """
    
    def export_to_json(self, result: NetlistExportResult) -> str:
        """Export netlist to JSON format."""
        export_data = {
            'netlist': {
                'nets': [
                    {
                        'name': net.name,
                        'connections': [
                            {'component': conn.component_id, 'pin': conn.pin}
                            for conn in net.connections
                        ],
                        'confidence': net.confidence
                    }
                    for net in result.netlist.nets
                ],
                'unresolved': [
                    {
                        'reason': unres.reason,
                        'component': unres.component_id,
                        'pin': unres.pin
                    }
                    for unres in result.netlist.unresolved
                ]
            },
            'components': [
                {
                    'reference': comp.reference,
                    'type': comp.type,
                    'value': comp.value,
                    'pins': comp.pins,
                    'attributes': comp.attributes,
                    'position': {'x': comp.position[0], 'y': comp.position[1]},
                    'page': comp.page,
                    'confidence': comp.confidence
                }
                for comp in result.component_catalog
            ],
            'statistics': result.statistics,
            'warnings': result.warnings
        }
        
        return json.dumps(export_data, indent=2)
    
    def export_to_graphml(self, result: NetlistExportResult) -> str:
        """Export netlist to GraphML format."""
        
        # Create XML root
        root = ET.Element('graphml')
        root.set('xmlns', 'http://graphml.graphdrawing.org/xmlns')
        
        # Define attributes
        key_elements = [
            ('node_type', 'node', 'type'),
            ('node_value', 'node', 'value'),
            ('node_reference', 'node', 'reference'),
            ('edge_pin', 'edge', 'pin'),
            ('edge_net', 'edge', 'net_name')
        ]
        
        for key_id, for_elem, attr_name in key_elements:
            key = ET.SubElement(root, 'key')
            key.set('id', key_id)
            key.set('for', for_elem)
            key.set('attr.name', attr_name)
            key.set('attr.type', 'string')
        
        # Create graph
        graph = ET.SubElement(root, 'graph')
        graph.set('id', 'schematic')
        graph.set('edgedefault', 'undirected')
        
        # Add component nodes
        for comp in result.component_catalog:
            node = ET.SubElement(graph, 'node')
            node.set('id', comp.reference)
            
            # Add data elements
            data_type = ET.SubElement(node, 'data')
            data_type.set('key', 'node_type')
            data_type.text = comp.type
            
            data_ref = ET.SubElement(node, 'data')
            data_ref.set('key', 'node_reference')
            data_ref.text = comp.reference
            
            if comp.value:
                data_val = ET.SubElement(node, 'data')
                data_val.set('key', 'node_value')
                data_val.text = comp.value
        
        # Add net edges
        edge_id = 0
        for net in result.netlist.nets:
            # Connect all components in this net
            components_in_net = list(set(conn.component_id for conn in net.connections))
            
            for i, comp1 in enumerate(components_in_net):
                for comp2 in components_in_net[i+1:]:
                    edge = ET.SubElement(graph, 'edge')
                    edge.set('id', f'e{edge_id}')
                    edge.set('source', comp1)
                    edge.set('target', comp2)
                    
                    data_net = ET.SubElement(edge, 'data')
                    data_net.set('key', 'edge_net')
                    data_net.text = net.name
                    
                    edge_id += 1
        
        return ET.tostring(root, encoding='unicode')
    
    def export_to_ndjson(self, result: NetlistExportResult) -> str:
        """Export netlist to NDJSON format (one entity per line)."""
        lines = []
        
        # Export components
        for comp in result.component_catalog:
            comp_obj = {
                'type': 'component',
                'reference': comp.reference,
                'component_type': comp.type,
                'value': comp.value,
                'pins': comp.pins,
                'attributes': comp.attributes,
                'position': {'x': comp.position[0], 'y': comp.position[1]},
                'page': comp.page,
                'confidence': comp.confidence
            }
            lines.append(json.dumps(comp_obj))
        
        # Export nets
        for net in result.netlist.nets:
            net_obj = {
                'type': 'net',
                'name': net.name,
                'connections': [
                    {'component': conn.component_id, 'pin': conn.pin}
                    for conn in net.connections
                ],
                'confidence': net.confidence
            }
            lines.append(json.dumps(net_obj))
        
        # Export statistics
        stats_obj = {
            'type': 'statistics',
            'data': result.statistics
        }
        lines.append(json.dumps(stats_obj))
        
        return '\n'.join(lines)
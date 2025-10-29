"""
Text-to-symbol association using spatial analysis and nearest neighbor algorithms.
"""
import math
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
from sklearn.neighbors import KDTree

from ..core.schemas import TextSpan
from .detect import Detection, ComponentDetector


@dataclass
class Association:
    """Represents an association between text and a symbol/component."""
    text_span: TextSpan
    component: Detection
    association_type: str  # "label", "value", "reference", "pin_label"
    confidence: float
    distance: float
    relationship: str  # "adjacent", "contained", "aligned", "nearest"


@dataclass 
class PinAssociation:
    """Represents text associated with a component pin."""
    pin_number: str
    pin_name: Optional[str]
    text_span: TextSpan
    confidence: float
    position: Tuple[float, float]


@dataclass
class ComponentWithText:
    """Component enriched with associated text labels and values."""
    component: Detection
    reference_label: Optional[TextSpan] = None  # e.g., "R1", "U1"
    value_label: Optional[TextSpan] = None      # e.g., "10k", "LM358"
    pin_labels: List[PinAssociation] = None
    additional_labels: List[TextSpan] = None    # Other nearby text
    confidence: float = 0.0


class TextSymbolAssociator:
    """
    Associates text labels with detected symbols using spatial analysis.
    """
    
    def __init__(
        self,
        max_association_distance: float = 50.0,
        alignment_tolerance: float = 10.0,
        containment_tolerance: float = 5.0,
        reference_patterns: List[str] = None,
        value_patterns: List[str] = None
    ):
        """
        Initialize text-symbol associator.
        
        Args:
            max_association_distance: Maximum distance for text-symbol association
            alignment_tolerance: Tolerance for considering elements aligned
            containment_tolerance: Tolerance for considering text contained in component
            reference_patterns: Regex patterns for reference designators
            value_patterns: Regex patterns for component values
        """
        self.max_association_distance = max_association_distance
        self.alignment_tolerance = alignment_tolerance
        self.containment_tolerance = containment_tolerance
        
        # Default patterns for reference designators and values
        self.reference_patterns = reference_patterns or [
            r'^[RULCQDT]\d+$',  # R1, U1, L1, C1, Q1, D1, T1
            r'^IC\d+$',         # IC1, IC2
            r'^SW\d+$',         # SW1, SW2 (switches)
            r'^J\d+$',          # J1, J2 (connectors)
            r'^TP\d+$',         # TP1, TP2 (test points)
        ]
        
        self.value_patterns = value_patterns or [
            r'^\d+[kKmM]?[ΩΩohm]*$',      # Resistor values: 10k, 1M, 470Ω
            r'^\d+[pnumµμ]*[Ff]$',        # Capacitor values: 100nF, 1µF
            r'^\d+[pnumµμ]*[Hh]$',        # Inductor values: 10µH, 1mH
            r'^\d+(\.\d+)?[Vv]$',         # Voltage values: 5V, 3.3V
            r'^LM\d+$',                   # IC part numbers: LM358, LM7805
            r'^TL\d+$',                   # IC part numbers: TL074
            r'^[12]N\d+$',                # Transistor part numbers: 2N2222
        ]
    
    def associate_text_with_symbols(
        self, 
        text_spans: List[TextSpan], 
        components: List[Detection]
    ) -> List[ComponentWithText]:
        """
        Associate text spans with detected components.
        
        Args:
            text_spans: List of detected text spans
            components: List of detected components
            
        Returns:
            List of components enriched with associated text
        """
        if not text_spans or not components:
            return [ComponentWithText(component=comp) for comp in components]
        
        # Build spatial index for efficient nearest neighbor queries
        text_tree = self._build_text_spatial_index(text_spans)
        component_tree = self._build_component_spatial_index(components)
        
        # Find associations for each component
        enriched_components = []
        used_text_spans = set()
        
        for component in components:
            associations = self._find_component_associations(
                component, text_spans, text_tree, used_text_spans
            )
            
            enriched_component = self._create_enriched_component(
                component, associations, text_spans
            )
            
            enriched_components.append(enriched_component)
            
            # Mark text spans as used
            for assoc in associations:
                used_text_spans.add(id(assoc.text_span))
        
        return enriched_components
    
    def _build_text_spatial_index(self, text_spans: List[TextSpan]) -> KDTree:
        """Build KDTree for efficient spatial queries on text spans."""
        if not text_spans:
            return None
        
        # Extract center points of text spans
        centers = []
        for span in text_spans:
            center_x = (span.bbox[0] + span.bbox[2]) / 2
            center_y = (span.bbox[1] + span.bbox[3]) / 2
            centers.append([center_x, center_y])
        
        return KDTree(np.array(centers))
    
    def _build_component_spatial_index(self, components: List[Detection]) -> KDTree:
        """Build KDTree for efficient spatial queries on components."""
        if not components:
            return None
        
        # Extract center points of components
        centers = []
        for comp in components:
            center_x = (comp.bbox[0] + comp.bbox[2]) / 2
            center_y = (comp.bbox[1] + comp.bbox[3]) / 2
            centers.append([center_x, center_y])
        
        return KDTree(np.array(centers))
    
    def _find_component_associations(
        self, 
        component: Detection, 
        text_spans: List[TextSpan], 
        text_tree: KDTree,
        used_text_spans: Set[int]
    ) -> List[Association]:
        """Find all text associations for a given component."""
        associations = []
        
        if text_tree is None:
            return associations
        
        # Component center point
        comp_center = (
            (component.bbox[0] + component.bbox[2]) / 2,
            (component.bbox[1] + component.bbox[3]) / 2
        )
        
        # Find nearby text spans
        nearby_indices = text_tree.query_radius(
            [comp_center], 
            r=self.max_association_distance
        )[0]
        
        for idx in nearby_indices:
            if idx >= len(text_spans):
                continue
                
            text_span = text_spans[idx]
            
            # Skip already used text spans
            if id(text_span) in used_text_spans:
                continue
            
            # Calculate association metrics
            association = self._evaluate_text_component_association(
                text_span, component
            )
            
            if association and association.confidence > 0.3:
                associations.append(association)
        
        # Sort by confidence (highest first)
        associations.sort(key=lambda a: a.confidence, reverse=True)
        
        return associations
    
    def _evaluate_text_component_association(
        self, 
        text_span: TextSpan, 
        component: Detection
    ) -> Optional[Association]:
        """Evaluate the association between a text span and component."""
        
        # Calculate spatial relationship
        relationship, distance = self._calculate_spatial_relationship(text_span, component)
        
        if distance > self.max_association_distance:
            return None
        
        # Classify text type
        text_type = self._classify_text_type(text_span.text)
        
        # Calculate base confidence from spatial relationship
        spatial_confidence = self._calculate_spatial_confidence(relationship, distance)
        
        # Adjust confidence based on text type and component type
        type_confidence = self._calculate_type_confidence(text_type, component.label)
        
        # Calculate final confidence
        final_confidence = spatial_confidence * type_confidence
        
        return Association(
            text_span=text_span,
            component=component,
            association_type=text_type,
            confidence=final_confidence,
            distance=distance,
            relationship=relationship
        )
    
    def _calculate_spatial_relationship(
        self, 
        text_span: TextSpan, 
        component: Detection
    ) -> Tuple[str, float]:
        """Calculate spatial relationship between text and component."""
        
        # Text and component centers
        text_center = (
            (text_span.bbox[0] + text_span.bbox[2]) / 2,
            (text_span.bbox[1] + text_span.bbox[3]) / 2
        )
        comp_center = (
            (component.bbox[0] + component.bbox[2]) / 2,
            (component.bbox[1] + component.bbox[3]) / 2
        )
        
        # Calculate distance
        distance = math.sqrt(
            (text_center[0] - comp_center[0])**2 + 
            (text_center[1] - comp_center[1])**2
        )
        
        # Check for containment (text inside component)
        if self._is_text_contained_in_component(text_span, component):
            return "contained", distance
        
        # Check for overlap
        if self._do_boxes_overlap(text_span.bbox, component.bbox):
            return "overlapping", distance
        
        # Check for alignment
        if self._are_elements_aligned(text_span, component):
            return "aligned", distance
        
        # Check adjacency direction
        adjacency = self._get_adjacency_direction(text_span, component)
        if adjacency:
            return adjacency, distance
        
        return "nearest", distance
    
    def _is_text_contained_in_component(
        self, 
        text_span: TextSpan, 
        component: Detection
    ) -> bool:
        """Check if text span is contained within component bounding box."""
        tx1, ty1, tx2, ty2 = text_span.bbox
        cx1, cy1, cx2, cy2 = component.bbox
        
        # Add tolerance
        tol = self.containment_tolerance
        
        return (cx1 - tol <= tx1 and tx2 <= cx2 + tol and 
                cy1 - tol <= ty1 and ty2 <= cy2 + tol)
    
    def _do_boxes_overlap(
        self, 
        box1: Tuple[float, float, float, float], 
        box2: Tuple[float, float, float, float]
    ) -> bool:
        """Check if two bounding boxes overlap."""
        x1a, y1a, x2a, y2a = box1
        x1b, y1b, x2b, y2b = box2
        
        return not (x2a < x1b or x2b < x1a or y2a < y1b or y2b < y1a)
    
    def _are_elements_aligned(
        self, 
        text_span: TextSpan, 
        component: Detection
    ) -> bool:
        """Check if text and component are aligned horizontally or vertically."""
        
        # Centers
        text_center = (
            (text_span.bbox[0] + text_span.bbox[2]) / 2,
            (text_span.bbox[1] + text_span.bbox[3]) / 2
        )
        comp_center = (
            (component.bbox[0] + component.bbox[2]) / 2,
            (component.bbox[1] + component.bbox[3]) / 2
        )
        
        # Check horizontal alignment
        if abs(text_center[1] - comp_center[1]) <= self.alignment_tolerance:
            return True
        
        # Check vertical alignment
        if abs(text_center[0] - comp_center[0]) <= self.alignment_tolerance:
            return True
        
        return False
    
    def _get_adjacency_direction(
        self, 
        text_span: TextSpan, 
        component: Detection
    ) -> Optional[str]:
        """Determine the direction of adjacency between text and component."""
        
        # Text and component bounds
        tx1, ty1, tx2, ty2 = text_span.bbox
        cx1, cy1, cx2, cy2 = component.bbox
        
        # Text center
        text_center_x = (tx1 + tx2) / 2
        text_center_y = (ty1 + ty2) / 2
        
        # Component center
        comp_center_x = (cx1 + cx2) / 2
        comp_center_y = (cy1 + cy2) / 2
        
        # Calculate relative position
        dx = text_center_x - comp_center_x
        dy = text_center_y - comp_center_y
        
        # Determine primary direction
        if abs(dx) > abs(dy):
            if dx > 0:
                return "adjacent_right"
            else:
                return "adjacent_left"
        else:
            if dy > 0:
                return "adjacent_below"
            else:
                return "adjacent_above"
    
    def _calculate_spatial_confidence(self, relationship: str, distance: float) -> float:
        """Calculate confidence based on spatial relationship."""
        
        # Base confidence by relationship type
        base_confidence = {
            "contained": 1.0,
            "overlapping": 0.9,
            "aligned": 0.8,
            "adjacent_above": 0.7,
            "adjacent_below": 0.7,
            "adjacent_left": 0.6,
            "adjacent_right": 0.6,
            "nearest": 0.4
        }.get(relationship, 0.2)
        
        # Adjust by distance (closer is better)
        distance_factor = max(0.1, 1.0 - (distance / self.max_association_distance))
        
        return base_confidence * distance_factor
    
    def _classify_text_type(self, text: str) -> str:
        """Classify text as reference, value, or generic label."""
        import re
        
        text_clean = text.strip()
        
        # Check reference designator patterns
        for pattern in self.reference_patterns:
            if re.match(pattern, text_clean, re.IGNORECASE):
                return "reference"
        
        # Check value patterns  
        for pattern in self.value_patterns:
            if re.match(pattern, text_clean, re.IGNORECASE):
                return "value"
        
        # Check for pin labels (numbers or short names)
        if text_clean.isdigit() and len(text_clean) <= 3:
            return "pin_label"
        
        # Check for common pin names
        pin_names = ['VCC', 'GND', 'IN', 'OUT', 'CLK', 'RST', 'EN', 'CS', 'WR', 'RD']
        if text_clean.upper() in pin_names:
            return "pin_label"
        
        return "label"
    
    def _calculate_type_confidence(self, text_type: str, component_type: str) -> float:
        """Calculate confidence based on text type and component type compatibility."""
        
        # Compatibility matrix
        compatibility = {
            ("reference", "resistor"): 1.0,
            ("reference", "capacitor"): 1.0,
            ("reference", "inductor"): 1.0,
            ("reference", "ic"): 1.0,
            ("reference", "transistor"): 1.0,
            ("reference", "diode"): 1.0,
            
            ("value", "resistor"): 1.0,
            ("value", "capacitor"): 1.0,
            ("value", "inductor"): 1.0,
            ("value", "ic"): 0.8,
            ("value", "transistor"): 0.8,
            ("value", "diode"): 0.6,
            
            ("pin_label", "ic"): 1.0,
            ("pin_label", "connector"): 1.0,
            ("pin_label", "transistor"): 0.8,
            ("pin_label", "resistor"): 0.3,
            ("pin_label", "capacitor"): 0.2,
            
            ("label", "resistor"): 0.5,
            ("label", "capacitor"): 0.5,
            ("label", "ic"): 0.7,
            ("label", "connector"): 0.8,
        }
        
        return compatibility.get((text_type, component_type), 0.5)
    
    def _create_enriched_component(
        self, 
        component: Detection, 
        associations: List[Association],
        text_spans: List[TextSpan]
    ) -> ComponentWithText:
        """Create enriched component with associated text labels."""
        
        enriched = ComponentWithText(component=component)
        enriched.pin_labels = []
        enriched.additional_labels = []
        
        # Separate associations by type
        reference_associations = [a for a in associations if a.association_type == "reference"]
        value_associations = [a for a in associations if a.association_type == "value"]
        pin_associations = [a for a in associations if a.association_type == "pin_label"]
        label_associations = [a for a in associations if a.association_type == "label"]
        
        # Assign reference label (highest confidence)
        if reference_associations:
            enriched.reference_label = reference_associations[0].text_span
        
        # Assign value label (highest confidence)
        if value_associations:
            enriched.value_label = value_associations[0].text_span
        
        # Assign pin labels
        for assoc in pin_associations:
            pin_assoc = PinAssociation(
                pin_number=assoc.text_span.text,
                pin_name=None,
                text_span=assoc.text_span,
                confidence=assoc.confidence,
                position=(
                    (assoc.text_span.bbox[0] + assoc.text_span.bbox[2]) / 2,
                    (assoc.text_span.bbox[1] + assoc.text_span.bbox[3]) / 2
                )
            )
            enriched.pin_labels.append(pin_assoc)
        
        # Add additional labels
        for assoc in label_associations:
            enriched.additional_labels.append(assoc.text_span)
        
        # Calculate overall confidence
        if associations:
            enriched.confidence = sum(a.confidence for a in associations) / len(associations)
        
        return enriched
    
    def find_pin_associations(
        self, 
        component: ComponentWithText, 
        estimated_pins: List[Tuple[float, float]]
    ) -> List[PinAssociation]:
        """
        Associate pin labels with estimated pin positions.
        
        Args:
            component: Component with associated text
            estimated_pins: List of estimated pin positions
            
        Returns:
            List of pin associations with positions
        """
        pin_associations = []
        
        if not component.pin_labels or not estimated_pins:
            return pin_associations
        
        # Build spatial index for pins
        pin_array = np.array(estimated_pins)
        pin_tree = KDTree(pin_array)
        
        for pin_label in component.pin_labels:
            # Find closest pin position
            distances, indices = pin_tree.query([pin_label.position], k=1)
            
            if distances[0] <= 20.0:  # 20 pixel tolerance
                closest_pin = estimated_pins[indices[0]]
                
                # Update pin association with physical position
                updated_pin = PinAssociation(
                    pin_number=pin_label.pin_number,
                    pin_name=pin_label.pin_name,
                    text_span=pin_label.text_span,
                    confidence=pin_label.confidence * (1.0 - distances[0] / 20.0),
                    position=closest_pin
                )
                
                pin_associations.append(updated_pin)
        
        return pin_associations
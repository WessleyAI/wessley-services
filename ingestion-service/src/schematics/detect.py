"""
Symbol and component detection for electrical schematics.
"""
import os
import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from ..core.schemas import PageImage, Component, ComponentType, Pin


@dataclass
class Detection:
    """Represents a detected component/symbol."""
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    
    def to_component(self, component_id: str, page: int) -> Component:
        """Convert detection to Component schema."""
        # Map class name to ComponentType enum
        component_type = self._map_class_to_type(self.class_name)
        
        # Estimate pins based on component type
        pins = self._estimate_pins(component_type)
        
        return Component(
            id=component_id,
            type=component_type,
            page=page,
            bbox=self.bbox,
            pins=pins,
            confidence=self.confidence,
            provenance={"detection_class": self.class_name}
        )
    
    def _map_class_to_type(self, class_name: str) -> ComponentType:
        """Map detection class name to ComponentType enum."""
        mapping = {
            "resistor": ComponentType.RESISTOR,
            "capacitor": ComponentType.CAPACITOR,
            "polarized_cap": ComponentType.POLARIZED_CAP,
            "inductor": ComponentType.INDUCTOR,
            "diode": ComponentType.DIODE,
            "zener": ComponentType.ZENER,
            "bjt_npn": ComponentType.BJT_NPN,
            "bjt_pnp": ComponentType.BJT_PNP,
            "mosfet_n": ComponentType.MOSFET_N,
            "mosfet_p": ComponentType.MOSFET_P,
            "opamp": ComponentType.OPAMP,
            "ground": ComponentType.GROUND,
            "power_flag": ComponentType.POWER_FLAG,
            "connector": ComponentType.CONNECTOR,
            "ic": ComponentType.IC,
            "fuse": ComponentType.FUSE,
            "relay": ComponentType.RELAY,
            "lamp": ComponentType.LAMP,
            "switch": ComponentType.SWITCH,
            "net_label": ComponentType.NET_LABEL,
            "junction": ComponentType.JUNCTION,
            "arrow": ComponentType.ARROW,
        }
        return mapping.get(class_name.lower(), ComponentType.IC)
    
    def _estimate_pins(self, component_type: ComponentType) -> List[Pin]:
        """Estimate pin locations based on component type and bbox."""
        pins = []
        x1, y1, x2, y2 = self.bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        if component_type in [ComponentType.RESISTOR, ComponentType.CAPACITOR, 
                             ComponentType.INDUCTOR, ComponentType.DIODE]:
            # Two-terminal components - pins on left and right
            pins = [
                Pin(name="1", bbox=[x1-2, center_y-2, x1+2, center_y+2], page=1),
                Pin(name="2", bbox=[x2-2, center_y-2, x2+2, center_y+2], page=1)
            ]
        elif component_type in [ComponentType.BJT_NPN, ComponentType.BJT_PNP]:
            # Three-terminal transistors - base, collector, emitter
            pins = [
                Pin(name="B", bbox=[x1-2, center_y-2, x1+2, center_y+2], page=1),  # Base
                Pin(name="C", bbox=[center_x-2, y1-2, center_x+2, y1+2], page=1),  # Collector
                Pin(name="E", bbox=[center_x-2, y2-2, center_x+2, y2+2], page=1)   # Emitter
            ]
        elif component_type == ComponentType.OPAMP:
            # Op-amp - typical 8-pin or simplified 5-pin
            pins = [
                Pin(name="+", bbox=[x1-2, y1+5, x1+2, y1+9], page=1),    # Non-inverting
                Pin(name="-", bbox=[x1-2, y2-9, x1+2, y2-5], page=1),    # Inverting
                Pin(name="OUT", bbox=[x2-2, center_y-2, x2+2, center_y+2], page=1),  # Output
                Pin(name="VCC", bbox=[center_x-2, y1-2, center_x+2, y1+2], page=1),  # Power
                Pin(name="GND", bbox=[center_x-2, y2-2, center_x+2, y2+2], page=1)   # Ground
            ]
        elif component_type == ComponentType.IC:
            # Generic IC - estimate pins along edges
            width = x2 - x1
            height = y2 - y1
            
            # Simple dual-in-line estimate
            pin_count = max(4, min(64, int((width + height) / 10)))  # Rough estimate
            pins_per_side = pin_count // 4
            
            for i in range(pins_per_side):
                # Left side pins
                pin_y = y1 + (i + 1) * height / (pins_per_side + 1)
                pins.append(Pin(name=str(i+1), bbox=[x1-2, pin_y-2, x1+2, pin_y+2], page=1))
                
                # Right side pins (if enough pins)
                if i < pins_per_side:
                    right_pin_y = y1 + (i + 1) * height / (pins_per_side + 1)
                    pins.append(Pin(name=str(pins_per_side + i + 1), 
                                  bbox=[x2-2, right_pin_y-2, x2+2, right_pin_y+2], page=1))
        
        return pins


class ComponentDetector(ABC):
    """Abstract base class for component detection algorithms."""
    
    @abstractmethod
    async def detect_components(self, page_image: PageImage) -> List[Detection]:
        """
        Detect components in a schematic page image.
        
        Args:
            page_image: Page image to analyze
            
        Returns:
            List of detected components with bounding boxes and confidence
        """
        pass
    
    @property
    @abstractmethod
    def supported_classes(self) -> List[str]:
        """Return list of component classes this detector can identify."""
        pass


class YOLOComponentDetector(ComponentDetector):
    """
    YOLO-based component detector for electrical schematics.
    """
    
    def __init__(
        self, 
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        device: str = "cpu"
    ):
        """
        Initialize YOLO detector.
        
        Args:
            model_path: Path to trained YOLO model weights
            confidence_threshold: Minimum confidence for detections
            nms_threshold: Non-maximum suppression threshold
            device: Device to run inference on ("cpu", "cuda")
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.device = device
        self.model = None
        self.class_names = self._get_default_classes()
        
        # Try to load YOLO model
        self._load_model()
    
    def _get_default_classes(self) -> List[str]:
        """Get default component classes from CLAUDE.md specification."""
        return [
            "resistor", "capacitor", "polarized_cap", "inductor", "diode", "zener",
            "bjt_npn", "bjt_pnp", "mosfet_n", "mosfet_p", "opamp", "ground", 
            "power_flag", "connector", "ic", "fuse", "relay", "lamp", "switch",
            "net_label", "junction", "arrow"
        ]
    
    def _load_model(self):
        """Load YOLO model if available."""
        try:
            # Try to import YOLO (ultralytics)
            from ultralytics import YOLO
            
            if self.model_path and os.path.exists(self.model_path):
                # Load custom trained model
                self.model = YOLO(self.model_path)
                print(f"Loaded custom YOLO model from {self.model_path}")
            else:
                # Use pre-trained model as base (will need fine-tuning for schematics)
                self.model = YOLO('yolov8n.pt')  # Nano model for speed
                print("Loaded YOLOv8 base model (requires fine-tuning for schematics)")
                
        except ImportError:
            print("YOLO (ultralytics) not available. Install with: pip install ultralytics")
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
    
    async def detect_components(self, page_image: PageImage) -> List[Detection]:
        """Detect components using YOLO model."""
        if self.model is None:
            print("YOLO model not available, falling back to placeholder detection")
            return self._fallback_detection(page_image)
        
        try:
            # Load image
            image = cv2.imread(page_image.file_path)
            if image is None:
                return []
            
            # Run YOLO inference
            results = self.model(image, conf=self.confidence_threshold, device=self.device)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i in range(len(boxes)):
                        # Extract detection data
                        bbox = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                        confidence = float(boxes.conf[i].cpu().numpy())
                        class_id = int(boxes.cls[i].cpu().numpy())
                        
                        # Map class ID to name
                        if hasattr(result, 'names') and class_id in result.names:
                            class_name = result.names[class_id]
                        else:
                            class_name = self.class_names[class_id] if class_id < len(self.class_names) else "unknown"
                        
                        detection = Detection(
                            bbox=bbox.tolist(),
                            confidence=confidence,
                            class_id=class_id,
                            class_name=class_name
                        )
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"YOLO detection failed: {e}")
            return self._fallback_detection(page_image)
    
    def _fallback_detection(self, page_image: PageImage) -> List[Detection]:
        """Fallback detection using traditional computer vision."""
        # Use traditional CV methods as fallback
        cv_detector = TraditionalComponentDetector()
        return cv_detector.detect_components_sync(page_image)
    
    @property
    def supported_classes(self) -> List[str]:
        """Return supported component classes."""
        return self.class_names


class TraditionalComponentDetector(ComponentDetector):
    """
    Traditional computer vision-based component detector.
    
    Uses template matching, contour analysis, and geometric heuristics
    to detect common electrical components.
    """
    
    def __init__(self):
        self.component_classes = self._get_default_classes()
        self.templates = {}
        self._load_templates()
    
    def _get_default_classes(self) -> List[str]:
        """Get component classes that can be detected with traditional CV."""
        return [
            "resistor", "capacitor", "inductor", "diode",
            "ground", "junction", "ic"
        ]
    
    def _load_templates(self):
        """Load or create component templates for matching."""
        # For now, create synthetic templates
        # In production, these would be loaded from files
        self.templates = {
            "resistor": self._create_resistor_template(),
            "capacitor": self._create_capacitor_template(),
            "inductor": self._create_inductor_template(),
            "diode": self._create_diode_template(),
            "ground": self._create_ground_template(),
        }
    
    def _create_resistor_template(self) -> np.ndarray:
        """Create resistor template (zigzag pattern)."""
        template = np.zeros((40, 80), dtype=np.uint8)
        
        # Draw zigzag resistor symbol
        points = []
        for i in range(8):
            x = 10 + i * 8
            y = 20 + (10 if i % 2 == 0 else -10)
            points.append([x, y])
        
        points = np.array(points, np.int32)
        cv2.polylines(template, [points], False, 255, 2)
        
        return template
    
    def _create_capacitor_template(self) -> np.ndarray:
        """Create capacitor template (two parallel lines)."""
        template = np.zeros((40, 60), dtype=np.uint8)
        
        # Draw two vertical lines
        cv2.line(template, (25, 5), (25, 35), 255, 2)
        cv2.line(template, (35, 5), (35, 35), 255, 2)
        
        return template
    
    def _create_inductor_template(self) -> np.ndarray:
        """Create inductor template (coil/loops)."""
        template = np.zeros((40, 80), dtype=np.uint8)
        
        # Draw coil loops
        for i in range(4):
            center = (20 + i * 15, 20)
            cv2.ellipse(template, center, (7, 10), 0, 0, 180, 255, 2)
        
        return template
    
    def _create_diode_template(self) -> np.ndarray:
        """Create diode template (triangle with line)."""
        template = np.zeros((40, 60), dtype=np.uint8)
        
        # Draw triangle
        triangle = np.array([[20, 10], [40, 20], [20, 30]], np.int32)
        cv2.fillPoly(template, [triangle], 255)
        
        # Draw cathode line
        cv2.line(template, (40, 10), (40, 30), 255, 2)
        
        return template
    
    def _create_ground_template(self) -> np.ndarray:
        """Create ground symbol template."""
        template = np.zeros((30, 40), dtype=np.uint8)
        
        # Draw ground symbol (multiple horizontal lines)
        for i in range(3):
            y = 15 + i * 4
            width = 20 - i * 4
            x_start = 20 - width // 2
            cv2.line(template, (x_start, y), (x_start + width, y), 255, 2)
        
        return template
    
    async def detect_components(self, page_image: PageImage) -> List[Detection]:
        """Detect components using traditional computer vision."""
        return self.detect_components_sync(page_image)
    
    def detect_components_sync(self, page_image: PageImage) -> List[Detection]:
        """Synchronous component detection."""
        try:
            # Load and preprocess image
            image = cv2.imread(page_image.file_path)
            if image is None:
                return []
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            detections = []
            
            # Template matching for each component type
            for class_name, template in self.templates.items():
                component_detections = self._template_match(gray, template, class_name)
                detections.extend(component_detections)
            
            # Geometric pattern detection
            geometric_detections = self._detect_geometric_patterns(gray)
            detections.extend(geometric_detections)
            
            # Filter overlapping detections
            detections = self._non_maximum_suppression(detections)
            
            return detections
            
        except Exception as e:
            print(f"Traditional CV detection failed: {e}")
            return []
    
    def _template_match(self, image: np.ndarray, template: np.ndarray, class_name: str) -> List[Detection]:
        """Perform template matching for a component type."""
        detections = []
        
        # Multi-scale template matching
        scales = [0.5, 0.75, 1.0, 1.25, 1.5]
        
        for scale in scales:
            # Resize template
            template_scaled = cv2.resize(template, None, fx=scale, fy=scale)
            
            if template_scaled.shape[0] > image.shape[0] or template_scaled.shape[1] > image.shape[1]:
                continue
            
            # Template matching
            result = cv2.matchTemplate(image, template_scaled, cv2.TM_CCOEFF_NORMED)
            
            # Find matches above threshold
            threshold = 0.6
            locations = np.where(result >= threshold)
            
            for pt in zip(*locations[::-1]):  # Switch x and y coordinates
                # Calculate bounding box
                x1, y1 = pt
                x2 = x1 + template_scaled.shape[1]
                y2 = y1 + template_scaled.shape[0]
                
                confidence = result[y1, x1]
                
                detection = Detection(
                    bbox=[x1, y1, x2, y2],
                    confidence=float(confidence),
                    class_id=self._get_class_id(class_name),
                    class_name=class_name
                )
                detections.append(detection)
        
        return detections
    
    def _detect_geometric_patterns(self, image: np.ndarray) -> List[Detection]:
        """Detect components based on geometric patterns."""
        detections = []
        
        # Detect circles (for connection points, junctions)
        circles = cv2.HoughCircles(
            image,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=3,
            maxRadius=15
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                detection = Detection(
                    bbox=[x-r, y-r, x+r, y+r],
                    confidence=0.7,
                    class_id=self._get_class_id("junction"),
                    class_name="junction"
                )
                detections.append(detection)
        
        # Detect rectangles (for ICs, connectors)
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Filter contours by area
            area = cv2.contourArea(contour)
            if 100 < area < 5000:
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Check if it's roughly rectangular
                if len(approx) >= 4:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Aspect ratio check for IC identification
                    aspect_ratio = w / h
                    if 0.3 < aspect_ratio < 3.0:  # Reasonable IC proportions
                        detection = Detection(
                            bbox=[x, y, x+w, y+h],
                            confidence=0.6,
                            class_id=self._get_class_id("ic"),
                            class_name="ic"
                        )
                        detections.append(detection)
        
        return detections
    
    def _non_maximum_suppression(self, detections: List[Detection], overlap_threshold: float = 0.3) -> List[Detection]:
        """Apply non-maximum suppression to remove overlapping detections."""
        if not detections:
            return []
        
        # Sort by confidence
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        keep = []
        while detections:
            # Keep the highest confidence detection
            current = detections.pop(0)
            keep.append(current)
            
            # Remove overlapping detections
            remaining = []
            for detection in detections:
                iou = self._calculate_iou(current.bbox, detection.bbox)
                if iou < overlap_threshold:
                    remaining.append(detection)
            
            detections = remaining
        
        return keep
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) for two bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _get_class_id(self, class_name: str) -> int:
        """Get class ID for a component class name."""
        if class_name in self.component_classes:
            return self.component_classes.index(class_name)
        return 0
    
    @property
    def supported_classes(self) -> List[str]:
        """Return supported component classes."""
        return self.component_classes


class HybridComponentDetector(ComponentDetector):
    """
    Hybrid detector that combines YOLO and traditional CV approaches.
    """
    
    def __init__(self, yolo_model_path: Optional[str] = None):
        self.yolo_detector = YOLOComponentDetector(yolo_model_path)
        self.cv_detector = TraditionalComponentDetector()
        
        # Combine class lists
        yolo_classes = set(self.yolo_detector.supported_classes)
        cv_classes = set(self.cv_detector.supported_classes)
        self._supported_classes = list(yolo_classes.union(cv_classes))
    
    async def detect_components(self, page_image: PageImage) -> List[Detection]:
        """Detect components using both YOLO and traditional CV."""
        # Run both detectors
        yolo_detections = await self.yolo_detector.detect_components(page_image)
        cv_detections = await self.cv_detector.detect_components(page_image)
        
        # Combine and deduplicate results
        all_detections = yolo_detections + cv_detections
        
        # Apply confidence-based filtering and NMS
        filtered_detections = self._merge_detections(all_detections)
        
        return filtered_detections
    
    def _merge_detections(self, detections: List[Detection]) -> List[Detection]:
        """Merge detections from multiple sources."""
        if not detections:
            return []
        
        # Group overlapping detections
        groups = self._group_overlapping_detections(detections)
        
        merged = []
        for group in groups:
            if len(group) == 1:
                merged.append(group[0])
            else:
                # Merge multiple detections of the same region
                best_detection = max(group, key=lambda x: x.confidence)
                merged.append(best_detection)
        
        return merged
    
    def _group_overlapping_detections(self, detections: List[Detection]) -> List[List[Detection]]:
        """Group detections that overlap significantly."""
        groups = []
        used = set()
        
        for i, detection1 in enumerate(detections):
            if i in used:
                continue
                
            group = [detection1]
            used.add(i)
            
            for j, detection2 in enumerate(detections[i+1:], i+1):
                if j in used:
                    continue
                
                iou = self.cv_detector._calculate_iou(detection1.bbox, detection2.bbox)
                if iou > 0.3:  # Overlap threshold
                    group.append(detection2)
                    used.add(j)
            
            groups.append(group)
        
        return groups
    
    @property
    def supported_classes(self) -> List[str]:
        """Return all supported component classes."""
        return self._supported_classes
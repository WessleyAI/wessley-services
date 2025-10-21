"""
Computer vision for wire path analysis and tracing in electrical systems.
Uses advanced image processing and machine learning to trace wire paths.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.cluster import DBSCAN
from skimage import morphology, measure
from scipy import ndimage
import networkx as nx
import logging

logger = logging.getLogger(__name__)

@dataclass
class WireSegment:
    """A detected wire segment."""
    id: str
    start_point: Tuple[int, int]
    end_point: Tuple[int, int]
    path_points: List[Tuple[int, int]]
    thickness: float
    color: Tuple[int, int, int]  # RGB
    confidence: float
    length: float

@dataclass
class WireConnection:
    """Connection between wire segments or to components."""
    wire1_id: str
    wire2_id: Optional[str]  # None if connected to component
    component_id: Optional[str]
    connection_point: Tuple[int, int]
    connection_type: str  # "wire_to_wire", "wire_to_component"
    confidence: float

@dataclass
class WireHarness:
    """Collection of related wires forming a harness."""
    id: str
    wire_ids: List[str]
    main_path: List[Tuple[int, int]]
    branch_points: List[Tuple[int, int]]
    estimated_gauge: float
    color_coding: str

class WireColorClassifier:
    """Classifier for wire colors based on electrical standards."""
    
    def __init__(self):
        # Standard automotive wire colors
        self.color_standards = {
            'power_positive': [(255, 0, 0), (200, 0, 0)],  # Red variants
            'power_negative': [(0, 0, 0), (50, 50, 50)],   # Black variants
            'ground': [(0, 255, 0), (0, 200, 0)],          # Green variants
            'signal': [(0, 0, 255), (100, 100, 255)],      # Blue variants
            'switched_power': [(255, 255, 0), (200, 200, 0)], # Yellow variants
            'ignition': [(255, 165, 0), (200, 140, 0)],    # Orange variants
            'lighting': [(255, 255, 255), (200, 200, 200)], # White variants
            'can_high': [(255, 0, 255), (200, 0, 200)],    # Purple variants
            'can_low': [(165, 42, 42), (140, 35, 35)]      # Brown variants
        }
    
    def classify_color(self, rgb_color: Tuple[int, int, int]) -> Tuple[str, float]:
        """Classify wire color and return confidence."""
        best_match = "unknown"
        best_distance = float('inf')
        
        for wire_type, color_variants in self.color_standards.items():
            for standard_color in color_variants:
                distance = np.linalg.norm(np.array(rgb_color) - np.array(standard_color))
                if distance < best_distance:
                    best_distance = distance
                    best_match = wire_type
        
        # Calculate confidence based on distance
        max_distance = 255 * np.sqrt(3)  # Maximum possible RGB distance
        confidence = max(0, 1 - best_distance / max_distance)
        
        return best_match, confidence

class LineDetectionCNN(nn.Module):
    """CNN for detecting wire-like structures in images."""
    
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class WireTracer:
    """Complete wire tracing system using computer vision."""
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.color_classifier = WireColorClassifier()
        
        # Initialize CNN model for wire detection
        self.line_detector = LineDetectionCNN().to(device)
        
        # Image processing parameters
        self.gaussian_blur_kernel = (5, 5)
        self.canny_low_threshold = 50
        self.canny_high_threshold = 150
        self.hough_threshold = 80
        self.min_line_length = 30
        self.max_line_gap = 10
        
        # Wire detection parameters
        self.min_wire_thickness = 2
        self.max_wire_thickness = 20
        self.wire_confidence_threshold = 0.6
    
    def trace_wires_in_image(self, image: np.ndarray) -> Tuple[List[WireSegment], List[WireConnection]]:
        """Trace all wires in an image."""
        logger.info("Starting wire tracing analysis")
        
        # Preprocess image
        processed_image = self._preprocess_image(image)
        
        # Detect potential wire regions using CNN
        wire_mask = self._detect_wire_regions_cnn(image)
        
        # Traditional computer vision approach as backup
        cv_segments = self._detect_wires_traditional_cv(processed_image)
        
        # Combine CNN and traditional CV results
        all_segments = self._combine_detection_results(wire_mask, cv_segments, image)
        
        # Trace individual wire paths
        traced_segments = self._trace_wire_paths(all_segments, image)
        
        # Analyze wire connections
        connections = self._analyze_connections(traced_segments, image)
        
        # Filter and validate results
        validated_segments = self._validate_wire_segments(traced_segments, image)
        validated_connections = self._validate_connections(connections, validated_segments)
        
        logger.info(f"Detected {len(validated_segments)} wire segments and {len(validated_connections)} connections")
        
        return validated_segments, validated_connections
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for wire detection."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, self.gaussian_blur_kernel, 0)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        return enhanced
    
    def _detect_wire_regions_cnn(self, image: np.ndarray) -> np.ndarray:
        """Use CNN to detect wire-like regions."""
        # Preprocess for CNN
        if len(image.shape) == 3:
            input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            input_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Resize to standard input size
        resized = cv2.resize(input_image, (512, 512))
        
        # Convert to tensor
        tensor_image = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0
        tensor_image = tensor_image.unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            wire_mask = self.line_detector(tensor_image)
        
        # Post-process
        mask = wire_mask.squeeze().cpu().numpy()
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        
        return (mask > 0.5).astype(np.uint8) * 255
    
    def _detect_wires_traditional_cv(self, processed_image: np.ndarray) -> List[WireSegment]:
        """Detect wires using traditional computer vision techniques."""
        segments = []
        
        # Edge detection
        edges = cv2.Canny(processed_image, self.canny_low_threshold, self.canny_high_threshold)
        
        # Morphological operations to connect broken edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Line detection using Hough transform
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length, 
            maxLineGap=self.max_line_gap
        )
        
        if lines is not None:
            for i, line in enumerate(lines):
                x1, y1, x2, y2 = line[0]
                
                # Calculate segment properties
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                segment = WireSegment(
                    id=f"cv_segment_{i}",
                    start_point=(x1, y1),
                    end_point=(x2, y2),
                    path_points=[(x1, y1), (x2, y2)],
                    thickness=self._estimate_line_thickness(processed_image, x1, y1, x2, y2),
                    color=(128, 128, 128),  # Will be refined later
                    confidence=0.7,
                    length=length
                )
                segments.append(segment)
        
        return segments
    
    def _estimate_line_thickness(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
        """Estimate thickness of a line segment."""
        # Create a perpendicular profile across the line
        line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if line_length == 0:
            return 1.0
        
        # Unit vector along the line
        line_dx = (x2 - x1) / line_length
        line_dy = (y2 - y1) / line_length
        
        # Perpendicular unit vector
        perp_dx = -line_dy
        perp_dy = line_dx
        
        # Sample point at line center
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Create intensity profile perpendicular to line
        profile_length = 20  # pixels
        intensities = []
        
        for i in range(-profile_length, profile_length + 1):
            sample_x = int(center_x + i * perp_dx)
            sample_y = int(center_y + i * perp_dy)
            
            if 0 <= sample_x < image.shape[1] and 0 <= sample_y < image.shape[0]:
                intensities.append(image[sample_y, sample_x])
            else:
                intensities.append(255)  # Background value
        
        # Find wire edges (transitions from dark to light)
        intensities = np.array(intensities)
        gradient = np.gradient(intensities)
        
        # Find peaks in gradient (edge locations)
        edges = []
        for i in range(1, len(gradient) - 1):
            if abs(gradient[i]) > 20:  # Threshold for edge detection
                edges.append(i)
        
        # Estimate thickness as distance between outermost edges
        if len(edges) >= 2:
            thickness = abs(edges[-1] - edges[0])
        else:
            thickness = 3  # Default thickness
        
        return max(self.min_wire_thickness, min(thickness, self.max_wire_thickness))
    
    def _combine_detection_results(self, cnn_mask: np.ndarray, cv_segments: List[WireSegment], 
                                 original_image: np.ndarray) -> List[WireSegment]:
        """Combine CNN and traditional CV detection results."""
        combined_segments = cv_segments.copy()
        
        # Extract segments from CNN mask
        # Connected component analysis
        labeled_mask = measure.label(cnn_mask)
        regions = measure.regionprops(labeled_mask)
        
        for i, region in enumerate(regions):
            if region.area > 100:  # Filter small regions
                # Extract skeleton of the region
                skeleton = morphology.skeletonize(labeled_mask == region.label)
                
                # Find endpoints and trace path
                endpoints = self._find_skeleton_endpoints(skeleton)
                
                if len(endpoints) >= 2:
                    # Create wire segment from skeleton
                    path_points = self._extract_skeleton_path(skeleton, endpoints[0], endpoints[-1])
                    
                    if len(path_points) > 5:  # Minimum path length
                        segment = WireSegment(
                            id=f"cnn_segment_{i}",
                            start_point=endpoints[0],
                            end_point=endpoints[-1],
                            path_points=path_points,
                            thickness=self._estimate_region_thickness(region),
                            color=self._extract_dominant_color(original_image, labeled_mask == region.label),
                            confidence=0.8,
                            length=len(path_points) * 1.0  # Approximate length
                        )
                        combined_segments.append(segment)
        
        return combined_segments
    
    def _find_skeleton_endpoints(self, skeleton: np.ndarray) -> List[Tuple[int, int]]:
        """Find endpoints of a skeletonized wire."""
        # Endpoint detection: pixels with only one neighbor
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
        neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
        
        endpoints = []
        y_coords, x_coords = np.where((skeleton > 0) & (neighbor_count == 1))
        
        for x, y in zip(x_coords, y_coords):
            endpoints.append((x, y))
        
        return endpoints
    
    def _extract_skeleton_path(self, skeleton: np.ndarray, start: Tuple[int, int], 
                              end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Extract path from skeleton between two points."""
        # Use A* pathfinding on skeleton
        def heuristic(a, b):
            return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        
        def get_neighbors(pos):
            x, y = pos
            neighbors = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < skeleton.shape[1] and 0 <= ny < skeleton.shape[0] and 
                        skeleton[ny, nx] > 0):
                        neighbors.append((nx, ny))
            return neighbors
        
        # A* algorithm
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, end)}
        
        while open_set:
            current = min(open_set, key=lambda x: f_score.get(x[1], float('inf')))[1]
            open_set = [x for x in open_set if x[1] != current]
            
            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            for neighbor in get_neighbors(current):
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, end)
                    
                    if (f_score[neighbor], neighbor) not in open_set:
                        open_set.append((f_score[neighbor], neighbor))
        
        # If no path found, return straight line approximation
        return [start, end]
    
    def _estimate_region_thickness(self, region) -> float:
        """Estimate thickness of a region."""
        # Use the minor axis length as thickness estimate
        return max(self.min_wire_thickness, min(region.minor_axis_length, self.max_wire_thickness))
    
    def _extract_dominant_color(self, image: np.ndarray, mask: np.ndarray) -> Tuple[int, int, int]:
        """Extract dominant color from masked region."""
        if len(image.shape) == 3:
            masked_pixels = image[mask > 0]
            if len(masked_pixels) > 0:
                # Use median color as representative
                return tuple(np.median(masked_pixels, axis=0).astype(int))
        
        return (128, 128, 128)  # Default gray
    
    def _trace_wire_paths(self, segments: List[WireSegment], image: np.ndarray) -> List[WireSegment]:
        """Refine wire path tracing for better accuracy."""
        traced_segments = []
        
        for segment in segments:
            # Improve color classification
            color_type, color_confidence = self.color_classifier.classify_color(segment.color)
            
            # Refine path points using active contours or other methods
            refined_path = self._refine_wire_path(segment.path_points, image)
            
            # Create improved segment
            improved_segment = WireSegment(
                id=segment.id,
                start_point=refined_path[0] if refined_path else segment.start_point,
                end_point=refined_path[-1] if refined_path else segment.end_point,
                path_points=refined_path,
                thickness=segment.thickness,
                color=segment.color,
                confidence=segment.confidence * color_confidence,
                length=self._calculate_path_length(refined_path)
            )
            
            traced_segments.append(improved_segment)
        
        return traced_segments
    
    def _refine_wire_path(self, initial_path: List[Tuple[int, int]], 
                         image: np.ndarray) -> List[Tuple[int, int]]:
        """Refine wire path using advanced tracking."""
        if len(initial_path) < 2:
            return initial_path
        
        # Simple path smoothing for now
        # In a complete implementation, this could use:
        # - Active contours (snakes)
        # - Kalman filtering
        # - Template matching
        # - Particle filters
        
        smoothed_path = []
        window_size = 3
        
        for i in range(len(initial_path)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(initial_path), i + window_size // 2 + 1)
            
            window_points = initial_path[start_idx:end_idx]
            avg_x = sum(p[0] for p in window_points) // len(window_points)
            avg_y = sum(p[1] for p in window_points) // len(window_points)
            
            smoothed_path.append((avg_x, avg_y))
        
        return smoothed_path
    
    def _calculate_path_length(self, path_points: List[Tuple[int, int]]) -> float:
        """Calculate total length of wire path."""
        if len(path_points) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(path_points) - 1):
            dx = path_points[i+1][0] - path_points[i][0]
            dy = path_points[i+1][1] - path_points[i][1]
            total_length += np.sqrt(dx*dx + dy*dy)
        
        return total_length
    
    def _analyze_connections(self, segments: List[WireSegment], 
                           image: np.ndarray) -> List[WireConnection]:
        """Analyze connections between wire segments."""
        connections = []
        connection_threshold = 10  # pixels
        
        # Check for wire-to-wire connections
        for i, seg1 in enumerate(segments):
            for j, seg2 in enumerate(segments[i+1:], i+1):
                # Check if endpoints are close
                distances = [
                    np.linalg.norm(np.array(seg1.start_point) - np.array(seg2.start_point)),
                    np.linalg.norm(np.array(seg1.start_point) - np.array(seg2.end_point)),
                    np.linalg.norm(np.array(seg1.end_point) - np.array(seg2.start_point)),
                    np.linalg.norm(np.array(seg1.end_point) - np.array(seg2.end_point))
                ]
                
                min_distance = min(distances)
                if min_distance < connection_threshold:
                    # Find connection point
                    min_idx = distances.index(min_distance)
                    if min_idx == 0:
                        conn_point = seg1.start_point
                    elif min_idx == 1:
                        conn_point = seg1.start_point
                    elif min_idx == 2:
                        conn_point = seg1.end_point
                    else:
                        conn_point = seg1.end_point
                    
                    connection = WireConnection(
                        wire1_id=seg1.id,
                        wire2_id=seg2.id,
                        component_id=None,
                        connection_point=conn_point,
                        connection_type="wire_to_wire",
                        confidence=1.0 - min_distance / connection_threshold
                    )
                    connections.append(connection)
        
        return connections
    
    def _validate_wire_segments(self, segments: List[WireSegment], 
                              image: np.ndarray) -> List[WireSegment]:
        """Validate and filter wire segments."""
        validated = []
        
        for segment in segments:
            # Check minimum length
            if segment.length < 20:  # pixels
                continue
            
            # Check confidence threshold
            if segment.confidence < self.wire_confidence_threshold:
                continue
            
            # Check thickness reasonableness
            if segment.thickness < self.min_wire_thickness or segment.thickness > self.max_wire_thickness:
                continue
            
            validated.append(segment)
        
        return validated
    
    def _validate_connections(self, connections: List[WireConnection], 
                            segments: List[WireSegment]) -> List[WireConnection]:
        """Validate wire connections."""
        segment_ids = {seg.id for seg in segments}
        validated = []
        
        for connection in connections:
            # Check that referenced segments exist
            if connection.wire1_id in segment_ids:
                if connection.wire2_id is None or connection.wire2_id in segment_ids:
                    if connection.confidence > 0.5:
                        validated.append(connection)
        
        return validated
    
    def detect_wire_harnesses(self, segments: List[WireSegment], 
                            connections: List[WireConnection]) -> List[WireHarness]:
        """Detect and group wires into harnesses."""
        # Build graph of wire connections
        G = nx.Graph()
        
        for segment in segments:
            G.add_node(segment.id, segment=segment)
        
        for connection in connections:
            if connection.wire2_id:
                G.add_edge(connection.wire1_id, connection.wire2_id, 
                          connection=connection)
        
        # Find connected components (harnesses)
        harnesses = []
        for i, component in enumerate(nx.connected_components(G)):
            if len(component) > 2:  # Harness should have at least 3 wires
                harness_segments = [G.nodes[wire_id]['segment'] for wire_id in component]
                
                # Analyze harness properties
                main_path = self._find_harness_main_path(harness_segments)
                branch_points = self._find_branch_points(harness_segments, connections)
                estimated_gauge = self._estimate_harness_gauge(harness_segments)
                
                harness = WireHarness(
                    id=f"harness_{i}",
                    wire_ids=list(component),
                    main_path=main_path,
                    branch_points=branch_points,
                    estimated_gauge=estimated_gauge,
                    color_coding=self._analyze_color_coding(harness_segments)
                )
                harnesses.append(harness)
        
        return harnesses
    
    def _find_harness_main_path(self, segments: List[WireSegment]) -> List[Tuple[int, int]]:
        """Find the main path of a wire harness."""
        # Find the longest continuous path
        all_points = []
        for segment in segments:
            all_points.extend(segment.path_points)
        
        if not all_points:
            return []
        
        # Use minimum spanning tree to find main path
        # Simplified: return the path of the longest segment
        longest_segment = max(segments, key=lambda s: s.length)
        return longest_segment.path_points
    
    def _find_branch_points(self, segments: List[WireSegment], 
                          connections: List[WireConnection]) -> List[Tuple[int, int]]:
        """Find branch points in wire harness."""
        branch_points = []
        
        # Count connections at each point
        point_connections = {}
        for connection in connections:
            point = connection.connection_point
            point_connections[point] = point_connections.get(point, 0) + 1
        
        # Points with more than 2 connections are branch points
        for point, count in point_connections.items():
            if count > 2:
                branch_points.append(point)
        
        return branch_points
    
    def _estimate_harness_gauge(self, segments: List[WireSegment]) -> float:
        """Estimate the overall gauge of a wire harness."""
        # Use the average thickness as gauge estimate
        if not segments:
            return 1.0
        
        avg_thickness = sum(seg.thickness for seg in segments) / len(segments)
        
        # Convert thickness to AWG (simplified mapping)
        if avg_thickness < 3:
            return 22  # AWG
        elif avg_thickness < 5:
            return 20
        elif avg_thickness < 8:
            return 18
        elif avg_thickness < 12:
            return 16
        else:
            return 14
    
    def _analyze_color_coding(self, segments: List[WireSegment]) -> str:
        """Analyze color coding pattern of harness."""
        colors = [seg.color for seg in segments]
        
        # Classify colors
        color_types = []
        for color in colors:
            color_type, _ = self.color_classifier.classify_color(color)
            color_types.append(color_type)
        
        # Determine coding scheme
        unique_colors = set(color_types)
        if len(unique_colors) == 1:
            return f"single_color_{list(unique_colors)[0]}"
        elif "power_positive" in unique_colors and "power_negative" in unique_colors:
            return "power_harness"
        elif "signal" in unique_colors:
            return "signal_harness"
        else:
            return "mixed_harness"
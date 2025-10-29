"""
Wire and line segment extraction for electrical schematics.
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass
import math
from collections import defaultdict

from ..core.schemas import PageImage


@dataclass
class LineSegment:
    """Represents a line segment in the schematic."""
    start: Tuple[float, float]  # (x, y)
    end: Tuple[float, float]    # (x, y)
    thickness: float
    confidence: float
    
    def length(self) -> float:
        """Calculate line segment length."""
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return math.sqrt(dx*dx + dy*dy)
    
    def angle(self) -> float:
        """Calculate line angle in radians."""
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return math.atan2(dy, dx)
    
    def midpoint(self) -> Tuple[float, float]:
        """Calculate midpoint of the line."""
        return (
            (self.start[0] + self.end[0]) / 2,
            (self.start[1] + self.end[1]) / 2
        )
    
    def is_horizontal(self, tolerance: float = 0.1) -> bool:
        """Check if line is approximately horizontal."""
        return abs(self.angle()) < tolerance or abs(abs(self.angle()) - math.pi) < tolerance
    
    def is_vertical(self, tolerance: float = 0.1) -> bool:
        """Check if line is approximately vertical."""
        return abs(abs(self.angle()) - math.pi/2) < tolerance
    
    def distance_to_point(self, point: Tuple[float, float]) -> float:
        """Calculate perpendicular distance from point to line."""
        x0, y0 = point
        x1, y1 = self.start
        x2, y2 = self.end
        
        # Line equation: ax + by + c = 0
        a = y2 - y1
        b = x1 - x2
        c = x2*y1 - x1*y2
        
        # Distance formula
        if a == 0 and b == 0:
            return math.sqrt((x0-x1)**2 + (y0-y1)**2)
        
        return abs(a*x0 + b*y0 + c) / math.sqrt(a*a + b*b)
    
    def intersects(self, other: 'LineSegment', tolerance: float = 5.0) -> Optional[Tuple[float, float]]:
        """Find intersection point with another line segment."""
        x1, y1 = self.start
        x2, y2 = self.end
        x3, y3 = other.start
        x4, y4 = other.end
        
        # Calculate intersection using parametric equations
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        
        if abs(denom) < 1e-6:  # Lines are parallel
            return None
        
        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
        u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / denom
        
        # Check if intersection is within both line segments
        if 0 <= t <= 1 and 0 <= u <= 1:
            x = x1 + t*(x2-x1)
            y = y1 + t*(y2-y1)
            return (x, y)
        
        return None


@dataclass
class Junction:
    """Represents a wire junction or connection point."""
    position: Tuple[float, float]
    connected_lines: List[int]  # Indices of connected line segments
    type: str  # "T", "cross", "corner", "endpoint"
    confidence: float


@dataclass
class NetLabel:
    """Represents a text label on a wire net."""
    text: str
    position: Tuple[float, float]
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    label_type: str  # "voltage", "signal", "ground", "power", "bus", "custom"


@dataclass 
class WireNet:
    """Represents a connected network of wires."""
    segments: List[int]  # Indices of line segments in this net
    junctions: List[int]  # Indices of junctions in this net
    endpoints: List[Tuple[float, float]]  # Connection points to components
    label: Optional[str] = None
    net_labels: List[NetLabel] = None  # Associated net labels
    propagated_name: Optional[str] = None  # Final propagated net name
    voltage_level: Optional[str] = None  # e.g., "5V", "GND", "VCC"
    is_bus: bool = False  # Whether this is a bus (multi-bit signal)
    bus_width: Optional[int] = None  # Number of bits if bus


class WireExtractor:
    """
    Extracts wire segments and traces connections in electrical schematics.
    """
    
    def __init__(
        self,
        min_line_length: float = 10.0,
        hough_threshold: int = 80,
        hough_min_line_length: int = 30,
        hough_max_line_gap: int = 10,
        junction_tolerance: float = 8.0
    ):
        """
        Initialize wire extractor.
        
        Args:
            min_line_length: Minimum length for valid wire segments
            hough_threshold: Hough transform threshold
            hough_min_line_length: Minimum line length for Hough transform
            hough_max_line_gap: Maximum gap in line for Hough transform
            junction_tolerance: Distance tolerance for junction detection
        """
        self.min_line_length = min_line_length
        self.hough_threshold = hough_threshold
        self.hough_min_line_length = hough_min_line_length
        self.hough_max_line_gap = hough_max_line_gap
        self.junction_tolerance = junction_tolerance
    
    def extract_wires(self, page_image: PageImage, text_spans: List = None) -> Tuple[List[LineSegment], List[Junction], List[WireNet]]:
        """
        Extract wire segments, junctions, and nets from schematic image.
        
        Args:
            page_image: Input schematic image
            text_spans: Optional list of detected text spans for net labeling
            
        Returns:
            Tuple of (line_segments, junctions, wire_nets)
        """
        # Load and preprocess image
        image = cv2.imread(page_image.file_path)
        if image is None:
            return [], [], []
        
        # Preprocess for line detection
        processed_image = self._preprocess_for_lines(image)
        
        # Extract line segments
        line_segments = self._detect_line_segments(processed_image)
        
        # Filter and clean line segments
        line_segments = self._filter_line_segments(line_segments)
        
        # Detect junctions with enhanced algorithm
        junctions = self._detect_junctions_enhanced(line_segments)
        
        # Trace wire networks
        wire_nets = self._trace_wire_networks(line_segments, junctions)
        
        # Associate text labels with nets if available
        if text_spans:
            wire_nets = self._associate_net_labels(wire_nets, line_segments, text_spans)
        
        # Propagate net names and resolve conflicts
        wire_nets = self._propagate_net_names(wire_nets)
        
        return line_segments, junctions, wire_nets
    
    def _preprocess_for_lines(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for optimal line detection.
        
        Args:
            image: Input BGR image
            
        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply morphological operations to clean up thin lines
        kernel = np.ones((3, 3), np.uint8)
        
        # Close small gaps in lines
        closed = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            closed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Apply edge detection
        edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
        
        return edges
    
    def _detect_line_segments(self, image: np.ndarray) -> List[LineSegment]:
        """
        Detect line segments using Hough Line Transform.
        
        Args:
            image: Preprocessed binary image
            
        Returns:
            List of detected line segments
        """
        line_segments = []
        
        # Probabilistic Hough Line Transform
        lines = cv2.HoughLinesP(
            image,
            rho=1,                                    # Distance resolution
            theta=np.pi/180,                         # Angular resolution  
            threshold=self.hough_threshold,          # Minimum votes
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap
        )
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Create line segment
                segment = LineSegment(
                    start=(float(x1), float(y1)),
                    end=(float(x2), float(y2)),
                    thickness=2.0,  # Default thickness
                    confidence=1.0  # Default confidence
                )
                
                line_segments.append(segment)
        
        # Detect additional line segments using contour analysis
        contour_lines = self._detect_lines_from_contours(image)
        line_segments.extend(contour_lines)
        
        return line_segments
    
    def _detect_lines_from_contours(self, image: np.ndarray) -> List[LineSegment]:
        """
        Detect line segments from contours (for curved or complex lines).
        
        Args:
            image: Binary image
            
        Returns:
            List of line segments from contour analysis
        """
        line_segments = []
        
        # Find contours
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Filter by area (remove very small contours)
            area = cv2.contourArea(contour)
            if area < 20:
                continue
            
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Convert polygon edges to line segments
            for i in range(len(approx)):
                p1 = approx[i][0]
                p2 = approx[(i + 1) % len(approx)][0]
                
                segment = LineSegment(
                    start=(float(p1[0]), float(p1[1])),
                    end=(float(p2[0]), float(p2[1])),
                    thickness=2.0,
                    confidence=0.8
                )
                
                # Only add if long enough
                if segment.length() >= self.min_line_length:
                    line_segments.append(segment)
        
        return line_segments
    
    def _filter_line_segments(self, line_segments: List[LineSegment]) -> List[LineSegment]:
        """
        Filter and clean line segments.
        
        Args:
            line_segments: Raw detected line segments
            
        Returns:
            Filtered line segments
        """
        filtered = []
        
        for segment in line_segments:
            # Filter by minimum length
            if segment.length() < self.min_line_length:
                continue
            
            # Filter out segments that are too short or at odd angles
            # (these might be noise or component parts)
            angle = segment.angle()
            normalized_angle = abs(angle) % (math.pi/2)
            
            # Prefer horizontal, vertical, or 45-degree lines for wires
            if (normalized_angle < 0.2 or  # ~11 degrees
                abs(normalized_angle - math.pi/4) < 0.2 or  # 45 degrees
                abs(normalized_angle - math.pi/2) < 0.2):   # 90 degrees
                filtered.append(segment)
            elif segment.length() > 50:  # Keep longer lines even if at odd angles
                filtered.append(segment)
        
        # Merge nearby collinear segments
        merged = self._merge_collinear_segments(filtered)
        
        return merged
    
    def _merge_collinear_segments(self, segments: List[LineSegment]) -> List[LineSegment]:
        """
        Merge nearby collinear line segments.
        
        Args:
            segments: Input line segments
            
        Returns:
            Merged line segments
        """
        if not segments:
            return []
        
        merged = []
        used = set()
        
        for i, seg1 in enumerate(segments):
            if i in used:
                continue
            
            # Find segments that can be merged with this one
            merge_group = [seg1]
            used.add(i)
            
            for j, seg2 in enumerate(segments[i+1:], i+1):
                if j in used:
                    continue
                
                # Check if segments are collinear and close
                if self._can_merge_segments(seg1, seg2):
                    merge_group.append(seg2)
                    used.add(j)
            
            # Merge the group into a single segment
            if len(merge_group) == 1:
                merged.append(merge_group[0])
            else:
                merged_segment = self._merge_segment_group(merge_group)
                merged.append(merged_segment)
        
        return merged
    
    def _can_merge_segments(self, seg1: LineSegment, seg2: LineSegment, 
                           angle_tolerance: float = 0.1, distance_tolerance: float = 10.0) -> bool:
        """
        Check if two segments can be merged (are collinear and close).
        
        Args:
            seg1, seg2: Line segments to check
            angle_tolerance: Maximum angle difference in radians
            distance_tolerance: Maximum distance between segments
            
        Returns:
            True if segments can be merged
        """
        # Check angle similarity
        angle_diff = abs(seg1.angle() - seg2.angle())
        angle_diff = min(angle_diff, abs(angle_diff - math.pi))  # Handle angle wraparound
        
        if angle_diff > angle_tolerance:
            return False
        
        # Check if endpoints are close
        distances = [
            math.sqrt((seg1.start[0] - seg2.start[0])**2 + (seg1.start[1] - seg2.start[1])**2),
            math.sqrt((seg1.start[0] - seg2.end[0])**2 + (seg1.start[1] - seg2.end[1])**2),
            math.sqrt((seg1.end[0] - seg2.start[0])**2 + (seg1.end[1] - seg2.start[1])**2),
            math.sqrt((seg1.end[0] - seg2.end[0])**2 + (seg1.end[1] - seg2.end[1])**2)
        ]
        
        return min(distances) <= distance_tolerance
    
    def _merge_segment_group(self, segments: List[LineSegment]) -> LineSegment:
        """
        Merge a group of collinear segments into one.
        
        Args:
            segments: List of segments to merge
            
        Returns:
            Single merged segment
        """
        # Collect all endpoints
        points = []
        for seg in segments:
            points.extend([seg.start, seg.end])
        
        # Find the two points that are farthest apart
        max_distance = 0
        best_start = points[0]
        best_end = points[1]
        
        for i, p1 in enumerate(points):
            for j, p2 in enumerate(points[i+1:], i+1):
                distance = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                if distance > max_distance:
                    max_distance = distance
                    best_start = p1
                    best_end = p2
        
        # Calculate average properties
        avg_thickness = sum(seg.thickness for seg in segments) / len(segments)
        avg_confidence = sum(seg.confidence for seg in segments) / len(segments)
        
        return LineSegment(
            start=best_start,
            end=best_end,
            thickness=avg_thickness,
            confidence=avg_confidence
        )
    
    def _detect_junctions(self, line_segments: List[LineSegment]) -> List[Junction]:
        """
        Detect wire junctions and connection points.
        
        Args:
            line_segments: List of line segments
            
        Returns:
            List of detected junctions
        """
        junctions = []
        
        # Find intersection points between line segments
        intersections = self._find_intersections(line_segments)
        
        # Find endpoints that are close to other lines (T-junctions)
        t_junctions = self._find_t_junctions(line_segments)
        
        # Combine and deduplicate
        all_junction_points = intersections + t_junctions
        deduplicated = self._deduplicate_junctions(all_junction_points)
        
        # Create Junction objects
        for i, (position, connected_lines, junction_type) in enumerate(deduplicated):
            junction = Junction(
                position=position,
                connected_lines=connected_lines,
                type=junction_type,
                confidence=0.9
            )
            junctions.append(junction)
        
        return junctions
    
    def _find_intersections(self, line_segments: List[LineSegment]) -> List[Tuple[Tuple[float, float], List[int], str]]:
        """Find intersection points between line segments."""
        intersections = []
        
        for i, seg1 in enumerate(line_segments):
            for j, seg2 in enumerate(line_segments[i+1:], i+1):
                intersection = seg1.intersects(seg2)
                if intersection:
                    intersections.append((intersection, [i, j], "cross"))
        
        return intersections
    
    def _find_t_junctions(self, line_segments: List[LineSegment]) -> List[Tuple[Tuple[float, float], List[int], str]]:
        """Find T-junctions where line endpoints meet other lines."""
        t_junctions = []
        
        for i, seg in enumerate(line_segments):
            for endpoint in [seg.start, seg.end]:
                # Find other lines that this endpoint is close to
                close_lines = []
                
                for j, other_seg in enumerate(line_segments):
                    if i == j:
                        continue
                    
                    distance = other_seg.distance_to_point(endpoint)
                    if distance <= self.junction_tolerance:
                        close_lines.append(j)
                
                if close_lines:
                    close_lines.append(i)  # Include the line with this endpoint
                    t_junctions.append((endpoint, close_lines, "T"))
        
        return t_junctions
    
    def _deduplicate_junctions(self, junctions: List[Tuple[Tuple[float, float], List[int], str]]) -> List[Tuple[Tuple[float, float], List[int], str]]:
        """Remove duplicate junction points that are very close."""
        if not junctions:
            return []
        
        deduplicated = []
        used = set()
        
        for i, (pos1, lines1, type1) in enumerate(junctions):
            if i in used:
                continue
            
            # Find nearby junctions to merge
            merge_group = [(pos1, lines1, type1)]
            used.add(i)
            
            for j, (pos2, lines2, type2) in enumerate(junctions[i+1:], i+1):
                if j in used:
                    continue
                
                distance = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                if distance <= self.junction_tolerance:
                    merge_group.append((pos2, lines2, type2))
                    used.add(j)
            
            # Merge the group
            if len(merge_group) == 1:
                deduplicated.append(merge_group[0])
            else:
                # Calculate average position
                avg_x = sum(pos[0] for pos, _, _ in merge_group) / len(merge_group)
                avg_y = sum(pos[1] for pos, _, _ in merge_group) / len(merge_group)
                
                # Combine connected lines
                all_lines = set()
                for _, lines, _ in merge_group:
                    all_lines.update(lines)
                
                # Determine junction type based on number of connections
                if len(all_lines) >= 4:
                    junction_type = "cross"
                elif len(all_lines) == 3:
                    junction_type = "T"
                else:
                    junction_type = "corner"
                
                deduplicated.append(((avg_x, avg_y), list(all_lines), junction_type))
        
        return deduplicated
    
    def _trace_wire_networks(self, line_segments: List[LineSegment], junctions: List[Junction]) -> List[WireNet]:
        """
        Trace connected wire networks using graph traversal.
        
        Args:
            line_segments: List of line segments
            junctions: List of junctions
            
        Returns:
            List of connected wire networks
        """
        # Build adjacency graph
        adjacency = self._build_adjacency_graph(line_segments, junctions)
        
        # Find connected components
        wire_nets = []
        visited_segments = set()
        
        for i, segment in enumerate(line_segments):
            if i in visited_segments:
                continue
            
            # Start a new network from this segment
            net_segments = set()
            net_junctions = set()
            
            # DFS to find all connected segments
            self._dfs_trace_network(i, adjacency, net_segments, net_junctions, visited_segments)
            
            # Find endpoints (connections to components)
            endpoints = self._find_network_endpoints(list(net_segments), line_segments, junctions)
            
            wire_net = WireNet(
                segments=list(net_segments),
                junctions=list(net_junctions),
                endpoints=endpoints
            )
            
            wire_nets.append(wire_net)
        
        return wire_nets
    
    def _build_adjacency_graph(self, line_segments: List[LineSegment], junctions: List[Junction]) -> Dict[int, Set[int]]:
        """Build adjacency graph of connected line segments."""
        adjacency = defaultdict(set)
        
        # Connect segments through junctions
        for junction in junctions:
            connected_lines = junction.connected_lines
            # Connect each pair of lines in this junction
            for i in range(len(connected_lines)):
                for j in range(i + 1, len(connected_lines)):
                    line1 = connected_lines[i]
                    line2 = connected_lines[j]
                    adjacency[line1].add(line2)
                    adjacency[line2].add(line1)
        
        return adjacency
    
    def _dfs_trace_network(self, segment_idx: int, adjacency: Dict[int, Set[int]], 
                          net_segments: Set[int], net_junctions: Set[int], visited: Set[int]):
        """DFS traversal to trace connected wire network."""
        if segment_idx in visited:
            return
        
        visited.add(segment_idx)
        net_segments.add(segment_idx)
        
        # Visit all connected segments
        for connected_idx in adjacency[segment_idx]:
            if connected_idx not in visited:
                self._dfs_trace_network(connected_idx, adjacency, net_segments, net_junctions, visited)
    
    def _find_network_endpoints(self, segment_indices: List[int], line_segments: List[LineSegment], 
                               junctions: List[Junction]) -> List[Tuple[float, float]]:
        """Find endpoints of wire network (potential component connection points)."""
        endpoints = []
        
        # Collect all junction positions
        junction_positions = set(junction.position for junction in junctions)
        
        # Check segment endpoints
        for idx in segment_indices:
            segment = line_segments[idx]
            
            for endpoint in [segment.start, segment.end]:
                # Check if this endpoint is NOT at a junction
                is_at_junction = any(
                    math.sqrt((endpoint[0] - jp[0])**2 + (endpoint[1] - jp[1])**2) <= self.junction_tolerance
                    for jp in junction_positions
                )
                
                if not is_at_junction:
                    endpoints.append(endpoint)
        
        return endpoints
    
    def _detect_junctions_enhanced(self, line_segments: List[LineSegment]) -> List[Junction]:
        """
        Enhanced junction detection with improved intersection analysis.
        
        Args:
            line_segments: List of line segments
            
        Returns:
            List of detected junctions with enhanced classification
        """
        junctions = []
        
        # Find all intersection points with improved accuracy
        intersections = self._find_intersections_enhanced(line_segments)
        
        # Find T-junctions and endpoint connections
        t_junctions = self._find_t_junctions_enhanced(line_segments)
        
        # Find corner connections (L-shaped junctions)
        corner_junctions = self._find_corner_junctions(line_segments)
        
        # Combine all junction types
        all_junction_points = intersections + t_junctions + corner_junctions
        
        # Deduplicate with improved clustering
        deduplicated = self._deduplicate_junctions_enhanced(all_junction_points, line_segments)
        
        # Create Junction objects with enhanced properties
        for i, (position, connected_lines, junction_type, confidence) in enumerate(deduplicated):
            junction = Junction(
                position=position,
                connected_lines=connected_lines,
                type=junction_type,
                confidence=confidence
            )
            junctions.append(junction)
        
        return junctions
    
    def _find_intersections_enhanced(self, line_segments: List[LineSegment]) -> List[Tuple[Tuple[float, float], List[int], str, float]]:
        """Enhanced intersection detection with better handling of near-misses."""
        intersections = []
        
        for i, seg1 in enumerate(line_segments):
            for j, seg2 in enumerate(line_segments[i+1:], i+1):
                # Check for exact intersection first
                intersection = seg1.intersects(seg2)
                if intersection:
                    # Verify this is a real crossing, not just touching endpoints
                    confidence = self._calculate_intersection_confidence(seg1, seg2, intersection)
                    if confidence > 0.5:
                        intersections.append((intersection, [i, j], "cross", confidence))
                else:
                    # Check for near-miss intersections (segments that almost cross)
                    near_intersection = self._find_near_intersection(seg1, seg2)
                    if near_intersection:
                        confidence = self._calculate_intersection_confidence(seg1, seg2, near_intersection)
                        if confidence > 0.3:
                            intersections.append((near_intersection, [i, j], "cross", confidence))
        
        return intersections
    
    def _find_near_intersection(self, seg1: LineSegment, seg2: LineSegment, 
                               tolerance: float = 8.0) -> Optional[Tuple[float, float]]:
        """Find near-intersection points for segments that almost cross."""
        # Find closest points between the two line segments
        closest_points = self._closest_points_between_segments(seg1, seg2)
        if not closest_points:
            return None
        
        p1, p2 = closest_points
        distance = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        if distance <= tolerance:
            # Return midpoint as intersection
            return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        
        return None
    
    def _closest_points_between_segments(self, seg1: LineSegment, seg2: LineSegment) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Find closest points between two line segments."""
        x1, y1 = seg1.start
        x2, y2 = seg1.end
        x3, y3 = seg2.start
        x4, y4 = seg2.end
        
        # Direction vectors
        d1 = (x2 - x1, y2 - y1)
        d2 = (x4 - x3, y4 - y3)
        
        # Vector between start points
        w = (x1 - x3, y1 - y3)
        
        a = d1[0]*d1[0] + d1[1]*d1[1]  # |d1|²
        b = d1[0]*d2[0] + d1[1]*d2[1]  # d1·d2
        c = d2[0]*d2[0] + d2[1]*d2[1]  # |d2|²
        d = d1[0]*w[0] + d1[1]*w[1]    # d1·w
        e = d2[0]*w[0] + d2[1]*w[1]    # d2·w
        
        denom = a*c - b*b
        if abs(denom) < 1e-6:  # Lines are parallel
            return None
        
        # Parameters for closest points
        t1 = (b*e - c*d) / denom
        t2 = (a*e - b*d) / denom
        
        # Clamp to segment bounds
        t1 = max(0, min(1, t1))
        t2 = max(0, min(1, t2))
        
        # Calculate closest points
        p1 = (x1 + t1*d1[0], y1 + t1*d1[1])
        p2 = (x3 + t2*d2[0], y3 + t2*d2[1])
        
        return (p1, p2)
    
    def _calculate_intersection_confidence(self, seg1: LineSegment, seg2: LineSegment, 
                                         intersection: Tuple[float, float]) -> float:
        """Calculate confidence score for an intersection."""
        # Base confidence based on angle between segments
        angle1 = seg1.angle()
        angle2 = seg2.angle()
        angle_diff = abs(angle1 - angle2)
        angle_diff = min(angle_diff, abs(angle_diff - math.pi))
        
        # Higher confidence for perpendicular intersections
        angle_confidence = 1.0 - (2 * abs(angle_diff - math.pi/2) / math.pi)
        angle_confidence = max(0.1, angle_confidence)
        
        # Check if intersection is near middle of both segments
        pos_confidence1 = self._position_confidence(seg1, intersection)
        pos_confidence2 = self._position_confidence(seg2, intersection)
        
        return angle_confidence * pos_confidence1 * pos_confidence2
    
    def _position_confidence(self, segment: LineSegment, point: Tuple[float, float]) -> float:
        """Calculate confidence based on where intersection occurs on segment."""
        # Distance from start and end
        start_dist = math.sqrt((point[0] - segment.start[0])**2 + (point[1] - segment.start[1])**2)
        end_dist = math.sqrt((point[0] - segment.end[0])**2 + (point[1] - segment.end[1])**2)
        total_length = segment.length()
        
        if total_length < 1e-6:
            return 0.1
        
        # Relative position (0 = start, 1 = end)
        relative_pos = start_dist / total_length
        
        # Higher confidence for intersections in the middle
        if 0.1 <= relative_pos <= 0.9:
            return 1.0
        elif 0.05 <= relative_pos <= 0.95:
            return 0.8
        else:
            return 0.3  # Lower confidence for endpoint intersections
    
    def _find_t_junctions_enhanced(self, line_segments: List[LineSegment]) -> List[Tuple[Tuple[float, float], List[int], str, float]]:
        """Enhanced T-junction detection with better endpoint analysis."""
        t_junctions = []
        
        for i, seg in enumerate(line_segments):
            for endpoint in [seg.start, seg.end]:
                # Find other lines that this endpoint connects to
                connecting_lines = []
                connection_distances = []
                
                for j, other_seg in enumerate(line_segments):
                    if i == j:
                        continue
                    
                    # Check distance to the other segment
                    distance = other_seg.distance_to_point(endpoint)
                    if distance <= self.junction_tolerance:
                        connecting_lines.append(j)
                        connection_distances.append(distance)
                
                if connecting_lines:
                    # Calculate confidence based on connection quality
                    avg_distance = sum(connection_distances) / len(connection_distances)
                    confidence = 1.0 - (avg_distance / self.junction_tolerance)
                    confidence = max(0.1, confidence)
                    
                    connecting_lines.append(i)  # Include the segment with this endpoint
                    t_junctions.append((endpoint, connecting_lines, "T", confidence))
        
        return t_junctions
    
    def _find_corner_junctions(self, line_segments: List[LineSegment]) -> List[Tuple[Tuple[float, float], List[int], str, float]]:
        """Find L-shaped corner junctions where segments meet at endpoints."""
        corner_junctions = []
        
        for i, seg1 in enumerate(line_segments):
            for j, seg2 in enumerate(line_segments[i+1:], i+1):
                # Check if any endpoints are close to each other
                endpoints1 = [seg1.start, seg1.end]
                endpoints2 = [seg2.start, seg2.end]
                
                for ep1 in endpoints1:
                    for ep2 in endpoints2:
                        distance = math.sqrt((ep1[0] - ep2[0])**2 + (ep1[1] - ep2[1])**2)
                        if distance <= self.junction_tolerance:
                            # Check if this forms a reasonable corner (not collinear)
                            angle_diff = abs(seg1.angle() - seg2.angle())
                            angle_diff = min(angle_diff, abs(angle_diff - math.pi))
                            
                            # Good corner if not too close to parallel
                            if angle_diff > 0.2:  # ~11 degrees
                                connection_point = ((ep1[0] + ep2[0]) / 2, (ep1[1] + ep2[1]) / 2)
                                confidence = 1.0 - (distance / self.junction_tolerance)
                                corner_junctions.append((connection_point, [i, j], "corner", confidence))
        
        return corner_junctions
    
    def _deduplicate_junctions_enhanced(self, junctions: List[Tuple[Tuple[float, float], List[int], str, float]], 
                                       line_segments: List[LineSegment]) -> List[Tuple[Tuple[float, float], List[int], str, float]]:
        """Enhanced junction deduplication with clustering analysis."""
        if not junctions:
            return []
        
        # Sort by confidence (higher first)
        junctions.sort(key=lambda x: x[3], reverse=True)
        
        deduplicated = []
        used = set()
        
        for i, (pos1, lines1, type1, conf1) in enumerate(junctions):
            if i in used:
                continue
            
            # Find nearby junctions to merge
            merge_group = [(pos1, lines1, type1, conf1)]
            used.add(i)
            
            for j, (pos2, lines2, type2, conf2) in enumerate(junctions[i+1:], i+1):
                if j in used:
                    continue
                
                distance = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                if distance <= self.junction_tolerance:
                    merge_group.append((pos2, lines2, type2, conf2))
                    used.add(j)
            
            # Merge the group
            if len(merge_group) == 1:
                deduplicated.append(merge_group[0])
            else:
                # Calculate weighted average position based on confidence
                total_conf = sum(conf for _, _, _, conf in merge_group)
                if total_conf > 0:
                    avg_x = sum(pos[0] * conf for pos, _, _, conf in merge_group) / total_conf
                    avg_y = sum(pos[1] * conf for pos, _, _, conf in merge_group) / total_conf
                else:
                    avg_x = sum(pos[0] for pos, _, _, _ in merge_group) / len(merge_group)
                    avg_y = sum(pos[1] for pos, _, _, _ in merge_group) / len(merge_group)
                
                # Combine connected lines (remove duplicates)
                all_lines = set()
                for _, lines, _, _ in merge_group:
                    all_lines.update(lines)
                
                # Determine junction type and confidence
                max_conf = max(conf for _, _, _, conf in merge_group)
                junction_type = self._determine_junction_type(len(all_lines))
                
                deduplicated.append(((avg_x, avg_y), list(all_lines), junction_type, max_conf))
        
        return deduplicated
    
    def _determine_junction_type(self, num_connections: int) -> str:
        """Determine junction type based on number of connections."""
        if num_connections >= 4:
            return "cross"
        elif num_connections == 3:
            return "T"
        elif num_connections == 2:
            return "corner"
        else:
            return "endpoint"
    
    def _associate_net_labels(self, wire_nets: List[WireNet], line_segments: List[LineSegment], 
                             text_spans: List) -> List[WireNet]:
        """
        Associate text labels with wire networks.
        
        Args:
            wire_nets: List of wire networks
            line_segments: List of line segments
            text_spans: List of detected text spans
            
        Returns:
            Wire networks with associated labels
        """
        for net in wire_nets:
            net.net_labels = []
            
            # Find text spans that are close to this network
            for text_span in text_spans:
                if not hasattr(text_span, 'text') or not hasattr(text_span, 'bbox'):
                    continue
                
                # Calculate distance from text to network
                text_center = (
                    (text_span.bbox[0] + text_span.bbox[2]) / 2,
                    (text_span.bbox[1] + text_span.bbox[3]) / 2
                )
                
                min_distance = float('inf')
                for seg_idx in net.segments:
                    if seg_idx < len(line_segments):
                        segment = line_segments[seg_idx]
                        distance = segment.distance_to_point(text_center)
                        min_distance = min(min_distance, distance)
                
                # Associate label if close enough
                if min_distance <= 30.0:  # 30 pixel tolerance
                    label_type = self._classify_net_label(text_span.text)
                    
                    net_label = NetLabel(
                        text=text_span.text,
                        position=text_center,
                        bbox=text_span.bbox,
                        confidence=getattr(text_span, 'confidence', 1.0),
                        label_type=label_type
                    )
                    net.net_labels.append(net_label)
        
        return wire_nets
    
    def _classify_net_label(self, text: str) -> str:
        """Classify the type of net label based on text content."""
        text_upper = text.upper().strip()
        
        # Voltage labels
        voltage_patterns = ['VCC', 'VDD', 'V+', '+5V', '+3V', '3.3V', '5V', '12V', '24V']
        if any(pattern in text_upper for pattern in voltage_patterns):
            return "voltage"
        
        # Ground labels
        ground_patterns = ['GND', 'GROUND', 'VSS', 'V-', '0V']
        if any(pattern in text_upper for pattern in ground_patterns):
            return "ground"
        
        # Power labels
        power_patterns = ['PWR', 'POWER', 'VIN', 'VOUT', 'VBat', 'VBAT']
        if any(pattern in text_upper for pattern in power_patterns):
            return "power"
        
        # Bus labels (multi-bit signals)
        if '[' in text and ']' in text:
            return "bus"
        
        # Clock signals
        clock_patterns = ['CLK', 'CLOCK', 'OSC', 'XTAL']
        if any(pattern in text_upper for pattern in clock_patterns):
            return "signal"
        
        # Default to custom signal
        return "custom"
    
    def _propagate_net_names(self, wire_nets: List[WireNet]) -> List[WireNet]:
        """
        Propagate net names throughout networks and resolve conflicts.
        
        Args:
            wire_nets: List of wire networks
            
        Returns:
            Wire networks with propagated names
        """
        for net in wire_nets:
            if not net.net_labels:
                continue
            
            # Sort labels by priority and confidence
            labels_by_priority = self._prioritize_net_labels(net.net_labels)
            
            if labels_by_priority:
                # Use highest priority label as propagated name
                primary_label = labels_by_priority[0]
                net.propagated_name = primary_label.text
                net.voltage_level = self._extract_voltage_level(primary_label.text)
                
                # Check if this is a bus
                if primary_label.label_type == "bus" or '[' in primary_label.text:
                    net.is_bus = True
                    net.bus_width = self._extract_bus_width(primary_label.text)
        
        return wire_nets
    
    def _prioritize_net_labels(self, labels: List[NetLabel]) -> List[NetLabel]:
        """Sort net labels by priority (power > signal > custom)."""
        priority_order = {
            "voltage": 5,
            "power": 4, 
            "ground": 4,
            "signal": 3,
            "bus": 3,
            "custom": 1
        }
        
        return sorted(labels, 
                     key=lambda label: (priority_order.get(label.label_type, 0), label.confidence),
                     reverse=True)
    
    def _extract_voltage_level(self, text: str) -> Optional[str]:
        """Extract voltage level from label text."""
        import re
        
        # Look for voltage patterns like "5V", "3.3V", "+12V", etc.
        voltage_match = re.search(r'([+-]?\d+(?:\.\d+)?)\s*V', text.upper())
        if voltage_match:
            return voltage_match.group(0)
        
        # Special cases
        if 'VCC' in text.upper() or 'VDD' in text.upper():
            return "VCC"
        elif 'GND' in text.upper():
            return "GND"
        
        return None
    
    def _extract_bus_width(self, text: str) -> Optional[int]:
        """Extract bus width from label text like 'DATA[7:0]'."""
        import re
        
        # Look for patterns like [7:0], [15:0], etc.
        bus_match = re.search(r'\[(\d+):(\d+)\]', text)
        if bus_match:
            high_bit = int(bus_match.group(1))
            low_bit = int(bus_match.group(2))
            return abs(high_bit - low_bit) + 1
        
        return None
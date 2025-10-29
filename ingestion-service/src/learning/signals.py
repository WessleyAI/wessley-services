"""
Signal collection for continual learning.

This module collects various signals during processing that indicate
model performance and areas for improvement, enabling continuous learning.
"""
import time
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..core.logging import StructuredLogger
from ..core.metrics import metrics

logger = StructuredLogger(__name__)


class SignalType(str, Enum):
    """Types of learning signals."""
    OCR_DISAGREEMENT = "ocr_disagreement"
    SYMBOL_DETECTION_ERROR = "symbol_detection_error"
    NET_PROPAGATION_CONFLICT = "net_propagation_conflict"
    SEARCH_FEEDBACK = "search_feedback"
    GEOMETRY_INCONSISTENCY = "geometry_inconsistency"
    VALIDATION_FAILURE = "validation_failure"
    USER_CORRECTION = "user_correction"


@dataclass
class LearningSignal:
    """Individual learning signal captured during processing."""
    signal_type: SignalType
    timestamp: datetime
    job_id: str
    project_id: str
    page: Optional[int] = None
    confidence: float = 1.0
    severity: str = "medium"  # low, medium, high, critical
    
    # Context data
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata for training
    features: Dict[str, Any] = field(default_factory=dict)
    labels: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "signal_type": self.signal_type.value,
            "timestamp": self.timestamp.isoformat(),
            "job_id": self.job_id,
            "project_id": self.project_id,
            "page": self.page,
            "confidence": self.confidence,
            "severity": self.severity,
            "context": self.context,
            "features": self.features,
            "labels": self.labels
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LearningSignal':
        """Create from dictionary."""
        return cls(
            signal_type=SignalType(data["signal_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            job_id=data["job_id"],
            project_id=data["project_id"],
            page=data.get("page"),
            confidence=data.get("confidence", 1.0),
            severity=data.get("severity", "medium"),
            context=data.get("context", {}),
            features=data.get("features", {}),
            labels=data.get("labels", {})
        )


class SignalCollector:
    """Collects learning signals during ingestion processing."""
    
    def __init__(self, storage_backend=None):
        """
        Initialize signal collector.
        
        Args:
            storage_backend: Backend for storing signals (Supabase)
        """
        self.storage = storage_backend
        self.signals: List[LearningSignal] = []
        self.current_job_id: Optional[str] = None
        self.current_project_id: Optional[str] = None
        
    def set_context(self, job_id: str, project_id: str):
        """Set current processing context."""
        self.current_job_id = job_id
        self.current_project_id = project_id
    
    def collect_ocr_disagreement(self, 
                                page: int,
                                bbox: List[float],
                                engine_results: Dict[str, Any],
                                confidence_threshold: float = 0.3) -> Optional[LearningSignal]:
        """
        Collect OCR disagreement signal when engines produce different results.
        
        Args:
            page: Page number
            bbox: Bounding box of text region
            engine_results: Results from different OCR engines
            confidence_threshold: Minimum confidence difference to trigger signal
            
        Returns:
            Learning signal if significant disagreement found
        """
        if len(engine_results) < 2:
            return None
        
        # Calculate disagreement metrics
        texts = [result.get("text", "") for result in engine_results.values()]
        confidences = [result.get("confidence", 0.0) for result in engine_results.values()]
        
        # Check for text disagreement
        unique_texts = set(texts)
        if len(unique_texts) <= 1:
            return None  # All engines agree
        
        # Check for significant confidence differences
        conf_diff = max(confidences) - min(confidences)
        if conf_diff < confidence_threshold:
            return None
        
        # Calculate edit distance between texts
        edit_distances = []
        for i, text1 in enumerate(texts):
            for j, text2 in enumerate(texts[i+1:], i+1):
                edit_dist = self._levenshtein_distance(text1, text2)
                edit_distances.append(edit_dist)
        
        avg_edit_distance = sum(edit_distances) / len(edit_distances) if edit_distances else 0
        
        signal = LearningSignal(
            signal_type=SignalType.OCR_DISAGREEMENT,
            timestamp=datetime.utcnow(),
            job_id=self.current_job_id,
            project_id=self.current_project_id,
            page=page,
            confidence=conf_diff,
            severity=self._classify_severity(conf_diff, avg_edit_distance),
            context={
                "bbox": bbox,
                "engine_count": len(engine_results),
                "unique_results": len(unique_texts)
            },
            features={
                "confidence_difference": conf_diff,
                "avg_edit_distance": avg_edit_distance,
                "text_length_variance": self._calculate_length_variance(texts),
                "has_special_chars": any(not text.isalnum() and not text.isspace() for text in texts)
            },
            labels={
                "engine_results": engine_results,
                "best_engine": max(engine_results.keys(), key=lambda k: engine_results[k].get("confidence", 0))
            }
        )
        
        self._add_signal(signal)
        return signal
    
    def collect_symbol_detection_error(self,
                                     page: int,
                                     detected_symbols: List[Dict],
                                     validation_errors: List[str]) -> Optional[LearningSignal]:
        """
        Collect symbol detection error signal based on validation failures.
        
        Args:
            page: Page number
            detected_symbols: List of detected symbols
            validation_errors: List of validation error messages
            
        Returns:
            Learning signal for detection errors
        """
        if not validation_errors:
            return None
        
        # Analyze error patterns
        error_types = {}
        for error in validation_errors:
            error_type = self._classify_detection_error(error)
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # Calculate severity based on error types and count
        severity = "low"
        if "missing_pins" in error_types or "wrong_component_type" in error_types:
            severity = "high"
        elif len(validation_errors) > 3:
            severity = "medium"
        
        signal = LearningSignal(
            signal_type=SignalType.SYMBOL_DETECTION_ERROR,
            timestamp=datetime.utcnow(),
            job_id=self.current_job_id,
            project_id=self.current_project_id,
            page=page,
            confidence=min(1.0, len(validation_errors) / 10.0),
            severity=severity,
            context={
                "symbol_count": len(detected_symbols),
                "error_count": len(validation_errors),
                "error_types": error_types
            },
            features={
                "symbol_density": len(detected_symbols) / max(1, page),  # symbols per page
                "avg_confidence": sum(s.get("confidence", 0) for s in detected_symbols) / max(1, len(detected_symbols)),
                "error_diversity": len(error_types),
                "pin_count_errors": error_types.get("wrong_pin_count", 0)
            },
            labels={
                "validation_errors": validation_errors,
                "detected_symbols": detected_symbols
            }
        )
        
        self._add_signal(signal)
        return signal
    
    def collect_net_propagation_conflict(self,
                                       page: int,
                                       conflicting_nets: List[Dict],
                                       conflict_type: str) -> Optional[LearningSignal]:
        """
        Collect net propagation conflict signal.
        
        Args:
            page: Page number
            conflicting_nets: List of conflicting net definitions
            conflict_type: Type of conflict (voltage, name, topology)
            
        Returns:
            Learning signal for net conflicts
        """
        if not conflicting_nets:
            return None
        
        # Determine severity based on conflict type
        severity_map = {
            "voltage": "high",      # Different voltages on same net
            "name": "medium",       # Multiple names for same net
            "topology": "low"       # Minor topology inconsistencies
        }
        severity = severity_map.get(conflict_type, "medium")
        
        signal = LearningSignal(
            signal_type=SignalType.NET_PROPAGATION_CONFLICT,
            timestamp=datetime.utcnow(),
            job_id=self.current_job_id,
            project_id=self.current_project_id,
            page=page,
            confidence=0.8,
            severity=severity,
            context={
                "conflict_type": conflict_type,
                "net_count": len(conflicting_nets)
            },
            features={
                "conflict_severity": {"voltage": 3, "name": 2, "topology": 1}.get(conflict_type, 1),
                "nets_involved": len(conflicting_nets),
                "has_power_nets": any("VCC" in net.get("name", "") or "GND" in net.get("name", "") 
                                    for net in conflicting_nets)
            },
            labels={
                "conflicting_nets": conflicting_nets,
                "conflict_type": conflict_type
            }
        )
        
        self._add_signal(signal)
        return signal
    
    def collect_geometry_inconsistency(self,
                                     page: int,
                                     component_id: str,
                                     expected_geometry: Dict,
                                     actual_geometry: Dict) -> Optional[LearningSignal]:
        """
        Collect geometry inconsistency signal.
        
        Args:
            page: Page number
            component_id: Component identifier
            expected_geometry: Expected geometric properties
            actual_geometry: Actual detected geometry
            
        Returns:
            Learning signal for geometry issues
        """
        # Calculate geometric differences
        bbox_diff = self._calculate_bbox_difference(
            expected_geometry.get("bbox", []),
            actual_geometry.get("bbox", [])
        )
        
        pin_diff = self._calculate_pin_position_difference(
            expected_geometry.get("pins", []),
            actual_geometry.get("pins", [])
        )
        
        total_diff = bbox_diff + pin_diff
        
        if total_diff < 0.1:  # Threshold for significant difference
            return None
        
        signal = LearningSignal(
            signal_type=SignalType.GEOMETRY_INCONSISTENCY,
            timestamp=datetime.utcnow(),
            job_id=self.current_job_id,
            project_id=self.current_project_id,
            page=page,
            confidence=min(1.0, total_diff),
            severity=self._classify_geometry_severity(total_diff),
            context={
                "component_id": component_id,
                "geometry_difference": total_diff
            },
            features={
                "bbox_difference": bbox_diff,
                "pin_position_difference": pin_diff,
                "component_type": expected_geometry.get("type", "unknown"),
                "pin_count": len(expected_geometry.get("pins", []))
            },
            labels={
                "expected_geometry": expected_geometry,
                "actual_geometry": actual_geometry
            }
        )
        
        self._add_signal(signal)
        return signal
    
    def collect_search_feedback(self,
                              query: str,
                              results: List[Dict],
                              user_interactions: Dict) -> Optional[LearningSignal]:
        """
        Collect search feedback signal from user interactions.
        
        Args:
            query: Search query
            results: Search results
            user_interactions: User interaction data (clicks, dwell time)
            
        Returns:
            Learning signal for search quality
        """
        if not user_interactions:
            return None
        
        # Analyze interaction patterns
        clicked_positions = user_interactions.get("clicked_positions", [])
        dwell_times = user_interactions.get("dwell_times", [])
        
        # Calculate metrics
        click_through_rate = len(clicked_positions) / max(1, len(results))
        avg_clicked_position = sum(clicked_positions) / max(1, len(clicked_positions))
        avg_dwell_time = sum(dwell_times) / max(1, len(dwell_times))
        
        # Determine if this indicates poor search quality
        poor_quality_indicators = [
            click_through_rate < 0.1,  # Very low CTR
            avg_clicked_position > 5,   # Users clicking far down the list
            avg_dwell_time < 2.0       # Very short dwell time
        ]
        
        if not any(poor_quality_indicators):
            return None  # Good search quality, no signal needed
        
        signal = LearningSignal(
            signal_type=SignalType.SEARCH_FEEDBACK,
            timestamp=datetime.utcnow(),
            job_id=self.current_job_id or "search",
            project_id=self.current_project_id or "unknown",
            confidence=sum(poor_quality_indicators) / len(poor_quality_indicators),
            severity="medium" if sum(poor_quality_indicators) >= 2 else "low",
            context={
                "query": query,
                "result_count": len(results),
                "interaction_count": len(clicked_positions)
            },
            features={
                "click_through_rate": click_through_rate,
                "avg_clicked_position": avg_clicked_position,
                "avg_dwell_time": avg_dwell_time,
                "query_length": len(query.split()),
                "has_automotive_terms": any(term in query.lower() for term in 
                                          ["relay", "fuse", "ecu", "starter", "fuel"])
            },
            labels={
                "query": query,
                "user_interactions": user_interactions
            }
        )
        
        self._add_signal(signal)
        return signal
    
    async def flush_signals(self) -> int:
        """
        Flush collected signals to storage.
        
        Returns:
            Number of signals flushed
        """
        if not self.signals:
            return 0
        
        try:
            if self.storage:
                await self._store_signals(self.signals)
            
            # Record metrics
            signal_count = len(self.signals)
            metrics.record_external_service_call("learning", "flush_signals", "success", 0.1)
            
            # Log signal summary
            signal_types = {}
            for signal in self.signals:
                signal_types[signal.signal_type.value] = signal_types.get(signal.signal_type.value, 0) + 1
            
            logger.info(f"Flushed {signal_count} learning signals", 
                       signal_types=signal_types,
                       job_id=self.current_job_id)
            
            # Clear signals
            flushed_count = len(self.signals)
            self.signals.clear()
            
            return flushed_count
            
        except Exception as e:
            logger.error(f"Failed to flush learning signals: {e}")
            metrics.record_error("signal_flush_failed", "learning", "error")
            return 0
    
    def get_signal_summary(self) -> Dict[str, Any]:
        """Get summary of collected signals."""
        if not self.signals:
            return {"total": 0}
        
        summary = {
            "total": len(self.signals),
            "by_type": {},
            "by_severity": {},
            "avg_confidence": sum(s.confidence for s in self.signals) / len(self.signals),
            "time_range": {
                "start": min(s.timestamp for s in self.signals).isoformat(),
                "end": max(s.timestamp for s in self.signals).isoformat()
            }
        }
        
        for signal in self.signals:
            # Count by type
            signal_type = signal.signal_type.value
            summary["by_type"][signal_type] = summary["by_type"].get(signal_type, 0) + 1
            
            # Count by severity
            severity = signal.severity
            summary["by_severity"][severity] = summary["by_severity"].get(severity, 0) + 1
        
        return summary
    
    def _add_signal(self, signal: LearningSignal):
        """Add signal to collection."""
        self.signals.append(signal)
        
        # Record metric
        metrics.record_external_service_call("learning", "collect_signal", "success", 0.01)
        
        logger.debug(f"Collected {signal.signal_type.value} signal",
                    page=signal.page,
                    confidence=signal.confidence,
                    severity=signal.severity)
    
    async def _store_signals(self, signals: List[LearningSignal]):
        """Store signals to backend storage."""
        # Mock implementation - would store to Supabase in real system
        logger.debug(f"Storing {len(signals)} signals to backend")
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _calculate_length_variance(self, texts: List[str]) -> float:
        """Calculate variance in text lengths."""
        if len(texts) < 2:
            return 0.0
        
        lengths = [len(text) for text in texts]
        mean_length = sum(lengths) / len(lengths)
        variance = sum((length - mean_length) ** 2 for length in lengths) / len(lengths)
        return variance
    
    def _classify_severity(self, conf_diff: float, edit_distance: float) -> str:
        """Classify signal severity based on metrics."""
        if conf_diff > 0.7 or edit_distance > 10:
            return "high"
        elif conf_diff > 0.4 or edit_distance > 5:
            return "medium"
        else:
            return "low"
    
    def _classify_detection_error(self, error_msg: str) -> str:
        """Classify detection error type."""
        error_lower = error_msg.lower()
        if "pin" in error_lower:
            return "wrong_pin_count"
        elif "type" in error_lower:
            return "wrong_component_type"
        elif "missing" in error_lower:
            return "missing_component"
        elif "duplicate" in error_lower:
            return "duplicate_detection"
        else:
            return "other"
    
    def _calculate_bbox_difference(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate normalized difference between bounding boxes."""
        if len(bbox1) != 4 or len(bbox2) != 4:
            return 1.0
        
        # Calculate IoU (Intersection over Union)
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 1.0  # No overlap
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        iou = intersection / union if union > 0 else 0
        return 1.0 - iou  # Convert to difference (0 = perfect match, 1 = no overlap)
    
    def _calculate_pin_position_difference(self, pins1: List[Dict], pins2: List[Dict]) -> float:
        """Calculate difference in pin positions."""
        if not pins1 or not pins2:
            return 0.5  # Moderate difference if one is empty
        
        if len(pins1) != len(pins2):
            return 1.0  # High difference for pin count mismatch
        
        # Calculate average position difference
        total_diff = 0.0
        for p1, p2 in zip(pins1, pins2):
            bbox1 = p1.get("bbox", [0, 0, 0, 0])
            bbox2 = p2.get("bbox", [0, 0, 0, 0])
            diff = self._calculate_bbox_difference(bbox1, bbox2)
            total_diff += diff
        
        return total_diff / len(pins1)
    
    def _classify_geometry_severity(self, difference: float) -> str:
        """Classify geometry difference severity."""
        if difference > 0.7:
            return "high"
        elif difference > 0.3:
            return "medium"
        else:
            return "low"


# Global signal collector instance
_global_collector: Optional[SignalCollector] = None

def get_signal_collector() -> SignalCollector:
    """Get global signal collector."""
    global _global_collector
    if _global_collector is None:
        _global_collector = SignalCollector()
    return _global_collector

def set_signal_collector(collector: SignalCollector):
    """Set global signal collector."""
    global _global_collector
    _global_collector = collector
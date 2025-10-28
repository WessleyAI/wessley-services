"""
Multi-engine OCR fusion system for combining results from multiple OCR providers.
"""
import asyncio
from typing import List, Dict, Optional, Set, Tuple
import numpy as np
from collections import defaultdict
from dataclasses import dataclass

from .base import OcrProvider
from ..core.schemas import PageImage, TextSpan, OcrEngine


@dataclass
class BboxOverlap:
    """Represents overlap between two bounding boxes."""
    iou: float  # Intersection over Union
    overlap_area: float
    confidence_diff: float


class OcrFusionEngine:
    """
    Combines results from multiple OCR engines using late fusion techniques.
    
    Implements various fusion strategies:
    - Confidence-based selection
    - Geometric consistency checks
    - Text similarity analysis
    - Ensemble voting
    """
    
    def __init__(
        self,
        providers: List[OcrProvider],
        fusion_strategy: str = "confidence_weighted",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.3
    ):
        """
        Initialize OCR fusion engine.
        
        Args:
            providers: List of OCR providers to use
            fusion_strategy: Strategy for combining results
            confidence_threshold: Minimum confidence for considering results
            iou_threshold: IoU threshold for considering bboxes as same region
        """
        self.providers = providers
        self.fusion_strategy = fusion_strategy
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Validate providers
        if not providers:
            raise ValueError("At least one OCR provider is required")
    
    async def extract_text_fused(self, page_image: PageImage) -> List[TextSpan]:
        """
        Extract text using multiple OCR engines and fuse results.
        
        Args:
            page_image: Page image to process
            
        Returns:
            Fused list of TextSpan objects
        """
        # Run all OCR engines concurrently
        tasks = [
            provider.extract_text(page_image) 
            for provider in self.providers
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out failed results
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"OCR provider {self.providers[i].engine_name} failed: {result}")
            elif isinstance(result, list):
                valid_results.append(result)
        
        if not valid_results:
            return []
        
        # Apply fusion strategy
        return self._fuse_results(valid_results, page_image)
    
    def _fuse_results(
        self, 
        results_list: List[List[TextSpan]], 
        page_image: PageImage
    ) -> List[TextSpan]:
        """
        Fuse results from multiple OCR engines.
        
        Args:
            results_list: List of TextSpan lists from different engines
            page_image: Original page image
            
        Returns:
            Fused TextSpan list
        """
        if len(results_list) == 1:
            return results_list[0]
        
        # Apply the selected fusion strategy
        if self.fusion_strategy == "confidence_weighted":
            return self._confidence_weighted_fusion(results_list)
        elif self.fusion_strategy == "geometric_consensus":
            return self._geometric_consensus_fusion(results_list)
        elif self.fusion_strategy == "text_similarity":
            return self._text_similarity_fusion(results_list)
        elif self.fusion_strategy == "ensemble_voting":
            return self._ensemble_voting_fusion(results_list)
        else:
            # Default: simple confidence-based selection
            return self._simple_confidence_fusion(results_list)
    
    def _confidence_weighted_fusion(self, results_list: List[List[TextSpan]]) -> List[TextSpan]:
        """
        Fuse results using confidence-weighted selection.
        
        For each spatial region, select the result with highest confidence.
        """
        # Flatten all results
        all_spans = []
        for spans in results_list:
            all_spans.extend(spans)
        
        # Filter by confidence threshold
        filtered_spans = [
            span for span in all_spans 
            if span.confidence >= self.confidence_threshold
        ]
        
        # Group overlapping spans
        groups = self._group_overlapping_spans(filtered_spans)
        
        # Select best span from each group
        fused_spans = []
        for group in groups:
            if len(group) == 1:
                fused_spans.append(group[0])
            else:
                # Select span with highest confidence
                best_span = max(group, key=lambda x: x.confidence)
                fused_spans.append(best_span)
        
        return fused_spans
    
    def _geometric_consensus_fusion(self, results_list: List[List[TextSpan]]) -> List[TextSpan]:
        """
        Fuse results using geometric consensus.
        
        Only keep results that are confirmed by multiple engines
        in similar spatial locations.
        """
        # Flatten all results
        all_spans = []
        for spans in results_list:
            all_spans.extend(spans)
        
        # Group overlapping spans
        groups = self._group_overlapping_spans(all_spans)
        
        # Keep only groups with multiple engines
        fused_spans = []
        for group in groups:
            # Check if multiple engines contributed to this group
            engines = set(span.engine for span in group)
            
            if len(engines) >= 2:  # Confirmed by at least 2 engines
                # Create consensus span
                consensus_span = self._create_consensus_span(group)
                fused_spans.append(consensus_span)
            elif len(group) == 1 and group[0].confidence > 0.8:
                # High confidence single detection
                fused_spans.append(group[0])
        
        return fused_spans
    
    def _text_similarity_fusion(self, results_list: List[List[TextSpan]]) -> List[TextSpan]:
        """
        Fuse results using text similarity analysis.
        
        Combine results where text content is similar across engines.
        """
        # Flatten all results
        all_spans = []
        for spans in results_list:
            all_spans.extend(spans)
        
        # Group overlapping spans
        groups = self._group_overlapping_spans(all_spans)
        
        fused_spans = []
        for group in groups:
            if len(group) == 1:
                fused_spans.append(group[0])
            else:
                # Analyze text similarity within group
                consensus_span = self._create_text_consensus(group)
                fused_spans.append(consensus_span)
        
        return fused_spans
    
    def _ensemble_voting_fusion(self, results_list: List[List[TextSpan]]) -> List[TextSpan]:
        """
        Fuse results using ensemble voting.
        
        Each engine votes on the presence and content of text regions.
        """
        # Create spatial grid for voting
        return self._confidence_weighted_fusion(results_list)  # Simplified for now
    
    def _simple_confidence_fusion(self, results_list: List[List[TextSpan]]) -> List[TextSpan]:
        """
        Simple fusion: take highest confidence result for each region.
        """
        return self._confidence_weighted_fusion(results_list)
    
    def _group_overlapping_spans(self, spans: List[TextSpan]) -> List[List[TextSpan]]:
        """
        Group TextSpans that have overlapping bounding boxes.
        
        Args:
            spans: List of TextSpan objects
            
        Returns:
            List of groups, where each group contains overlapping spans
        """
        if not spans:
            return []
        
        groups = []
        used = set()
        
        for i, span1 in enumerate(spans):
            if i in used:
                continue
            
            group = [span1]
            used.add(i)
            
            for j, span2 in enumerate(spans[i+1:], i+1):
                if j in used:
                    continue
                
                # Check if spans overlap
                iou = self._calculate_iou(span1.bbox, span2.bbox)
                if iou > self.iou_threshold:
                    group.append(span2)
                    used.add(j)
            
            groups.append(group)
        
        return groups
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        Calculate Intersection over Union (IoU) for two bounding boxes.
        
        Args:
            bbox1: [x1, y1, x2, y2]
            bbox2: [x1, y1, x2, y2]
            
        Returns:
            IoU score (0.0 to 1.0)
        """
        # Calculate intersection
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        if union <= 0:
            return 0.0
        
        return intersection / union
    
    def _create_consensus_span(self, group: List[TextSpan]) -> TextSpan:
        """
        Create a consensus TextSpan from a group of overlapping spans.
        
        Args:
            group: List of overlapping TextSpan objects
            
        Returns:
            Consensus TextSpan
        """
        if len(group) == 1:
            return group[0]
        
        # Use weighted average for bounding box
        total_confidence = sum(span.confidence for span in group)
        
        weighted_bbox = [0.0, 0.0, 0.0, 0.0]
        for span in group:
            weight = span.confidence / total_confidence
            for i in range(4):
                weighted_bbox[i] += span.bbox[i] * weight
        
        # Select best text (highest confidence)
        best_span = max(group, key=lambda x: x.confidence)
        
        # Calculate average confidence
        avg_confidence = total_confidence / len(group)
        
        return TextSpan(
            page=best_span.page,
            bbox=weighted_bbox,
            text=best_span.text,
            rotation=best_span.rotation,
            confidence=min(avg_confidence, 1.0),
            engine=best_span.engine  # Keep the best engine
        )
    
    def _create_text_consensus(self, group: List[TextSpan]) -> TextSpan:
        """
        Create consensus TextSpan using text similarity analysis.
        
        Args:
            group: List of overlapping TextSpan objects
            
        Returns:
            Consensus TextSpan with best text
        """
        if len(group) == 1:
            return group[0]
        
        # Find most common text or best quality text
        text_votes = defaultdict(list)
        
        for span in group:
            text_votes[span.text.lower().strip()].append(span)
        
        # Select text with highest total confidence
        best_text_group = max(
            text_votes.values(),
            key=lambda spans: sum(s.confidence for s in spans)
        )
        
        # Use the highest confidence span from the best text group
        best_span = max(best_text_group, key=lambda x: x.confidence)
        
        # Create consensus bbox from all spans in group
        consensus_span = self._create_consensus_span(group)
        
        # But use the best text
        consensus_span.text = best_span.text
        
        return consensus_span
    
    def get_fusion_metrics(self, results_list: List[List[TextSpan]]) -> Dict[str, float]:
        """
        Calculate metrics about the fusion process.
        
        Args:
            results_list: Results from multiple OCR engines
            
        Returns:
            Dictionary with fusion metrics
        """
        if not results_list:
            return {}
        
        # Count total detections per engine
        engine_counts = {}
        for i, results in enumerate(results_list):
            engine_name = self.providers[i].engine_name if i < len(self.providers) else f"engine_{i}"
            engine_counts[engine_name] = len(results)
        
        # Calculate agreement rate
        all_spans = [span for results in results_list for span in results]
        groups = self._group_overlapping_spans(all_spans)
        
        # Count groups with multiple engines
        multi_engine_groups = sum(
            1 for group in groups 
            if len(set(span.engine for span in group)) > 1
        )
        
        agreement_rate = multi_engine_groups / len(groups) if groups else 0.0
        
        return {
            "total_engines": len(results_list),
            "engine_counts": engine_counts,
            "total_groups": len(groups),
            "multi_engine_groups": multi_engine_groups,
            "agreement_rate": agreement_rate,
            "fusion_strategy": self.fusion_strategy
        }
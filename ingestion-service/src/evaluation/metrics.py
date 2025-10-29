"""
Evaluation metrics for semantic search and learning systems.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from enum import Enum
import numpy as np
from datetime import datetime, timezone

try:
    from sklearn.metrics import ndcg_score, precision_recall_fscore_support
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from ..semantic.search import SearchHit, SearchFilter
from ..semantic.ontology import AutomotiveComponent, ElectricalNet, VehicleSignature
from ..learning.signals import LearningSignal, SignalType


class EvaluationMetric(Enum):
    """Supported evaluation metrics."""
    NDCG_AT_K = "ndcg_at_k"
    PRECISION_AT_K = "precision_at_k"
    RECALL_AT_K = "recall_at_k"
    MAP = "mean_average_precision"
    MRR = "mean_reciprocal_rank"
    LATENCY_P95 = "latency_p95"
    LATENCY_MEAN = "latency_mean"
    HIT_RATE = "hit_rate"
    COMPONENT_ACCURACY = "component_accuracy"
    NET_ACCURACY = "net_accuracy"


@dataclass
class EvaluationQuery:
    """A query with ground truth for evaluation."""
    query_text: str
    relevant_doc_ids: Set[str]
    vehicle_context: Optional[VehicleSignature] = None
    system_filter: Optional[str] = None
    expected_components: List[str] = field(default_factory=list)
    expected_nets: List[str] = field(default_factory=list)
    difficulty: str = "medium"  # easy, medium, hard


@dataclass
class EvaluationResult:
    """Results from running evaluation."""
    metric_name: str
    value: float
    query_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSuite:
    """Collection of evaluation queries and expected results."""
    name: str
    queries: List[EvaluationQuery]
    description: str = ""
    automotive_focus: bool = True
    version: str = "1.0"


class SearchEvaluator:
    """Evaluates semantic search performance."""
    
    def __init__(self, k_values: List[int] = None):
        self.k_values = k_values or [1, 3, 5, 10]
        
    async def evaluate_search_quality(
        self,
        search_engine,
        benchmark: BenchmarkSuite
    ) -> Dict[str, List[EvaluationResult]]:
        """Evaluate search quality across multiple metrics."""
        results = {metric.value: [] for metric in EvaluationMetric}
        
        for i, query in enumerate(benchmark.queries):
            query_id = f"{benchmark.name}_{i}"
            
            # Perform search
            start_time = time.time()
            filters = SearchFilter(
                vehicle=query.vehicle_context,
                system=query.system_filter
            )
            
            search_hits = await search_engine.search(
                query=query.query_text,
                filters=filters,
                limit=max(self.k_values),
                strategy="hybrid"
            )
            
            latency = time.time() - start_time
            
            # Calculate metrics
            retrieved_ids = [hit.chunk_id for hit in search_hits]
            
            # Precision, Recall, NDCG at K
            for k in self.k_values:
                prec_k = self._precision_at_k(retrieved_ids, query.relevant_doc_ids, k)
                recall_k = self._recall_at_k(retrieved_ids, query.relevant_doc_ids, k)
                ndcg_k = self._ndcg_at_k(search_hits, query.relevant_doc_ids, k)
                
                results[EvaluationMetric.PRECISION_AT_K.value].append(
                    EvaluationResult(f"precision_at_{k}", prec_k, query_id)
                )
                results[EvaluationMetric.RECALL_AT_K.value].append(
                    EvaluationResult(f"recall_at_{k}", recall_k, query_id)
                )
                results[EvaluationMetric.NDCG_AT_K.value].append(
                    EvaluationResult(f"ndcg_at_{k}", ndcg_k, query_id)
                )
            
            # Mean Average Precision
            map_score = self._mean_average_precision(retrieved_ids, query.relevant_doc_ids)
            results[EvaluationMetric.MAP.value].append(
                EvaluationResult("map", map_score, query_id)
            )
            
            # Mean Reciprocal Rank
            mrr = self._mean_reciprocal_rank([retrieved_ids], [query.relevant_doc_ids])
            results[EvaluationMetric.MRR.value].append(
                EvaluationResult("mrr", mrr, query_id)
            )
            
            # Latency
            results[EvaluationMetric.LATENCY_MEAN.value].append(
                EvaluationResult("latency_mean", latency, query_id)
            )
            
            # Hit Rate
            hit_rate = 1.0 if any(doc_id in query.relevant_doc_ids for doc_id in retrieved_ids) else 0.0
            results[EvaluationMetric.HIT_RATE.value].append(
                EvaluationResult("hit_rate", hit_rate, query_id)
            )
            
            # Component/Net accuracy (if expected results provided)
            if query.expected_components:
                comp_acc = self._component_accuracy(search_hits, query.expected_components)
                results[EvaluationMetric.COMPONENT_ACCURACY.value].append(
                    EvaluationResult("component_accuracy", comp_acc, query_id)
                )
                
            if query.expected_nets:
                net_acc = self._net_accuracy(search_hits, query.expected_nets)
                results[EvaluationMetric.NET_ACCURACY.value].append(
                    EvaluationResult("net_accuracy", net_acc, query_id)
                )
        
        return results
    
    def _precision_at_k(self, retrieved: List[str], relevant: Set[str], k: int) -> float:
        """Calculate precision at k."""
        retrieved_k = retrieved[:k]
        if not retrieved_k:
            return 0.0
        relevant_retrieved = sum(1 for doc_id in retrieved_k if doc_id in relevant)
        return relevant_retrieved / len(retrieved_k)
    
    def _recall_at_k(self, retrieved: List[str], relevant: Set[str], k: int) -> float:
        """Calculate recall at k."""
        if not relevant:
            return 0.0
        retrieved_k = retrieved[:k]
        relevant_retrieved = sum(1 for doc_id in retrieved_k if doc_id in relevant)
        return relevant_retrieved / len(relevant)
    
    def _ndcg_at_k(self, search_hits: List[SearchHit], relevant: Set[str], k: int) -> float:
        """Calculate NDCG at k."""
        if not HAS_SKLEARN:
            # Fallback to DCG calculation
            return self._dcg_at_k(search_hits, relevant, k) / self._idcg_at_k(relevant, k)
        
        hits_k = search_hits[:k]
        if not hits_k:
            return 0.0
            
        # Create relevance scores (1 for relevant, 0 for irrelevant)
        relevance_scores = [1.0 if hit.chunk_id in relevant else 0.0 for hit in hits_k]
        
        if sum(relevance_scores) == 0:
            return 0.0
            
        # Calculate ideal relevance order
        ideal_relevance = sorted(relevance_scores, reverse=True)
        
        try:
            return ndcg_score([ideal_relevance], [relevance_scores], k=k)
        except Exception:
            return 0.0
    
    def _dcg_at_k(self, search_hits: List[SearchHit], relevant: Set[str], k: int) -> float:
        """Calculate DCG at k (fallback when sklearn unavailable)."""
        dcg = 0.0
        for i, hit in enumerate(search_hits[:k]):
            if hit.chunk_id in relevant:
                dcg += 1.0 / np.log2(i + 2)  # +2 because log2(1) = 0
        return dcg
    
    def _idcg_at_k(self, relevant: Set[str], k: int) -> float:
        """Calculate ideal DCG at k."""
        idcg = 0.0
        for i in range(min(len(relevant), k)):
            idcg += 1.0 / np.log2(i + 2)
        return idcg if idcg > 0 else 1.0
    
    def _mean_average_precision(self, retrieved: List[str], relevant: Set[str]) -> float:
        """Calculate Mean Average Precision."""
        if not relevant:
            return 0.0
            
        precision_sum = 0.0
        relevant_found = 0
        
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                relevant_found += 1
                precision_at_i = relevant_found / (i + 1)
                precision_sum += precision_at_i
        
        return precision_sum / len(relevant) if relevant else 0.0
    
    def _mean_reciprocal_rank(self, retrieved_lists: List[List[str]], relevant_lists: List[Set[str]]) -> float:
        """Calculate Mean Reciprocal Rank."""
        reciprocal_ranks = []
        
        for retrieved, relevant in zip(retrieved_lists, relevant_lists):
            for i, doc_id in enumerate(retrieved):
                if doc_id in relevant:
                    reciprocal_ranks.append(1.0 / (i + 1))
                    break
            else:
                reciprocal_ranks.append(0.0)
        
        return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    def _component_accuracy(self, search_hits: List[SearchHit], expected_components: List[str]) -> float:
        """Calculate component identification accuracy."""
        if not expected_components:
            return 0.0
            
        found_components = set()
        for hit in search_hits[:10]:  # Check top 10 results
            # Extract component mentions from hit metadata
            if hit.metadata and 'components' in hit.metadata:
                components = hit.metadata.get('components', [])
                if isinstance(components, list):
                    found_components.update(comp.get('id', '') for comp in components)
        
        expected_set = set(expected_components)
        intersection = found_components.intersection(expected_set)
        return len(intersection) / len(expected_set)
    
    def _net_accuracy(self, search_hits: List[SearchHit], expected_nets: List[str]) -> float:
        """Calculate net identification accuracy."""
        if not expected_nets:
            return 0.0
            
        found_nets = set()
        for hit in search_hits[:10]:  # Check top 10 results
            # Extract net mentions from hit metadata
            if hit.metadata and 'nets' in hit.metadata:
                nets = hit.metadata.get('nets', [])
                if isinstance(nets, list):
                    found_nets.update(net.get('name', '') for net in nets)
        
        expected_set = set(expected_nets)
        intersection = found_nets.intersection(expected_set)
        return len(intersection) / len(expected_set)


class LearningEvaluator:
    """Evaluates continual learning system performance."""
    
    async def evaluate_learning_signals(
        self,
        signals: List[LearningSignal],
        signal_collector
    ) -> Dict[str, EvaluationResult]:
        """Evaluate quality of collected learning signals."""
        results = {}
        
        # Group signals by type
        signal_groups = {}
        for signal in signals:
            signal_type = signal.signal_type.value
            if signal_type not in signal_groups:
                signal_groups[signal_type] = []
            signal_groups[signal_type].append(signal)
        
        # Evaluate each signal type
        for signal_type, group_signals in signal_groups.items():
            # Signal quality metrics
            avg_confidence = np.mean([s.confidence for s in group_signals])
            signal_coverage = len(set(s.context.get('page', 0) for s in group_signals))
            
            results[f"{signal_type}_avg_confidence"] = EvaluationResult(
                f"{signal_type}_avg_confidence", avg_confidence, "signal_eval"
            )
            results[f"{signal_type}_coverage"] = EvaluationResult(
                f"{signal_type}_coverage", signal_coverage, "signal_eval"
            )
        
        # Overall signal collection rate
        total_signals = len(signals)
        results["total_signals"] = EvaluationResult(
            "total_signals", total_signals, "signal_eval"
        )
        
        return results
    
    async def evaluate_model_improvement(
        self,
        model_registry,
        baseline_version: str,
        current_version: str,
        test_queries: List[EvaluationQuery]
    ) -> Dict[str, EvaluationResult]:
        """Evaluate improvement from baseline to current model."""
        results = {}
        
        # This would require actual model comparison
        # For now, mock the evaluation
        improvement_metrics = {
            "embedding_quality": 0.05,  # 5% improvement
            "search_latency": -0.02,    # 2% latency reduction
            "search_accuracy": 0.08     # 8% accuracy improvement
        }
        
        for metric, improvement in improvement_metrics.items():
            results[f"{metric}_improvement"] = EvaluationResult(
                f"{metric}_improvement", improvement, "model_comparison",
                metadata={
                    "baseline_version": baseline_version,
                    "current_version": current_version
                }
            )
        
        return results


class PerformanceMonitor:
    """Monitors system performance in real-time."""
    
    def __init__(self):
        self.latency_samples = []
        self.error_counts = {}
        self.search_counts = 0
        
    def record_search_latency(self, latency: float):
        """Record search latency sample."""
        self.latency_samples.append(latency)
        self.search_counts += 1
        
        # Keep only recent samples (sliding window)
        if len(self.latency_samples) > 1000:
            self.latency_samples = self.latency_samples[-1000:]
    
    def record_error(self, error_type: str):
        """Record an error occurrence."""
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        if not self.latency_samples:
            return {"latency_mean": 0.0, "latency_p95": 0.0, "error_rate": 0.0}
        
        latencies = np.array(self.latency_samples)
        total_errors = sum(self.error_counts.values())
        
        return {
            "latency_mean": float(np.mean(latencies)),
            "latency_p95": float(np.percentile(latencies, 95)),
            "latency_p99": float(np.percentile(latencies, 99)),
            "error_rate": total_errors / max(self.search_counts, 1),
            "total_searches": self.search_counts,
            "total_errors": total_errors
        }
    
    def reset_metrics(self):
        """Reset performance metrics."""
        self.latency_samples.clear()
        self.error_counts.clear()
        self.search_counts = 0
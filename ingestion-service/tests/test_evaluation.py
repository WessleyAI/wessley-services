"""
Tests for evaluation and benchmarking system.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from typing import List

from src.evaluation.metrics import (
    SearchEvaluator, LearningEvaluator, PerformanceMonitor,
    EvaluationQuery, EvaluationResult, BenchmarkSuite, EvaluationMetric
)
from src.evaluation.benchmarks import (
    BenchmarkManager, AutomotiveBenchmarks, MockDataGenerator
)
from src.evaluation.runner import (
    ComprehensiveEvaluationRunner, EvaluationConfig, EvaluationSummary
)
from src.semantic.search import SearchHit, SearchResult, SearchFilter
from src.semantic.ontology import VehicleSignature
from src.learning.signals import LearningSignal, SignalType


class TestSearchEvaluator:
    """Test search quality evaluation."""
    
    @pytest.fixture
    def search_evaluator(self):
        return SearchEvaluator(k_values=[1, 3, 5])
    
    @pytest.fixture
    def sample_benchmark(self):
        queries = [
            EvaluationQuery(
                query_text="starter relay",
                relevant_doc_ids={"doc_1", "doc_2"},
                expected_components=["K1", "STARTER_RELAY"],
                difficulty="easy"
            ),
            EvaluationQuery(
                query_text="ECU power circuit",
                relevant_doc_ids={"doc_3", "doc_4", "doc_5"},
                expected_components=["ECU", "F15"],
                expected_nets=["ECU_PWR", "VBATT"],
                difficulty="medium"
            )
        ]
        return BenchmarkSuite(
            name="test_suite",
            queries=queries,
            description="Test benchmark suite"
        )
    
    @pytest.fixture
    def mock_search_engine(self):
        engine = Mock()
        engine.search = AsyncMock()
        return engine
    
    def test_precision_at_k(self, search_evaluator):
        """Test precision at k calculation."""
        retrieved = ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"]
        relevant = {"doc_1", "doc_3", "doc_5"}
        
        # Precision@3 = 2/3 (doc_1, doc_3 are relevant out of first 3)
        precision = search_evaluator._precision_at_k(retrieved, relevant, 3)
        assert precision == 2/3
        
        # Precision@5 = 3/5 (doc_1, doc_3, doc_5 are relevant out of all 5)
        precision = search_evaluator._precision_at_k(retrieved, relevant, 5)
        assert precision == 3/5
    
    def test_recall_at_k(self, search_evaluator):
        """Test recall at k calculation."""
        retrieved = ["doc_1", "doc_2", "doc_3"]
        relevant = {"doc_1", "doc_3", "doc_5", "doc_7"}  # 4 relevant docs total
        
        # Recall@3 = 2/4 (doc_1, doc_3 found out of 4 relevant)
        recall = search_evaluator._recall_at_k(retrieved, relevant, 3)
        assert recall == 2/4
    
    def test_mean_average_precision(self, search_evaluator):
        """Test MAP calculation."""
        retrieved = ["doc_1", "doc_2", "doc_3", "doc_4"]
        relevant = {"doc_1", "doc_3"}
        
        # AP = (1/1 + 2/3) / 2 = (1 + 0.667) / 2 = 0.833
        map_score = search_evaluator._mean_average_precision(retrieved, relevant)
        expected = (1.0 + 2.0/3.0) / 2.0
        assert abs(map_score - expected) < 0.001
    
    def test_mean_reciprocal_rank(self, search_evaluator):
        """Test MRR calculation."""
        retrieved_lists = [
            ["doc_1", "doc_2", "doc_3"],  # First relevant at position 1 -> RR = 1/1 = 1.0
            ["doc_2", "doc_3", "doc_1"],  # First relevant at position 3 -> RR = 1/3 = 0.333
        ]
        relevant_lists = [
            {"doc_1", "doc_4"},
            {"doc_1", "doc_4"}
        ]
        
        mrr = search_evaluator._mean_reciprocal_rank(retrieved_lists, relevant_lists)
        expected = (1.0 + 1.0/3.0) / 2.0
        assert abs(mrr - expected) < 0.001
    
    @pytest.mark.asyncio
    async def test_evaluate_search_quality(self, search_evaluator, sample_benchmark, mock_search_engine):
        """Test full search quality evaluation."""
        # Mock search results
        mock_hits = [
            SearchHit(
                chunk_id="doc_1",
                score=0.9,
                text="Starter relay information",
                metadata={"components": [{"id": "K1"}]}
            ),
            SearchHit(
                chunk_id="doc_2", 
                score=0.8,
                text="More relay info",
                metadata={"components": [{"id": "STARTER_RELAY"}]}
            ),
            SearchHit(
                chunk_id="doc_6",
                score=0.7,
                text="Irrelevant content",
                metadata={}
            )
        ]
        mock_search_engine.search.return_value = mock_hits
        
        # Run evaluation
        results = await search_evaluator.evaluate_search_quality(
            mock_search_engine, sample_benchmark
        )
        
        # Check that all metrics were calculated
        assert EvaluationMetric.PRECISION_AT_K.value in results
        assert EvaluationMetric.RECALL_AT_K.value in results
        assert EvaluationMetric.MAP.value in results
        assert EvaluationMetric.MRR.value in results
        assert EvaluationMetric.HIT_RATE.value in results
        
        # Check precision@1 for first query (doc_1 is relevant)
        precision_1_results = [r for r in results[EvaluationMetric.PRECISION_AT_K.value] 
                              if r.metric_name == "precision_at_1" and r.query_id == "test_suite_0"]
        assert len(precision_1_results) == 1
        assert precision_1_results[0].value == 1.0  # First result is relevant


class TestLearningEvaluator:
    """Test learning system evaluation."""
    
    @pytest.fixture
    def learning_evaluator(self):
        return LearningEvaluator()
    
    @pytest.fixture
    def sample_signals(self):
        return [
            LearningSignal(
                signal_type=SignalType.OCR_DISAGREEMENT,
                confidence=0.8,
                context={"page": 1, "bbox": [100, 100, 200, 200]},
                data={"engines": ["tesseract", "deepseek"]}
            ),
            LearningSignal(
                signal_type=SignalType.SYMBOL_DETECTION_ERROR,
                confidence=0.9,
                context={"page": 2, "component_type": "resistor"},
                data={"missed_detection": True}
            ),
            LearningSignal(
                signal_type=SignalType.OCR_DISAGREEMENT,
                confidence=0.7,
                context={"page": 1, "bbox": [300, 300, 400, 400]},
                data={"engines": ["tesseract", "mistral"]}
            )
        ]
    
    @pytest.mark.asyncio
    async def test_evaluate_learning_signals(self, learning_evaluator, sample_signals):
        """Test learning signal quality evaluation."""
        mock_signal_collector = Mock()
        
        results = await learning_evaluator.evaluate_learning_signals(
            sample_signals, mock_signal_collector
        )
        
        # Check that metrics were calculated for each signal type
        assert "ocr_disagreement_avg_confidence" in results
        assert "symbol_detection_error_avg_confidence" in results
        assert "total_signals" in results
        
        # Check confidence calculation
        ocr_signals = [s for s in sample_signals if s.signal_type == SignalType.OCR_DISAGREEMENT]
        expected_ocr_confidence = sum(s.confidence for s in ocr_signals) / len(ocr_signals)
        
        assert abs(results["ocr_disagreement_avg_confidence"].value - expected_ocr_confidence) < 0.001
        assert results["total_signals"].value == len(sample_signals)
    
    @pytest.mark.asyncio
    async def test_evaluate_model_improvement(self, learning_evaluator):
        """Test model improvement evaluation."""
        mock_registry = Mock()
        
        results = await learning_evaluator.evaluate_model_improvement(
            mock_registry, "v1.0.0", "v1.1.0", []
        )
        
        # Check that improvement metrics are present
        assert "embedding_quality_improvement" in results
        assert "search_latency_improvement" in results
        assert "search_accuracy_improvement" in results
        
        # Check metadata contains version info
        embedding_result = results["embedding_quality_improvement"]
        assert embedding_result.metadata["baseline_version"] == "v1.0.0"
        assert embedding_result.metadata["current_version"] == "v1.1.0"


class TestPerformanceMonitor:
    """Test performance monitoring."""
    
    @pytest.fixture
    def performance_monitor(self):
        return PerformanceMonitor()
    
    def test_record_search_latency(self, performance_monitor):
        """Test latency recording."""
        latencies = [0.1, 0.2, 0.15, 0.3, 0.25]
        
        for latency in latencies:
            performance_monitor.record_search_latency(latency)
        
        metrics = performance_monitor.get_performance_metrics()
        
        assert metrics["latency_mean"] == sum(latencies) / len(latencies)
        assert metrics["total_searches"] == len(latencies)
        assert metrics["latency_p95"] > 0
        assert metrics["error_rate"] == 0  # No errors recorded
    
    def test_record_error(self, performance_monitor):
        """Test error recording."""
        performance_monitor.record_search_latency(0.1)
        performance_monitor.record_error("timeout")
        performance_monitor.record_error("timeout")
        performance_monitor.record_error("connection_error")
        
        metrics = performance_monitor.get_performance_metrics()
        
        assert metrics["total_errors"] == 3
        assert metrics["total_searches"] == 1
        assert metrics["error_rate"] == 3.0  # 3 errors / 1 search
    
    def test_reset_metrics(self, performance_monitor):
        """Test metrics reset."""
        performance_monitor.record_search_latency(0.1)
        performance_monitor.record_error("test_error")
        
        performance_monitor.reset_metrics()
        metrics = performance_monitor.get_performance_metrics()
        
        assert metrics["total_searches"] == 0
        assert metrics["total_errors"] == 0
        assert metrics["latency_mean"] == 0.0


class TestBenchmarkManager:
    """Test benchmark management."""
    
    @pytest.fixture
    def benchmark_manager(self, tmp_path):
        return BenchmarkManager(results_dir=str(tmp_path / "benchmarks"))
    
    @pytest.fixture
    def sample_suite(self):
        return AutomotiveBenchmarks.get_component_identification_suite()
    
    @pytest.mark.asyncio
    async def test_run_benchmark_suite(self, benchmark_manager, sample_suite):
        """Test running a benchmark suite."""
        mock_evaluator = Mock()
        mock_engine = Mock()
        
        # Mock evaluation results
        mock_results = {
            "precision_at_1_mean": [EvaluationResult("precision_at_1", 0.8, "query_1")],
            "recall_at_1_mean": [EvaluationResult("recall_at_1", 0.7, "query_1")]
        }
        mock_evaluator.evaluate_search_quality = AsyncMock(return_value=mock_results)
        
        results = await benchmark_manager.run_benchmark_suite(
            sample_suite, mock_evaluator, mock_engine
        )
        
        # Check that aggregated results are computed
        assert "precision_at_1_mean_mean" in results
        assert "recall_at_1_mean_mean" in results
        
        # Check that evaluator was called with correct parameters
        mock_evaluator.evaluate_search_quality.assert_called_once_with(mock_engine, sample_suite)
    
    def test_compare_results(self, benchmark_manager):
        """Test result comparison."""
        baseline = {"precision_at_5": 0.8, "recall_at_5": 0.7}
        current = {"precision_at_5": 0.85, "recall_at_5": 0.75}
        
        comparison = benchmark_manager.compare_results(baseline, current)
        
        # Check precision improvement
        assert "precision_at_5" in comparison
        precision_comp = comparison["precision_at_5"]
        assert precision_comp["baseline"] == 0.8
        assert precision_comp["current"] == 0.85
        assert abs(precision_comp["improvement"] - 0.0625) < 0.001  # (0.85-0.8)/0.8 = 0.0625
        
        # Check recall improvement
        assert "recall_at_5" in comparison
        recall_comp = comparison["recall_at_5"]
        assert abs(recall_comp["improvement"] - (0.75-0.7)/0.7) < 0.001


class TestAutomotiveBenchmarks:
    """Test automotive benchmark suites."""
    
    def test_component_identification_suite(self):
        """Test component identification benchmark."""
        suite = AutomotiveBenchmarks.get_component_identification_suite()
        
        assert suite.name == "component_identification"
        assert len(suite.queries) > 0
        assert suite.automotive_focus is True
        
        # Check that queries have expected structure
        for query in suite.queries:
            assert query.query_text
            assert len(query.relevant_doc_ids) > 0
            assert query.difficulty in ["easy", "medium", "hard"]
    
    def test_wiring_analysis_suite(self):
        """Test wiring analysis benchmark."""
        suite = AutomotiveBenchmarks.get_wiring_analysis_suite()
        
        assert suite.name == "wiring_analysis"
        assert len(suite.queries) > 0
        
        # Check that queries focus on wiring/nets
        for query in suite.queries:
            assert any(term in query.query_text.lower() 
                      for term in ["wiring", "circuit", "signal", "power", "trace"])
    
    def test_troubleshooting_suite(self):
        """Test troubleshooting benchmark."""
        suite = AutomotiveBenchmarks.get_troubleshooting_suite()
        
        assert suite.name == "troubleshooting"
        assert len(suite.queries) > 0
        
        # Check that queries focus on diagnostics
        for query in suite.queries:
            assert any(term in query.query_text.lower() 
                      for term in ["test", "diagnosis", "no start", "dim", "stall"])
    
    def test_get_all_suites(self):
        """Test getting all benchmark suites."""
        all_suites = AutomotiveBenchmarks.get_all_suites()
        
        assert len(all_suites) >= 3
        suite_names = [s.name for s in all_suites]
        assert "component_identification" in suite_names
        assert "wiring_analysis" in suite_names
        assert "troubleshooting" in suite_names


class TestComprehensiveEvaluationRunner:
    """Test comprehensive evaluation runner."""
    
    @pytest.fixture
    def evaluation_runner(self):
        mock_search_engine = Mock()
        mock_model_registry = Mock()
        mock_signal_collector = Mock()
        
        config = EvaluationConfig(
            benchmark_suites=["component_identification"],
            search_strategies=["hybrid"],
            include_learning_eval=True,
            include_performance_monitoring=True
        )
        
        return ComprehensiveEvaluationRunner(
            search_engine=mock_search_engine,
            model_registry=mock_model_registry,
            signal_collector=mock_signal_collector,
            config=config
        )
    
    @pytest.mark.asyncio
    async def test_run_comprehensive_evaluation(self, evaluation_runner):
        """Test full evaluation run."""
        # Mock the benchmark manager
        with patch.object(evaluation_runner.benchmark_manager, 'run_benchmark_suite') as mock_run:
            mock_run.return_value = {
                "precision_at_1_mean": 0.8,
                "recall_at_1_mean": 0.7,
                "ndcg_at_1_mean": 0.75
            }
            
            # Mock search strategy comparison
            with patch.object(evaluation_runner.search_engine, 'search') as mock_search:
                mock_search.return_value = [
                    SearchHit("doc_1", 0.9, "test content", {})
                ]
                
                summary = await evaluation_runner.run_comprehensive_evaluation()
                
                # Check summary structure
                assert isinstance(summary, EvaluationSummary)
                assert summary.benchmarks_run > 0
                assert summary.total_duration > 0
                assert len(summary.overall_scores) > 0
                assert len(summary.strategy_comparison) > 0
    
    def test_compute_overall_scores(self, evaluation_runner):
        """Test overall score computation."""
        benchmark_results = {
            "component_identification": {
                "precision_at_5_mean": 0.8,
                "recall_at_5_mean": 0.7,
                "ndcg_at_5_mean": 0.75,
                "component_accuracy_mean": 0.85
            }
        }
        
        overall_scores = evaluation_runner._compute_overall_scores(benchmark_results)
        
        assert "overall_precision" in overall_scores
        assert "overall_recall" in overall_scores
        assert "overall_ndcg" in overall_scores
        assert "overall_f1" in overall_scores
        assert "overall_component_accuracy" in overall_scores
        
        # Check F1 calculation
        p = overall_scores["overall_precision"]
        r = overall_scores["overall_recall"]
        expected_f1 = 2 * p * r / (p + r)
        assert abs(overall_scores["overall_f1"] - expected_f1) < 0.001


class TestMockDataGenerator:
    """Test mock data generation for testing."""
    
    def test_create_mock_search_hits(self):
        """Test mock search hit generation."""
        query = "starter relay"
        hits = MockDataGenerator.create_mock_search_hits(query, num_hits=5)
        
        assert len(hits) == 5
        
        for i, hit in enumerate(hits):
            assert "chunk_id" in hit
            assert "score" in hit
            assert "text" in hit
            assert "metadata" in hit
            
            # Check decreasing relevance scores
            if i > 0:
                assert hit["score"] <= hits[i-1]["score"]
            
            # Check that query is mentioned in text
            assert query in hit["text"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
"""
Evaluation runner and orchestration for comprehensive benchmarking.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from .metrics import SearchEvaluator, LearningEvaluator, PerformanceMonitor
from .benchmarks import BenchmarkManager, AutomotiveBenchmarks
from ..semantic.search import HybridAutomotiveSearch
from ..learning.signals import SignalCollector
from ..core.models import ModelRegistry


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs."""
    search_strategies: List[str] = None
    k_values: List[int] = None
    benchmark_suites: List[str] = None
    output_dir: str = "benchmarks/results"
    include_learning_eval: bool = True
    include_performance_monitoring: bool = True
    save_detailed_results: bool = True
    
    def __post_init__(self):
        if self.search_strategies is None:
            self.search_strategies = ["dense", "sparse", "hybrid"]
        if self.k_values is None:
            self.k_values = [1, 3, 5, 10]
        if self.benchmark_suites is None:
            self.benchmark_suites = ["component_identification", "wiring_analysis", "troubleshooting"]


@dataclass
class EvaluationSummary:
    """Summary of evaluation run results."""
    timestamp: datetime
    total_duration: float
    benchmarks_run: int
    total_queries: int
    overall_scores: Dict[str, float]
    strategy_comparison: Dict[str, Dict[str, float]]
    learning_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]
    config: EvaluationConfig


class ComprehensiveEvaluationRunner:
    """Orchestrates comprehensive evaluation of the automotive search system."""
    
    def __init__(
        self,
        search_engine: HybridAutomotiveSearch,
        model_registry: ModelRegistry,
        signal_collector: SignalCollector,
        config: EvaluationConfig = None
    ):
        self.search_engine = search_engine
        self.model_registry = model_registry
        self.signal_collector = signal_collector
        self.config = config or EvaluationConfig()
        
        # Initialize evaluators
        self.search_evaluator = SearchEvaluator(k_values=self.config.k_values)
        self.learning_evaluator = LearningEvaluator()
        self.performance_monitor = PerformanceMonitor()
        self.benchmark_manager = BenchmarkManager(results_dir=self.config.output_dir)
        
        # Results storage
        self.results_dir = Path(self.config.output_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    async def run_comprehensive_evaluation(self) -> EvaluationSummary:
        """Run complete evaluation suite."""
        start_time = time.time()
        timestamp = datetime.now(timezone.utc)
        
        print("🚀 Starting comprehensive evaluation of automotive search system")
        print(f"Timestamp: {timestamp}")
        print(f"Configuration: {asdict(self.config)}")
        
        # Initialize results tracking
        all_results = {}
        strategy_results = {}
        learning_results = {}
        performance_results = {}
        
        # 1. Run search quality benchmarks
        print("\n📊 Running search quality benchmarks...")
        search_results = await self._run_search_benchmarks()
        all_results.update(search_results)
        
        # 2. Compare search strategies
        print("\n🔍 Comparing search strategies...")
        strategy_results = await self._compare_search_strategies()
        
        # 3. Evaluate learning system
        if self.config.include_learning_eval:
            print("\n🧠 Evaluating learning system...")
            learning_results = await self._evaluate_learning_system()
        
        # 4. Monitor performance
        if self.config.include_performance_monitoring:
            print("\n⚡ Collecting performance metrics...")
            performance_results = await self._collect_performance_metrics()
        
        # 5. Generate overall scores
        print("\n📈 Computing overall scores...")
        overall_scores = self._compute_overall_scores(all_results)
        
        total_duration = time.time() - start_time
        
        # Create evaluation summary
        summary = EvaluationSummary(
            timestamp=timestamp,
            total_duration=total_duration,
            benchmarks_run=len(self.config.benchmark_suites),
            total_queries=sum(len(AutomotiveBenchmarks.get_component_identification_suite().queries) 
                            for _ in self.config.benchmark_suites),  # Simplified
            overall_scores=overall_scores,
            strategy_comparison=strategy_results,
            learning_metrics=learning_results,
            performance_metrics=performance_results,
            config=self.config
        )
        
        # Save comprehensive results
        await self._save_comprehensive_results(summary, all_results)
        
        print(f"\n✅ Evaluation completed in {total_duration:.2f}s")
        self._print_summary(summary)
        
        return summary
    
    async def _run_search_benchmarks(self) -> Dict[str, Any]:
        """Run all configured benchmark suites."""
        results = {}
        
        # Get benchmark suites
        available_suites = {
            "component_identification": AutomotiveBenchmarks.get_component_identification_suite(),
            "wiring_analysis": AutomotiveBenchmarks.get_wiring_analysis_suite(),
            "troubleshooting": AutomotiveBenchmarks.get_troubleshooting_suite()
        }
        
        for suite_name in self.config.benchmark_suites:
            if suite_name in available_suites:
                print(f"  Running {suite_name} benchmark...")
                suite = available_suites[suite_name]
                
                suite_results = await self.benchmark_manager.run_benchmark_suite(
                    suite, self.search_evaluator, self.search_engine
                )
                results[suite_name] = suite_results
            else:
                print(f"  Warning: Unknown benchmark suite '{suite_name}'")
        
        return results
    
    async def _compare_search_strategies(self) -> Dict[str, Dict[str, float]]:
        """Compare different search strategies."""
        strategy_results = {}
        
        # Use a sample query for strategy comparison
        sample_query = "starter relay circuit wiring"
        
        for strategy in self.config.search_strategies:
            print(f"  Testing {strategy} strategy...")
            
            # Measure search performance
            start_time = time.time()
            try:
                search_hits = await self.search_engine.search(
                    query=sample_query,
                    filters=None,
                    limit=10,
                    strategy=strategy
                )
                latency = time.time() - start_time
                
                # Mock relevance evaluation (in real implementation, would use ground truth)
                avg_score = sum(hit.score for hit in search_hits) / len(search_hits) if search_hits else 0.0
                
                strategy_results[strategy] = {
                    "latency": latency,
                    "avg_relevance_score": avg_score,
                    "num_results": len(search_hits),
                    "top_score": search_hits[0].score if search_hits else 0.0
                }
                
            except Exception as e:
                print(f"    Error testing {strategy}: {e}")
                strategy_results[strategy] = {"error": str(e)}
        
        return strategy_results
    
    async def _evaluate_learning_system(self) -> Dict[str, float]:
        """Evaluate the continual learning system."""
        # Mock learning signals for evaluation
        from ..learning.signals import LearningSignal, SignalType
        
        mock_signals = [
            LearningSignal(
                signal_type=SignalType.OCR_DISAGREEMENT,
                confidence=0.8,
                context={"page": 1, "bbox": [100, 100, 200, 200]},
                data={"engine_disagreement": True}
            ),
            LearningSignal(
                signal_type=SignalType.SYMBOL_DETECTION_ERROR,
                confidence=0.9,
                context={"page": 2, "component_type": "resistor"},
                data={"detection_missed": True}
            )
        ]
        
        # Evaluate signal quality
        signal_results = await self.learning_evaluator.evaluate_learning_signals(
            mock_signals, self.signal_collector
        )
        
        # Evaluate model improvement (mock)
        model_results = await self.learning_evaluator.evaluate_model_improvement(
            self.model_registry,
            baseline_version="v1.0.0",
            current_version="v1.1.0",
            test_queries=[]
        )
        
        # Combine results
        learning_metrics = {}
        for metric_name, result in signal_results.items():
            learning_metrics[metric_name] = result.value
        
        for metric_name, result in model_results.items():
            learning_metrics[metric_name] = result.value
        
        return learning_metrics
    
    async def _collect_performance_metrics(self) -> Dict[str, float]:
        """Collect system performance metrics."""
        # Simulate some search operations to collect metrics
        sample_queries = [
            "ECU power supply",
            "ground connections",
            "relay testing procedures",
            "CAN bus wiring"
        ]
        
        for query in sample_queries:
            start_time = time.time()
            try:
                await self.search_engine.search(query=query, filters=None, limit=5)
                latency = time.time() - start_time
                self.performance_monitor.record_search_latency(latency)
            except Exception as e:
                self.performance_monitor.record_error("search_error")
        
        return self.performance_monitor.get_performance_metrics()
    
    def _compute_overall_scores(self, benchmark_results: Dict[str, Any]) -> Dict[str, float]:
        """Compute overall performance scores."""
        overall_scores = {}
        
        # Aggregate precision and recall scores
        precision_scores = []
        recall_scores = []
        ndcg_scores = []
        
        for suite_name, suite_results in benchmark_results.items():
            for metric, value in suite_results.items():
                if "precision" in metric and "mean" in metric:
                    precision_scores.append(value)
                elif "recall" in metric and "mean" in metric:
                    recall_scores.append(value)
                elif "ndcg" in metric and "mean" in metric:
                    ndcg_scores.append(value)
        
        if precision_scores:
            overall_scores["overall_precision"] = sum(precision_scores) / len(precision_scores)
        if recall_scores:
            overall_scores["overall_recall"] = sum(recall_scores) / len(recall_scores)
        if ndcg_scores:
            overall_scores["overall_ndcg"] = sum(ndcg_scores) / len(ndcg_scores)
        
        # Compute F1 score
        if "overall_precision" in overall_scores and "overall_recall" in overall_scores:
            p = overall_scores["overall_precision"]
            r = overall_scores["overall_recall"]
            if p + r > 0:
                overall_scores["overall_f1"] = 2 * p * r / (p + r)
        
        # Add component and net accuracy if available
        comp_accuracies = []
        net_accuracies = []
        
        for suite_results in benchmark_results.values():
            for metric, value in suite_results.items():
                if "component_accuracy" in metric and "mean" in metric:
                    comp_accuracies.append(value)
                elif "net_accuracy" in metric and "mean" in metric:
                    net_accuracies.append(value)
        
        if comp_accuracies:
            overall_scores["overall_component_accuracy"] = sum(comp_accuracies) / len(comp_accuracies)
        if net_accuracies:
            overall_scores["overall_net_accuracy"] = sum(net_accuracies) / len(net_accuracies)
        
        return overall_scores
    
    async def _save_comprehensive_results(
        self,
        summary: EvaluationSummary,
        detailed_results: Dict[str, Any]
    ):
        """Save comprehensive evaluation results."""
        timestamp_str = summary.timestamp.strftime("%Y%m%d_%H%M%S")
        
        # Save summary
        summary_file = self.results_dir / f"evaluation_summary_{timestamp_str}.json"
        with open(summary_file, 'w') as f:
            json.dump(asdict(summary), f, indent=2, default=str)
        
        # Save detailed results if configured
        if self.config.save_detailed_results:
            detailed_file = self.results_dir / f"evaluation_detailed_{timestamp_str}.json"
            with open(detailed_file, 'w') as f:
                json.dump(detailed_results, f, indent=2, default=str)
        
        print(f"Results saved to: {summary_file}")
    
    def _print_summary(self, summary: EvaluationSummary):
        """Print evaluation summary to console."""
        print("\n" + "="*60)
        print("🎯 EVALUATION SUMMARY")
        print("="*60)
        
        print(f"⏱️  Duration: {summary.total_duration:.2f}s")
        print(f"📊 Benchmarks: {summary.benchmarks_run}")
        print(f"🔍 Queries: {summary.total_queries}")
        
        print("\n📈 Overall Scores:")
        for metric, score in summary.overall_scores.items():
            print(f"  {metric}: {score:.4f}")
        
        print("\n🔍 Strategy Comparison:")
        for strategy, metrics in summary.strategy_comparison.items():
            if "error" not in metrics:
                print(f"  {strategy}: latency={metrics.get('latency', 0):.3f}s, "
                      f"relevance={metrics.get('avg_relevance_score', 0):.3f}")
        
        if summary.learning_metrics:
            print("\n🧠 Learning System:")
            for metric, value in list(summary.learning_metrics.items())[:3]:  # Show top 3
                print(f"  {metric}: {value:.4f}")
        
        if summary.performance_metrics:
            print("\n⚡ Performance:")
            perf = summary.performance_metrics
            print(f"  Avg Latency: {perf.get('latency_mean', 0):.3f}s")
            print(f"  P95 Latency: {perf.get('latency_p95', 0):.3f}s")
            print(f"  Error Rate: {perf.get('error_rate', 0):.3f}")
        
        print("="*60)


async def main():
    """Example usage of the evaluation runner."""
    # This would be called with actual system components
    print("Evaluation runner ready - import and configure with actual system components")


if __name__ == "__main__":
    asyncio.run(main())
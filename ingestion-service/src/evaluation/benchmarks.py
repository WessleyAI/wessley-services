"""
Benchmark suites and datasets for automotive electronics evaluation.
"""

import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict

from .metrics import BenchmarkSuite, EvaluationQuery, EvaluationResult
from ..semantic.ontology import VehicleSignature


class AutomotiveBenchmarks:
    """Pre-defined benchmark suites for automotive electronics."""
    
    @classmethod
    def get_component_identification_suite(cls) -> BenchmarkSuite:
        """Benchmark for component identification accuracy."""
        queries = [
            EvaluationQuery(
                query_text="starter relay location",
                relevant_doc_ids={"doc_starter_1", "doc_relay_panel_1"},
                vehicle_context=VehicleSignature(make="Toyota", model="Camry", year=2015),
                expected_components=["K1", "STARTER_RELAY"],
                difficulty="easy"
            ),
            EvaluationQuery(
                query_text="ECU power supply circuit",
                relevant_doc_ids={"doc_ecu_power_1", "doc_fuse_panel_2"},
                vehicle_context=VehicleSignature(make="Honda", model="Accord", year=2018),
                expected_components=["ECU", "F15", "F16"],
                expected_nets=["ECU_PWR", "VBATT"],
                difficulty="medium"
            ),
            EvaluationQuery(
                query_text="ground point for dashboard cluster",
                relevant_doc_ids={"doc_ground_1", "doc_cluster_1"},
                vehicle_context=VehicleSignature(make="Ford", model="F150", year=2020),
                expected_components=["G105", "CLUSTER"],
                expected_nets=["GND", "CHASSIS_GND"],
                difficulty="medium"
            ),
            EvaluationQuery(
                query_text="CAN bus termination resistor",
                relevant_doc_ids={"doc_can_bus_1", "doc_network_1"},
                vehicle_context=VehicleSignature(make="BMW", model="X5", year=2019),
                system_filter="communication",
                expected_components=["R120", "CAN_TERM"],
                expected_nets=["CAN_H", "CAN_L"],
                difficulty="hard"
            ),
            EvaluationQuery(
                query_text="alternator charging circuit wiring",
                relevant_doc_ids={"doc_alternator_1", "doc_charging_1"},
                vehicle_context=VehicleSignature(make="Hyundai", model="Elantra", year=2017),
                system_filter="charging",
                expected_components=["ALT", "F3", "D1"],
                expected_nets=["BATT+", "ALT_S", "ALT_L"],
                difficulty="medium"
            )
        ]
        
        return BenchmarkSuite(
            name="component_identification",
            queries=queries,
            description="Tests ability to identify automotive electrical components from search queries",
            version="1.0"
        )
    
    @classmethod
    def get_wiring_analysis_suite(cls) -> BenchmarkSuite:
        """Benchmark for wiring and net analysis."""
        queries = [
            EvaluationQuery(
                query_text="trace power from battery to headlights",
                relevant_doc_ids={"doc_headlight_1", "doc_power_dist_1"},
                expected_nets=["BATT+", "HEAD_LP", "HEAD_LO", "HEAD_HI"],
                difficulty="medium"
            ),
            EvaluationQuery(
                query_text="fuel pump relay control signal",
                relevant_doc_ids={"doc_fuel_system_1", "doc_pcm_1"},
                expected_components=["K_FUEL", "PCM"],
                expected_nets=["FUEL_PUMP_CTRL", "PCM_PWR"],
                difficulty="hard"
            ),
            EvaluationQuery(
                query_text="oxygen sensor heater circuit",
                relevant_doc_ids={"doc_o2_sensor_1", "doc_emissions_1"},
                system_filter="emissions",
                expected_components=["O2_SENSOR", "F8"],
                expected_nets=["O2_HTR", "O2_SIG"],
                difficulty="hard"
            )
        ]
        
        return BenchmarkSuite(
            name="wiring_analysis",
            queries=queries,
            description="Tests ability to analyze wiring connections and signal flow",
            version="1.0"
        )
    
    @classmethod
    def get_troubleshooting_suite(cls) -> BenchmarkSuite:
        """Benchmark for troubleshooting scenarios."""
        queries = [
            EvaluationQuery(
                query_text="no start condition relay test",
                relevant_doc_ids={"doc_no_start_1", "doc_relay_test_1"},
                expected_components=["K1", "K2", "STARTER"],
                difficulty="medium"
            ),
            EvaluationQuery(
                query_text="dim headlight diagnosis",
                relevant_doc_ids={"doc_headlight_dim_1", "doc_ground_poor_1"},
                expected_components=["G101", "HEAD_LP"],
                difficulty="easy"
            ),
            EvaluationQuery(
                query_text="intermittent engine stall ECU codes",
                relevant_doc_ids={"doc_stall_1", "doc_ecu_codes_1"},
                expected_components=["ECU", "CRANK_SENSOR", "CAM_SENSOR"],
                difficulty="hard"
            )
        ]
        
        return BenchmarkSuite(
            name="troubleshooting",
            queries=queries,
            description="Tests diagnostic and troubleshooting query capabilities",
            version="1.0"
        )
    
    @classmethod
    def get_all_suites(cls) -> List[BenchmarkSuite]:
        """Get all available benchmark suites."""
        return [
            cls.get_component_identification_suite(),
            cls.get_wiring_analysis_suite(),
            cls.get_troubleshooting_suite()
        ]


class BenchmarkManager:
    """Manages benchmark execution and result storage."""
    
    def __init__(self, results_dir: str = "benchmarks/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    async def run_benchmark_suite(
        self,
        benchmark: BenchmarkSuite,
        search_evaluator,
        search_engine
    ) -> Dict[str, Any]:
        """Run a complete benchmark suite."""
        print(f"Running benchmark suite: {benchmark.name}")
        
        # Run evaluation
        metric_results = await search_evaluator.evaluate_search_quality(
            search_engine, benchmark
        )
        
        # Aggregate results
        aggregated = self._aggregate_results(metric_results)
        
        # Store results
        result_file = self.results_dir / f"{benchmark.name}_results.json"
        await self._save_results(result_file, {
            "benchmark": {
                "name": benchmark.name,
                "description": benchmark.description,
                "version": benchmark.version,
                "query_count": len(benchmark.queries)
            },
            "results": aggregated,
            "raw_metrics": {k: [asdict(r) for r in v] for k, v in metric_results.items()}
        })
        
        return aggregated
    
    def _aggregate_results(self, metric_results: Dict[str, List[EvaluationResult]]) -> Dict[str, float]:
        """Aggregate evaluation results into summary metrics."""
        aggregated = {}
        
        for metric_name, results in metric_results.items():
            if not results:
                continue
                
            values = [r.value for r in results]
            aggregated[f"{metric_name}_mean"] = sum(values) / len(values)
            aggregated[f"{metric_name}_min"] = min(values)
            aggregated[f"{metric_name}_max"] = max(values)
            
            # Add percentiles for latency metrics
            if "latency" in metric_name:
                sorted_values = sorted(values)
                n = len(sorted_values)
                aggregated[f"{metric_name}_p95"] = sorted_values[int(0.95 * n)]
                aggregated[f"{metric_name}_p99"] = sorted_values[int(0.99 * n)]
        
        return aggregated
    
    async def _save_results(self, file_path: Path, results: Dict[str, Any]):
        """Save benchmark results to file."""
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to: {file_path}")
    
    async def run_all_benchmarks(self, search_evaluator, search_engine) -> Dict[str, Dict[str, float]]:
        """Run all automotive benchmark suites."""
        all_results = {}
        
        for benchmark in AutomotiveBenchmarks.get_all_suites():
            try:
                results = await self.run_benchmark_suite(
                    benchmark, search_evaluator, search_engine
                )
                all_results[benchmark.name] = results
                print(f"✓ Completed {benchmark.name}")
            except Exception as e:
                print(f"✗ Failed {benchmark.name}: {e}")
                all_results[benchmark.name] = {"error": str(e)}
        
        # Save combined results
        combined_file = self.results_dir / "all_benchmarks_summary.json"
        await self._save_results(combined_file, {
            "timestamp": str(datetime.now()),
            "benchmarks": all_results
        })
        
        return all_results
    
    def load_historical_results(self, benchmark_name: str) -> List[Dict[str, Any]]:
        """Load historical results for trend analysis."""
        pattern = f"{benchmark_name}_results_*.json"
        result_files = list(self.results_dir.glob(pattern))
        
        historical_results = []
        for file_path in sorted(result_files):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    historical_results.append(data)
            except Exception as e:
                print(f"Failed to load {file_path}: {e}")
        
        return historical_results
    
    def compare_results(
        self,
        baseline_results: Dict[str, float],
        current_results: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """Compare current results with baseline."""
        comparison = {}
        
        for metric in baseline_results:
            if metric in current_results:
                baseline_val = baseline_results[metric]
                current_val = current_results[metric]
                
                if baseline_val != 0:
                    improvement = (current_val - baseline_val) / baseline_val
                else:
                    improvement = float('inf') if current_val > 0 else 0.0
                
                comparison[metric] = {
                    "baseline": baseline_val,
                    "current": current_val,
                    "improvement": improvement,
                    "absolute_change": current_val - baseline_val
                }
        
        return comparison


# Mock data for testing (in real implementation, would come from actual documents)
from datetime import datetime


class MockDataGenerator:
    """Generates mock evaluation data for testing."""
    
    @staticmethod
    def create_mock_search_hits(query: str, num_hits: int = 10) -> List[Dict[str, Any]]:
        """Create mock search hits for evaluation."""
        hits = []
        
        for i in range(num_hits):
            hit = {
                "chunk_id": f"mock_doc_{i}",
                "score": 0.9 - (i * 0.05),  # Decreasing relevance
                "text": f"Mock document {i} about {query}",
                "metadata": {
                    "components": [{"id": f"COMP_{i}", "type": "resistor"}],
                    "nets": [{"name": f"NET_{i}"}],
                    "page": i % 3 + 1
                }
            }
            hits.append(hit)
        
        return hits
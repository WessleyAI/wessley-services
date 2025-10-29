#!/usr/bin/env python3
"""
Command-line interface for running automotive electronics evaluations.

Usage:
    python -m evaluation.cli run --benchmark component_identification
    python -m evaluation.cli compare --baseline v1.0 --current v1.1
    python -m evaluation.cli report --benchmark all --format json
"""

import asyncio
import click
import json
from pathlib import Path
from typing import List, Optional

from .runner import ComprehensiveEvaluationRunner, EvaluationConfig
from .benchmarks import BenchmarkManager, AutomotiveBenchmarks
from .metrics import SearchEvaluator, LearningEvaluator


@click.group()
def cli():
    """Automotive electronics evaluation CLI."""
    pass


@cli.command()
@click.option('--benchmark', '-b', multiple=True, 
              help='Benchmark suite to run (can specify multiple)')
@click.option('--strategy', '-s', multiple=True,
              help='Search strategy to test (dense, sparse, hybrid)')
@click.option('--output-dir', '-o', default='benchmarks/results',
              help='Output directory for results')
@click.option('--include-learning/--no-learning', default=True,
              help='Include learning system evaluation')
@click.option('--include-performance/--no-performance', default=True,
              help='Include performance monitoring')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def run(benchmark: List[str], strategy: List[str], output_dir: str,
        include_learning: bool, include_performance: bool, verbose: bool):
    """Run comprehensive evaluation."""
    
    async def _run_evaluation():
        # Default values
        if not benchmark:
            benchmark_suites = ["component_identification", "wiring_analysis", "troubleshooting"]
        else:
            benchmark_suites = list(benchmark)
            
        if not strategy:
            search_strategies = ["dense", "sparse", "hybrid"]
        else:
            search_strategies = list(strategy)
        
        if verbose:
            click.echo(f"Running evaluation with:")
            click.echo(f"  Benchmarks: {', '.join(benchmark_suites)}")
            click.echo(f"  Strategies: {', '.join(search_strategies)}")
            click.echo(f"  Output: {output_dir}")
            click.echo(f"  Learning eval: {include_learning}")
            click.echo(f"  Performance eval: {include_performance}")
            click.echo()
        
        # Create configuration
        config = EvaluationConfig(
            benchmark_suites=benchmark_suites,
            search_strategies=search_strategies,
            output_dir=output_dir,
            include_learning_eval=include_learning,
            include_performance_monitoring=include_performance
        )
        
        try:
            # Mock system components for CLI usage
            from ..semantic.search import HybridAutomotiveSearch
            from ..core.models import ModelRegistry
            from ..learning.signals import SignalCollector
            
            # Initialize with mock components
            search_engine = HybridAutomotiveSearch()
            model_registry = ModelRegistry()
            signal_collector = SignalCollector()
            
            # Create and run evaluator
            evaluator = ComprehensiveEvaluationRunner(
                search_engine=search_engine,
                model_registry=model_registry,
                signal_collector=signal_collector,
                config=config
            )
            
            click.echo("🚀 Starting comprehensive evaluation...")
            summary = await evaluator.run_comprehensive_evaluation()
            
            # Display results
            click.echo("\n" + "="*60)
            click.echo("✅ EVALUATION COMPLETED")
            click.echo("="*60)
            click.echo(f"Duration: {summary.total_duration:.2f}s")
            click.echo(f"Benchmarks: {summary.benchmarks_run}")
            click.echo(f"Queries: {summary.total_queries}")
            
            if summary.overall_scores:
                click.echo("\nOverall Scores:")
                for metric, score in summary.overall_scores.items():
                    click.echo(f"  {metric}: {score:.4f}")
            
            click.echo(f"\nResults saved to: {output_dir}")
            
        except Exception as e:
            click.echo(f"❌ Evaluation failed: {e}", err=True)
            raise click.ClickException(str(e))
    
    asyncio.run(_run_evaluation())


@cli.command()
@click.option('--baseline', '-b', required=True, help='Baseline version/timestamp')
@click.option('--current', '-c', required=True, help='Current version/timestamp')
@click.option('--benchmark', help='Specific benchmark to compare')
@click.option('--output', '-o', help='Output file for comparison results')
def compare(baseline: str, current: str, benchmark: Optional[str], output: Optional[str]):
    """Compare evaluation results between versions."""
    
    async def _compare_results():
        try:
            benchmark_manager = BenchmarkManager()
            
            # Load baseline and current results
            if benchmark:
                baseline_results = benchmark_manager.load_historical_results(f"{benchmark}_{baseline}")
                current_results = benchmark_manager.load_historical_results(f"{benchmark}_{current}")
                benchmarks_to_compare = [benchmark]
            else:
                # Compare all benchmarks
                benchmarks_to_compare = ["component_identification", "wiring_analysis", "troubleshooting"]
                baseline_results = {}
                current_results = {}
                
                for bench in benchmarks_to_compare:
                    base_res = benchmark_manager.load_historical_results(f"{bench}_{baseline}")
                    curr_res = benchmark_manager.load_historical_results(f"{bench}_{current}")
                    if base_res:
                        baseline_results[bench] = base_res[-1]  # Latest result
                    if curr_res:
                        current_results[bench] = curr_res[-1]  # Latest result
            
            # Perform comparison
            comparison_results = {}
            
            for bench in benchmarks_to_compare:
                if bench in baseline_results and bench in current_results:
                    base_metrics = baseline_results[bench].get("results", {})
                    curr_metrics = current_results[bench].get("results", {})
                    
                    bench_comparison = benchmark_manager.compare_results(base_metrics, curr_metrics)
                    comparison_results[bench] = bench_comparison
            
            # Display comparison
            click.echo(f"\n📊 COMPARISON: {baseline} vs {current}")
            click.echo("="*60)
            
            for bench, comparison in comparison_results.items():
                click.echo(f"\n🔍 {bench.upper()}:")
                
                for metric, data in comparison.items():
                    improvement = data["improvement"]
                    if improvement > 0.05:  # 5% improvement
                        symbol = "🟢"
                    elif improvement < -0.05:  # 5% degradation
                        symbol = "🔴"
                    else:
                        symbol = "🟡"
                    
                    click.echo(f"  {symbol} {metric}: {data['baseline']:.4f} → {data['current']:.4f} "
                              f"({improvement:+.2%})")
            
            # Save results if output specified
            if output:
                output_path = Path(output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w') as f:
                    json.dump({
                        "baseline": baseline,
                        "current": current,
                        "comparison": comparison_results,
                        "timestamp": str(datetime.now())
                    }, f, indent=2, default=str)
                
                click.echo(f"\nComparison saved to: {output_path}")
            
        except Exception as e:
            click.echo(f"❌ Comparison failed: {e}", err=True)
            raise click.ClickException(str(e))
    
    asyncio.run(_compare_results())


@cli.command()
@click.option('--benchmark', '-b', help='Specific benchmark to report on')
@click.option('--format', '-f', type=click.Choice(['json', 'md', 'csv']), 
              default='json', help='Output format')
@click.option('--limit', '-l', default=10, help='Number of recent results to include')
@click.option('--output', '-o', help='Output file (default: stdout)')
def report(benchmark: Optional[str], format: str, limit: int, output: Optional[str]):
    """Generate evaluation report."""
    
    async def _generate_report():
        try:
            benchmark_manager = BenchmarkManager()
            
            if benchmark:
                benchmarks_to_report = [benchmark]
            else:
                benchmarks_to_report = ["component_identification", "wiring_analysis", "troubleshooting"]
            
            report_data = {}
            
            for bench in benchmarks_to_report:
                historical_results = benchmark_manager.load_historical_results(bench)
                recent_results = sorted(
                    historical_results,
                    key=lambda x: x.get("timestamp", ""),
                    reverse=True
                )[:limit]
                
                report_data[bench] = {
                    "total_runs": len(historical_results),
                    "recent_results": recent_results,
                    "latest_metrics": recent_results[0]["results"] if recent_results else {}
                }
            
            # Format output
            if format == "json":
                report_content = json.dumps(report_data, indent=2, default=str)
            elif format == "md":
                report_content = _format_markdown_report(report_data)
            elif format == "csv":
                report_content = _format_csv_report(report_data)
            
            # Output results
            if output:
                output_path = Path(output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w') as f:
                    f.write(report_content)
                
                click.echo(f"Report saved to: {output_path}")
            else:
                click.echo(report_content)
            
        except Exception as e:
            click.echo(f"❌ Report generation failed: {e}", err=True)
            raise click.ClickException(str(e))
    
    asyncio.run(_generate_report())


def _format_markdown_report(report_data: dict) -> str:
    """Format report data as Markdown."""
    lines = ["# Automotive Electronics Evaluation Report\n"]
    
    for benchmark, data in report_data.items():
        lines.append(f"## {benchmark.replace('_', ' ').title()}\n")
        lines.append(f"- Total runs: {data['total_runs']}")
        lines.append(f"- Recent results: {len(data['recent_results'])}\n")
        
        if data['latest_metrics']:
            lines.append("### Latest Metrics\n")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            
            for metric, value in data['latest_metrics'].items():
                if isinstance(value, (int, float)):
                    lines.append(f"| {metric} | {value:.4f} |")
                else:
                    lines.append(f"| {metric} | {value} |")
            
            lines.append("")
    
    return "\n".join(lines)


def _format_csv_report(report_data: dict) -> str:
    """Format report data as CSV."""
    lines = ["benchmark,metric,value,timestamp"]
    
    for benchmark, data in report_data.items():
        for result in data['recent_results']:
            timestamp = result.get('timestamp', '')
            for metric, value in result.get('results', {}).items():
                lines.append(f"{benchmark},{metric},{value},{timestamp}")
    
    return "\n".join(lines)


@cli.command()
@click.option('--suite', '-s', help='Benchmark suite name')
@click.option('--description', '-d', help='Description of the benchmark')
@click.option('--output', '-o', help='Output file for benchmark definition')
def create(suite: str, description: str, output: str):
    """Create a new benchmark suite template."""
    
    template = {
        "name": suite or "custom_benchmark",
        "description": description or "Custom automotive benchmark suite",
        "version": "1.0",
        "queries": [
            {
                "query_text": "example query",
                "relevant_doc_ids": ["doc_1", "doc_2"],
                "vehicle_context": {
                    "make": "Toyota",
                    "model": "Camry", 
                    "year": 2020
                },
                "expected_components": ["K1", "F5"],
                "expected_nets": ["BATT+", "GND"],
                "difficulty": "medium"
            }
        ]
    }
    
    output_path = Path(output) if output else Path(f"{template['name']}_benchmark.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    click.echo(f"✅ Benchmark template created: {output_path}")
    click.echo("Edit the file to add your specific test cases.")


if __name__ == "__main__":
    cli()
"""
Benchmark harness for OCR engine evaluation with CER/WER metrics.
"""
import argparse
import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import statistics

from ..src.ocr.tesseract import TesseractProvider
from ..src.ocr.deepseek import DeepSeekProvider
from ..src.ocr.mistral import MistralProvider
from ..src.ocr.fusion import OcrFusionEngine
from ..src.preprocess.image import ImagePreprocessor
from ..src.preprocess.pdf import PdfProcessor
from ..src.core.schemas import PageImage, TextSpan, OcrEngine


class BenchmarkMetrics:
    """Calculate OCR performance metrics."""
    
    @staticmethod
    def calculate_cer(reference: str, hypothesis: str) -> float:
        """
        Calculate Character Error Rate (CER).
        
        CER = (S + D + I) / N
        where S=substitutions, D=deletions, I=insertions, N=total chars in reference
        
        Args:
            reference: Ground truth text
            hypothesis: OCR output text
            
        Returns:
            CER as float (0.0 = perfect, higher = worse)
        """
        if not reference:
            return 1.0 if hypothesis else 0.0
        
        # Use Levenshtein distance for character-level comparison
        distance = BenchmarkMetrics._levenshtein_distance(reference, hypothesis)
        return distance / len(reference)
    
    @staticmethod
    def calculate_wer(reference: str, hypothesis: str) -> float:
        """
        Calculate Word Error Rate (WER).
        
        WER = (S + D + I) / N  
        where S=substitutions, D=deletions, I=insertions, N=total words in reference
        
        Args:
            reference: Ground truth text  
            hypothesis: OCR output text
            
        Returns:
            WER as float (0.0 = perfect, higher = worse)
        """
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        
        if not ref_words:
            return 1.0 if hyp_words else 0.0
        
        distance = BenchmarkMetrics._levenshtein_distance(ref_words, hyp_words)
        return distance / len(ref_words)
    
    @staticmethod
    def _levenshtein_distance(seq1, seq2) -> int:
        """Calculate Levenshtein distance between two sequences."""
        if len(seq1) < len(seq2):
            return BenchmarkMetrics._levenshtein_distance(seq2, seq1)
        
        if len(seq2) == 0:
            return len(seq1)
        
        previous_row = list(range(len(seq2) + 1))
        for i, c1 in enumerate(seq1):
            current_row = [i + 1]
            for j, c2 in enumerate(seq2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]


class BenchmarkDataset:
    """Manages benchmark datasets with ground truth."""
    
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir
        self.samples = []
        self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset samples with ground truth."""
        if not self.dataset_dir.exists():
            print(f"Warning: Dataset directory not found: {self.dataset_dir}")
            return
        
        # Look for ground truth files
        for gt_file in self.dataset_dir.glob("*.json"):
            try:
                with open(gt_file, 'r') as f:
                    gt_data = json.load(f)
                
                # Find corresponding image/PDF file
                base_name = gt_file.stem
                image_file = None
                
                for ext in ['.pdf', '.png', '.jpg', '.jpeg']:
                    candidate = self.dataset_dir / f"{base_name}{ext}"
                    if candidate.exists():
                        image_file = candidate
                        break
                
                if image_file:
                    self.samples.append({
                        'name': base_name,
                        'file_path': str(image_file),
                        'ground_truth': gt_data
                    })
                    
            except Exception as e:
                print(f"Failed to load ground truth {gt_file}: {e}")
    
    def get_samples(self, category: Optional[str] = None) -> List[Dict]:
        """Get dataset samples, optionally filtered by category."""
        if category:
            return [s for s in self.samples if s.get('category') == category]
        return self.samples


class BenchmarkRunner:
    """Main benchmark runner for OCR evaluation."""
    
    def __init__(self):
        self.results = []
        self.preprocessor = ImagePreprocessor()
        self.pdf_processor = PdfProcessor()
    
    async def run_benchmark(
        self,
        engine: str = "all",
        dataset: str = "all",
        report_format: str = "json"
    ) -> Dict:
        """
        Run benchmark suite.
        
        Args:
            engine: OCR engine to test ("tesseract", "deepseek", "mistral", "all")
            dataset: Dataset category ("clean", "noisy", "all")
            report_format: Output format ("json", "md")
            
        Returns:
            Benchmark results dictionary
        """
        print(f"Starting benchmark - Engine: {engine}, Dataset: {dataset}")
        
        # Load datasets
        datasets = self._load_datasets(dataset)
        
        # Initialize OCR engines
        engines = self._initialize_engines(engine)
        
        # Run benchmarks
        all_results = []
        
        for dataset_name, samples in datasets.items():
            print(f"\nProcessing dataset: {dataset_name} ({len(samples)} samples)")
            
            for engine_name, ocr_engine in engines.items():
                print(f"  Testing engine: {engine_name}")
                
                engine_results = await self._test_engine_on_dataset(
                    ocr_engine, samples, engine_name, dataset_name
                )
                all_results.extend(engine_results)
        
        # Generate summary report
        summary = self._generate_summary(all_results)
        
        # Save results
        await self._save_results(all_results, summary, report_format)
        
        return summary
    
    def _load_datasets(self, dataset_filter: str) -> Dict[str, List]:
        """Load benchmark datasets."""
        base_dir = Path(__file__).parent / "datasets"
        datasets = {}
        
        categories = ["clean", "noisy", "handdrawn", "negative"] if dataset_filter == "all" else [dataset_filter]
        
        for category in categories:
            dataset_dir = base_dir / category
            if dataset_dir.exists():
                benchmark_dataset = BenchmarkDataset(dataset_dir)
                samples = benchmark_dataset.get_samples()
                if samples:
                    datasets[category] = samples
                    print(f"Loaded {len(samples)} samples from {category} dataset")
            else:
                print(f"Dataset directory not found: {dataset_dir}")
        
        return datasets
    
    def _initialize_engines(self, engine_filter: str) -> Dict[str, object]:
        """Initialize OCR engines based on filter."""
        engines = {}
        
        if engine_filter == "all" or engine_filter == "tesseract":
            try:
                engines["tesseract"] = TesseractProvider()
                print("Initialized Tesseract engine")
            except Exception as e:
                print(f"Failed to initialize Tesseract: {e}")
        
        if engine_filter == "all" or engine_filter == "deepseek":
            try:
                engines["deepseek"] = DeepSeekProvider()
                print("Initialized DeepSeek engine")
            except Exception as e:
                print(f"Failed to initialize DeepSeek: {e}")
        
        if engine_filter == "all" or engine_filter == "mistral":
            try:
                engines["mistral"] = MistralProvider()
                print("Initialized Mistral engine")
            except Exception as e:
                print(f"Failed to initialize Mistral: {e}")
        
        if engine_filter == "all" or engine_filter == "fusion":
            available_providers = []
            for name, provider in engines.items():
                available_providers.append(provider)
            
            if len(available_providers) > 1:
                engines["fusion"] = OcrFusionEngine(available_providers)
                print("Initialized Fusion engine")
        
        return engines
    
    async def _test_engine_on_dataset(
        self,
        ocr_engine,
        samples: List[Dict],
        engine_name: str,
        dataset_name: str
    ) -> List[Dict]:
        """Test a single OCR engine on a dataset."""
        results = []
        
        for i, sample in enumerate(samples):
            print(f"    Sample {i+1}/{len(samples)}: {sample['name']}")
            
            try:
                # Process document to images
                page_images = await self._process_document(sample['file_path'])
                
                if not page_images:
                    print(f"      Failed to process document")
                    continue
                
                # Test on first page (most samples are single page)
                page_image = page_images[0]
                
                # Run OCR
                start_time = time.time()
                
                if isinstance(ocr_engine, OcrFusionEngine):
                    text_spans = await ocr_engine.extract_text_fused(page_image)
                else:
                    text_spans = await ocr_engine.extract_text(page_image)
                
                processing_time = time.time() - start_time
                
                # Calculate metrics
                metrics = self._calculate_metrics(text_spans, sample['ground_truth'])
                
                # Record result
                result = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'sample_name': sample['name'],
                    'dataset': dataset_name,
                    'engine': engine_name,
                    'processing_time': processing_time,
                    'text_spans_count': len(text_spans),
                    'metrics': metrics
                }
                
                results.append(result)
                
                print(f"      CER: {metrics['cer']:.3f}, WER: {metrics['wer']:.3f}, Time: {processing_time:.2f}s")
                
            except Exception as e:
                print(f"      Error processing sample: {e}")
                continue
        
        return results
    
    async def _process_document(self, file_path: str) -> List[PageImage]:
        """Process document (PDF or image) to PageImage objects."""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.pdf':
            # Convert PDF to images
            page_images = await self.pdf_processor.convert_pdf_to_images(str(file_path))
        else:
            # Process single image
            page_image = await self.preprocessor.preprocess_image(str(file_path))
            page_images = [page_image]
        
        return page_images
    
    def _calculate_metrics(self, text_spans: List[TextSpan], ground_truth: Dict) -> Dict:
        """Calculate OCR metrics against ground truth."""
        # Extract OCR text
        ocr_text = " ".join([span.text for span in text_spans])
        
        # Extract ground truth text
        gt_text_regions = ground_truth.get('text_regions', [])
        gt_text = " ".join([region['text'] for region in gt_text_regions])
        
        # Calculate basic metrics
        cer = BenchmarkMetrics.calculate_cer(gt_text, ocr_text)
        wer = BenchmarkMetrics.calculate_wer(gt_text, ocr_text)
        
        # Calculate additional metrics
        precision = self._calculate_precision(text_spans, gt_text_regions)
        recall = self._calculate_recall(text_spans, gt_text_regions)
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'cer': cer,
            'wer': wer,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'detected_count': len(text_spans),
            'expected_count': len(gt_text_regions)
        }
    
    def _calculate_precision(self, text_spans: List[TextSpan], gt_regions: List[Dict]) -> float:
        """Calculate precision (detected text that matches ground truth)."""
        if not text_spans:
            return 0.0
        
        matched = 0
        for span in text_spans:
            # Check if this span matches any ground truth region
            for gt_region in gt_regions:
                if self._text_matches(span.text, gt_region['text']):
                    matched += 1
                    break
        
        return matched / len(text_spans)
    
    def _calculate_recall(self, text_spans: List[TextSpan], gt_regions: List[Dict]) -> float:
        """Calculate recall (ground truth text that was detected)."""
        if not gt_regions:
            return 1.0 if not text_spans else 0.0
        
        matched = 0
        for gt_region in gt_regions:
            # Check if this ground truth region was detected
            for span in text_spans:
                if self._text_matches(span.text, gt_region['text']):
                    matched += 1
                    break
        
        return matched / len(gt_regions)
    
    def _text_matches(self, text1: str, text2: str) -> bool:
        """Check if two text strings match (with normalization)."""
        # Normalize text for comparison
        norm1 = text1.lower().strip().replace(' ', '')
        norm2 = text2.lower().strip().replace(' ', '')
        
        # Allow for small differences (edit distance)
        if not norm1 or not norm2:
            return norm1 == norm2
        
        distance = BenchmarkMetrics._levenshtein_distance(norm1, norm2)
        max_len = max(len(norm1), len(norm2))
        
        # Allow up to 20% character differences
        return (distance / max_len) <= 0.2
    
    def _generate_summary(self, all_results: List[Dict]) -> Dict:
        """Generate summary statistics from all results."""
        if not all_results:
            return {}
        
        summary = {
            'total_samples': len(all_results),
            'engines': {},
            'datasets': {},
            'overall': {}
        }
        
        # Group by engine
        by_engine = {}
        for result in all_results:
            engine = result['engine']
            if engine not in by_engine:
                by_engine[engine] = []
            by_engine[engine].append(result)
        
        # Calculate engine stats
        for engine, results in by_engine.items():
            metrics = [r['metrics'] for r in results]
            
            summary['engines'][engine] = {
                'sample_count': len(results),
                'avg_cer': statistics.mean([m['cer'] for m in metrics]),
                'avg_wer': statistics.mean([m['wer'] for m in metrics]),
                'avg_f1': statistics.mean([m['f1'] for m in metrics]),
                'avg_processing_time': statistics.mean([r['processing_time'] for r in results])
            }
        
        # Group by dataset
        by_dataset = {}
        for result in all_results:
            dataset = result['dataset']
            if dataset not in by_dataset:
                by_dataset[dataset] = []
            by_dataset[dataset].append(result)
        
        # Calculate dataset stats
        for dataset, results in by_dataset.items():
            metrics = [r['metrics'] for r in results]
            
            summary['datasets'][dataset] = {
                'sample_count': len(results),
                'avg_cer': statistics.mean([m['cer'] for m in metrics]),
                'avg_wer': statistics.mean([m['wer'] for m in metrics]),
                'avg_f1': statistics.mean([m['f1'] for m in metrics])
            }
        
        # Overall stats
        all_metrics = [r['metrics'] for r in all_results]
        summary['overall'] = {
            'avg_cer': statistics.mean([m['cer'] for m in all_metrics]),
            'avg_wer': statistics.mean([m['wer'] for m in all_metrics]),
            'avg_f1': statistics.mean([m['f1'] for m in all_metrics]),
            'total_processing_time': sum([r['processing_time'] for r in all_results])
        }
        
        return summary
    
    async def _save_results(self, results: List[Dict], summary: Dict, format: str):
        """Save benchmark results to files."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        
        if format == "json":
            # Save detailed results
            results_file = results_dir / f"benchmark_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump({
                    'timestamp': timestamp,
                    'summary': summary,
                    'detailed_results': results
                }, f, indent=2)
            
            print(f"\nResults saved to: {results_file}")
        
        elif format == "md":
            # Save markdown report
            report_file = results_dir / f"benchmark_report_{timestamp}.md"
            with open(report_file, 'w') as f:
                f.write(self._generate_markdown_report(summary, results))
            
            print(f"\nReport saved to: {report_file}")
    
    def _generate_markdown_report(self, summary: Dict, results: List[Dict]) -> str:
        """Generate markdown benchmark report."""
        report = f"""# OCR Benchmark Report

**Generated:** {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}

## Summary

Total samples processed: {summary.get('total_samples', 0)}

### Overall Performance

| Metric | Score |
|--------|-------|
| Average CER | {summary['overall']['avg_cer']:.3f} |
| Average WER | {summary['overall']['avg_wer']:.3f} |
| Average F1 | {summary['overall']['avg_f1']:.3f} |
| Total Processing Time | {summary['overall']['total_processing_time']:.2f}s |

### Performance by Engine

| Engine | Samples | CER | WER | F1 | Avg Time (s) |
|--------|---------|-----|-----|----|-----------| 
"""
        
        for engine, stats in summary.get('engines', {}).items():
            report += f"| {engine} | {stats['sample_count']} | {stats['avg_cer']:.3f} | {stats['avg_wer']:.3f} | {stats['avg_f1']:.3f} | {stats['avg_processing_time']:.2f} |\n"
        
        report += """
### Performance by Dataset

| Dataset | Samples | CER | WER | F1 |
|---------|---------|-----|-----|----|
"""
        
        for dataset, stats in summary.get('datasets', {}).items():
            report += f"| {dataset} | {stats['sample_count']} | {stats['avg_cer']:.3f} | {stats['avg_wer']:.3f} | {stats['avg_f1']:.3f} |\n"
        
        report += "\n## Detailed Results\n\n"
        
        for result in results[:10]:  # Show first 10 detailed results
            metrics = result['metrics']
            report += f"""### {result['sample_name']} ({result['dataset']} - {result['engine']})

- **CER:** {metrics['cer']:.3f}
- **WER:** {metrics['wer']:.3f}  
- **F1:** {metrics['f1']:.3f}
- **Processing Time:** {result['processing_time']:.2f}s
- **Detected/Expected:** {metrics['detected_count']}/{metrics['expected_count']}

"""
        
        return report


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Run OCR benchmark suite")
    parser.add_argument("--engine", default="all", 
                       choices=["tesseract", "deepseek", "mistral", "fusion", "all"],
                       help="OCR engine to test")
    parser.add_argument("--dataset", default="all",
                       choices=["clean", "noisy", "handdrawn", "negative", "all"], 
                       help="Dataset category to test")
    parser.add_argument("--report", default="json",
                       choices=["json", "md"],
                       help="Report format")
    
    args = parser.parse_args()
    
    runner = BenchmarkRunner()
    summary = await runner.run_benchmark(
        engine=args.engine,
        dataset=args.dataset, 
        report_format=args.report
    )
    
    print("\n" + "="*50)
    print("BENCHMARK COMPLETE")
    print("="*50)
    print(f"Overall CER: {summary['overall']['avg_cer']:.3f}")
    print(f"Overall WER: {summary['overall']['avg_wer']:.3f}")
    print(f"Overall F1:  {summary['overall']['avg_f1']:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
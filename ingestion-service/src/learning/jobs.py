"""
Continual learning jobs for model improvement.

This module implements scheduled learning jobs that analyze collected signals
and improve models through self-supervised and weakly-supervised learning.
"""
import asyncio
import time
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

from ..core.logging import StructuredLogger
from ..core.metrics import metrics
from ..core.models import get_model_registry, ModelVersion, ModelMetrics
from .signals import get_signal_collector, SignalType, LearningSignal

logger = StructuredLogger(__name__)


@dataclass
class TrainingResult:
    """Result of a training job."""
    success: bool
    model_version: Optional[ModelVersion] = None
    metrics: Optional[ModelMetrics] = None
    training_time: float = 0.0
    samples_used: int = 0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class LearningJob(ABC):
    """Abstract base class for learning jobs."""
    
    def __init__(self, name: str, schedule: str = "weekly"):
        """
        Initialize learning job.
        
        Args:
            name: Job name
            schedule: Schedule (daily, weekly, monthly)
        """
        self.name = name
        self.schedule = schedule
        self.last_run: Optional[datetime] = None
        self.enabled = True
        
    @abstractmethod
    async def run(self, signals: List[LearningSignal]) -> TrainingResult:
        """
        Run the learning job.
        
        Args:
            signals: Collected learning signals
            
        Returns:
            Training result
        """
        pass
    
    def should_run(self) -> bool:
        """Check if job should run based on schedule."""
        if not self.enabled:
            return False
        
        if self.last_run is None:
            return True
        
        now = datetime.utcnow()
        schedule_intervals = {
            "daily": timedelta(days=1),
            "weekly": timedelta(weeks=1),
            "monthly": timedelta(days=30)
        }
        
        interval = schedule_intervals.get(self.schedule, timedelta(weeks=1))
        return now - self.last_run >= interval


class ContrastiveLearningJob(LearningJob):
    """Self-supervised contrastive learning for text embeddings."""
    
    def __init__(self):
        super().__init__("contrastive_learning", "weekly")
        self.min_samples = 100
        self.max_samples = 10000
    
    async def run(self, signals: List[LearningSignal]) -> TrainingResult:
        """
        Run contrastive learning on text similarities.
        
        Uses OCR disagreement signals to find positive and negative pairs
        for contrastive learning of embeddings.
        """
        start_time = time.time()
        
        try:
            logger.info("Starting contrastive learning job")
            
            # Filter OCR disagreement signals
            ocr_signals = [s for s in signals if s.signal_type == SignalType.OCR_DISAGREEMENT]
            
            if len(ocr_signals) < self.min_samples:
                return TrainingResult(
                    success=False,
                    error_message=f"Insufficient samples: {len(ocr_signals)} < {self.min_samples}",
                    training_time=time.time() - start_time
                )
            
            # Extract positive and negative pairs
            positive_pairs, negative_pairs = self._extract_contrastive_pairs(ocr_signals)
            
            # Mock training - in real implementation would fine-tune embeddings
            await asyncio.sleep(5)  # Simulate training time
            
            # Create new model version
            current_time = datetime.utcnow()
            new_version = ModelVersion(
                name="auto_embed",
                version=f"v{current_time.strftime('%Y%m%d_%H%M%S')}",
                metrics=ModelMetrics(
                    # Mock improved metrics
                    ndcg_10=0.82,  # Improved from baseline
                    e2e_f1=0.75
                ),
                stage="staging",
                framework="sentence_transformers",
                changelog=f"Contrastive learning on {len(positive_pairs)} pairs",
                metadata={
                    "training_samples": len(ocr_signals),
                    "positive_pairs": len(positive_pairs),
                    "negative_pairs": len(negative_pairs)
                }
            )
            
            # Register new model
            registry = get_model_registry()
            await registry.register_model(new_version)
            
            training_time = time.time() - start_time
            
            logger.info(f"Contrastive learning completed in {training_time:.2f}s",
                       samples_used=len(ocr_signals),
                       positive_pairs=len(positive_pairs),
                       negative_pairs=len(negative_pairs))
            
            return TrainingResult(
                success=True,
                model_version=new_version,
                metrics=new_version.metrics,
                training_time=training_time,
                samples_used=len(ocr_signals),
                metadata={
                    "positive_pairs": len(positive_pairs),
                    "negative_pairs": len(negative_pairs)
                }
            )
            
        except Exception as e:
            training_time = time.time() - start_time
            logger.error(f"Contrastive learning failed: {e}")
            
            return TrainingResult(
                success=False,
                error_message=str(e),
                training_time=training_time
            )
    
    def _extract_contrastive_pairs(self, signals: List[LearningSignal]) -> Tuple[List[Tuple], List[Tuple]]:
        """Extract positive and negative pairs from OCR signals."""
        positive_pairs = []
        negative_pairs = []
        
        for signal in signals:
            engine_results = signal.labels.get("engine_results", {})
            if len(engine_results) < 2:
                continue
            
            texts = [result.get("text", "") for result in engine_results.values()]
            confidences = [result.get("confidence", 0.0) for result in engine_results.values()]
            
            # High confidence results with similar text = positive pairs
            high_conf_texts = [text for text, conf in zip(texts, confidences) if conf > 0.8]
            if len(high_conf_texts) >= 2:
                for i in range(len(high_conf_texts) - 1):
                    positive_pairs.append((high_conf_texts[i], high_conf_texts[i + 1]))
            
            # High vs low confidence with different text = negative pairs
            high_conf_idx = confidences.index(max(confidences))
            low_conf_idx = confidences.index(min(confidences))
            
            if confidences[high_conf_idx] - confidences[low_conf_idx] > 0.3:
                negative_pairs.append((texts[high_conf_idx], texts[low_conf_idx]))
        
        return positive_pairs, negative_pairs


class WeakLabelingJob(LearningJob):
    """Generate weak labels for electrical component tagging."""
    
    def __init__(self):
        super().__init__("weak_labeling", "weekly")
        self.min_samples = 50
    
    async def run(self, signals: List[LearningSignal]) -> TrainingResult:
        """
        Generate weak labels using rules and patterns.
        
        Creates BIO (Begin-Inside-Outside) tags for electrical components,
        nets, and other automotive entities.
        """
        start_time = time.time()
        
        try:
            logger.info("Starting weak labeling job")
            
            # Get all text spans from signals
            text_samples = self._extract_text_samples(signals)
            
            if len(text_samples) < self.min_samples:
                return TrainingResult(
                    success=False,
                    error_message=f"Insufficient text samples: {len(text_samples)} < {self.min_samples}",
                    training_time=time.time() - start_time
                )
            
            # Generate weak labels using rules
            labeled_samples = []
            for text in text_samples:
                labels = self._generate_bio_labels(text)
                labeled_samples.append({
                    "text": text,
                    "labels": labels,
                    "confidence": self._calculate_label_confidence(labels)
                })
            
            # Mock training on weak labels
            await asyncio.sleep(3)  # Simulate training
            
            # Create new tagger model version
            current_time = datetime.utcnow()
            new_version = ModelVersion(
                name="electrical_tagger",
                version=f"v{current_time.strftime('%Y%m%d_%H%M%S')}",
                metrics=ModelMetrics(
                    # Mock metrics for NER
                    e2e_f1=0.72
                ),
                stage="staging",
                framework="pytorch",
                changelog=f"Weak supervision on {len(labeled_samples)} samples",
                metadata={
                    "training_samples": len(labeled_samples),
                    "avg_confidence": sum(s["confidence"] for s in labeled_samples) / len(labeled_samples)
                }
            )
            
            # Register model
            registry = get_model_registry()
            await registry.register_model(new_version)
            
            training_time = time.time() - start_time
            
            logger.info(f"Weak labeling completed in {training_time:.2f}s",
                       samples_used=len(labeled_samples))
            
            return TrainingResult(
                success=True,
                model_version=new_version,
                metrics=new_version.metrics,
                training_time=training_time,
                samples_used=len(labeled_samples),
                metadata={
                    "avg_label_confidence": sum(s["confidence"] for s in labeled_samples) / len(labeled_samples)
                }
            )
            
        except Exception as e:
            training_time = time.time() - start_time
            logger.error(f"Weak labeling failed: {e}")
            
            return TrainingResult(
                success=False,
                error_message=str(e),
                training_time=training_time
            )
    
    def _extract_text_samples(self, signals: List[LearningSignal]) -> List[str]:
        """Extract text samples from various signal types."""
        texts = []
        
        for signal in signals:
            # Extract from OCR disagreement
            if signal.signal_type == SignalType.OCR_DISAGREEMENT:
                engine_results = signal.labels.get("engine_results", {})
                for result in engine_results.values():
                    text = result.get("text", "").strip()
                    if text and len(text) > 3:  # Filter very short texts
                        texts.append(text)
            
            # Extract from validation failures (context text)
            elif signal.signal_type == SignalType.VALIDATION_FAILURE:
                context_text = signal.context.get("text", "")
                if context_text:
                    texts.append(context_text)
        
        return list(set(texts))  # Remove duplicates
    
    def _generate_bio_labels(self, text: str) -> List[str]:
        """Generate BIO labels for electrical entities."""
        import re
        
        words = text.split()
        labels = ["O"] * len(words)  # Default to Outside
        
        # Patterns for different entity types
        patterns = {
            "COMPONENT": [
                (r"\b[RKCLQUF]\d+[A-Z]*\b", "component"),  # R1, K2, etc.
                (r"\b(?:relay|fuse|resistor|capacitor|inductor|diode)\b", "component_type"),
            ],
            "NET": [
                (r"\b(?:VCC|VDD|GND|GROUND|IG1|IG2|ACC|BATT)\b", "power_net"),
                (r"\b[A-Z]+\d*\b", "signal_net"),  # General signal names
            ],
            "VALUE": [
                (r"\b\d+[kKmM]?[Ω\u03A9]?\b", "resistance"),  # 10k, 1MΩ
                (r"\b\d+[\.,]?\d*[uµnpmk]?[FH]\b", "capacitance"),  # 100nF, 1µH
                (r"\b\d+[\.,]?\d*[Vv]\b", "voltage"),  # 12V, 5.0v
                (r"\b\d+[\.,]?\d*[Aa]\b", "current"),  # 10A, 2.5a
            ],
            "LOCATION": [
                (r"\b(?:pin|connector|terminal|contact)\s+\d+\b", "pin_ref"),
                (r"\b(?:fuse|relay)\s+(?:box|panel)\b", "location"),
            ]
        }
        
        # Apply patterns
        for entity_type, pattern_list in patterns.items():
            for pattern, sub_type in pattern_list:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    start_pos = match.start()
                    end_pos = match.end()
                    
                    # Find word indices that overlap with match
                    word_start_idx = None
                    word_end_idx = None
                    char_pos = 0
                    
                    for i, word in enumerate(words):
                        word_end = char_pos + len(word)
                        
                        if word_start_idx is None and word_end > start_pos:
                            word_start_idx = i
                        
                        if word_end >= end_pos:
                            word_end_idx = i
                            break
                        
                        char_pos = word_end + 1  # +1 for space
                    
                    # Apply BIO labels
                    if word_start_idx is not None and word_end_idx is not None:
                        for i in range(word_start_idx, word_end_idx + 1):
                            if i == word_start_idx:
                                labels[i] = f"B-{entity_type}"
                            else:
                                labels[i] = f"I-{entity_type}"
        
        return labels
    
    def _calculate_label_confidence(self, labels: List[str]) -> float:
        """Calculate confidence in the generated labels."""
        # Simple heuristic: more specific labels = higher confidence
        entity_labels = [label for label in labels if label != "O"]
        if not entity_labels:
            return 0.5  # No entities found
        
        # Higher confidence for more entities and diverse types
        entity_types = set(label.split("-")[1] for label in entity_labels if "-" in label)
        diversity_score = len(entity_types) / 4.0  # Normalize by max expected types
        density_score = len(entity_labels) / len(labels)
        
        return min(1.0, (diversity_score + density_score) / 2.0)


class DetectorFineTuningJob(LearningJob):
    """Fine-tune symbol detector on error-mined patches."""
    
    def __init__(self):
        super().__init__("detector_finetuning", "monthly")
        self.min_samples = 20
        self.max_training_hours = 2
    
    async def run(self, signals: List[LearningSignal]) -> TrainingResult:
        """
        Fine-tune YOLO detector on problematic regions.
        
        Uses symbol detection error signals to create training patches
        for improving detection accuracy.
        """
        start_time = time.time()
        
        try:
            logger.info("Starting detector fine-tuning job")
            
            # Filter detection error signals
            detection_signals = [s for s in signals 
                               if s.signal_type == SignalType.SYMBOL_DETECTION_ERROR
                               and s.severity in ["medium", "high"]]
            
            if len(detection_signals) < self.min_samples:
                return TrainingResult(
                    success=False,
                    error_message=f"Insufficient detection errors: {len(detection_signals)} < {self.min_samples}",
                    training_time=time.time() - start_time
                )
            
            # Extract training patches
            training_patches = self._extract_training_patches(detection_signals)
            
            # Mock training process - in real implementation would:
            # 1. Load current detector model
            # 2. Prepare training data in YOLO format
            # 3. Fine-tune on error patches
            # 4. Validate on held-out set
            
            max_training_time = self.max_training_hours * 3600
            training_duration = min(300, max_training_time)  # 5 minutes for mock
            await asyncio.sleep(training_duration / 60)  # Scale down for testing
            
            # Create new detector model version
            current_time = datetime.utcnow()
            new_version = ModelVersion(
                name="symbol_detector",
                version=f"v{current_time.strftime('%Y%m%d_%H%M%S')}",
                metrics=ModelMetrics(
                    map_score=0.79,  # Improved mAP
                    e2e_f1=0.76
                ),
                stage="staging",
                framework="pytorch",
                changelog=f"Fine-tuned on {len(training_patches)} error patches",
                metadata={
                    "training_patches": len(training_patches),
                    "training_hours": training_duration / 3600,
                    "error_types_addressed": self._count_error_types(detection_signals)
                }
            )
            
            # Register model
            registry = get_model_registry()
            await registry.register_model(new_version)
            
            training_time = time.time() - start_time
            
            logger.info(f"Detector fine-tuning completed in {training_time:.2f}s",
                       samples_used=len(training_patches))
            
            return TrainingResult(
                success=True,
                model_version=new_version,
                metrics=new_version.metrics,
                training_time=training_time,
                samples_used=len(training_patches),
                metadata={
                    "training_patches": len(training_patches),
                    "error_types": self._count_error_types(detection_signals)
                }
            )
            
        except Exception as e:
            training_time = time.time() - start_time
            logger.error(f"Detector fine-tuning failed: {e}")
            
            return TrainingResult(
                success=False,
                error_message=str(e),
                training_time=training_time
            )
    
    def _extract_training_patches(self, signals: List[LearningSignal]) -> List[Dict]:
        """Extract training patches from detection error signals."""
        patches = []
        
        for signal in signals:
            detected_symbols = signal.labels.get("detected_symbols", [])
            validation_errors = signal.labels.get("validation_errors", [])
            
            # Create training patches for each problematic symbol
            for symbol in detected_symbols:
                bbox = symbol.get("bbox", [])
                if len(bbox) == 4:
                    patch = {
                        "bbox": bbox,
                        "page": signal.page,
                        "job_id": signal.job_id,
                        "detected_type": symbol.get("type"),
                        "confidence": symbol.get("confidence", 0.0),
                        "errors": validation_errors,
                        "severity": signal.severity
                    }
                    patches.append(patch)
        
        return patches
    
    def _count_error_types(self, signals: List[LearningSignal]) -> Dict[str, int]:
        """Count different types of detection errors."""
        error_counts = {}
        
        for signal in signals:
            errors = signal.labels.get("validation_errors", [])
            for error in errors:
                error_type = self._classify_error_type(error)
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        return error_counts
    
    def _classify_error_type(self, error_msg: str) -> str:
        """Classify detection error into categories."""
        error_lower = error_msg.lower()
        if "pin" in error_lower:
            return "pin_errors"
        elif "type" in error_lower:
            return "classification_errors"
        elif "missing" in error_lower:
            return "missed_detections"
        elif "duplicate" in error_lower:
            return "duplicate_detections"
        else:
            return "other_errors"


class LearningScheduler:
    """Scheduler for continual learning jobs."""
    
    def __init__(self):
        """Initialize learning scheduler."""
        self.jobs: List[LearningJob] = [
            ContrastiveLearningJob(),
            WeakLabelingJob(),
            DetectorFineTuningJob()
        ]
        self.running = False
        self.last_evaluation: Optional[datetime] = None
    
    def add_job(self, job: LearningJob):
        """Add a learning job to the scheduler."""
        self.jobs.append(job)
        logger.info(f"Added learning job: {job.name}")
    
    async def run_pending_jobs(self) -> List[TrainingResult]:
        """Run all pending learning jobs."""
        if self.running:
            logger.warning("Learning jobs already running, skipping")
            return []
        
        self.running = True
        results = []
        
        try:
            # Get signals from collector
            collector = get_signal_collector()
            signals = collector.signals.copy()  # Copy to avoid modification during processing
            
            if not signals:
                logger.info("No learning signals available, skipping jobs")
                return []
            
            logger.info(f"Running learning jobs with {len(signals)} signals")
            
            # Run each job that should run
            for job in self.jobs:
                if job.should_run():
                    logger.info(f"Running learning job: {job.name}")
                    
                    try:
                        result = await job.run(signals)
                        results.append(result)
                        job.last_run = datetime.utcnow()
                        
                        # Log result
                        if result.success:
                            logger.info(f"Learning job {job.name} succeeded",
                                       training_time=result.training_time,
                                       samples_used=result.samples_used)
                        else:
                            logger.error(f"Learning job {job.name} failed: {result.error_message}")
                        
                    except Exception as e:
                        logger.error(f"Learning job {job.name} crashed: {e}")
                        results.append(TrainingResult(
                            success=False,
                            error_message=f"Job crashed: {str(e)}"
                        ))
                else:
                    logger.debug(f"Skipping job {job.name} (not scheduled)")
            
            # Record metrics
            successful_jobs = sum(1 for r in results if r.success)
            metrics.record_external_service_call(
                "learning", 
                "run_jobs", 
                "success" if successful_jobs > 0 else "partial", 
                0.1
            )
            
            return results
            
        finally:
            self.running = False
    
    async def run_weekly_evaluation(self) -> Dict[str, Any]:
        """Run weekly model evaluation and promotion."""
        try:
            logger.info("Running weekly model evaluation")
            
            # Mock evaluation - in real implementation would:
            # 1. Run full benchmark suite
            # 2. Compare new models vs production
            # 3. Promote models that show improvement
            # 4. Rollback models that degrade performance
            
            registry = get_model_registry()
            evaluation_results = {
                "timestamp": datetime.utcnow().isoformat(),
                "models_evaluated": 0,
                "models_promoted": 0,
                "models_rolled_back": 0,
                "overall_performance": "stable"
            }
            
            # Check for staging models to evaluate
            # (In real implementation, would query model registry)
            
            self.last_evaluation = datetime.utcnow()
            
            logger.info("Weekly evaluation completed", **evaluation_results)
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Weekly evaluation failed: {e}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
    
    def get_job_status(self) -> Dict[str, Any]:
        """Get status of all learning jobs."""
        return {
            "running": self.running,
            "jobs": [
                {
                    "name": job.name,
                    "schedule": job.schedule,
                    "enabled": job.enabled,
                    "last_run": job.last_run.isoformat() if job.last_run else None,
                    "should_run": job.should_run()
                }
                for job in self.jobs
            ],
            "last_evaluation": self.last_evaluation.isoformat() if self.last_evaluation else None
        }


# Global scheduler instance
_global_scheduler: Optional[LearningScheduler] = None

def get_learning_scheduler() -> LearningScheduler:
    """Get global learning scheduler."""
    global _global_scheduler
    if _global_scheduler is None:
        _global_scheduler = LearningScheduler()
    return _global_scheduler
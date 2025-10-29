"""
Model registry and versioning system for continual learning.

This module manages different model versions, hot-swapping, and model 
lifecycle management for the automotive ingestion service.
"""
import os
import asyncio
import hashlib
import tempfile
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from abc import ABC, abstractmethod

from ..core.logging import StructuredLogger
from ..core.metrics import metrics

logger = StructuredLogger(__name__)


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    map_score: Optional[float] = None  # mAP for object detection
    net_f1: Optional[float] = None     # Net connectivity F1
    cer: Optional[float] = None        # Character error rate
    wer: Optional[float] = None        # Word error rate  
    ndcg_10: Optional[float] = None    # Search ranking quality
    e2e_f1: Optional[float] = None     # End-to-end F1 score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class ModelVersion:
    """Model version metadata."""
    name: str
    version: str
    s3_uri: Optional[str] = None
    local_path: Optional[str] = None
    metrics: ModelMetrics = field(default_factory=ModelMetrics)
    stage: str = "development"  # development, staging, production
    changelog: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    file_hash: Optional[str] = None
    size_bytes: Optional[int] = None
    framework: str = "unknown"
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "name": self.name,
            "version": self.version,
            "s3_uri": self.s3_uri,
            "local_path": self.local_path,
            "metrics": self.metrics.to_dict(),
            "stage": self.stage,
            "changelog": self.changelog,
            "created_at": self.created_at.isoformat(),
            "file_hash": self.file_hash,
            "size_bytes": self.size_bytes,
            "framework": self.framework,
            "dependencies": self.dependencies,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        """Create from dictionary."""
        # Parse metrics
        metrics_data = data.get("metrics", {})
        model_metrics = ModelMetrics(**metrics_data)
        
        # Parse datetime
        created_at = datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow()
        
        return cls(
            name=data["name"],
            version=data["version"],
            s3_uri=data.get("s3_uri"),
            local_path=data.get("local_path"),
            metrics=model_metrics,
            stage=data.get("stage", "development"),
            changelog=data.get("changelog"),
            created_at=created_at,
            file_hash=data.get("file_hash"),
            size_bytes=data.get("size_bytes"),
            framework=data.get("framework", "unknown"),
            dependencies=data.get("dependencies", []),
            metadata=data.get("metadata", {})
        )


class ModelLoader(ABC):
    """Abstract model loader interface."""
    
    @abstractmethod
    async def load_model(self, model_path: str) -> Any:
        """Load model from path."""
        pass
    
    @abstractmethod
    async def health_check(self, model: Any) -> bool:
        """Check if model is healthy."""
        pass
    
    @abstractmethod
    def get_model_info(self, model: Any) -> Dict[str, Any]:
        """Get model information."""
        pass


class TorchModelLoader(ModelLoader):
    """PyTorch model loader."""
    
    def __init__(self):
        self.torch_available = False
        try:
            import torch
            self.torch = torch
            self.torch_available = True
        except ImportError:
            logger.warning("PyTorch not available")
    
    async def load_model(self, model_path: str) -> Any:
        """Load PyTorch model."""
        if not self.torch_available:
            raise RuntimeError("PyTorch not available")
        
        logger.info(f"Loading PyTorch model from {model_path}")
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        model = await loop.run_in_executor(
            None,
            lambda: self.torch.load(model_path, map_location='cpu')
        )
        
        # Set to eval mode
        if hasattr(model, 'eval'):
            model.eval()
        
        logger.info("PyTorch model loaded successfully")
        return model
    
    async def health_check(self, model: Any) -> bool:
        """Check PyTorch model health."""
        try:
            # Basic check - ensure model is callable
            if hasattr(model, '__call__') or hasattr(model, 'forward'):
                return True
            return False
        except Exception as e:
            logger.warning(f"PyTorch model health check failed: {e}")
            return False
    
    def get_model_info(self, model: Any) -> Dict[str, Any]:
        """Get PyTorch model information."""
        info = {"framework": "pytorch"}
        
        try:
            if hasattr(model, 'parameters'):
                param_count = sum(p.numel() for p in model.parameters())
                info["parameter_count"] = param_count
            
            if hasattr(model, '__class__'):
                info["model_class"] = model.__class__.__name__
                
        except Exception as e:
            logger.warning(f"Failed to get PyTorch model info: {e}")
        
        return info


class ONNXModelLoader(ModelLoader):
    """ONNX model loader."""
    
    def __init__(self):
        self.onnx_available = False
        try:
            import onnxruntime as ort
            self.ort = ort
            self.onnx_available = True
        except ImportError:
            logger.warning("ONNX Runtime not available")
    
    async def load_model(self, model_path: str) -> Any:
        """Load ONNX model."""
        if not self.onnx_available:
            raise RuntimeError("ONNX Runtime not available")
        
        logger.info(f"Loading ONNX model from {model_path}")
        
        # Run in thread pool
        loop = asyncio.get_event_loop()
        session = await loop.run_in_executor(
            None,
            lambda: self.ort.InferenceSession(model_path)
        )
        
        logger.info("ONNX model loaded successfully")
        return session
    
    async def health_check(self, model: Any) -> bool:
        """Check ONNX model health."""
        try:
            # Check if we can get input/output info
            if hasattr(model, 'get_inputs') and hasattr(model, 'get_outputs'):
                inputs = model.get_inputs()
                outputs = model.get_outputs()
                return len(inputs) > 0 and len(outputs) > 0
            return False
        except Exception as e:
            logger.warning(f"ONNX model health check failed: {e}")
            return False
    
    def get_model_info(self, model: Any) -> Dict[str, Any]:
        """Get ONNX model information."""
        info = {"framework": "onnx"}
        
        try:
            if hasattr(model, 'get_inputs'):
                inputs = model.get_inputs()
                info["input_shapes"] = [inp.shape for inp in inputs]
                info["input_types"] = [inp.type for inp in inputs]
            
            if hasattr(model, 'get_outputs'):
                outputs = model.get_outputs()
                info["output_shapes"] = [out.shape for out in outputs]
                info["output_types"] = [out.type for out in outputs]
                
        except Exception as e:
            logger.warning(f"Failed to get ONNX model info: {e}")
        
        return info


class MockModelLoader(ModelLoader):
    """Mock model loader for testing."""
    
    async def load_model(self, model_path: str) -> Any:
        """Load mock model."""
        logger.info(f"Loading mock model from {model_path}")
        
        # Return a simple mock object
        class MockModel:
            def __init__(self, path):
                self.path = path
                self.loaded_at = datetime.utcnow()
            
            def predict(self, x):
                return f"mock_prediction_for_{x}"
            
            def __call__(self, x):
                return self.predict(x)
        
        return MockModel(model_path)
    
    async def health_check(self, model: Any) -> bool:
        """Check mock model health."""
        return hasattr(model, 'predict')
    
    def get_model_info(self, model: Any) -> Dict[str, Any]:
        """Get mock model information."""
        return {
            "framework": "mock",
            "loaded_at": getattr(model, 'loaded_at', datetime.utcnow()).isoformat()
        }


class ModelRegistry:
    """Central model registry with versioning and hot-swapping."""
    
    def __init__(self, 
                 storage_backend: Optional[Any] = None,
                 cache_dir: Optional[str] = None):
        """
        Initialize model registry.
        
        Args:
            storage_backend: Storage backend for model metadata (Supabase)
            cache_dir: Local cache directory for models
        """
        self.storage = storage_backend
        self.cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "wessley_models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Model loaders by framework
        self.loaders = {
            "pytorch": TorchModelLoader(),
            "onnx": ONNXModelLoader(), 
            "mock": MockModelLoader()
        }
        
        # Active models
        self.active_models: Dict[str, Any] = {}
        self.model_versions: Dict[str, ModelVersion] = {}
        
        # Model update callbacks
        self.update_callbacks: Dict[str, List[Callable]] = {}
        
        logger.info(f"ModelRegistry initialized with cache dir: {self.cache_dir}")
    
    async def register_model(self, model_version: ModelVersion) -> bool:
        """
        Register a new model version.
        
        Args:
            model_version: Model version to register
            
        Returns:
            True if successful
        """
        try:
            # Store metadata
            if self.storage:
                await self._store_model_metadata(model_version)
            
            # Cache locally if we have model data
            if model_version.s3_uri:
                local_path = await self._download_model(model_version)
                model_version.local_path = str(local_path)
            
            # Calculate file hash if local file exists
            if model_version.local_path and os.path.exists(model_version.local_path):
                model_version.file_hash = await self._calculate_file_hash(model_version.local_path)
                model_version.size_bytes = os.path.getsize(model_version.local_path)
            
            # Store in memory
            key = f"{model_version.name}:{model_version.version}"
            self.model_versions[key] = model_version
            
            logger.info(f"Registered model {key}")
            metrics.record_external_service_call("model_registry", "register", "success", 0.1)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to register model {model_version.name}:{model_version.version}: {e}")
            metrics.record_error("model_registration_failed", "model_registry", "error")
            return False
    
    async def load_model(self, 
                        model_name: str, 
                        version: str = "latest",
                        force_reload: bool = False) -> Optional[Any]:
        """
        Load a model by name and version.
        
        Args:
            model_name: Name of the model
            version: Model version ("latest" for most recent)
            force_reload: Force reload even if already loaded
            
        Returns:
            Loaded model or None if failed
        """
        try:
            # Get model version
            model_version = await self._get_model_version(model_name, version)
            if not model_version:
                logger.error(f"Model version not found: {model_name}:{version}")
                return None
            
            # Check if already loaded
            model_key = f"{model_name}:{version}"
            if model_key in self.active_models and not force_reload:
                logger.debug(f"Using cached model: {model_key}")
                return self.active_models[model_key]
            
            # Ensure local file exists
            if not model_version.local_path or not os.path.exists(model_version.local_path):
                if model_version.s3_uri:
                    local_path = await self._download_model(model_version)
                    model_version.local_path = str(local_path)
                else:
                    logger.error(f"No local path or S3 URI for model {model_key}")
                    return None
            
            # Load model using appropriate loader
            loader = self.loaders.get(model_version.framework)
            if not loader:
                logger.error(f"No loader available for framework: {model_version.framework}")
                return None
            
            start_time = time.time()
            model = await loader.load_model(model_version.local_path)
            load_time = time.time() - start_time
            
            # Health check
            if not await loader.health_check(model):
                logger.error(f"Model health check failed: {model_key}")
                return None
            
            # Store active model
            self.active_models[model_key] = model
            
            # Record metrics
            metrics.record_external_service_call("model_registry", "load", "success", load_time)
            
            logger.info(f"Loaded model {model_key} in {load_time:.2f}s")
            
            # Notify callbacks
            await self._notify_model_update(model_name, model)
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}:{version}: {e}")
            metrics.record_error("model_load_failed", "model_registry", "error")
            return None
    
    async def promote_model(self, 
                           model_name: str, 
                           version: str,
                           target_stage: str) -> bool:
        """
        Promote model to target stage (staging -> production).
        
        Args:
            model_name: Model name
            version: Model version
            target_stage: Target stage ("staging" or "production")
            
        Returns:
            True if successful
        """
        try:
            model_version = await self._get_model_version(model_name, version)
            if not model_version:
                return False
            
            # Update stage
            old_stage = model_version.stage
            model_version.stage = target_stage
            
            # Update in storage
            if self.storage:
                await self._store_model_metadata(model_version)
            
            # Update in memory
            key = f"{model_name}:{version}"
            self.model_versions[key] = model_version
            
            # If promoting to production, update current production tag
            if target_stage == "production":
                await self._update_production_tag(model_name, version)
            
            logger.info(f"Promoted model {key} from {old_stage} to {target_stage}")
            metrics.record_external_service_call("model_registry", "promote", "success", 0.1)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to promote model {model_name}:{version}: {e}")
            metrics.record_error("model_promotion_failed", "model_registry", "error")
            return False
    
    async def rollback_model(self, model_name: str) -> bool:
        """
        Rollback to previous production version.
        
        Args:
            model_name: Model name to rollback
            
        Returns:
            True if successful
        """
        try:
            # Get current and previous production versions
            current_prod = await self._get_production_model(model_name)
            previous_versions = await self._get_model_history(model_name, stage="production")
            
            if len(previous_versions) < 2:
                logger.warning(f"No previous version to rollback to for {model_name}")
                return False
            
            # Get previous version (second most recent)
            previous_version = previous_versions[1]
            
            # Update production tag
            await self._update_production_tag(model_name, previous_version.version)
            
            # Reload model
            await self.load_model(model_name, "production", force_reload=True)
            
            logger.warning(f"Rolled back model {model_name} to version {previous_version.version}")
            metrics.record_external_service_call("model_registry", "rollback", "success", 0.1)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to rollback model {model_name}: {e}")
            metrics.record_error("model_rollback_failed", "model_registry", "error")
            return False
    
    def register_update_callback(self, model_name: str, callback: Callable):
        """Register callback for model updates."""
        if model_name not in self.update_callbacks:
            self.update_callbacks[model_name] = []
        self.update_callbacks[model_name].append(callback)
    
    async def _get_model_version(self, model_name: str, version: str) -> Optional[ModelVersion]:
        """Get model version from cache or storage."""
        # Check cache first
        if version == "latest":
            # Find latest version for this model
            matching_versions = [v for k, v in self.model_versions.items() if v.name == model_name]
            if matching_versions:
                return max(matching_versions, key=lambda x: x.created_at)
        elif version == "production":
            return await self._get_production_model(model_name)
        else:
            key = f"{model_name}:{version}"
            if key in self.model_versions:
                return self.model_versions[key]
        
        # Load from storage if available
        if self.storage:
            return await self._load_model_metadata(model_name, version)
        
        return None
    
    async def _get_production_model(self, model_name: str) -> Optional[ModelVersion]:
        """Get current production model version."""
        # In a real implementation, this would query the production tag
        # For now, return the latest model marked as production
        matching_versions = [
            v for k, v in self.model_versions.items() 
            if v.name == model_name and v.stage == "production"
        ]
        if matching_versions:
            return max(matching_versions, key=lambda x: x.created_at)
        return None
    
    async def _get_model_history(self, model_name: str, stage: str = None) -> List[ModelVersion]:
        """Get model version history."""
        matching_versions = [
            v for k, v in self.model_versions.items() 
            if v.name == model_name and (stage is None or v.stage == stage)
        ]
        return sorted(matching_versions, key=lambda x: x.created_at, reverse=True)
    
    async def _download_model(self, model_version: ModelVersion) -> Path:
        """Download model from S3 to local cache."""
        # Mock implementation - in real system would download from S3
        cache_path = self.cache_dir / f"{model_version.name}_{model_version.version}.model"
        
        # Create dummy file for testing
        cache_path.write_text(f"Mock model data for {model_version.name}:{model_version.version}")
        
        logger.info(f"Downloaded model to {cache_path}")
        return cache_path
    
    async def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    async def _store_model_metadata(self, model_version: ModelVersion):
        """Store model metadata to storage backend."""
        # Mock implementation - would store to Supabase in real system
        logger.debug(f"Storing metadata for {model_version.name}:{model_version.version}")
    
    async def _load_model_metadata(self, model_name: str, version: str) -> Optional[ModelVersion]:
        """Load model metadata from storage backend."""
        # Mock implementation - would load from Supabase in real system
        return None
    
    async def _update_production_tag(self, model_name: str, version: str):
        """Update production tag pointer."""
        # Mock implementation - would update tag in storage
        logger.info(f"Updated production tag for {model_name} to {version}")
    
    async def _notify_model_update(self, model_name: str, model: Any):
        """Notify registered callbacks of model update."""
        if model_name in self.update_callbacks:
            for callback in self.update_callbacks[model_name]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(model)
                    else:
                        callback(model)
                except Exception as e:
                    logger.error(f"Model update callback failed: {e}")


# Global model registry instance
_global_registry: Optional[ModelRegistry] = None

def get_model_registry() -> ModelRegistry:
    """Get global model registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ModelRegistry()
    return _global_registry

def set_model_registry(registry: ModelRegistry):
    """Set global model registry."""
    global _global_registry
    _global_registry = registry
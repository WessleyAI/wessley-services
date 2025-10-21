# Learning/ML Service Architecture

## Overview
**Language**: Python (FastAPI + Ray)  
**Primary Function**: Continuous improvement of 3D generation algorithms through machine learning  
**Status**: 🔄 **Future Implementation**

## Core Responsibilities
- Analyze user interactions and model generation patterns
- Optimize component placement algorithms using reinforcement learning
- Enhance wire harness routing strategies with physics-based ML models
- Train models for component recognition and classification
- Provide algorithmic improvements to 3D Model Service via gRPC
- Learn from user feedback to improve spatial layouts
- Develop predictive models for electrical system optimization

## Technology Stack
```json
{
  "framework": "FastAPI",
  "language": "Python 3.11+",
  "ml_framework": "PyTorch + TensorFlow",
  "distributed": "Ray",
  "3d_library": "Open3D + Trimesh",
  "database": "PostgreSQL + Redis",
  "messaging": "gRPC",
  "monitoring": "MLflow + Weights & Biases",
  "testing": "pytest + hypothesis",
  "deployment": "Docker + Kubernetes"
}
```

## Service Architecture
```
src/
├── algorithms/
│   ├── spatial/
│   │   ├── placement_optimizer.py    # Genetic algorithms for component placement
│   │   ├── routing_optimizer.py      # A* pathfinding with physics constraints
│   │   └── layout_scorer.py          # Multi-objective optimization scoring
│   ├── recognition/
│   │   ├── component_detector.py     # CNN models for visual component detection
│   │   ├── wire_tracer.py           # Computer vision for wire path analysis
│   │   └── pattern_matcher.py       # Pattern recognition for electrical layouts
│   └── feedback/
│       ├── user_behavior.py         # User interaction analysis
│       ├── generation_quality.py   # Model quality assessment
│       └── continuous_learning.py  # Online learning algorithms
├── models/
│   ├── reinforcement/
│   │   ├── placement_agent.py       # RL agent for component placement
│   │   ├── routing_agent.py         # RL agent for wire routing
│   │   └── environment.py           # 3D electrical system environment
│   ├── supervised/
│   │   ├── component_classifier.py  # Component type classification
│   │   ├── quality_predictor.py     # Generation quality prediction
│   │   └── layout_evaluator.py      # Layout quality evaluation
│   └── unsupervised/
│       ├── pattern_discovery.py     # Clustering and pattern discovery
│       ├── anomaly_detection.py     # Anomalous layout detection
│       └── embedding_learner.py     # Representation learning
├── training/
│   ├── data_pipeline.py             # Data preprocessing and augmentation
│   ├── experiment_runner.py         # MLflow experiment management
│   ├── model_trainer.py             # Distributed training with Ray
│   └── validation.py                # Cross-validation and testing
├── services/
│   ├── optimization_service.py      # gRPC service for 3D Model Service
│   ├── training_service.py          # Model training coordination
│   ├── inference_service.py         # Real-time inference endpoints
│   └── feedback_service.py          # User feedback processing
├── api/
│   ├── optimization.py              # Optimization endpoints
│   ├── training.py                  # Training management endpoints
│   ├── models.py                    # Model management endpoints
│   └── health.py                    # Health check endpoints
└── utils/
    ├── ray_utils.py                 # Ray cluster utilities
    ├── ml_utils.py                  # Common ML utilities
    └── 3d_utils.py                  # 3D geometry utilities
```

## Key Algorithms

### 1. Spatial Optimization
- **Genetic Algorithm**: Multi-objective optimization for component placement
- **Simulated Annealing**: Local optimization for fine-tuning layouts
- **Particle Swarm**: Distributed optimization for large electrical systems
- **Physics Simulation**: Real-world constraints for realistic placements

### 2. Wire Routing Enhancement
- **A* Pathfinding**: Physics-constrained pathfinding with cost functions
- **Neural Pathfinding**: Deep learning-based routing optimization
- **Collision Avoidance**: Dynamic obstacle avoidance algorithms
- **Harness Optimization**: Bundle routing and strain relief optimization

### 3. Component Recognition
- **Computer Vision**: CNN models for component detection in images
- **3D Recognition**: Point cloud analysis for 3D component identification
- **Transfer Learning**: Pre-trained models for automotive components
- **Active Learning**: Human-in-the-loop model improvement

### 4. Quality Assessment
- **Multi-objective Scoring**: Electrical performance, mechanical feasibility, aesthetics
- **User Preference Learning**: Personalized quality metrics
- **Industry Standards**: Compliance with automotive wiring standards
- **Performance Prediction**: Electrical system performance modeling

## gRPC Integration with 3D Model Service

### Service Definition
```protobuf
service LearningService {
  rpc OptimizeComponentPlacement(PlacementRequest) returns (PlacementResponse);
  rpc EnhanceWireRouting(RoutingRequest) returns (RoutingResponse);
  rpc ScoreLayout(LayoutRequest) returns (LayoutScore);
  rpc LearnFromFeedback(FeedbackRequest) returns (FeedbackResponse);
  rpc GetOptimizationSuggestions(SuggestionRequest) returns (SuggestionResponse);
}
```

### Request/Response Types
```python
@dataclass
class PlacementRequest:
    components: List[ComponentSpec]
    constraints: List[Constraint]
    vehicle_signature: str
    optimization_goals: List[OptimizationGoal]

@dataclass
class PlacementResponse:
    optimized_positions: List[Position3D]
    confidence_scores: List[float]
    improvement_metrics: Dict[str, float]
    execution_time_ms: int
```

## Machine Learning Pipeline

### 1. Data Collection
```python
# Real-time learning from 3D Model Service
async def collect_generation_data(generation_event):
    features = extract_layout_features(generation_event.layout)
    user_feedback = await get_user_feedback(generation_event.request_id)
    
    training_sample = {
        'features': features,
        'quality_score': user_feedback.quality,
        'user_preferences': user_feedback.preferences,
        'performance_metrics': generation_event.metrics
    }
    
    await store_training_sample(training_sample)
```

### 2. Model Training
```python
# Distributed training with Ray
@ray.remote
class ModelTrainer:
    def train_placement_model(self, training_data):
        model = PlacementOptimizer()
        trainer = RayTrainer(
            model=model,
            datasets=training_data,
            num_workers=8,
            use_gpu=True
        )
        return trainer.fit()
```

### 3. Inference Integration
```python
# Real-time optimization for 3D Model Service
async def optimize_component_placement(components, constraints):
    # Load latest trained model
    model = await load_latest_model('placement_optimizer')
    
    # Generate optimized layout
    optimized_layout = await model.optimize(
        components=components,
        constraints=constraints,
        timeout_ms=5000
    )
    
    return optimized_layout
```

## API Endpoints

### Optimization APIs
```python
# Component placement optimization
POST /api/v1/optimize/placement
{
  "components": [...],
  "constraints": [...],
  "vehicle_signature": "pajero_pinin_2001",
  "optimization_goals": ["minimize_wire_length", "maximize_accessibility"]
}

# Wire routing enhancement
POST /api/v1/optimize/routing
{
  "harnesses": [...],
  "obstacles": [...],
  "constraints": [...],
  "optimization_type": "physics_based"
}

# Layout quality scoring
POST /api/v1/evaluate/layout
{
  "layout": {...},
  "evaluation_criteria": ["electrical", "mechanical", "aesthetic"]
}
```

### Training Management APIs
```python
# Start model training
POST /api/v1/training/start
{
  "model_type": "placement_optimizer",
  "training_config": {...},
  "dataset_version": "v2.1"
}

# Get training status
GET /api/v1/training/:job_id/status

# Deploy trained model
POST /api/v1/models/:model_id/deploy
```

## Performance Characteristics
- **Optimization Response Time**: < 5 seconds for typical electrical systems
- **Model Training**: Distributed across 8+ GPU workers
- **Inference Throughput**: 100+ optimizations per second
- **Memory Usage**: < 4GB per worker process
- **Model Accuracy**: Target 95%+ layout quality improvement

## Integration Points

### With 3D Model Service (TypeScript/NestJS)
```typescript
// 3D Model Service calls Learning Service for optimization
interface ILearningService {
  optimizeComponentPlacement(components: ComponentEntity[]): Promise<OptimizedLayout>;
  enhanceWireRouting(harnesses: HarnessEntity[]): Promise<RoutingStrategy>;
  scoreLayout(layout: LayoutEntity): Promise<QualityScore>;
  learnFromFeedback(modelId: string, feedback: UserFeedback): Promise<void>;
}
```

### With Neo4j Knowledge Graph
```python
# Access electrical system data for training
async def fetch_training_data(vehicle_signature: str):
    query = """
    MATCH (c:Component {vehicle_signature: $signature})
    OPTIONAL MATCH (c)-[r:CONNECTS_TO]->(other)
    RETURN c, r, other
    """
    return await neo4j_session.run(query, signature=vehicle_signature)
```

### With Supabase
```python
# Store ML model metadata and results
async def store_model_metadata(model_info):
    await supabase.table('ml_models').upsert({
        'model_id': model_info.id,
        'version': model_info.version,
        'accuracy_metrics': model_info.metrics,
        'deployment_status': 'active'
    })
```

## Environment Configuration
```bash
# ML Framework
PYTHON_VERSION=3.11
PYTORCH_VERSION=2.0
TENSORFLOW_VERSION=2.13

# Ray Cluster
RAY_HEAD_NODE=localhost:10001
RAY_DASHBOARD_PORT=8265
RAY_NUM_WORKERS=8

# Model Storage
MODEL_REGISTRY_URL=s3://wessley-ml-models
MLFLOW_TRACKING_URI=http://localhost:5000

# Database
POSTGRES_URL=postgresql://localhost:5432/learning_db
REDIS_URL=redis://localhost:6379/1

# 3D Model Service Integration
MODEL_SERVICE_GRPC_URL=localhost:50051
MODEL_SERVICE_API_KEY=your-service-key

# Service Configuration
PORT=3002
LOG_LEVEL=info
ENABLE_GPU=true
```

## Future Enhancements

### Phase 1: Foundation (Current)
- Basic pattern analysis from generation data
- Simple optimization algorithms
- gRPC integration with 3D Model Service

### Phase 2: Advanced ML
- Reinforcement learning for component placement
- Computer vision for component recognition
- Real-time user preference learning

### Phase 3: Generative AI
- Generative models for electrical layouts
- Natural language to 3D model generation
- Predictive maintenance recommendations

### Phase 4: Collaborative AI
- Multi-user collaborative optimization
- Real-time AI assistance during design
- Automated electrical system validation

## Monitoring & Observability
- **MLflow**: Experiment tracking and model registry
- **Weights & Biases**: Advanced experiment visualization
- **Ray Dashboard**: Distributed computing monitoring
- **Prometheus**: Service metrics collection
- **Grafana**: ML performance dashboards

This service will continuously improve the quality and efficiency of 3D electrical model generation through advanced machine learning techniques.
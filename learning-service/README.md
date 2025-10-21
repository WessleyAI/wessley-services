# Learning Service

Advanced ML-powered optimization and learning services for electrical systems in the Wessley.ai platform.

## 🚀 Features

### Spatial Optimization Algorithms
- **Genetic Algorithm Placement Optimizer**: Multi-objective optimization for component placement
- **A* 3D Wire Routing**: Physics-constrained pathfinding for optimal wire routing
- **Multi-Objective Layout Scorer**: Comprehensive layout evaluation across safety, efficiency, cost, and reliability

### Recognition & Computer Vision
- **Component Detection**: CNN-based detection and classification of electrical components
- **Wire Tracing**: Advanced computer vision for wire path analysis and harness detection
- **Pattern Matching**: Recognition of electrical layout patterns and standards

### Reinforcement Learning Agents
- **Placement Agent**: Deep Q-Network and Actor-Critic models for intelligent component placement
- **Routing Agent**: RL-based wire routing optimization
- **Environment Simulation**: 3D electrical system simulation for training

### User Behavior Analytics
- **Interaction Analysis**: Real-time user behavior tracking and pattern recognition
- **Feedback Processing**: Continuous learning from user feedback and preferences
- **Personalized Recommendations**: AI-driven suggestions for improved user experience

### Training & Experiment Management
- **MLflow Integration**: Comprehensive experiment tracking and model versioning
- **Ray Distributed Computing**: Scalable distributed training and inference
- **Hyperparameter Optimization**: Automated hyperparameter tuning and model selection

### API Services
- **FastAPI REST Endpoints**: High-performance API for optimization services
- **gRPC Services**: Low-latency communication with 3D Model Service
- **Real-time Analytics**: WebSocket support for real-time updates

## 🏗️ Architecture

```
apps/services/learning-service/
├── src/
│   ├── algorithms/
│   │   ├── spatial/              # Spatial optimization algorithms
│   │   ├── recognition/          # Computer vision and ML models
│   │   └── feedback/             # User behavior analysis
│   ├── models/
│   │   ├── reinforcement/        # RL agents and environments
│   │   ├── supervised/           # Supervised learning models
│   │   └── unsupervised/         # Clustering and pattern discovery
│   ├── training/                 # Training pipeline and experiment management
│   ├── services/                 # gRPC services
│   ├── api/                      # FastAPI endpoints
│   ├── utils/                    # Utilities and helpers
│   ├── schemas/                  # Data models and schemas
│   └── config/                   # Configuration management
├── tests/                        # Test suites
├── docs/                         # Documentation
├── models/                       # Trained model artifacts
├── data/                         # Training and test data
├── docker-compose.yml            # Multi-service deployment
├── Dockerfile                    # Container configuration
└── requirements.txt              # Python dependencies
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+ 
- Docker & Docker Compose
- CUDA-compatible GPU (optional, for accelerated training)

### Local Development Setup

1. **Clone and Navigate**
   ```bash
   cd apps/services/learning-service
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   # or using poetry
   poetry install
   ```

3. **Environment Configuration**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Start Services**
   ```bash
   docker-compose up -d postgres redis mlflow ray-head
   ```

5. **Run the Service**
   ```bash
   python -m uvicorn src.api.optimization:app --reload --port 8000
   ```

### Docker Deployment

1. **Build and Start All Services**
   ```bash
   docker-compose up -d
   ```

2. **Check Service Health**
   ```bash
   curl http://localhost:8000/health
   ```

3. **Access Web Interfaces**
   - API Documentation: http://localhost:8000/docs
   - MLflow UI: http://localhost:5000
   - Ray Dashboard: http://localhost:8265
   - Grafana Monitoring: http://localhost:3000

## 📚 API Usage

### Component Placement Optimization

```python
import httpx

# Start optimization
response = httpx.post("http://localhost:8000/api/v1/optimization/placement", json={
    "components": [
        {
            "id": "fuse_1",
            "type": "fuse",
            "dimensions": [20, 10, 10],
            "weight": 0.05,
            "electrical_properties": {"voltage": 12, "current": 15}
        }
    ],
    "workspace_bounds": [500, 500, 300],
    "use_rl_agent": True
})

request_id = response.json()["request_id"]

# Check status
status = httpx.get(f"http://localhost:8000/api/v1/optimization/placement/{request_id}/status")

# Get results
result = httpx.get(f"http://localhost:8000/api/v1/optimization/placement/{request_id}/result")
```

### Wire Routing Optimization

```python
# Route multiple wires
response = httpx.post("http://localhost:8000/api/v1/optimization/routing", json={
    "wire_specifications": [
        {
            "id": "power_wire",
            "type": "power",
            "gauge": 2.5,
            "max_current": 20,
            "voltage_rating": 12
        }
    ],
    "start_points": [[0, 0, 0]],
    "end_points": [[100, 100, 50]],
    "workspace_bounds": [500, 500, 300]
})
```

### Layout Scoring

```python
# Score a complete layout
response = httpx.post("http://localhost:8000/api/v1/optimization/score", json={
    "components": [...],  # Component layout data
    "wires": [...],       # Wire routing data
    "scoring_weights": {
        "safety": 0.4,
        "efficiency": 0.3,
        "cost": 0.3
    }
})
```

### User Analytics

```python
# Record user interaction
httpx.post("http://localhost:8000/api/v1/analytics/interaction", json={
    "session_id": "session_123",
    "user_id": "user_456",
    "interaction_type": "component_placement",
    "target_object_id": "fuse_1",
    "duration_ms": 1500,
    "success": True
})

# Get personalized recommendations
recommendations = httpx.get("http://localhost:8000/api/v1/analytics/user/user_456/recommendations")
```

## 🧠 ML Models

### Pre-trained Models
- **Component Detector**: ResNet-50 based CNN for electrical component classification
- **Placement Agent**: DQN trained on 10M+ placement scenarios
- **Wire Tracer**: U-Net architecture for wire path segmentation

### Training New Models

```python
from src.training.experiment_runner import MLflowExperimentRunner, ExperimentConfig

# Configure experiment
config = ExperimentConfig(
    experiment_name="component_detector_v2",
    model_type="component_detector",
    dataset_path="/data/electrical_components",
    hyperparameters={
        "learning_rate": 1e-4,
        "batch_size": 32,
        "epochs": 100
    },
    training_config={
        "optimizer": "adam",
        "scheduler": "cosine"
    },
    evaluation_config={
        "metrics": ["accuracy", "precision", "recall", "f1"]
    },
    hardware_config={
        "device": "cuda",
        "num_gpus": 1
    },
    tags={"version": "2.0", "experiment_type": "baseline"},
    description="Improved component detection with data augmentation"
)

# Run experiment
runner = MLflowExperimentRunner()
result = runner.run_experiment(config)
```

### Hyperparameter Optimization

```python
# Define parameter grid
param_grid = {
    "learning_rate": [1e-5, 1e-4, 1e-3],
    "batch_size": [16, 32, 64],
    "dropout_rate": [0.1, 0.3, 0.5]
}

# Run sweep
results = runner.run_hyperparameter_sweep(config, param_grid, n_trials=20)

# Get best model
best_run_id, best_score = runner.get_best_model("component_detector_v2", "accuracy")
```

## 🔧 Configuration

### Environment Variables

```bash
# Service Configuration
SERVICE_NAME=learning-service
LOG_LEVEL=INFO

# Database
DATABASE_URL=postgresql://user:pass@host:5432/db
REDIS_URL=redis://host:6379/0

# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_ARTIFACT_ROOT=/data/mlflow/artifacts

# Ray
RAY_ADDRESS=ray://ray-head:10001

# API
API_HOST=0.0.0.0
API_PORT=8000

# GPU Support
CUDA_VISIBLE_DEVICES=0
```

### Model Configuration

```yaml
# config/models.yaml
component_detector:
  backbone: "resnet50"
  num_classes: 15
  pretrained: true
  dropout_rate: 0.3

placement_agent:
  state_dim: 1000
  action_dim: 1000
  hidden_dim: 512
  learning_rate: 1e-4
  epsilon_decay: 0.995

routing_optimizer:
  grid_resolution: 5.0
  min_wire_clearance: 2.0
  max_iterations: 1000
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit                    # Unit tests only
pytest -m integration            # Integration tests only
pytest -m "not slow"             # Skip slow tests

# Run with coverage
pytest --cov=src --cov-report=html

# Run GPU tests (if available)
pytest -m gpu
```

## 📊 Monitoring & Observability

### Metrics
- **API Performance**: Request latency, throughput, error rates
- **ML Model Performance**: Inference time, accuracy, drift detection
- **Resource Utilization**: CPU, memory, GPU usage
- **Training Metrics**: Loss curves, validation scores, convergence rates

### Dashboards
- **Grafana**: Real-time service metrics and alerts
- **MLflow**: Experiment tracking and model comparison
- **Ray Dashboard**: Distributed computing status
- **Prometheus**: Metric collection and alerting

## 🤝 Integration

### With 3D Model Service
```python
# gRPC client example
import grpc
from src.services.optimization_service import optimization_pb2_grpc

channel = grpc.insecure_channel('3d-model-service:50052')
stub = optimization_pb2_grpc.OptimizationServiceStub(channel)

# Request optimization
request = optimization_pb2.PlacementRequest(
    components=[...],
    workspace_bounds=[500, 500, 300]
)

response = stub.OptimizeComponentPlacement(request)
```

### With Web Frontend
```javascript
// WebSocket connection for real-time updates
const ws = new WebSocket('ws://localhost:8000/ws/optimization');

ws.onmessage = (event) => {
    const update = JSON.parse(event.data);
    if (update.type === 'optimization_progress') {
        updateProgressBar(update.progress);
    }
};

// REST API calls
async function optimizeLayout(components, wires) {
    const response = await fetch('/api/v1/optimization/full', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ components, wires })
    });
    return response.json();
}
```

## 🚀 Production Deployment

### Kubernetes
```yaml
# k8s/learning-service.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: learning-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: learning-service
  template:
    metadata:
      labels:
        app: learning-service
    spec:
      containers:
      - name: learning-service
        image: wessley/learning-service:latest
        ports:
        - containerPort: 8000
        - containerPort: 50051
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"
            nvidia.com/gpu: 1
```

### Scaling Considerations
- **Horizontal Pod Autoscaling**: Scale based on CPU/memory usage
- **GPU Node Pools**: Dedicated nodes for ML workloads
- **Ray Cluster**: Auto-scaling distributed computing
- **Model Caching**: Redis for frequently accessed models
- **Load Balancing**: Nginx for API traffic distribution

## 🔒 Security

### Authentication & Authorization
- JWT token-based authentication
- Role-based access control (RBAC)
- API rate limiting
- Request validation and sanitization

### Data Protection
- Encryption at rest and in transit
- Secure model artifact storage
- PII data anonymization
- Audit logging

### Network Security
- mTLS for gRPC communications
- WAF for HTTP traffic
- Network policies for pod isolation
- Secret management with Kubernetes secrets

## 📈 Performance Optimization

### Model Optimization
- **Model Quantization**: INT8 quantization for faster inference
- **Model Pruning**: Remove redundant parameters
- **ONNX Export**: Cross-platform model deployment
- **TensorRT**: GPU-accelerated inference

### Caching Strategies
- **Model Caching**: Keep hot models in memory
- **Result Caching**: Cache optimization results
- **Preprocessing Caching**: Cache feature extraction
- **Distributed Caching**: Redis cluster for scale

### Resource Management
- **GPU Memory Pooling**: Efficient GPU utilization
- **Batch Processing**: Group inference requests
- **Async Processing**: Non-blocking operations
- **Resource Limits**: Prevent resource exhaustion

## 🐛 Troubleshooting

### Common Issues

1. **GPU Out of Memory**
   ```bash
   # Reduce batch size or model size
   export CUDA_VISIBLE_DEVICES=0
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
   ```

2. **Ray Connection Issues**
   ```bash
   # Check Ray cluster status
   ray status
   # Restart Ray cluster
   ray stop && ray start --head
   ```

3. **MLflow Tracking Issues**
   ```bash
   # Check MLflow server
   curl http://localhost:5000/health
   # Reset tracking URI
   export MLFLOW_TRACKING_URI=http://localhost:5000
   ```

### Debugging

```python
# Enable debug logging
import logging
logging.getLogger("src").setLevel(logging.DEBUG)

# Profile memory usage
from memory_profiler import profile

@profile
def optimize_placement():
    # Your optimization code here
    pass

# Profile GPU usage
import torch.profiler

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    # Your model inference code
    pass

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**: Follow the coding standards
4. **Run tests**: `pytest` and `black src/`
5. **Commit changes**: `git commit -m 'Add amazing feature'`
6. **Push to branch**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

### Development Guidelines
- Follow PEP 8 style guidelines
- Write comprehensive tests
- Document all public APIs
- Use type hints
- Update CHANGELOG.md

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyTorch Team**: For the excellent deep learning framework
- **Ray Team**: For distributed computing capabilities
- **MLflow Team**: For experiment tracking tools
- **FastAPI Team**: For the high-performance API framework
- **OpenCV Team**: For computer vision tools

## 📞 Support

- **Documentation**: https://docs.wessley.ai/learning-service
- **Issues**: https://github.com/wessley-ai/learning-service/issues
- **Discussions**: https://github.com/wessley-ai/learning-service/discussions
- **Email**: support@wessley.ai

---

**Made with ❤️ by the Wessley.ai Team**
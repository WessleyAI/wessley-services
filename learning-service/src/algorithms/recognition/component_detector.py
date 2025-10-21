"""
CNN models for visual component detection in electrical systems.
Detects and classifies electrical components from images using deep learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from PIL import Image
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    """Result from component detection."""
    component_type: str
    confidence: float
    bounding_box: Tuple[int, int, int, int]  # x, y, width, height
    keypoints: List[Tuple[int, int]]  # Important points on component
    properties: Dict[str, Any]  # Estimated properties (size, rating, etc.)

@dataclass
class ComponentClass:
    """Definition of a component class."""
    id: int
    name: str
    description: str
    typical_size_range: Tuple[float, float]  # min, max size in mm
    color_characteristics: List[str]
    shape_characteristics: List[str]

class ElectricalComponentDataset(Dataset):
    """Dataset for electrical component images."""
    
    def __init__(self, image_paths: List[str], labels: List[int], 
                 transform=None, augment=True):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.augment = augment
        
        # Default transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        
        # Augmentation transforms
        if augment:
            self.augment_transform = transforms.Compose([
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                                     saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), 
                                      scale=(0.9, 1.1))
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply augmentation during training
        if self.augment and torch.rand(1) < 0.5:
            image = self.augment_transform(image)
        
        # Apply standard transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label

class ResNetComponentDetector(nn.Module):
    """ResNet-based component detection and classification model."""
    
    def __init__(self, num_classes: int, backbone='resnet50', pretrained=True):
        super().__init__()
        
        # Load pretrained backbone
        if backbone == 'resnet50':
            from torchvision.models import resnet50
            self.backbone = resnet50(pretrained=pretrained)
            feature_dim = 2048
        elif backbone == 'resnet34':
            from torchvision.models import resnet34
            self.backbone = resnet34(pretrained=pretrained)
            feature_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
        
        # Bounding box regression head
        self.bbox_regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4)  # x, y, width, height
        )
        
        # Keypoint detection head
        self.keypoint_detector = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(feature_dim * 49, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 20)  # 10 keypoints (x, y) pairs
        )
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Classification
        class_logits = self.classifier(features)
        
        # Bounding box regression
        bbox_coords = self.bbox_regressor(features)
        
        # Keypoint detection
        keypoints = self.keypoint_detector(features)
        
        return {
            'classification': class_logits,
            'bbox': bbox_coords,
            'keypoints': keypoints
        }

class YOLOComponentDetector(nn.Module):
    """YOLO-style detector optimized for electrical components."""
    
    def __init__(self, num_classes: int, grid_size=13):
        super().__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.num_anchors = 3
        
        # Backbone - simplified for demonstration
        self.backbone = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            
            # Conv Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            
            # Conv Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            
            # Conv Block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            
            # Conv Block 5
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
        )
        
        # Detection head
        # Output: (batch, anchors * (5 + num_classes), grid, grid)
        # 5 = x, y, w, h, confidence
        self.detection_head = nn.Conv2d(
            512, 
            self.num_anchors * (5 + num_classes), 
            1
        )
        
        # Predefined anchor boxes for different component sizes
        self.anchors = torch.tensor([
            [10, 13], [16, 30], [33, 23],  # Small components
        ], dtype=torch.float32)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Extract features
        features = self.backbone(x)
        
        # Detection
        detection = self.detection_head(features)
        
        # Reshape to (batch, anchors, grid, grid, 5 + num_classes)
        detection = detection.view(
            batch_size, 
            self.num_anchors, 
            5 + self.num_classes, 
            self.grid_size, 
            self.grid_size
        ).permute(0, 1, 3, 4, 2).contiguous()
        
        return detection

class ComponentDetectionPipeline:
    """Complete pipeline for component detection in electrical system images."""
    
    def __init__(self, model_path: Optional[str] = None, 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Component classes
        self.component_classes = self._define_component_classes()
        self.num_classes = len(self.component_classes)
        
        # Initialize model
        self.model = ResNetComponentDetector(self.num_classes)
        self.model.to(self.device)
        
        # Load pretrained weights if available
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        
        # Preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Post-processing parameters
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
    
    def _define_component_classes(self) -> List[ComponentClass]:
        """Define the electrical component classes."""
        return [
            ComponentClass(0, "fuse", "Electrical fuse", (5, 50), ["black", "clear"], ["cylindrical"]),
            ComponentClass(1, "relay", "Electrical relay", (15, 40), ["black", "blue"], ["rectangular"]),
            ComponentClass(2, "capacitor", "Capacitor", (5, 30), ["black", "blue", "silver"], ["cylindrical"]),
            ComponentClass(3, "resistor", "Resistor", (3, 15), ["brown", "red", "orange"], ["cylindrical"]),
            ComponentClass(4, "diode", "Diode", (2, 10), ["black", "silver"], ["cylindrical"]),
            ComponentClass(5, "transistor", "Transistor", (3, 20), ["black"], ["rectangular"]),
            ComponentClass(6, "connector", "Wire connector", (5, 50), ["white", "black", "clear"], ["rectangular"]),
            ComponentClass(7, "switch", "Switch", (10, 30), ["black", "red"], ["rectangular"]),
            ComponentClass(8, "led", "LED indicator", (3, 10), ["red", "green", "blue"], ["cylindrical"]),
            ComponentClass(9, "battery", "Battery", (20, 100), ["black", "red"], ["rectangular"]),
            ComponentClass(10, "motor", "Electric motor", (30, 200), ["black", "silver"], ["cylindrical"]),
            ComponentClass(11, "sensor", "Sensor", (5, 30), ["black", "white"], ["rectangular"]),
            ComponentClass(12, "wire", "Wire/Cable", (1, 20), ["red", "black", "blue"], ["linear"]),
            ComponentClass(13, "terminal", "Terminal block", (5, 30), ["white", "black"], ["rectangular"]),
            ComponentClass(14, "ground", "Ground connection", (5, 20), ["black", "green"], ["varied"])
        ]
    
    def detect_components(self, image: np.ndarray) -> List[DetectionResult]:
        """Detect electrical components in an image."""
        self.model.eval()
        
        # Preprocess image
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        pil_image = Image.fromarray(image_rgb)
        input_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        # Post-process results
        detections = self._post_process_outputs(outputs, image.shape[:2])
        
        return detections
    
    def detect_components_batch(self, images: List[np.ndarray]) -> List[List[DetectionResult]]:
        """Detect components in a batch of images."""
        self.model.eval()
        
        # Preprocess batch
        batch_tensors = []
        for image in images:
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            pil_image = Image.fromarray(image_rgb)
            tensor = self.preprocess(pil_image)
            batch_tensors.append(tensor)
        
        batch_tensor = torch.stack(batch_tensors).to(self.device)
        
        # Batch inference
        with torch.no_grad():
            outputs = self.model(batch_tensor)
        
        # Post-process each image
        all_detections = []
        for i, image in enumerate(images):
            image_outputs = {
                'classification': outputs['classification'][i:i+1],
                'bbox': outputs['bbox'][i:i+1],
                'keypoints': outputs['keypoints'][i:i+1]
            }
            detections = self._post_process_outputs(image_outputs, image.shape[:2])
            all_detections.append(detections)
        
        return all_detections
    
    def _post_process_outputs(self, outputs: Dict[str, torch.Tensor], 
                            original_shape: Tuple[int, int]) -> List[DetectionResult]:
        """Post-process model outputs to detection results."""
        detections = []
        
        # Get predictions
        class_probs = F.softmax(outputs['classification'], dim=1)
        max_prob, predicted_class = torch.max(class_probs, dim=1)
        bbox_coords = outputs['bbox']
        keypoints = outputs['keypoints']
        
        # Filter by confidence threshold
        confident_mask = max_prob > self.confidence_threshold
        
        if confident_mask.any():
            for i in range(confident_mask.sum()):
                confidence = max_prob[confident_mask][i].item()
                class_id = predicted_class[confident_mask][i].item()
                bbox = bbox_coords[confident_mask][i].cpu().numpy()
                kpts = keypoints[confident_mask][i].cpu().numpy()
                
                # Convert normalized bbox to image coordinates
                height, width = original_shape
                x = int(bbox[0] * width)
                y = int(bbox[1] * height)
                w = int(bbox[2] * width)
                h = int(bbox[3] * height)
                
                # Convert keypoints
                keypoint_pairs = []
                for j in range(0, len(kpts), 2):
                    if j + 1 < len(kpts):
                        kx = int(kpts[j] * width)
                        ky = int(kpts[j+1] * height)
                        keypoint_pairs.append((kx, ky))
                
                # Estimate component properties
                properties = self._estimate_component_properties(
                    class_id, (w, h), original_shape)
                
                detection = DetectionResult(
                    component_type=self.component_classes[class_id].name,
                    confidence=confidence,
                    bounding_box=(x, y, w, h),
                    keypoints=keypoint_pairs,
                    properties=properties
                )
                
                detections.append(detection)
        
        return detections
    
    def _estimate_component_properties(self, class_id: int, 
                                     bbox_size: Tuple[int, int],
                                     image_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Estimate physical properties of detected component."""
        component_class = self.component_classes[class_id]
        
        # Estimate physical size based on bounding box
        # This would require calibration in a real system
        pixel_to_mm_ratio = 0.1  # Simplified assumption
        width_mm = bbox_size[0] * pixel_to_mm_ratio
        height_mm = bbox_size[1] * pixel_to_mm_ratio
        
        properties = {
            'estimated_width_mm': width_mm,
            'estimated_height_mm': height_mm,
            'size_category': self._categorize_size(width_mm, height_mm, component_class),
            'confidence_size_estimate': 0.7  # Would be based on calibration accuracy
        }
        
        # Add component-specific property estimates
        if component_class.name == "fuse":
            properties['estimated_rating_amps'] = self._estimate_fuse_rating(width_mm, height_mm)
        elif component_class.name == "resistor":
            properties['estimated_resistance_ohms'] = self._estimate_resistance(width_mm)
        elif component_class.name == "capacitor":
            properties['estimated_capacitance_uf'] = self._estimate_capacitance(width_mm, height_mm)
        
        return properties
    
    def _categorize_size(self, width: float, height: float, 
                        component_class: ComponentClass) -> str:
        """Categorize component size relative to its class."""
        avg_size = (width + height) / 2
        min_size, max_size = component_class.typical_size_range
        
        if avg_size < min_size * 1.2:
            return "small"
        elif avg_size > max_size * 0.8:
            return "large"
        else:
            return "standard"
    
    def _estimate_fuse_rating(self, width: float, height: float) -> Optional[float]:
        """Estimate fuse current rating based on size."""
        # Simplified heuristic - in reality would use trained regression model
        volume = width * height
        if volume < 50:
            return 5.0
        elif volume < 200:
            return 15.0
        elif volume < 500:
            return 30.0
        else:
            return 60.0
    
    def _estimate_resistance(self, width: float) -> Optional[float]:
        """Estimate resistor value based on size."""
        # Color band detection would be more accurate
        if width < 5:
            return 1000  # 1K ohm
        elif width < 10:
            return 10000  # 10K ohm
        else:
            return 100000  # 100K ohm
    
    def _estimate_capacitance(self, width: float, height: float) -> Optional[float]:
        """Estimate capacitor value based on size."""
        volume = width * height
        if volume < 100:
            return 0.1  # 0.1 uF
        elif volume < 300:
            return 1.0  # 1 uF
        else:
            return 10.0  # 10 uF
    
    def train_model(self, train_dataset: Dataset, val_dataset: Dataset,
                   num_epochs: int = 100, learning_rate: float = 0.001):
        """Train the component detection model."""
        logger.info(f"Training model for {num_epochs} epochs")
        
        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        classification_loss = nn.CrossEntropyLoss()
        bbox_loss = nn.SmoothL1Loss()
        keypoint_loss = nn.MSELoss()
        
        # Training loop
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                
                # Multi-task loss
                cls_loss = classification_loss(outputs['classification'], labels)
                # Note: In a real implementation, you'd need ground truth bboxes and keypoints
                # For now, using simplified loss
                loss = cls_loss
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    
                    loss = classification_loss(outputs['classification'], labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs['classification'], 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                       f"Train Loss: {train_loss/len(train_loader):.4f}, "
                       f"Val Loss: {val_loss/len(val_loader):.4f}, "
                       f"Accuracy: {accuracy:.2f}%")
    
    def save_model(self, path: str):
        """Save trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'component_classes': self.component_classes,
            'num_classes': self.num_classes
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.component_classes = checkpoint['component_classes']
        self.num_classes = checkpoint['num_classes']
        logger.info(f"Model loaded from {path}")
    
    def visualize_detections(self, image: np.ndarray, 
                           detections: List[DetectionResult]) -> np.ndarray:
        """Visualize detection results on image."""
        vis_image = image.copy()
        
        for detection in detections:
            x, y, w, h = detection.bounding_box
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw label with confidence
            label = f"{detection.component_type}: {detection.confidence:.2f}"
            cv2.putText(vis_image, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw keypoints
            for kx, ky in detection.keypoints:
                cv2.circle(vis_image, (kx, ky), 3, (0, 0, 255), -1)
        
        return vis_image
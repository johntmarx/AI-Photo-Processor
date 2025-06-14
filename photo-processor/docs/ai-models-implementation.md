# AI Models Implementation Guide

## Model Pipeline Architecture

The AI processing pipeline consists of independent, composable stages that communicate through standardized data structures. Each model operates as a microservice with clear input/output contracts.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         AI Processing Pipeline                           │
├──────────────┬──────────────┬──────────────┬──────────────┬────────────┤
│  RT-DETR     │    SAM2      │    CLIP      │    NIMA      │  Qwen2.5   │
│  Detection   │ Segmentation │  Embeddings  │   Quality    │   VL Chat  │
├──────────────┴──────────────┴──────────────┴──────────────┴────────────┤
│                     Unified Data Translation Layer                       │
└─────────────────────────────────────────────────────────────────────────┘
```

## 1. RT-DETR Object Detection Implementation

### Model Setup and Configuration

```python
import torch
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np

@dataclass
class BoundingBox:
    """Standardized bounding box format"""
    x1: float  # Top-left x (normalized 0-1)
    y1: float  # Top-left y (normalized 0-1)
    x2: float  # Bottom-right x (normalized 0-1)
    y2: float  # Bottom-right y (normalized 0-1)
    confidence: float
    class_name: str
    class_id: int
    
    def to_absolute(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """Convert normalized coordinates to absolute pixels"""
        return (
            int(self.x1 * width),
            int(self.y1 * height),
            int(self.x2 * width),
            int(self.y2 * height)
        )
    
    def to_sam_prompt(self) -> Dict[str, Any]:
        """Convert to SAM2 input format"""
        return {
            "type": "box",
            "coordinates": [self.x1, self.y1, self.x2, self.y2],
            "label": self.class_name
        }

class RTDETRDetector:
    """
    RT-DETR wrapper for object detection
    """
    
    def __init__(self, model_name: str = "PekingU/rtdetr_r50vd"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and processor
        self.processor = RTDetrImageProcessor.from_pretrained(model_name)
        self.model = RTDetrForObjectDetection.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Detection thresholds
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.45
        
    def detect(self, image: np.ndarray) -> List[BoundingBox]:
        """
        Perform object detection on image
        
        Args:
            image: RGB numpy array (H, W, C)
            
        Returns:
            List of detected objects with bounding boxes
        """
        # Preprocess image
        inputs = self.processor(
            images=image, 
            return_tensors="pt"
        ).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process results
        target_sizes = torch.tensor([image.shape[:2]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, 
            target_sizes=target_sizes,
            threshold=self.confidence_threshold
        )[0]
        
        # Convert to standardized format
        detections = []
        height, width = image.shape[:2]
        
        for score, label, box in zip(
            results["scores"], 
            results["labels"], 
            results["boxes"]
        ):
            # Normalize coordinates
            x1, y1, x2, y2 = box.tolist()
            bbox = BoundingBox(
                x1=x1/width,
                y1=y1/height,
                x2=x2/width,
                y2=y2/height,
                confidence=score.item(),
                class_name=self.model.config.id2label[label.item()],
                class_id=label.item()
            )
            detections.append(bbox)
        
        # Apply NMS if needed
        detections = self._apply_nms(detections)
        
        return detections
    
    def _apply_nms(self, detections: List[BoundingBox]) -> List[BoundingBox]:
        """Apply non-maximum suppression"""
        if not detections:
            return detections
            
        # Group by class
        by_class = {}
        for det in detections:
            if det.class_id not in by_class:
                by_class[det.class_id] = []
            by_class[det.class_id].append(det)
        
        # Apply NMS per class
        final_detections = []
        for class_id, class_dets in by_class.items():
            # Sort by confidence
            class_dets.sort(key=lambda x: x.confidence, reverse=True)
            
            # Apply NMS
            keep = []
            for i, det in enumerate(class_dets):
                should_keep = True
                for kept_det in keep:
                    if self._compute_iou(det, kept_det) > self.nms_threshold:
                        should_keep = False
                        break
                if should_keep:
                    keep.append(det)
            
            final_detections.extend(keep)
        
        return final_detections
    
    def _compute_iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """Compute intersection over union"""
        # Calculate intersection area
        x1 = max(box1.x1, box2.x1)
        y1 = max(box1.y1, box2.y1)
        x2 = min(box1.x2, box2.x2)
        y2 = min(box1.y2, box2.y2)
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Calculate union area
        area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
        area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
```

## 2. SAM2 Segmentation Implementation

### Segment Anything Model Integration

```python
import torch
from segment_anything import sam_model_registry, SamPredictor
from typing import List, Dict, Any, Optional
import cv2

@dataclass
class SegmentationMask:
    """Standardized segmentation output"""
    mask: np.ndarray  # Binary mask (H, W)
    bbox: BoundingBox  # Associated bounding box
    area: float  # Relative area (0-1)
    stability_score: float
    predicted_iou: float
    
    def to_polygon(self) -> List[Tuple[int, int]]:
        """Convert mask to polygon points"""
        contours, _ = cv2.findContours(
            self.mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            # Get the largest contour
            largest = max(contours, key=cv2.contourArea)
            return [(p[0][0], p[0][1]) for p in largest]
        return []
    
    def to_rle(self) -> Dict[str, Any]:
        """Convert to run-length encoding"""
        from pycocotools import mask as mask_utils
        rle = mask_utils.encode(
            np.asfortranarray(self.mask.astype(np.uint8))
        )
        rle['counts'] = rle['counts'].decode('utf-8')
        return rle

class SAM2Segmenter:
    """
    SAM2 wrapper for image segmentation
    """
    
    def __init__(self, model_size: str = "vit_h", checkpoint_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load SAM model
        if checkpoint_path is None:
            # Download default checkpoint
            checkpoint_path = self._download_checkpoint(model_size)
        
        sam = sam_model_registry[model_size](checkpoint=checkpoint_path)
        sam.to(self.device)
        
        self.predictor = SamPredictor(sam)
        self.automatic_mask_generator = None  # Lazy load if needed
        
    def segment_with_boxes(
        self, 
        image: np.ndarray, 
        boxes: List[BoundingBox]
    ) -> List[SegmentationMask]:
        """
        Segment objects using bounding boxes from detection
        
        Args:
            image: RGB numpy array (H, W, C)
            boxes: List of bounding boxes from RT-DETR
            
        Returns:
            List of segmentation masks
        """
        # Set image
        self.predictor.set_image(image)
        
        segments = []
        height, width = image.shape[:2]
        
        for box in boxes:
            # Convert normalized coords to absolute
            x1, y1, x2, y2 = box.to_absolute(width, height)
            input_box = np.array([x1, y1, x2, y2])
            
            # Predict mask
            masks, scores, logits = self.predictor.predict(
                box=input_box,
                multimask_output=True  # Get multiple mask options
            )
            
            # Select best mask
            best_idx = np.argmax(scores)
            mask = masks[best_idx]
            
            # Calculate mask area
            area = np.sum(mask) / (height * width)
            
            # Create segmentation object
            segment = SegmentationMask(
                mask=mask,
                bbox=box,
                area=area,
                stability_score=scores[best_idx],
                predicted_iou=scores[best_idx]  # SAM returns IoU as score
            )
            segments.append(segment)
        
        return segments
    
    def segment_with_points(
        self,
        image: np.ndarray,
        points: List[Tuple[int, int]],
        labels: List[int] = None
    ) -> SegmentationMask:
        """
        Segment using point prompts (for user interaction)
        
        Args:
            image: RGB numpy array
            points: List of (x, y) coordinates
            labels: List of labels (1=foreground, 0=background)
        """
        self.predictor.set_image(image)
        
        if labels is None:
            labels = [1] * len(points)  # All foreground by default
        
        input_points = np.array(points)
        input_labels = np.array(labels)
        
        masks, scores, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True
        )
        
        # Select best mask
        best_idx = np.argmax(scores)
        mask = masks[best_idx]
        
        # Generate bounding box from mask
        y_indices, x_indices = np.where(mask)
        if len(x_indices) > 0:
            x1, x2 = x_indices.min(), x_indices.max()
            y1, y2 = y_indices.min(), y_indices.max()
            height, width = image.shape[:2]
            
            bbox = BoundingBox(
                x1=x1/width,
                y1=y1/height,
                x2=x2/width,
                y2=y2/height,
                confidence=scores[best_idx],
                class_name="user_selection",
                class_id=-1
            )
        else:
            bbox = None
        
        return SegmentationMask(
            mask=mask,
            bbox=bbox,
            area=np.sum(mask) / (image.shape[0] * image.shape[1]),
            stability_score=scores[best_idx],
            predicted_iou=scores[best_idx]
        )
    
    def auto_segment(self, image: np.ndarray) -> List[SegmentationMask]:
        """
        Automatically segment entire image without prompts
        """
        if self.automatic_mask_generator is None:
            from segment_anything import SamAutomaticMaskGenerator
            self.automatic_mask_generator = SamAutomaticMaskGenerator(
                self.predictor.model,
                points_per_side=32,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100
            )
        
        masks = self.automatic_mask_generator.generate(image)
        
        # Convert to our format
        segments = []
        height, width = image.shape[:2]
        
        for mask_data in masks:
            # Extract bounding box
            x, y, w, h = mask_data['bbox']
            bbox = BoundingBox(
                x1=x/width,
                y1=y/height,
                x2=(x+w)/width,
                y2=(y+h)/height,
                confidence=mask_data['predicted_iou'],
                class_name="auto_segment",
                class_id=-1
            )
            
            segment = SegmentationMask(
                mask=mask_data['segmentation'],
                bbox=bbox,
                area=mask_data['area'] / (height * width),
                stability_score=mask_data['stability_score'],
                predicted_iou=mask_data['predicted_iou']
            )
            segments.append(segment)
        
        return segments
```

## 3. CLIP Embeddings and Semantic Search

### CLIP Integration for Image Understanding

```python
import torch
from transformers import CLIPModel, CLIPProcessor
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any, Union
import faiss

@dataclass
class ImageEmbedding:
    """Standardized embedding representation"""
    vector: np.ndarray  # Embedding vector
    model_name: str
    dimension: int
    metadata: Dict[str, Any]
    
    def to_base64(self) -> str:
        """Convert to base64 for storage"""
        import base64
        return base64.b64encode(self.vector.tobytes()).decode('utf-8')
    
    @staticmethod
    def from_base64(b64_str: str, dimension: int) -> np.ndarray:
        """Reconstruct from base64"""
        import base64
        bytes_data = base64.b64decode(b64_str)
        return np.frombuffer(bytes_data, dtype=np.float32).reshape(-1, dimension)

class CLIPEmbedder:
    """
    CLIP-based embeddings for semantic search and classification
    """
    
    def __init__(self, model_name: str = "openai/clip-vit-large-patch14"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load CLIP model
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Also load sentence-transformers version for flexibility
        self.sentence_model = SentenceTransformer('clip-ViT-L-14')
        
        # Embedding dimension
        self.dimension = self.model.config.projection_dim
        
        # Initialize FAISS index for similarity search
        self.index = None
        self.indexed_ids = []
        
    def encode_image(self, image: np.ndarray) -> ImageEmbedding:
        """
        Generate embedding for single image
        
        Args:
            image: RGB numpy array (H, W, C)
            
        Returns:
            ImageEmbedding object
        """
        # Process image
        inputs = self.processor(
            images=image, 
            return_tensors="pt"
        ).to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        embedding_vector = image_features.cpu().numpy().squeeze()
        
        return ImageEmbedding(
            vector=embedding_vector,
            model_name="clip-vit-large-patch14",
            dimension=self.dimension,
            metadata={"normalized": True}
        )
    
    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embedding for text query
        
        Args:
            text: Text string or list of strings
            
        Returns:
            Normalized embedding vector(s)
        """
        if isinstance(text, str):
            text = [text]
        
        inputs = self.processor(
            text=text, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            # Normalize
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy()
    
    def compute_similarity(
        self, 
        image_embedding: ImageEmbedding, 
        text_queries: List[str]
    ) -> Dict[str, float]:
        """
        Compute similarity between image and text queries
        
        Returns:
            Dictionary mapping query to similarity score
        """
        text_embeddings = self.encode_text(text_queries)
        image_vec = image_embedding.vector.reshape(1, -1)
        
        # Compute cosine similarity
        similarities = np.dot(text_embeddings, image_vec.T).squeeze()
        
        return {
            query: float(sim) 
            for query, sim in zip(text_queries, similarities)
        }
    
    def classify_image(
        self, 
        image: np.ndarray, 
        categories: List[str],
        template: str = "a photo of {}"
    ) -> Dict[str, float]:
        """
        Zero-shot image classification
        
        Args:
            image: Input image
            categories: List of category names
            template: Template for text prompts
            
        Returns:
            Dictionary of category probabilities
        """
        # Generate prompts
        prompts = [template.format(cat) for cat in categories]
        
        # Get embeddings
        image_embedding = self.encode_image(image)
        text_embeddings = self.encode_text(prompts)
        
        # Compute similarities
        image_vec = image_embedding.vector.reshape(1, -1)
        logits = np.dot(text_embeddings, image_vec.T).squeeze()
        
        # Convert to probabilities
        probs = np.exp(logits) / np.sum(np.exp(logits))
        
        return {cat: float(prob) for cat, prob in zip(categories, probs)}
    
    def build_search_index(self, embeddings: List[ImageEmbedding], ids: List[str]):
        """
        Build FAISS index for efficient similarity search
        """
        # Stack embeddings
        vectors = np.vstack([emb.vector for emb in embeddings])
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        self.index.add(vectors)
        self.indexed_ids = ids
    
    def search_similar(
        self, 
        query_embedding: Union[ImageEmbedding, np.ndarray], 
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Search for similar images
        
        Returns:
            List of (image_id, similarity_score) tuples
        """
        if self.index is None:
            raise ValueError("Search index not built")
        
        if isinstance(query_embedding, ImageEmbedding):
            query_vec = query_embedding.vector
        else:
            query_vec = query_embedding
        
        query_vec = query_vec.reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_vec, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.indexed_ids):
                results.append((self.indexed_ids[idx], float(dist)))
        
        return results
```

## 4. NIMA Aesthetic Quality Scoring

### Neural Image Assessment Implementation

```python
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple

@dataclass
class QualityScores:
    """Standardized quality assessment output"""
    aesthetic_score: float  # Overall aesthetic quality (1-10)
    technical_score: float  # Technical quality (1-10)
    distribution: np.ndarray  # Full score distribution
    confidence: float
    details: Dict[str, float]  # Component scores
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "aesthetic_score": self.aesthetic_score,
            "technical_score": self.technical_score,
            "confidence": self.confidence,
            "details": self.details,
            "distribution": self.distribution.tolist()
        }
    
    def get_recommendation(self) -> str:
        """Get processing recommendation based on scores"""
        if self.aesthetic_score >= 7.5:
            return "excellent"
        elif self.aesthetic_score >= 6.0:
            return "good"
        elif self.aesthetic_score >= 4.5:
            return "moderate"
        else:
            return "needs_improvement"

class NIMAScorer:
    """
    NIMA-based aesthetic and technical quality scoring
    """
    
    def __init__(self, checkpoint_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load base model (MobileNet V2)
        base_model = models.mobilenet_v2(pretrained=True)
        
        # Modify for NIMA (10-class output for score distribution)
        self.model = nn.Sequential(
            base_model.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(1280, 10)  # 10 score buckets (1-10)
        )
        
        if checkpoint_path:
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Score buckets (1-10)
        self.score_range = np.arange(1, 11)
        
    def score_image(self, image: np.ndarray) -> QualityScores:
        """
        Calculate aesthetic and technical quality scores
        
        Args:
            image: RGB numpy array (H, W, C)
            
        Returns:
            QualityScores object
        """
        # Convert to PIL
        pil_image = Image.fromarray(image)
        
        # Preprocess
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
        
        # Calculate mean score
        probs_np = probs.cpu().numpy()[0]
        mean_score = np.sum(probs_np * self.score_range)
        
        # Calculate standard deviation (confidence)
        std = np.sqrt(np.sum(probs_np * (self.score_range - mean_score) ** 2))
        confidence = 1.0 - (std / 3.0)  # Normalize std to confidence
        
        # Analyze technical aspects
        technical_scores = self._analyze_technical_quality(image)
        
        # Combine aesthetic and technical
        technical_score = np.mean(list(technical_scores.values()))
        
        return QualityScores(
            aesthetic_score=float(mean_score),
            technical_score=float(technical_score),
            distribution=probs_np,
            confidence=float(confidence),
            details=technical_scores
        )
    
    def _analyze_technical_quality(self, image: np.ndarray) -> Dict[str, float]:
        """
        Analyze technical aspects of the image
        """
        scores = {}
        
        # Sharpness detection
        scores['sharpness'] = self._calculate_sharpness(image)
        
        # Exposure analysis
        scores['exposure'] = self._analyze_exposure(image)
        
        # Color balance
        scores['color_balance'] = self._analyze_color_balance(image)
        
        # Noise level
        scores['noise_level'] = self._estimate_noise(image)
        
        # Dynamic range
        scores['dynamic_range'] = self._calculate_dynamic_range(image)
        
        return scores
    
    def _calculate_sharpness(self, image: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize to 1-10 scale
        sharpness = np.clip(laplacian_var / 1000, 0, 10)
        return float(sharpness)
    
    def _analyze_exposure(self, image: np.ndarray) -> float:
        """Analyze exposure quality"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        
        # Calculate histogram
        hist, _ = np.histogram(l_channel, bins=256, range=(0, 255))
        hist = hist / hist.sum()
        
        # Check for clipping
        underexposed = hist[:10].sum()
        overexposed = hist[-10:].sum()
        
        # Calculate score
        clipping_penalty = (underexposed + overexposed) * 5
        exposure_score = 10 - clipping_penalty
        
        return float(np.clip(exposure_score, 1, 10))
    
    def _analyze_color_balance(self, image: np.ndarray) -> float:
        """Analyze color balance and saturation"""
        # Calculate mean and std for each channel
        means = []
        stds = []
        
        for channel in range(3):
            channel_data = image[:, :, channel].flatten()
            means.append(np.mean(channel_data))
            stds.append(np.std(channel_data))
        
        # Check color cast
        mean_diff = np.std(means) / np.mean(means)
        color_cast_penalty = mean_diff * 10
        
        # Check saturation
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1].mean() / 255
        
        # Optimal saturation around 0.4-0.6
        if 0.4 <= saturation <= 0.6:
            saturation_score = 10
        else:
            saturation_score = 10 - abs(saturation - 0.5) * 10
        
        # Combine scores
        color_score = (10 - color_cast_penalty + saturation_score) / 2
        
        return float(np.clip(color_score, 1, 10))
    
    def _estimate_noise(self, image: np.ndarray) -> float:
        """Estimate image noise level"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate noise using median absolute deviation
        median = np.median(gray)
        mad = np.median(np.abs(gray - median))
        noise_estimate = mad * 1.4826  # Scale factor for Gaussian noise
        
        # Convert to score (lower noise = higher score)
        noise_score = 10 - (noise_estimate / 10)
        
        return float(np.clip(noise_score, 1, 10))
    
    def _calculate_dynamic_range(self, image: np.ndarray) -> float:
        """Calculate dynamic range score"""
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        
        # Calculate percentiles
        p5 = np.percentile(l_channel, 5)
        p95 = np.percentile(l_channel, 95)
        
        # Dynamic range
        dr = (p95 - p5) / 255
        
        # Score based on dynamic range
        dr_score = dr * 10
        
        return float(np.clip(dr_score, 1, 10))
```

## 5. Qwen2.5-VL Visual Language Model Integration

### Advanced Visual Understanding and Analysis

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
import torch
from PIL import Image
import json
from typing import Dict, Any, List, Optional

@dataclass
class VLMAnalysis:
    """Comprehensive visual-language model analysis"""
    composition: Dict[str, Any]
    technical_assessment: Dict[str, Any]
    content_description: str
    suggested_edits: List[Dict[str, Any]]
    style_attributes: Dict[str, Any]
    narrative: str
    confidence_scores: Dict[str, float]
    
    def get_primary_suggestions(self) -> List[Dict[str, Any]]:
        """Get high-confidence editing suggestions"""
        return [
            edit for edit in self.suggested_edits 
            if edit.get('confidence', 0) > 0.7
        ]
    
    def to_recipe_operations(self) -> List[ProcessingOperation]:
        """Convert suggestions to processing operations"""
        operations = []
        
        for edit in self.suggested_edits:
            if edit['type'] == 'crop':
                op = ProcessingOperation(
                    type='crop',
                    parameters=edit['parameters'],
                    timestamp=datetime.now(),
                    source='ai_vlm'
                )
                operations.append(op)
            elif edit['type'] == 'color_correction':
                op = ProcessingOperation(
                    type='enhance',
                    parameters=edit['parameters'],
                    timestamp=datetime.now(),
                    source='ai_vlm'
                )
                operations.append(op)
        
        return operations

class QwenVLAnalyzer:
    """
    Qwen2.5-VL for comprehensive image analysis
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-VL"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and processors
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Analysis prompts
        self.analysis_prompts = {
            'composition': """Analyze the composition of this image:
            1. Rule of thirds alignment
            2. Leading lines and patterns
            3. Balance and symmetry
            4. Focal points and visual hierarchy
            5. Suggested crop coordinates (normalized 0-1)
            Provide specific coordinates and measurements.""",
            
            'technical': """Assess the technical quality:
            1. Focus and sharpness issues
            2. Exposure problems (over/under)
            3. Color cast or white balance issues
            4. Noise or artifacts
            5. Specific correction parameters needed
            Provide numerical adjustments where applicable.""",
            
            'content': """Describe the image content:
            1. Main subjects and their positions
            2. Background elements
            3. Scene type and context
            4. Mood and atmosphere
            5. Notable details or points of interest""",
            
            'style': """Identify style attributes:
            1. Photography style (portrait, landscape, street, etc.)
            2. Lighting characteristics
            3. Color palette and mood
            4. Artistic techniques used
            5. Similar photographer or style references"""
        }
    
    def analyze_comprehensive(
        self, 
        image: np.ndarray,
        include_detections: Optional[List[BoundingBox]] = None,
        include_quality: Optional[QualityScores] = None
    ) -> VLMAnalysis:
        """
        Perform comprehensive analysis using VLM
        
        Args:
            image: Input image
            include_detections: Object detections to consider
            include_quality: Quality scores to consider
            
        Returns:
            Comprehensive VLM analysis
        """
        # Convert to PIL
        pil_image = Image.fromarray(image)
        
        # Run all analysis types
        results = {}
        confidence_scores = {}
        
        for analysis_type, prompt in self.analysis_prompts.items():
            # Add context from other models if available
            enhanced_prompt = self._enhance_prompt(
                prompt, 
                include_detections, 
                include_quality
            )
            
            # Get analysis
            response = self._query_model(pil_image, enhanced_prompt)
            parsed = self._parse_response(response, analysis_type)
            
            results[analysis_type] = parsed['data']
            confidence_scores[analysis_type] = parsed['confidence']
        
        # Extract specific components
        composition = results.get('composition', {})
        technical = results.get('technical', {})
        content = results.get('content', {})
        style = results.get('style', {})
        
        # Generate editing suggestions
        suggested_edits = self._generate_edit_suggestions(
            composition, 
            technical, 
            include_quality
        )
        
        # Create narrative description
        narrative = self._generate_narrative(content, style)
        
        return VLMAnalysis(
            composition=composition,
            technical_assessment=technical,
            content_description=content.get('description', ''),
            suggested_edits=suggested_edits,
            style_attributes=style,
            narrative=narrative,
            confidence_scores=confidence_scores
        )
    
    def _query_model(self, image: Image.Image, prompt: str) -> str:
        """Query the VLM with image and prompt"""
        # Prepare inputs
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Tokenize
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt"
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True
            )
        
        # Decode response
        response = self.processor.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return response
    
    def _enhance_prompt(
        self, 
        base_prompt: str,
        detections: Optional[List[BoundingBox]],
        quality: Optional[QualityScores]
    ) -> str:
        """Add context from other models to prompt"""
        enhanced = base_prompt
        
        if detections:
            objects = [f"{d.class_name} at ({d.x1:.2f}, {d.y1:.2f})" 
                      for d in detections[:5]]
            enhanced += f"\n\nDetected objects: {', '.join(objects)}"
        
        if quality:
            enhanced += f"\n\nQuality scores: aesthetic={quality.aesthetic_score:.1f}, "
            enhanced += f"technical={quality.technical_score:.1f}"
        
        return enhanced
    
    def _parse_response(self, response: str, analysis_type: str) -> Dict[str, Any]:
        """Parse VLM response into structured data"""
        try:
            # Try to extract JSON if present
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                confidence = 0.9
            else:
                # Parse text response
                data = self._parse_text_response(response, analysis_type)
                confidence = 0.7
        except:
            # Fallback
            data = {"raw_response": response}
            confidence = 0.5
        
        return {"data": data, "confidence": confidence}
    
    def _parse_text_response(self, response: str, analysis_type: str) -> Dict[str, Any]:
        """Parse unstructured text response"""
        lines = response.strip().split('\n')
        parsed = {}
        
        if analysis_type == 'composition':
            # Look for crop suggestions
            for line in lines:
                if 'crop' in line.lower():
                    # Extract coordinates
                    numbers = re.findall(r'0\.\d+', line)
                    if len(numbers) >= 4:
                        parsed['suggested_crop'] = {
                            'x1': float(numbers[0]),
                            'y1': float(numbers[1]),
                            'x2': float(numbers[2]),
                            'y2': float(numbers[3])
                        }
        
        # Add more parsing logic as needed
        parsed['raw_text'] = response
        return parsed
    
    def _generate_edit_suggestions(
        self,
        composition: Dict[str, Any],
        technical: Dict[str, Any],
        quality: Optional[QualityScores]
    ) -> List[Dict[str, Any]]:
        """Generate concrete editing suggestions"""
        suggestions = []
        
        # Crop suggestion
        if 'suggested_crop' in composition:
            suggestions.append({
                'type': 'crop',
                'parameters': composition['suggested_crop'],
                'reason': 'Improve composition',
                'confidence': 0.8
            })
        
        # Color corrections
        if quality and quality.details.get('exposure', 10) < 5:
            suggestions.append({
                'type': 'color_correction',
                'parameters': {
                    'exposure': 0.5,
                    'highlights': -0.3,
                    'shadows': 0.2
                },
                'reason': 'Fix exposure issues',
                'confidence': 0.9
            })
        
        return suggestions
    
    def _generate_narrative(
        self, 
        content: Dict[str, Any], 
        style: Dict[str, Any]
    ) -> str:
        """Generate a narrative description"""
        description = content.get('description', '')
        style_desc = style.get('description', '')
        
        narrative = f"{description} "
        if style_desc:
            narrative += f"The image exhibits {style_desc}"
        
        return narrative.strip()
```

## 6. Data Translation Layer

### Unified Data Flow Between Models

```python
from typing import Any, Dict, List, Optional, Union
import numpy as np

class ModelDataTranslator:
    """
    Handles data translation between different AI models
    """
    
    def __init__(self):
        self.translation_cache = {}
        
    def detection_to_segmentation(
        self, 
        detections: List[BoundingBox], 
        image_shape: Tuple[int, int]
    ) -> List[Dict[str, Any]]:
        """
        Convert RT-DETR detections to SAM2 prompts
        """
        sam_prompts = []
        
        for detection in detections:
            # Convert to absolute coordinates
            height, width = image_shape
            x1, y1, x2, y2 = detection.to_absolute(width, height)
            
            prompt = {
                'type': 'box',
                'coordinates': [x1, y1, x2, y2],
                'label': detection.class_name,
                'confidence': detection.confidence,
                'metadata': {
                    'source': 'rt-detr',
                    'class_id': detection.class_id
                }
            }
            sam_prompts.append(prompt)
        
        return sam_prompts
    
    def segmentation_to_vlm(
        self,
        segments: List[SegmentationMask],
        image: np.ndarray
    ) -> Dict[str, Any]:
        """
        Prepare segmentation data for VLM analysis
        """
        # Create visualization with masks
        viz_image = image.copy()
        mask_info = []
        
        for i, segment in enumerate(segments):
            # Overlay mask with transparency
            mask_color = self._get_mask_color(i)
            mask_overlay = np.zeros_like(image)
            mask_overlay[segment.mask > 0] = mask_color
            
            viz_image = cv2.addWeighted(viz_image, 0.7, mask_overlay, 0.3, 0)
            
            # Extract mask statistics
            mask_info.append({
                'id': i,
                'label': segment.bbox.class_name if segment.bbox else 'unknown',
                'area_ratio': segment.area,
                'center': self._get_mask_center(segment.mask),
                'bbox': segment.bbox.to_dict() if segment.bbox else None
            })
        
        return {
            'visualization': viz_image,
            'mask_info': mask_info,
            'num_objects': len(segments)
        }
    
    def quality_to_processing(
        self,
        quality: QualityScores,
        vlm_analysis: VLMAnalysis
    ) -> List[ProcessingOperation]:
        """
        Convert quality scores and VLM analysis to processing operations
        """
        operations = []
        
        # Base adjustments from quality scores
        if quality.details['exposure'] < 5:
            operations.append(ProcessingOperation(
                type='adjust_exposure',
                parameters={
                    'ev_adjustment': (5 - quality.details['exposure']) * 0.3
                },
                timestamp=datetime.now(),
                source='quality_analysis'
            ))
        
        if quality.details['color_balance'] < 6:
            operations.append(ProcessingOperation(
                type='color_correction',
                parameters={
                    'auto_white_balance': True,
                    'vibrance': 0.1
                },
                timestamp=datetime.now(),
                source='quality_analysis'
            ))
        
        # Add VLM suggestions
        operations.extend(vlm_analysis.to_recipe_operations())
        
        return operations
    
    def embeddings_to_search_metadata(
        self,
        embedding: ImageEmbedding,
        detections: List[BoundingBox],
        quality: QualityScores,
        vlm_analysis: VLMAnalysis
    ) -> Dict[str, Any]:
        """
        Combine all analysis into searchable metadata
        """
        # Extract object labels
        objects = [d.class_name for d in detections]
        
        # Generate search tags
        tags = []
        tags.extend(objects)
        tags.extend(vlm_analysis.style_attributes.get('tags', []))
        
        if quality.aesthetic_score > 7:
            tags.append('high_quality')
        
        # Create metadata
        metadata = {
            'embedding_vector': embedding.to_base64(),
            'embedding_model': embedding.model_name,
            'objects': objects,
            'tags': list(set(tags)),
            'quality_scores': quality.to_dict(),
            'description': vlm_analysis.content_description,
            'style': vlm_analysis.style_attributes,
            'technical_details': vlm_analysis.technical_assessment
        }
        
        return metadata
    
    def _get_mask_color(self, index: int) -> Tuple[int, int, int]:
        """Generate distinct colors for masks"""
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
        ]
        return colors[index % len(colors)]
    
    def _get_mask_center(self, mask: np.ndarray) -> Tuple[float, float]:
        """Calculate mask centroid"""
        y_indices, x_indices = np.where(mask)
        if len(x_indices) > 0:
            center_x = np.mean(x_indices) / mask.shape[1]
            center_y = np.mean(y_indices) / mask.shape[0]
            return (float(center_x), float(center_y))
        return (0.5, 0.5)
```

## 7. Pipeline Orchestrator

### Coordinating the Complete AI Pipeline

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

class AIProcessingPipeline:
    """
    Orchestrates the complete AI processing pipeline
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Initialize all models
        self.detector = RTDETRDetector()
        self.segmenter = SAM2Segmenter()
        self.embedder = CLIPEmbedder()
        self.quality_scorer = NIMAScorer()
        self.vlm_analyzer = QwenVLAnalyzer()
        
        # Data translator
        self.translator = ModelDataTranslator()
        
        # Threading for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Pipeline configuration
        self.config = config
        
    async def process_image(
        self, 
        image_path: Path,
        processing_mode: str = 'auto'
    ) -> ProcessingResult:
        """
        Run complete AI pipeline on image
        
        Args:
            image_path: Path to input image
            processing_mode: 'auto', 'semi-auto', or 'manual'
            
        Returns:
            Complete processing result with all AI outputs
        """
        start_time = time.time()
        
        # Load image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Stage 1: Object Detection (RT-DETR)
        detection_future = self.executor.submit(
            self.detector.detect, image
        )
        
        # Stage 2: Quality Assessment (NIMA) - Can run in parallel
        quality_future = self.executor.submit(
            self.quality_scorer.score_image, image
        )
        
        # Stage 3: Generate Embeddings (CLIP) - Can run in parallel
        embedding_future = self.executor.submit(
            self.embedder.encode_image, image
        )
        
        # Wait for detection results
        detections = detection_future.result()
        
        # Stage 4: Segmentation (SAM2) - Depends on detections
        sam_prompts = self.translator.detection_to_segmentation(
            detections, image.shape[:2]
        )
        segments = await self._segment_objects_async(image, detections)
        
        # Stage 5: VLM Analysis - Can use all previous results
        quality_scores = quality_future.result()
        vlm_analysis = await self._analyze_with_vlm_async(
            image, detections, quality_scores
        )
        
        # Stage 6: Generate Processing Recipe
        operations = self.translator.quality_to_processing(
            quality_scores, vlm_analysis
        )
        
        # Stage 7: Prepare Search Metadata
        embedding = embedding_future.result()
        search_metadata = self.translator.embeddings_to_search_metadata(
            embedding, detections, quality_scores, vlm_analysis
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        return ProcessingResult(
            image_path=image_path,
            detections=detections,
            segments=segments,
            quality_scores=quality_scores,
            embedding=embedding,
            vlm_analysis=vlm_analysis,
            suggested_operations=operations,
            search_metadata=search_metadata,
            processing_time=processing_time,
            mode=processing_mode
        )
    
    async def _segment_objects_async(
        self, 
        image: np.ndarray, 
        detections: List[BoundingBox]
    ) -> List[SegmentationMask]:
        """Async wrapper for segmentation"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.segmenter.segment_with_boxes,
            image,
            detections
        )
    
    async def _analyze_with_vlm_async(
        self,
        image: np.ndarray,
        detections: List[BoundingBox],
        quality: QualityScores
    ) -> VLMAnalysis:
        """Async wrapper for VLM analysis"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.vlm_analyzer.analyze_comprehensive,
            image,
            detections,
            quality
        )
    
    def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)

@dataclass
class ProcessingResult:
    """Complete result from AI pipeline"""
    image_path: Path
    detections: List[BoundingBox]
    segments: List[SegmentationMask]
    quality_scores: QualityScores
    embedding: ImageEmbedding
    vlm_analysis: VLMAnalysis
    suggested_operations: List[ProcessingOperation]
    search_metadata: Dict[str, Any]
    processing_time: float
    mode: str
    
    def to_recipe(self) -> ProcessingRecipe:
        """Convert to processing recipe"""
        return ProcessingRecipe(
            original_id=str(self.image_path),
            timestamp=datetime.now(),
            operations=self.suggested_operations,
            ai_metadata={
                'detections': [d.__dict__ for d in self.detections],
                'quality': self.quality_scores.to_dict(),
                'vlm_analysis': self.vlm_analysis.__dict__
            },
            user_overrides={}
        )
```

## Model Deployment Considerations

### 1. GPU Memory Management
```python
# Optimize memory usage by loading models on demand
class ModelLoader:
    def __init__(self):
        self.loaded_models = {}
        
    def get_model(self, model_type: str):
        if model_type not in self.loaded_models:
            if model_type == 'detector':
                self.loaded_models[model_type] = RTDETRDetector()
            # ... load other models
        return self.loaded_models[model_type]
    
    def unload_model(self, model_type: str):
        if model_type in self.loaded_models:
            del self.loaded_models[model_type]
            torch.cuda.empty_cache()
```

### 2. Batch Processing
```python
# Process multiple images efficiently
async def batch_process(images: List[Path], pipeline: AIProcessingPipeline):
    tasks = []
    for image_path in images:
        task = pipeline.process_image(image_path)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return results
```

### 3. Error Recovery
```python
# Graceful degradation when models fail
class ResilientPipeline(AIProcessingPipeline):
    async def process_image(self, image_path: Path, **kwargs):
        try:
            return await super().process_image(image_path, **kwargs)
        except Exception as e:
            # Fallback to basic processing
            return self.basic_process(image_path, error=str(e))
```

This comprehensive implementation provides a complete AI processing pipeline with proper data translation between models, allowing each component to work independently while maintaining a cohesive workflow.
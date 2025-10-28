"""
Image preprocessing pipeline for OCR optimization.
"""
import os
from typing import Tuple, Optional, List
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from scipy import ndimage
import tempfile

from ..core.schemas import PageImage


class ImagePreprocessor:
    """
    Image preprocessing pipeline to optimize images for OCR.
    """
    
    def __init__(self, target_dpi: int = 300):
        """
        Initialize image preprocessor.
        
        Args:
            target_dpi: Target DPI for processed images
        """
        self.target_dpi = target_dpi
    
    async def preprocess_image(
        self, 
        input_path: str, 
        page: int = 1,
        operations: Optional[List[str]] = None
    ) -> PageImage:
        """
        Apply full preprocessing pipeline to an image.
        
        Args:
            input_path: Path to input image
            page: Page number
            operations: List of operations to apply (None = all)
            
        Returns:
            PageImage with processed image path and metadata
        """
        if operations is None:
            operations = [
                "resize_to_target_dpi",
                "convert_to_grayscale", 
                "deskew",
                "denoise",
                "enhance_contrast",
                "binarize"
            ]
        
        # Load image
        image = self._load_image(input_path)
        original_height, original_width = image.shape[:2]
        
        # Apply preprocessing operations
        processed_image = image.copy()
        
        for operation in operations:
            if hasattr(self, f"_{operation}"):
                processed_image = getattr(self, f"_{operation}")(processed_image)
        
        # Save processed image
        output_path = self._save_processed_image(processed_image, page)
        
        # Get final dimensions
        final_height, final_width = processed_image.shape[:2]
        
        return PageImage(
            page=page,
            dpi=self.target_dpi,
            width=final_width,
            height=final_height,
            file_path=output_path
        )
    
    def _load_image(self, file_path: str) -> np.ndarray:
        """Load image from file path."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")
        
        # Load with OpenCV
        image = cv2.imread(file_path)
        if image is None:
            # Fallback to PIL
            pil_image = Image.open(file_path)
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return image
    
    def _resize_to_target_dpi(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target DPI.
        
        For OCR, 300 DPI is typically optimal. Higher DPI can improve
        accuracy for small text but increases processing time.
        """
        height, width = image.shape[:2]
        
        # Estimate current DPI (assuming typical scan is 150-600 DPI)
        # This is a heuristic - in practice, DPI would be from image metadata
        if max(width, height) > 3000:
            current_dpi = 600
        elif max(width, height) > 2000:
            current_dpi = 300  
        else:
            current_dpi = 150
        
        # Calculate scaling factor
        scale_factor = self.target_dpi / current_dpi
        
        if abs(scale_factor - 1.0) > 0.1:  # Only resize if significant difference
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            # Use appropriate interpolation
            interpolation = cv2.INTER_CUBIC if scale_factor > 1 else cv2.INTER_AREA
            image = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
        
        return image
    
    def _convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """
        Automatically detect and correct skew in the image.
        
        Uses Hough line transform to detect dominant lines and calculate skew angle.
        """
        # Ensure grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Hough line transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None:
            # Calculate angles of detected lines
            angles = []
            for rho, theta in lines[:, 0]:
                angle = theta * 180 / np.pi
                # Convert to skew angle (-90 to 90 degrees)
                if angle > 90:
                    angle = angle - 180
                angles.append(angle)
            
            # Find most common angle (mode)
            if angles:
                # Use median angle to avoid outliers
                skew_angle = np.median(angles)
                
                # Only correct if skew is significant (> 0.5 degrees)
                if abs(skew_angle) > 0.5:
                    # Rotate image to correct skew
                    height, width = image.shape[:2]
                    center = (width // 2, height // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
                    
                    # Calculate new dimensions to avoid cropping
                    cos_angle = abs(rotation_matrix[0, 0])
                    sin_angle = abs(rotation_matrix[0, 1])
                    new_width = int((height * sin_angle) + (width * cos_angle))
                    new_height = int((height * cos_angle) + (width * sin_angle))
                    
                    # Adjust translation
                    rotation_matrix[0, 2] += (new_width / 2) - center[0]
                    rotation_matrix[1, 2] += (new_height / 2) - center[1]
                    
                    # Apply rotation
                    image = cv2.warpAffine(
                        image, 
                        rotation_matrix, 
                        (new_width, new_height),
                        flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(255, 255, 255)  # White background
                    )
        
        return image
    
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Remove noise from the image using various denoising techniques.
        """
        # Ensure grayscale for denoising
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Non-local Means Denoising
        denoised = cv2.fastNlMeansDenoising(
            gray,
            h=10,  # Strength of denoising
            templateWindowSize=7,  # Size of template patch
            searchWindowSize=21   # Size of search window
        )
        
        # Apply median filter to remove salt-and-pepper noise
        denoised = cv2.medianBlur(denoised, 3)
        
        # Convert back to original format if needed
        if len(image.shape) == 3:
            denoised = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
        
        return denoised
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast using adaptive histogram equalization.
        """
        # Ensure grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(
            clipLimit=2.0,    # Contrast limit
            tileGridSize=(8, 8)  # Size of grid for local equalization
        )
        enhanced = clahe.apply(gray)
        
        # Convert back to original format if needed
        if len(image.shape) == 3:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        return enhanced
    
    def _binarize(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to binary (black and white) using adaptive thresholding.
        """
        # Ensure grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray,
            255,                          # Maximum value
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Adaptive method
            cv2.THRESH_BINARY,            # Threshold type
            11,                           # Block size for neighborhood
            2                             # Constant subtracted from mean
        )
        
        return binary
    
    def _save_processed_image(self, image: np.ndarray, page: int) -> str:
        """Save processed image to temporary file."""
        # Create temp file
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, f"processed_page_{page}.png")
        
        # Save image
        cv2.imwrite(output_path, image)
        
        return output_path
    
    def detect_orientation(self, image: np.ndarray) -> float:
        """
        Detect image orientation using OCR.
        
        Returns rotation angle needed to correct orientation.
        """
        try:
            import pytesseract
            
            # Try different rotations and measure confidence
            best_angle = 0
            best_confidence = 0
            
            for angle in [0, 90, 180, 270]:
                # Rotate image
                if angle == 0:
                    rotated = image
                else:
                    height, width = image.shape[:2]
                    center = (width // 2, height // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
                
                # Get OCR confidence
                try:
                    data = pytesseract.image_to_data(
                        rotated, 
                        output_type=pytesseract.Output.DICT
                    )
                    
                    # Calculate average confidence
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    if confidences:
                        avg_confidence = sum(confidences) / len(confidences)
                        if avg_confidence > best_confidence:
                            best_confidence = avg_confidence
                            best_angle = angle
                
                except Exception:
                    continue
            
            return best_angle
            
        except ImportError:
            # Fallback: no rotation
            return 0
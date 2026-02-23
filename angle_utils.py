"""
Angle calculation utilities for upper body joint analysis.
Provides deterministic angle calculations for shoulders, elbows, and wrists.
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class Point3D:
    """3D point representation"""
    x: float
    y: float
    z: float
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])
    
    def distance_to(self, other: 'Point3D') -> float:
        """Calculate Euclidean distance between two points"""
        return np.linalg.norm(self.to_array() - other.to_array())


class AngleCalculator:
    """Calculates joint angles from 3D landmark positions"""
    
    @staticmethod
    def calculate_angle(point1: Point3D, point2: Point3D, point3: Point3D) -> float:
        """
        Calculate the angle at point2 formed by points point1-point2-point3.
        Returns angle in degrees.
        
        Args:
            point1: First point in the angle
            point2: Vertex point (joint being measured)
            point3: Third point in the angle
            
        Returns:
            Angle in degrees (0-180)
        """
        try:
            # Convert to numpy arrays
            v1 = point1.to_array() - point2.to_array()
            v2 = point3.to_array() - point2.to_array()
            
            # Calculate norm with zero check
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 < 1e-10 or norm2 < 1e-10:
                return 0.0
            
            # Calculate cosine of the angle
            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            
            # Clamp to avoid numerical errors
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            
            # Convert to degrees
            angle_rad = np.arccos(cos_angle)
            angle_deg = np.degrees(angle_rad)
            
            # For joint angles, we want the smaller angle (0-180 range)
            # This handles cases where vectors point in opposite directions
            joint_angle = abs(180.0 - angle_deg)
            
            return float(joint_angle)
        except Exception:
            return 0.0
    
    @staticmethod
    def calculate_shoulder_flexion(
        shoulder: Point3D, 
        elbow: Point3D, 
        wrist: Point3D
    ) -> float:
        """
        Calculate shoulder flexion angle (forward raise).
        Normal range: 0-180 degrees
        Safe limit: 120 degrees
        """
        return AngleCalculator.calculate_angle(shoulder, elbow, wrist)
    
    @staticmethod
    def calculate_shoulder_extension(
        shoulder: Point3D,
        elbow: Point3D,
        wrist: Point3D
    ) -> float:
        """
        Calculate shoulder extension (backward movement).
        Normal range: 0-60 degrees
        Safe limit: 60 degrees
        """
        return AngleCalculator.calculate_angle(shoulder, elbow, wrist)
    
    @staticmethod
    def calculate_shoulder_abduction(
        shoulder: Point3D,
        elbow: Point3D,
        wrist: Point3D
    ) -> float:
        """
        Calculate shoulder abduction angle (side raise).
        Normal range: 0-180 degrees
        Safe limit: 150 degrees
        """
        return AngleCalculator.calculate_angle(shoulder, elbow, wrist)
    
    @staticmethod
    def calculate_elbow_flexion(
        shoulder: Point3D,
        elbow: Point3D,
        wrist: Point3D
    ) -> float:
        """
        Calculate elbow flexion angle.
        Normal range: 0-150 degrees
        Safe limit: 150 degrees
        """
        return AngleCalculator.calculate_angle(shoulder, elbow, wrist)
    
    @staticmethod
    def calculate_elbow_extension(
        shoulder: Point3D,
        elbow: Point3D,
        wrist: Point3D
    ) -> float:
        """
        Calculate elbow extension angle.
        Normal range: 0-10 degrees (hyperextension is negative)
        Safe limit: 10 degrees
        """
        return AngleCalculator.calculate_angle(shoulder, elbow, wrist)
    
    @staticmethod
    def calculate_wrist_flexion(
        elbow: Point3D,
        wrist: Point3D,
        middle_finger: Point3D
    ) -> float:
        """
        Calculate wrist flexion angle (bending palm toward forearm).
        Normal range: 0-80 degrees
        Safe limit: 80 degrees
        """
        return AngleCalculator.calculate_angle(elbow, wrist, middle_finger)
    
    @staticmethod
    def calculate_wrist_extension(
        elbow: Point3D,
        wrist: Point3D,
        middle_finger: Point3D
    ) -> float:
        """
        Calculate wrist extension angle (bending back of hand toward forearm).
        Normal range: 0-70 degrees
        Safe limit: 70 degrees
        """
        return AngleCalculator.calculate_angle(elbow, wrist, middle_finger)
    
    @staticmethod
    def extract_angles(landmarks: Dict[str, Point3D]) -> Dict[str, float]:
        """
        Extract all relevant upper body angles from landmarks.
        
        Args:
            landmarks: Dictionary of landmark names to Point3D objects
            
        Returns:
            Dictionary of angle names to angle values in degrees
        """
        angles = {}
        
        # Required landmarks for calculations
        required = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                   'left_wrist', 'right_wrist', 'left_index', 'right_index']
        
        # Check if required landmarks are present
        if not all(k in landmarks for k in required):
            return angles
        
        # Left side angles
        if all(k in landmarks for k in ['left_shoulder', 'left_elbow', 'left_wrist']):
            angles['left_shoulder_flexion'] = AngleCalculator.calculate_shoulder_flexion(
                landmarks['left_shoulder'], landmarks['left_elbow'], landmarks['left_wrist']
            )
            angles['left_shoulder_abduction'] = AngleCalculator.calculate_shoulder_abduction(
                landmarks['left_shoulder'], landmarks['left_elbow'], landmarks['left_wrist']
            )
            angles['left_elbow_flexion'] = AngleCalculator.calculate_elbow_flexion(
                landmarks['left_shoulder'], landmarks['left_elbow'], landmarks['left_wrist']
            )
            angles['left_elbow_extension'] = AngleCalculator.calculate_elbow_extension(
                landmarks['left_shoulder'], landmarks['left_elbow'], landmarks['left_wrist']
            )
        
        # Right side angles
        if all(k in landmarks for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
            angles['right_shoulder_flexion'] = AngleCalculator.calculate_shoulder_flexion(
                landmarks['right_shoulder'], landmarks['right_elbow'], landmarks['right_wrist']
            )
            angles['right_shoulder_abduction'] = AngleCalculator.calculate_shoulder_abduction(
                landmarks['right_shoulder'], landmarks['right_elbow'], landmarks['right_wrist']
            )
            angles['right_elbow_flexion'] = AngleCalculator.calculate_elbow_flexion(
                landmarks['right_shoulder'], landmarks['right_elbow'], landmarks['right_wrist']
            )
            angles['right_elbow_extension'] = AngleCalculator.calculate_elbow_extension(
                landmarks['right_shoulder'], landmarks['right_elbow'], landmarks['right_wrist']
            )
        
        # Wrist angles (using index finger as reference)
        if all(k in landmarks for k in ['left_elbow', 'left_wrist', 'left_index']):
            angles['left_wrist_flexion'] = AngleCalculator.calculate_wrist_flexion(
                landmarks['left_elbow'], landmarks['left_wrist'], landmarks['left_index']
            )
            angles['left_wrist_extension'] = AngleCalculator.calculate_wrist_extension(
                landmarks['left_elbow'], landmarks['left_wrist'], landmarks['left_index']
            )
        
        if all(k in landmarks for k in ['right_elbow', 'right_wrist', 'right_index']):
            angles['right_wrist_flexion'] = AngleCalculator.calculate_wrist_flexion(
                landmarks['right_elbow'], landmarks['right_wrist'], landmarks['right_index']
            )
            angles['right_wrist_extension'] = AngleCalculator.calculate_wrist_extension(
                landmarks['right_elbow'], landmarks['right_wrist'], landmarks['right_index']
            )
        
        return angles


class VectorOperations:
    """Utility class for vector operations used in angle calculations"""
    
    @staticmethod
    def normalize(vector: np.ndarray) -> np.ndarray:
        """Normalize a vector to unit length"""
        norm = np.linalg.norm(vector)
        if norm < 1e-10:
            return vector
        return vector / norm
    
    @staticmethod
    def cross_product(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Calculate cross product of two vectors"""
        return np.cross(v1, v2)
    
    @staticmethod
    def dot_product(v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate dot product of two vectors"""
        return float(np.dot(v1, v2))
    
    @staticmethod
    def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate angle between two vectors in degrees"""
        cos_angle = VectorOperations.dot_product(
            VectorOperations.normalize(v1),
            VectorOperations.normalize(v2)
        )
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))

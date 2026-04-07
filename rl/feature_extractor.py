"""
Feature extractor for converting observations to 30-dimensional feature vectors.
Handles patient vitals, context, complaint keywords, queue status, and step info.
"""

import os
import sys
import numpy as np
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Patient


class FeatureExtractor:
    """Extracts 30-dimensional feature vectors from environment observations."""
    
    def __init__(self):
        # Complaint keyword categories
        self.keyword_categories = {
            "cardiac": ["cardiac", "arrest", "chest", "heart", "mi"],
            "seizure": ["seizure", "epilepsy", "convulsion"],
            "headache": ["headache", "migraine", "thunderclap"],
            "chest": ["chest", "pressure", "pain", "radiating"],
            "breath": ["breath", "dyspnea", "shortness", "asthma"],
            "weakness": ["weakness", "stroke", "neuro", "slurred"],
            "bleed": ["bleed", "blood", "hemorrhage", "urine"],
            "pain": ["pain", "abdominal", "back"],
            "unconscious": ["unresponsive", "unconscious", "gcs"],
            "stroke": ["stroke", "cva", "weakness", "slurred"]
        }
        
        self.feature_names = (
            # Vitals (8 features)
            ["bp_systolic_norm", "bp_diastolic_norm", "heart_rate_norm", "respiratory_rate_norm",
             "spo2_norm", "temperature_norm", "gcs_norm", "pain_scale_norm"] +
            # Patient context (4 features)
            ["age_norm", "gender_norm", "wait_time_norm", "deteriorated_flag"] +
            # Complaint keywords (10 features)
            [f"kw_{cat}" for cat in self.keyword_categories.keys()] +
            # Queue status (5 features)
            ["pending_norm", "triaged_norm", "resus_available_norm", "acute_available_norm", "time_elapsed_norm"] +
            # Step info (3 features)
            ["step_progress_norm", "score_norm", "alert_active_flag"]
        )
    
    @staticmethod
    def _norm(value: float, low: float, high: float) -> float:
        """Normalize value to [0, 1] range with clipping."""
        if high - low == 0:
            return 0.0
        normalized = (value - low) / (high - low)
        return max(0.0, min(1.0, normalized))
    
    def _extract_vitals_features(self, vitals: Dict[str, Any]) -> np.ndarray:
        """Extract 8 normalized vital sign features."""
        features = np.zeros(8)
        
        features[0] = self._norm(vitals.get("bp_systolic", 120), 0, 200)
        features[1] = self._norm(vitals.get("bp_diastolic", 80), 0, 130)
        features[2] = self._norm(vitals.get("heart_rate", 70), 0, 200)
        features[3] = self._norm(vitals.get("respiratory_rate", 16), 0, 40)
        features[4] = self._norm(vitals.get("spo2", 98), 50, 100)
        features[5] = self._norm(vitals.get("temperature", 37.0), 34.0, 42.0)
        features[6] = self._norm(vitals.get("gcs", 15), 3, 15)
        features[7] = self._norm(vitals.get("pain_scale", 0), 0, 10)
        
        return features
    
    def _extract_context_features(self, patient: Dict[str, Any]) -> np.ndarray:
        """Extract 4 patient context features."""
        features = np.zeros(4)
        
        features[0] = self._norm(patient.get("age", 30), 0, 100)
        
        # Gender encoding: M=0, F=1, Other=0.5
        gender = patient.get("gender", "").lower()
        if gender == "f":
            features[1] = 1.0
        elif gender == "m":
            features[1] = 0.0
        else:
            features[1] = 0.5
        
        features[2] = self._norm(patient.get("wait_time_minutes", 0), 0, 60)
        features[3] = 1.0 if patient.get("deteriorated", False) else 0.0
        
        return features
    
    def _extract_keyword_features(self, complaint: str) -> np.ndarray:
        """Extract 10 binary keyword features from chief complaint."""
        features = np.zeros(10)
        complaint_lower = complaint.lower()
        
        for i, (category, keywords) in enumerate(self.keyword_categories.items()):
            for keyword in keywords:
                if keyword in complaint_lower:
                    features[i] = 1.0
                    break
        
        return features
    
    def _extract_queue_features(self, obs: Dict[str, Any]) -> np.ndarray:
        """Extract 5 queue status features."""
        features = np.zeros(5)
        
        pending = len(obs.get("pending_patients", []))
        triaged = len(obs.get("triaged_patients", []))
        beds = obs.get("bed_status", {})
        
        features[0] = self._norm(pending, 0, 15)
        features[1] = self._norm(triaged, 0, 15)
        features[2] = self._norm(beds.get("resuscitation_available", 0), 0, 3)
        features[3] = self._norm(beds.get("acute_care_available", 0), 0, 8)
        features[4] = self._norm(obs.get("time_elapsed_minutes", 0), 0, 200)
        
        return features
    
    def _extract_step_features(self, obs: Dict[str, Any]) -> np.ndarray:
        """Extract 3 step-related features."""
        features = np.zeros(3)
        
        max_steps = obs.get("max_steps", 40)
        current_step = obs.get("step", 0)
        
        features[0] = self._norm(current_step, 0, max_steps)
        features[1] = max(-1.0, min(1.0, obs.get("score_so_far", 0.0)))
        features[2] = 1.0 if obs.get("alerts", []) else 0.0
        
        return features
    
    def extract(self, obs_dict: Dict[str, Any]) -> np.ndarray:
        """Extract 30-dimensional feature vector from observation."""
        features = np.zeros(30)
        
        # Handle None patient gracefully
        patient = obs_dict.get("current_patient") or {}
        
        # Extract feature groups
        vitals_features = self._extract_vitals_features(patient.get("vitals", {}))
        context_features = self._extract_context_features(patient)
        keyword_features = self._extract_keyword_features(patient.get("chief_complaint", ""))
        queue_features = self._extract_queue_features(obs_dict)
        step_features = self._extract_step_features(obs_dict)
        
        # Concatenate all features
        features = np.concatenate([
            vitals_features,      # 8 features
            context_features,     # 4 features
            keyword_features,     # 10 features
            queue_features,       # 5 features
            step_features         # 3 features
        ])
        
        assert features.shape == (30,), f"Feature vector shape is {features.shape}, expected (30,)"
        return features.astype(np.float32)
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names for debugging."""
        return self.feature_names.copy()


if __name__ == "__main__":
    # Test the feature extractor
    extractor = FeatureExtractor()
    
    # Create sample observation
    sample_obs = {
        "task_id": 1,
        "step": 5,
        "max_steps": 20,
        "current_patient": {
            "patient_id": "P001",
            "name": "Test Patient",
            "age": 45,
            "gender": "M",
            "chief_complaint": "chest pain radiating to arm",
            "vitals": {
                "bp_systolic": 140,
                "bp_diastolic": 90,
                "heart_rate": 95,
                "respiratory_rate": 18,
                "spo2": 96,
                "temperature": 37.2,
                "gcs": 15,
                "pain_scale": 7
            },
            "medical_history": ["hypertension"],
            "current_medications": ["lisinopril"],
            "wait_time_minutes": 10,
            "deteriorated": False
        },
        "pending_patients": ["P001", "P002"],
        "triaged_patients": ["P003"],
        "bed_status": {
            "total_beds": 10,
            "available_beds": 7,
            "resuscitation_bays": 2,
            "resuscitation_available": 1,
            "acute_care_beds": 8,
            "acute_care_available": 6
        },
        "time_elapsed_minutes": 25,
        "alerts": [],
        "task_description": "Test task",
        "score_so_far": 0.15
    }
    
    # Extract features
    features = extractor.extract(sample_obs)
    
    print("Feature Extractor Test:")
    print(f"Feature vector shape: {features.shape}")
    print(f"Feature vector dtype: {features.dtype}")
    print(f"Feature range: [{features.min():.3f}, {features.max():.3f}]")
    
    print("\nFeature names and values:")
    for name, value in zip(extractor.get_feature_names(), features):
        print(f"{name:20s}: {value:6.3f}")
    
    print(f"\n✅ Feature extractor test passed!")

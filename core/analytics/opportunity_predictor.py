# core/analytics/opportunity_predictor.py
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import logging

class OpportunityPredictor:
    """Predict win probability for contract opportunities"""
    
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.logger = logging.getLogger("opportunity_predictor")
    
    def _prepare_features(self, opportunities: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert opportunity data to features for ML model"""
        # Extract relevant features
        data = []
        for opp in opportunities:
            features = {
                # Basic contract features
                'contract_value': opp.get('value', 0),
                'duration_days': self._calculate_duration(opp),
                
                # Entity counts
                'tech_count': len(opp.get('entity_summary', {}).get('tech_stack', [])),
                'reg_count': len(opp.get('entity_summary', {}).get('regulatory_requirements', [])),
                'clearance_count': len(opp.get('entity_summary', {}).get('clearance_requirements', [])),
                
                # Boolean features
                'requires_clearance': int(len(opp.get('entity_summary', {}).get('clearance_requirements', [])) > 0),
                'is_defense_agency': int('Department of Defense' in str(opp.get('agency_name', ''))),
                
                # Target variable - 1 if won, 0 if lost
                'won': 1 if opp.get('outcome') == 'won' else 0
            }
            
            # Add specific tech requirements as one-hot features
            tech_stack = opp.get('entity_summary', {}).get('tech_stack', [])
            for tech in ['cloud', 'cybersecurity', 'ai', 'ml', 'blockchain', 'iot']:
                features[f'tech_{tech}'] = int(any(tech in t.lower() for t in tech_stack))
            
            data.append(features)
        
        return pd.DataFrame(data)
    
    def _calculate_duration(self, opportunity: Dict[str, Any]) -> int:
        """Calculate contract duration in days"""
        # In a real implementation, extract start and end dates
        # For now, return a random value for demonstration
        return np.random.randint(30, 1825)  # 1 month to 5 years
    
    def train(self, historical_opportunities: List[Dict[str, Any]]):
        """Train the win probability prediction model"""
        # Prepare features
        df = self._prepare_features(historical_opportunities)
        
        if df.empty or 'won' not in df.columns:
            self.logger.error("No valid training data available")
            return False
        
        # Split features and target
        X = df.drop('won', axis=1)
        y = df['won']
        
        self.feature_columns = X.columns
        
        # Split into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred)
        self.logger.info(f"Model trained with performance:\n{report}")
        
        return True
    
    def predict_win_probability(self, opportunity: Dict[str, Any]) -> float:
        """Predict win probability for a given opportunity"""
        if not self.model or not self.feature_columns:
            self.logger.error("Model not trained yet")
            return 0.0
        
        # Prepare features for single opportunity
        df = self._prepare_features([opportunity])
        
        # Ensure all feature columns exist
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Use only the columns the model was trained on, in the same order
        X = df[self.feature_columns]
        
        # Return probability of winning
        return self.model.predict_proba(X)[0][1]
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get the importance of each feature in the model"""
        if not self.model or not self.feature_columns:
            self.logger.error("Model not trained yet")
            return {}
        
        # Get feature importance
        importance = self.model.feature_importances_
        
        # Return as dict
        return {feature: float(importance) for feature, importance in zip(self.feature_columns, importance)}
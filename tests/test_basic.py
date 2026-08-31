import unittest
import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path to import system
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rop_prediction_system import ROPDataProcessor, DEFAULT_CONFIG

class TestROPSystem(unittest.TestCase):
    
    def setUp(self):
        self.processor = ROPDataProcessor(DEFAULT_CONFIG)
        
    def test_feature_calculation(self):
        """Test if physics features (MSE, HSI) are calculated correctly"""
        # Create dummy dataframe
        df = pd.DataFrame({
            'WOB': [10, 20],
            'RPM': [100, 120],
            'TORQUE': [5, 6],
            'ROP': [20, 30],
            'SPP': [3000, 3100],
            'FLOW': [800, 850],
            'MW': [10, 10.5]
        })
        
        # Calculate features using the processor's internal method
        # Note: _calculate_features is protected but we test it for logic verification
        processed = self.processor._calculate_features(df, 'Test_Well')
        
        # Check MSE existence and non-zero value
        self.assertIn('MSE', processed.columns)
        self.assertTrue(all(processed['MSE'] > 0))
        
        # Check HSI existence
        self.assertIn('HSI', processed.columns)
        
    def test_sequence_creation(self):
        """Test shape of LSTM sequences"""
        seq_len = DEFAULT_CONFIG['data']['sequence_length'] # 50
        n_samples = 100
        n_features = 5
        
        df = pd.DataFrame(np.random.rand(n_samples, n_features), 
                         columns=[f'F{i}' for i in range(n_features)])
        df['WELL_NAME'] = 'Test_Well'
        df['MD'] = np.linspace(0, 100, n_samples)
        df['ROP'] = np.random.rand(n_samples)
        
        feature_cols = [f'F{i}' for i in range(n_features)]
        
        X, y, groups = self.processor.create_sequences(df, feature_cols)
        
        # Check X shape: (samples, seq_len, features)
        self.assertEqual(len(X.shape), 3)
        self.assertEqual(X.shape[1], seq_len)
        self.assertEqual(X.shape[2], n_features)
        
        # Check y shape
        self.assertEqual(len(y), len(X))

if __name__ == '__main__':
    unittest.main()

# Hyperparameter Optimization Report

**Date:** 2026-01-08 13:52

## 1. Optimization Results
| Trial | LSTM Units | Dense Units | Dropout | LR | Batch | Val MAE (Scaled) |
|---|---|---|---|---|---|---|
| 1 | 256 | 32 | 0.3 | 0.0001 | 16 | **0.2113** |
| 2 | 64 | 128 | 0.2 | 0.0001 | 64 | **0.1871** |
| 3 | 256 | 128 | 0.5 | 0.001 | 64 | **0.2137** |
| 4 | 128 | 32 | 0.2 | 0.0005 | 32 | **0.1292** |
| 5 | 128 | 32 | 0.1 | 0.0005 | 32 | **0.1300** |

## 2. Best Configuration
```json
{
    "boosting_layers": 2,
    "lstm_layers": 2,
    "attention_heads": 4,
    "boosting_filters": 64,
    "lstm_units": 128,
    "dense_units": 32,
    "dropout_rate": 0.2,
    "learning_rate": 0.0005,
    "kernel_sizes": [
        3,
        5,
        7
    ]
}
```

## 3. Deployment
- **Final Model Saved:** `models\cbt_lstm_OPTIMIZED_20260108_135213.h5`
- **Final Test MAE (Scaled):** 0.0807
- **Recommendation:** Use this model for future predictions if Test MAE < 0.1 (good for scaled data).

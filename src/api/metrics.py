"""
Custom metrics for Prometheus
"""

from prometheus_client import Counter, Histogram, Gauge

TOTAL_PREDICTIONS = Counter(
    'total_predictions',
    'Total of predictions',
    ['threshold', 'user']
)

TOTAL_ERRORS = Counter(
    'total_errors',
    'Total of errors of the API',
    ['error_type']
)

PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Latency of predictions in seconds',
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
)

MODEL_LOADED = Gauge(
    'model_loaded',
    'Model is loaded, 1 (yes) or 0 (no)'
)

CURRENT_THRESHOLD = Gauge(
    'current_threshold',
    'Threshold used to define the probability boundary that separates churn from non-churn'
)
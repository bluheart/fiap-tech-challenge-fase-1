"""
Custom metrics for Prometheus monitoring.

This module defines application-specific metrics that track the performance,
health, and behavior of the MLP churn prediction API. These metrics are
exposed via the /metrics endpoint and scraped by Prometheus for monitoring,
alerting, and observability purposes.

Metrics defined:
    - TOTAL_PREDICTIONS: Counter tracking prediction volume by threshold
    - TOTAL_ERRORS: Counter tracking API errors by type
    - PREDICTION_LATENCY: Histogram measuring prediction response times
    - MODEL_LOADED: Gauge indicating model availability status
    - CURRENT_THRESHOLD: Gauge showing the active churn probability threshold

Usage:
    from metrics import TOTAL_PREDICTIONS, MODEL_LOADED
    
    # Increment prediction counter for a specific threshold
    TOTAL_PREDICTIONS.labels(threshold='0.5').inc()
    
    # Set model loaded status
    MODEL_LOADED.set(1)  # Model is loaded and ready
"""

from prometheus_client import Counter, Histogram, Gauge

# Prediction Volume Counter
# Tracks total number of churn predictions made, labeled by the probability
# threshold used. This enables monitoring prediction volume per threshold
# and analyzing threshold distribution over time.
TOTAL_PREDICTIONS = Counter(
    'total_predictions',
    'Total of predictions',
    ['threshold']  # Label: probability threshold used for classification
)

# API Error Counter
# Counts errors occurring in the API, categorized by error type.
# Useful for monitoring error rates, identifying common failure modes,
# and triggering alerts when error frequency increases.
TOTAL_ERRORS = Counter(
    'total_errors',
    'Total of errors of the API',
    ['error_type']  # Label: category of error (e.g., 'validation', 'model', 'server')
)

# Prediction Latency Histogram
# Measures the time taken to generate churn predictions in seconds.
# Buckets are optimized for API response times, with finer granularity
# in the sub-second range to capture typical latencies. The histogram
# enables calculation of averages, percentiles, and Apdex scores.
PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Latency of predictions in seconds',
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
    # Buckets: 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s
)

# Model Availability Gauge
# Indicates whether the ML model is loaded and ready to serve predictions.
# 1 = model loaded successfully, 0 = model not loaded or failed to load.
# Critical for determining if the service can handle prediction requests.
MODEL_LOADED = Gauge(
    'model_loaded',
    'Model is loaded, 1 (yes) or 0 (no)'
)

# Current Threshold Gauge
# Shows the current churn probability threshold being used for classification.
# This threshold determines the boundary between predicting churn vs. non-churn.
# Tracking this as a metric allows monitoring of threshold changes and their
# correlation with prediction volume and model performance over time.
CURRENT_THRESHOLD = Gauge(
    'current_threshold',
    'Threshold used to define the probability boundary that separates churn from non-churn'
)
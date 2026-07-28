from prometheus_client import Counter, Histogram

HTTP_REQUESTS = Counter(
    "whisperx_api_http_requests_total",
    "HTTP requests handled by the API.",
    ("method", "path", "status"),
)
HTTP_DURATION = Histogram(
    "whisperx_api_http_request_duration_seconds",
    "HTTP request latency.",
    ("method", "path"),
)

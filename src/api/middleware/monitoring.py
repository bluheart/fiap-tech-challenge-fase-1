import time
import uuid
from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import json
from ..logging_config import logger

class APILoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging API request/response details including:
    - Trace ID (generated or passed via headers)
    - Client IP
    - Request method and path
    - Response status code
    - Latency
    - User agent
    """
    
    def __init__(
        self, 
        app, 
        trace_id_header: str = "X-Trace-ID",
        exclude_paths: list = None,
        log_request_body: bool = False,
        log_response_body: bool = False,
        max_body_length: int = 1000
    ):
        super().__init__(app)
        self.trace_id_header = trace_id_header
        self.exclude_paths = exclude_paths or ["/health", "/metrics", "/docs", "/openapi.json", "/redoc"]
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.max_body_length = max_body_length

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip logging for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        # Generate or extract trace ID
        trace_id = request.headers.get(
            self.trace_id_header.lower(), 
            str(uuid.uuid4())
        )
        
        # Extract client IP
        client_ip = self._get_client_ip(request)
        
        # Record start time
        start_time = time.time()
        
        # Add trace_id to request state for use in route handlers
        request.state.trace_id = trace_id
        
        # Log request
        request_log = {
            "trace_id": trace_id,
            "type": "request",
            "method": request.method,
            "path": request.url.path,
            "query_params": str(request.query_params),
            "client_ip": client_ip,
            "user_agent": request.headers.get("user-agent", ""),
        }
        
        if self.log_request_body and request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if len(body) <= self.max_body_length:
                    request_log["body"] = body.decode("utf-8", errors="ignore")
                else:
                    request_log["body"] = f"<body too large: {len(body)} bytes>"
                # Reset the request body so it can be read by the route handler
                async def receive():
                    return {"type": "http.request", "body": body}
                request._receive = receive
            except Exception:
                request_log["body"] = "<unable to read body>"
        
        logger.info(f"API Request: {json.dumps(request_log)}")
        
        # Process the request
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            
            # Log response
            response_log = {
                "trace_id": trace_id,
                "type": "response",
                "status_code": status_code,
            }
            
            if self.log_response_body:
                try:
                    response_body = b""
                    async for chunk in response.body_iterator:
                        response_body += chunk
                    
                    if len(response_body) <= self.max_body_length:
                        response_log["body"] = response_body.decode("utf-8", errors="ignore")
                    else:
                        response_log["body"] = f"<body too large: {len(response_body)} bytes>"
                    
                    # Reconstruct response with the body we read
                    response = Response(
                        content=response_body,
                        status_code=response.status_code,
                        headers=dict(response.headers),
                        media_type=response.media_type,
                    )
                except Exception:
                    response_log["body"] = "<unable to read body>"
            
            logger.info(f"API Response: {json.dumps(response_log)}")
            
        except Exception as e:
            logger.error(
                f"API Error: trace_id={trace_id}, error={str(e)}", 
                exc_info=True
            )
            raise
        
        finally:
            # Calculate and log latency
            latency = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            log_data = {
                "trace_id": trace_id,
                "type": "summary",
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "client_ip": client_ip,
                "latency_ms": round(latency, 2),
                "user_agent": request.headers.get("user-agent", ""),
            }
            
            logger.info(f"API Summary: {json.dumps(log_data)}")
            
            # Add trace_id to response headers
            response.headers[self.trace_id_header] = trace_id
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request headers or direct client."""
        # Check common proxy headers first
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Get the first IP in the chain (original client)
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to direct client
        if request.client:
            return request.client.host
        
        return "unknown"
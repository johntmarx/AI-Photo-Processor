"""
Photo Processor API - Main Application

A FastAPI backend for the photo processor with no authentication.
Designed for local network use where all users have full access.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response, JSONResponse
from contextlib import asynccontextmanager
from typing import Set
import asyncio
import logging
import traceback
import json
from datetime import datetime
from pathlib import Path

from routes import photos, processing, recipes, stats, upload, files, recipe_builder, batch_processing
from services.websocket_manager import WebSocketManager

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/app/logs/api.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Suppress verbose aiosqlite logging
logging.getLogger('aiosqlite').setLevel(logging.WARNING)
# Suppress multipart debug messages
logging.getLogger('python_multipart').setLevel(logging.WARNING)
# Suppress PIL debug messages
logging.getLogger('PIL').setLevel(logging.WARNING)
# Suppress websocket debug messages
logging.getLogger('services.websocket_manager').setLevel(logging.WARNING)

# Ensure logs directory exists
Path('/app/logs').mkdir(exist_ok=True)

# WebSocket manager instance
ws_manager = WebSocketManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    logger.info("Starting Photo Processor API...")
    
    # Import services
    from services.processing_service_v2 import processing_service
    
    # Create background task for processing photos
    async def processing_loop():
        """Continuously process photos from the queue"""
        logger.info("Starting background processing loop...")
        while True:
            try:
                # Process next item in queue
                result = await processing_service.process_next_item()
                if result:
                    logger.info(f"✓ Processing complete: {result['photo_id']}")
                else:
                    # No items to process, wait a bit
                    await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                logger.error(f"Stack trace: {traceback.format_exc()}")
                await asyncio.sleep(5)  # Wait longer on error
    
    # Start background processing task
    processing_task = asyncio.create_task(processing_loop())
    logger.info("Background processing loop started")
    
    yield
    
    # Shutdown tasks here
    logger.info("Shutting down Photo Processor API...")
    processing_task.cancel()
    try:
        await processing_task
    except asyncio.CancelledError:
        pass
    await ws_manager.disconnect_all()

# Create FastAPI app
app = FastAPI(
    title="Photo Processor API",
    description="Local network API for photo processing control and monitoring",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS - permissive for local network
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins on local network
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handling middleware
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with detailed logging"""
    error_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    # Log the full error details
    logger.error(f"Error ID: {error_id} - Unhandled exception")
    logger.error(f"Request: {request.method} {request.url}")
    logger.error(f"Headers: {dict(request.headers)}")
    logger.error(f"Exception type: {type(exc).__name__}")
    logger.error(f"Exception message: {str(exc)}")
    logger.error(f"Stack trace:\n{traceback.format_exc()}")
    
    # Return user-friendly error with ID for tracking
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "error_id": error_id,
            "message": str(exc),
            "type": type(exc).__name__,
            "path": str(request.url.path),
            "method": request.method
        }
    )

# Request/Response logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests and responses with timing"""
    start_time = datetime.now()
    request_id = start_time.strftime("%Y%m%d_%H%M%S_%f")
    
    # Only log important requests
    if request.url.path in ["/api/upload/single", "/api/upload/session"] or request.method == "POST":
        logger.info(f"→ {request.method} {request.url.path}")
    
    try:
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds()
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration:.3f}s"
        
        return response
        
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"Request ID: {request_id} - Exception during request processing - Duration: {duration:.3f}s")
        logger.error(f"Exception: {type(e).__name__}: {str(e)}")
        logger.error(f"Stack trace:\n{traceback.format_exc()}")
        raise

# Include routers
app.include_router(photos.router, prefix="/api/photos", tags=["photos"])
app.include_router(processing.router, prefix="/api/processing", tags=["processing"])
app.include_router(recipes.router, prefix="/api/recipes", tags=["recipes"])
app.include_router(stats.router, prefix="/api/stats", tags=["statistics"])
app.include_router(upload.router, prefix="/api/upload", tags=["upload"])
app.include_router(files.router, prefix="/api/files", tags=["files"])
app.include_router(recipe_builder.router, prefix="/api", tags=["recipe-builder"])
app.include_router(batch_processing.router, prefix="/api", tags=["batch-processing"])

# Inject WebSocket manager into processing routes
processing.set_websocket_manager(ws_manager)

# Inject WebSocket manager into batch processing
batch_processing.set_websocket_manager(ws_manager)

# Also inject into processing service
from services.processing_service_v2 import processing_service
processing_service.set_websocket_manager(ws_manager)

# Inject WebSocket manager into recipe builder
recipe_builder.set_websocket_manager(ws_manager)

# Inject into upload service
from services.upload_service import upload_service
upload_service.set_websocket_manager(ws_manager)

# Serve static files (images)
data_path = Path("/app/data")
if data_path.exists():
    app.mount("/images/originals", StaticFiles(directory=str(data_path / "originals")), name="originals")
    app.mount("/images/processed", StaticFiles(directory=str(data_path / "processed")), name="processed")
    app.mount("/images/thumbnails", StaticFiles(directory=str(data_path / "thumbnails")), name="thumbnails")
    app.mount("/images/web", StaticFiles(directory=str(data_path / "web")), name="web")

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Photo Processor API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "photos": "/api/photos",
            "processing": "/api/processing",
            "recipes": "/api/recipes",
            "stats": "/api/stats",
            "upload": "/api/upload",
            "websocket": "/ws"
        }
    }

# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "photo-processor-api",
        "websocket_connections": len(ws_manager.active_connections)
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "photo-processor-api",
        "timestamp": datetime.now().isoformat(),
        "websocket_connections": len(ws_manager.active_connections) if ws_manager else 0
    }


# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    client_id = None
    try:
        await ws_manager.connect(websocket)
        client_id = f"{websocket.client.host}:{websocket.client.port}"
        logger.info(f"WebSocket client connected: {client_id}")
        
        while True:
            try:
                # Keep connection alive and handle incoming messages
                # Use receive_json to handle both text and binary messages
                message = await websocket.receive()
                
                if message["type"] == "websocket.receive":
                    if "text" in message:
                        data = message["text"]
                        # Handle ping/pong for keepalive
                        if data == "ping":
                            await websocket.send_text("pong")
                    elif "bytes" in message:
                        # Handle binary messages if needed
                        pass
                        
                elif message["type"] == "websocket.disconnect":
                    break
                    
            except asyncio.TimeoutError:
                # Send periodic ping to check if client is still alive
                try:
                    await websocket.send_text("ping")
                except Exception:
                    break
                    
            except Exception as e:
                logger.error(f"Error handling WebSocket message from {client_id}: {e}")
                # Continue listening unless it's a connection error
                if isinstance(e, (ConnectionError, OSError)):
                    break
                    
    except WebSocketDisconnect:
        pass  # Normal disconnection
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
    finally:
        ws_manager.disconnect(websocket)

# Direct image access endpoint
@app.get("/images/{image_type}/{file_path:path}")
async def get_image(image_type: str, file_path: str):
    """Serve images directly from storage"""
    if image_type not in ["originals", "processed", "thumbnails", "web"]:
        return {"error": "Invalid image type"}
    
    full_path = data_path / image_type / file_path
    if not full_path.exists():
        return {"error": "Image not found"}
    
    return FileResponse(full_path)

# API file access endpoint (for transformed paths)
@app.get("/api/files/{file_type}/{file_path:path}")
async def get_api_file(file_type: str, file_path: str):
    """Serve files through API paths (transformed from /app/data/)"""
    if file_type not in ["originals", "processed", "thumbnails", "web", "inbox"]:
        return {"error": "Invalid file type"}
    
    full_path = data_path / file_type / file_path
    if not full_path.exists():
        return {"error": "File not found"}
    
    return FileResponse(full_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
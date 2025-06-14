"""
Celery configuration for background processing tasks

This module provides a scalable architecture for running background jobs:
- AI analysis tasks (NIMA, object detection, scene analysis, etc.)
- Image processing tasks (RAW conversion, filters, etc.)
- Batch operations (culling, grouping, etc.)
- System maintenance tasks (cleanup, optimization, etc.)
"""

from celery import Celery
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Discover all task modules automatically
TASK_MODULES = [
    "tasks.ai_tasks",      # AI analysis tasks (NIMA, detection, etc.)
    "tasks.image_tasks",   # Image processing tasks
    "tasks.batch_tasks",   # Batch operations (culling, grouping)
    "tasks.system_tasks",  # System maintenance
]

# Create Celery app with auto-discovery
celery_app = Celery(
    "photo_processor",
    broker=os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0"),
    include=TASK_MODULES
)

# Celery configuration
celery_app.conf.update(
    # Task routing - organize tasks by type into queues
    task_routes={
        # AI Analysis Queue (GPU-intensive)
        'tasks.ai_tasks.analyze_photo_nima': {'queue': 'ai_analysis'},
        'tasks.ai_tasks.analyze_photo_onealign': {'queue': 'ai_analysis'},
        'tasks.ai_tasks.analyze_batch_onealign': {'queue': 'ai_analysis'},
        'tasks.ai_tasks.analyze_rotation_onealign': {'queue': 'ai_analysis'},
        'tasks.ai_tasks.reanalyze_photo_onealign': {'queue': 'ai_analysis'},
        'tasks.ai_tasks.analyze_composition_vlm': {'queue': 'ai_analysis'},
        'tasks.ai_tasks.generate_crop_bbox_vlm': {'queue': 'ai_analysis'},
        'tasks.ai_tasks.cleanup_gpu_memory': {'queue': 'ai_analysis'},
        'tasks.ai_tasks.detect_objects': {'queue': 'ai_analysis'},
        'tasks.ai_tasks.analyze_scene': {'queue': 'ai_analysis'},
        'tasks.ai_tasks.classify_image': {'queue': 'ai_analysis'},
        
        # Image Processing Queue (CPU-intensive)
        'tasks.image_tasks.convert_raw': {'queue': 'image_processing'},
        'tasks.image_tasks.apply_filters': {'queue': 'image_processing'},
        'tasks.image_tasks.generate_thumbnails': {'queue': 'image_processing'},
        'tasks.image_tasks.optimize_image': {'queue': 'image_processing'},
        
        # Batch Operations Queue (high-memory)
        'tasks.batch_tasks.cull_photos': {'queue': 'batch_operations'},
        'tasks.batch_tasks.group_bursts': {'queue': 'batch_operations'},
        'tasks.batch_tasks.batch_export': {'queue': 'batch_operations'},
        
        # System Maintenance Queue (low-priority)
        'tasks.system_tasks.cleanup_temp_files': {'queue': 'system'},
        'tasks.system_tasks.optimize_database': {'queue': 'system'},
        'tasks.system_tasks.generate_reports': {'queue': 'system'},
    },
    
    # Queue priorities (higher number = higher priority)
    task_queue_max_priority=10,
    task_default_priority=5,
    
    # Worker configuration
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=50,
    worker_disable_rate_limits=True,
    
    # Task configuration
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Task time limits (different for different task types)
    task_soft_time_limit=300,   # 5 minutes default
    task_time_limit=600,        # 10 minutes default
    task_timeout=3600,          # 1 hour max for any task
    
    # Redis configuration
    broker_connection_retry_on_startup=True,
    broker_transport_options={
        'visibility_timeout': 3600,  # 1 hour
        'retry_on_timeout': True,
    },
    
    # Result backend configuration
    result_expires=3600,  # 1 hour
    result_backend_transport_options={
        'retry_on_timeout': True,
    },
    
    # Error handling
    task_reject_on_worker_lost=True,
    task_ignore_result=False,
    
    # Beat schedule for periodic tasks
    beat_schedule={
        'cleanup-temp-files': {
            'task': 'tasks.system_tasks.cleanup_temp_files',
            'schedule': 3600.0,  # Every hour
            'options': {'queue': 'system', 'priority': 1}
        },
        'cleanup-old-results': {
            'task': 'tasks.system_tasks.cleanup_old_results',
            'schedule': 86400.0,  # Daily
            'options': {'queue': 'system', 'priority': 1}
        },
        'cleanup-gpu-memory': {
            'task': 'tasks.ai_tasks.cleanup_gpu_memory',
            'schedule': 7200.0,  # Every 2 hours
            'options': {'queue': 'ai_analysis', 'priority': 3},
            'kwargs': {}  # No arguments needed
        },
    },
)

# Task retry configuration
celery_app.conf.task_annotations = {
    # AI tasks - retry with exponential backoff
    'tasks.ai_tasks.*': {
        'retry_kwargs': {'max_retries': 3, 'countdown': 60},
        'retry_backoff': True,
        'retry_jitter': True,
    },
    
    # Image processing - retry quickly
    'tasks.image_tasks.*': {
        'retry_kwargs': {'max_retries': 2, 'countdown': 30},
        'retry_backoff': False,
    },
    
    # Batch operations - no auto-retry (user initiated)
    'tasks.batch_tasks.*': {
        'retry_kwargs': {'max_retries': 0},
    },
    
    # System tasks - retry with long delay
    'tasks.system_tasks.*': {
        'retry_kwargs': {'max_retries': 1, 'countdown': 300},
    },
}

# Auto-discover tasks on startup
celery_app.autodiscover_tasks()

if __name__ == '__main__':
    celery_app.start()
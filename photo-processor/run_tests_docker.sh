#!/bin/bash
# Run tests inside Docker container

echo "Building Docker image for tests..."
docker build -f Dockerfile.v2 -t photo-processor-test .

echo "Running tests in Docker container..."
docker run --rm \
  -v $(pwd):/app \
  -w /app \
  photo-processor-test \
  bash -c "
    # Install pytest if not already installed
    pip install pytest pytest-cov
    
    # Run all unit tests
    echo '======================================'
    echo 'Running Recipe Storage Tests'
    echo '======================================'
    python -m pytest tests/unit/test_recipe_storage.py -v
    
    echo ''
    echo '======================================'
    echo 'Running Immich Client Tests'
    echo '======================================'
    python -m pytest tests/unit/test_immich_client_v2.py -v
    
    echo ''
    echo '======================================'
    echo 'Running Main Processor Tests'
    echo '======================================'
    python -m pytest tests/unit/test_main_v2.py -v
    
    echo ''
    echo '======================================'
    echo 'Running Integration Tests'
    echo '======================================'
    python -m pytest tests/integration/test_dual_upload_integration.py -v
    
    echo ''
    echo '======================================'
    echo 'Running All Tests with Coverage'
    echo '======================================'
    python -m pytest tests/ --cov=. --cov-report=term-missing -v
  "
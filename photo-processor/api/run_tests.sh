#!/bin/bash
# Run API tests in Docker

echo "Building API test image..."
docker build -f Dockerfile.test -t photo-processor-api-test .

echo -e "\nRunning unit tests..."
docker run --rm photo-processor-api-test pytest tests/unit -v -m "not integration"

echo -e "\nRunning integration tests..."
docker run --rm photo-processor-api-test pytest tests/integration -v -m "not slow"

echo -e "\nRunning all tests with coverage..."
docker run --rm photo-processor-api-test pytest -v --cov=. --cov-report=term-missing --cov-report=html

echo -e "\nTest run complete!"
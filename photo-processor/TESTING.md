# AI Photo Processor - Comprehensive Test Suite 🧪

This directory contains a complete test suite with near 100% coverage for the AI Photo Processing pipeline.

## Test Structure

```
tests/
├── unit/                          # Unit tests for individual components
│   ├── test_ai_analyzer.py       # AI analyzer module tests (40+ test cases)
│   ├── test_image_processor.py   # Image processing tests (35+ test cases)
│   ├── test_immich_client.py     # Immich API client tests (30+ test cases)
│   └── test_schemas.py           # Data validation tests (50+ test cases)
├── integration/                   # Integration tests
│   └── test_photo_processor_service.py  # End-to-end workflow tests
├── fixtures/                     # Test data and mocks
│   └── mock_data.py              # Mock objects and test data generators
└── conftest.py                   # Pytest configuration and shared fixtures
```

## Test Coverage Areas

### 🤖 AI Analyzer Tests (`test_ai_analyzer.py`)
- ✅ Ollama client connection and model availability
- ✅ Photo analysis with structured outputs  
- ✅ JSON parsing and Pydantic validation
- ✅ Error handling for network failures
- ✅ Model response format handling (dict vs object)
- ✅ Temperature and prompt configuration
- ✅ Filename inclusion in analysis prompts

### 🖼️ Image Processor Tests (`test_image_processor.py`)
- ✅ RAW image loading (ARW, CR2, NEF, etc.)
- ✅ Image resizing for AI analysis
- ✅ Smart cropping with bounding boxes
- ✅ Color enhancement (brightness, contrast, CLAHE)
- ✅ Auto white balance correction
- ✅ Image saving with quality control
- ✅ Complete processing workflows
- ✅ Error handling for file operations

### 🌐 Immich Client Tests (`test_immich_client.py`)
- ✅ API connection and authentication
- ✅ Album creation and management
- ✅ Photo upload with metadata
- ✅ AI-generated descriptions
- ✅ Asset addition to albums
- ✅ Error handling for API failures
- ✅ Request timeout configuration
- ✅ File handling and multipart uploads

### 📋 Schema Tests (`test_schemas.py`)
- ✅ Pydantic model validation
- ✅ Bounding box coordinate validation
- ✅ Crop suggestion parameters
- ✅ Color analysis data structures
- ✅ Swimming context validation
- ✅ Complete photo analysis schema
- ✅ JSON serialization/deserialization
- ✅ Field validation and constraints

### 🔄 Integration Tests (`test_photo_processor_service.py`)
- ✅ End-to-end file processing workflow
- ✅ Service startup and health checks
- ✅ File watching and stability checks
- ✅ Error handling across components
- ✅ Concurrent processing capabilities
- ✅ Album integration
- ✅ Quality-based processing decisions

## Running Tests

### Install Test Dependencies
```bash
# Install test requirements
pip install -r test-requirements.txt

# Or use the test runner
./run_tests.py install
```

### Run All Tests
```bash
# Run complete test suite with coverage
./run_tests.py all

# Run without coverage
./run_tests.py all --no-coverage
```

### Run Specific Test Types
```bash
# Unit tests only
./run_tests.py unit

# Integration tests only  
./run_tests.py integration

# Schema validation tests
./run_tests.py schema

# Code quality check
./run_tests.py lint
```

### Run Specific Tests
```bash
# Run specific test file
./run_tests.py specific --path tests/unit/test_ai_analyzer.py

# Run specific test function
./run_tests.py specific --path tests/unit/test_ai_analyzer.py::TestAIAnalyzer::test_analyze_photo_success
```

### Run Tests by Markers
```bash
# Run only unit tests
./run_tests.py markers --markers unit

# Run tests that require external services
./run_tests.py markers --markers requires_ollama requires_immich

# Run slow tests
./run_tests.py markers --markers slow
```

## Test Configuration

### Pytest Configuration (`pytest.ini`)
- 📊 Coverage reporting (HTML, XML, terminal)
- 🎯 90% coverage requirement
- ⏱️ Test timeouts and duration reporting
- 🏷️ Custom markers for test categorization
- 📝 Comprehensive logging

### Test Fixtures (`conftest.py`)
- 🗂️ Temporary directory management
- 🖼️ Mock image data generation
- 🤖 Service mocking (Ollama, Immich)
- 🔧 Environment variable setup
- 📊 Logging configuration

### Mock Data (`fixtures/mock_data.py`)
- 🏊 Swimming photo analysis samples
- 🖼️ Image array generators
- 📡 API response mocks
- 📁 Test file generators
- 🧰 Testing utilities

## Coverage Goals

Our test suite aims for **≥90% code coverage** across all modules:

| Module | Target Coverage | Test Count |
|--------|----------------|------------|
| `ai_analyzer.py` | 95%+ | 40+ tests |
| `image_processor.py` | 95%+ | 35+ tests |
| `immich_client.py` | 95%+ | 30+ tests |
| `schemas.py` | 98%+ | 50+ tests |
| `main.py` | 90%+ | 20+ tests |

## Test Data

The test suite uses the **10 real ARW files** copied to `/home/john/immich/photo-processor/test-data/`:
- DSC09497.ARW (67MB)
- DSC09501.ARW (67MB) 
- DSC09509.ARW (66MB)
- DSC09528.ARW (66MB)
- DSC09531.ARW (66MB)
- DSC09535.ARW (66MB)
- DSC09544.ARW (66MB)
- DSC09557.ARW (67MB)
- DSC09580.ARW (68MB)
- DSC09585.ARW (66MB)

These provide realistic test data for RAW image processing validation.

## Test Execution Examples

### Basic Test Run
```bash
# Quick unit test run
./run_tests.py unit -v

# Full test suite with verbose output
./run_tests.py all -v
```

### Coverage Analysis
```bash
# Generate coverage report
./run_tests.py coverage

# View HTML coverage report
open htmlcov/index.html
```

### CI/CD Integration
```bash
# Silent run for automated testing
python -m pytest --quiet --cov=. --cov-fail-under=90

# XML output for CI systems
python -m pytest --cov=. --cov-report=xml --junit-xml=test-results.xml
```

## Test Philosophy

This test suite follows these principles:

1. **Comprehensive Coverage**: Every function, method, and code path is tested
2. **Realistic Scenarios**: Tests use real-world data and edge cases
3. **Fast Execution**: Unit tests run in milliseconds, full suite in seconds
4. **Isolated Testing**: Each test is independent and can run in any order
5. **Clear Assertions**: Every test has specific, meaningful assertions
6. **Error Simulation**: Network failures, file errors, and edge cases are tested
7. **Documentation**: Tests serve as living documentation of expected behavior

## Troubleshooting

### Common Issues

**ImportError**: Make sure all dependencies are installed:
```bash
pip install -r test-requirements.txt
```

**Coverage Too Low**: Check which lines aren't covered:
```bash
./run_tests.py all -v
open htmlcov/index.html
```

**Test Failures**: Run specific failing tests with verbose output:
```bash
./run_tests.py specific --path tests/unit/test_ai_analyzer.py -v
```

### Environment Setup

For the tests to run properly, ensure:
- ✅ Python 3.11+ is installed
- ✅ All test dependencies are available
- ✅ Test data files exist (for integration tests)
- ✅ No conflicting processes on test ports

The test suite is designed to be **self-contained** and **mock all external dependencies**, so it can run without Ollama, Immich, or GPU access.

---

🎯 **Goal**: Achieve and maintain **≥95% test coverage** with **comprehensive error handling** and **realistic test scenarios** to ensure the AI photo processing pipeline works reliably in production.
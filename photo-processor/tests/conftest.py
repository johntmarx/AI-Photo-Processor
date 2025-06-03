"""
Pytest configuration and shared fixtures
"""
import pytest
import tempfile
import shutil
import os
from unittest.mock import Mock, patch
from tests.fixtures.mock_data import (
    MockImageData, MockPhotoAnalysisData, MockOllamaResponses, 
    MockImmichResponses, create_temp_directory_structure
)


@pytest.fixture(scope="session")
def temp_base_dir():
    """Create a temporary base directory for all tests"""
    temp_dir = tempfile.mkdtemp(prefix="photo_processor_tests_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_dirs(temp_base_dir):
    """Create temporary directories for each test"""
    test_structure = create_temp_directory_structure()
    yield test_structure
    # Cleanup is handled by session fixture


@pytest.fixture
def sample_rgb_image():
    """Create a sample RGB image array for testing"""
    return MockImageData.create_rgb_array(800, 600)


@pytest.fixture
def sample_low_contrast_image():
    """Create a low contrast image for testing enhancement"""
    return MockImageData.create_low_contrast_image(200, 200)


@pytest.fixture
def test_image_file():
    """Create a temporary test image file"""
    filepath = MockImageData.create_test_image_file()
    yield filepath
    try:
        os.unlink(filepath)
    except OSError:
        pass


@pytest.fixture
def test_raw_file():
    """Create a temporary test RAW file"""
    filepath = MockImageData.create_mock_raw_file(size_mb=10)
    yield filepath
    try:
        os.unlink(filepath)
    except OSError:
        pass


@pytest.fixture
def crisp_analysis():
    """Sample crisp photo analysis"""
    return MockPhotoAnalysisData.crisp_swimmer_analysis()


@pytest.fixture
def blurry_analysis():
    """Sample blurry photo analysis"""
    return MockPhotoAnalysisData.blurry_photo_analysis()


@pytest.fixture
def multiple_swimmers_analysis():
    """Sample multiple swimmers analysis"""
    return MockPhotoAnalysisData.multiple_swimmers_analysis()


@pytest.fixture
def enhance_only_analysis():
    """Sample enhance-only analysis"""
    return MockPhotoAnalysisData.enhance_only_analysis()


@pytest.fixture
def mock_ollama_client():
    """Mock Ollama client for testing"""
    with patch('ai_analyzer.ollama.Client') as mock_client_class:
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Setup default successful responses
        mock_client.list.return_value = MockOllamaResponses.successful_model_list()
        mock_client.chat.return_value = MockOllamaResponses.successful_chat_response(
            MockPhotoAnalysisData.crisp_swimmer_analysis()
        )
        
        yield mock_client


@pytest.fixture
def mock_immich_client():
    """Mock Immich client for testing"""
    with patch('immich_client.requests') as mock_requests:
        # Setup default successful responses
        mock_requests.get.return_value = MockImmichResponses.successful_ping()
        mock_requests.post.return_value = MockImmichResponses.successful_upload()
        mock_requests.put.return_value = MockImmichResponses.successful_album_add()
        
        yield mock_requests


@pytest.fixture
def mock_image_processor():
    """Mock image processor for testing"""
    with patch('image_processor.rawpy') as mock_rawpy, \
         patch('image_processor.cv2') as mock_cv2, \
         patch('image_processor.Image') as mock_pil:
        
        # Setup mock RAW processing
        mock_raw = Mock()
        mock_raw.postprocess.return_value = MockImageData.create_rgb_array()
        mock_rawpy.imread.return_value = mock_raw
        
        # Setup mock OpenCV
        mock_cv2.resize.return_value = MockImageData.create_rgb_array(512, 512)
        mock_cv2.cvtColor.return_value = MockImageData.create_rgb_array(512, 512)
        
        yield {
            'rawpy': mock_rawpy,
            'cv2': mock_cv2,
            'pil': mock_pil
        }


@pytest.fixture
def mock_file_watcher():
    """Mock file watcher for testing"""
    with patch('main.Observer') as mock_observer, \
         patch('main.FileSystemEventHandler') as mock_handler:
        
        mock_observer_instance = Mock()
        mock_observer.return_value = mock_observer_instance
        
        mock_handler_instance = Mock()
        mock_handler.return_value = mock_handler_instance
        
        yield {
            'observer': mock_observer_instance,
            'handler': mock_handler_instance
        }


@pytest.fixture
def mock_all_services():
    """Mock all external services for integration testing"""
    with patch('main.AIAnalyzer') as mock_ai, \
         patch('main.ImageProcessor') as mock_img, \
         patch('main.ImmichClient') as mock_immich:
        
        # AI Analyzer
        mock_ai_instance = Mock()
        mock_ai_instance.test_connection.return_value = True
        mock_ai_instance.ensure_model_available.return_value = True
        mock_ai_instance.analyze_photo.return_value = MockPhotoAnalysisData.crisp_swimmer_analysis()
        mock_ai.return_value = mock_ai_instance
        
        # Image Processor
        mock_img_instance = Mock()
        mock_img_instance.process_photo.return_value = True
        mock_img.return_value = mock_img_instance
        
        # Immich Client
        mock_immich_instance = Mock()
        mock_immich_instance.test_connection.return_value = True
        mock_immich_instance.get_or_create_album.return_value = "test-album-id"
        mock_immich_instance.upload_photo.return_value = "test-asset-id"
        mock_immich.return_value = mock_immich_instance
        
        yield {
            'ai_analyzer': mock_ai_instance,
            'image_processor': mock_img_instance,
            'immich_client': mock_immich_instance
        }


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging configuration for each test"""
    import logging
    
    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Setup basic configuration for tests
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    yield
    
    # Cleanup after test
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)


@pytest.fixture
def environment_variables():
    """Set up environment variables for testing"""
    import os
    
    original_env = os.environ.copy()
    
    # Set test environment variables
    test_env = {
        'OLLAMA_HOST': 'http://test-ollama:11434',
        'IMMICH_API_URL': 'http://test-immich:2283',
        'IMMICH_API_KEY': 'test-api-key',
        'WATCH_FOLDER': '/test/inbox',
        'OUTPUT_FOLDER': '/test/processed',
    }
    
    os.environ.update(test_env)
    
    yield test_env
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


# Custom pytest markers
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "requires_ollama: mark test as requiring Ollama service"
    )
    config.addinivalue_line(
        "markers", "requires_immich: mark test as requiring Immich service"
    )


# Test collection customization
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file location"""
    for item in items:
        # Add unit test marker for tests in unit directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Add integration test marker for tests in integration directory
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Mark tests that use external services
        if "ollama" in item.name.lower() or "ai_analyzer" in item.name.lower():
            item.add_marker(pytest.mark.requires_ollama)
        
        if "immich" in item.name.lower():
            item.add_marker(pytest.mark.requires_immich)
        
        # Mark slow tests
        if "workflow" in item.name.lower() or "end_to_end" in item.name.lower():
            item.add_marker(pytest.mark.slow)
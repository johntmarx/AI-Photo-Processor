"""
Comprehensive unit tests for AI Analyzer module
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import json
from datetime import datetime
from ai_analyzer import AIAnalyzer
from schemas import PhotoAnalysis, BoundingBox, CropSuggestion, ColorAnalysis, SwimmingContext


class TestAIAnalyzer:
    """Test suite for AIAnalyzer class"""

    @pytest.fixture
    def ai_analyzer(self):
        """Create AIAnalyzer instance for testing"""
        return AIAnalyzer(ollama_host="http://test-ollama:11434", model="test-model")

    @pytest.fixture
    def mock_ollama_client(self):
        """Mock Ollama client"""
        return Mock()

    @pytest.fixture
    def sample_photo_analysis(self):
        """Sample photo analysis response"""
        return {
            "description": "A crisp photo of a swimmer performing freestyle stroke in an indoor pool",
            "quality": "crisp",
            "primary_subject": "swimmer",
            "primary_subject_box": {
                "x": 25.0,
                "y": 30.0,
                "width": 50.0,
                "height": 40.0
            },
            "recommended_crop": {
                "crop_box": {
                    "x": 10.0,
                    "y": 15.0,
                    "width": 80.0,
                    "height": 70.0
                },
                "aspect_ratio": "16:9",
                "composition_rule": "rule_of_thirds",
                "confidence": 0.85
            },
            "color_analysis": {
                "dominant_colors": ["blue", "white"],
                "exposure_assessment": "properly_exposed",
                "white_balance_assessment": "neutral",
                "contrast_level": "good",
                "brightness_adjustment_needed": 0,
                "contrast_adjustment_needed": 5
            },
            "swimming_context": {
                "event_type": "freestyle",
                "pool_type": "indoor",
                "time_of_event": "mid_race",
                "lane_number": 4
            },
            "processing_recommendation": "crop_and_enhance"
        }

    def test_initialization_default_params(self):
        """Test AIAnalyzer initialization with default parameters"""
        analyzer = AIAnalyzer()
        assert analyzer.ollama_host == "http://ollama:11434"
        assert analyzer.model == "gemma3:4b"
        assert analyzer.client is not None

    def test_initialization_custom_params(self):
        """Test AIAnalyzer initialization with custom parameters"""
        custom_host = "http://custom-ollama:12345"
        custom_model = "custom-model"
        analyzer = AIAnalyzer(ollama_host=custom_host, model=custom_model)
        assert analyzer.ollama_host == custom_host
        assert analyzer.model == custom_model

    @patch('ai_analyzer.ollama.Client')
    def test_analyze_photo_success(self, mock_client_class, ai_analyzer, sample_photo_analysis):
        """Test successful photo analysis"""
        # Setup mock
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        ai_analyzer.client = mock_client
        
        mock_response = Mock()
        mock_response.message.content = json.dumps(sample_photo_analysis)
        mock_client.chat.return_value = mock_response

        # Test
        result = ai_analyzer.analyze_photo("/test/image.jpg", "test_image.jpg")

        # Assertions
        assert result is not None
        assert isinstance(result, PhotoAnalysis)
        assert result.quality == "crisp"
        assert result.primary_subject == "swimmer"
        assert result.processing_recommendation == "crop_and_enhance"
        
        # Verify client was called correctly
        mock_client.chat.assert_called_once()
        call_args = mock_client.chat.call_args
        assert call_args[1]['model'] == ai_analyzer.model
        assert len(call_args[1]['messages']) == 1
        assert call_args[1]['messages'][0]['role'] == 'user'
        assert 'test_image.jpg' in call_args[1]['messages'][0]['content']
        assert call_args[1]['messages'][0]['images'] == ["/test/image.jpg"]

    @patch('ai_analyzer.ollama.Client')
    def test_analyze_photo_json_decode_error(self, mock_client_class, ai_analyzer):
        """Test photo analysis with JSON decode error"""
        # Setup mock
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        ai_analyzer.client = mock_client
        
        mock_response = Mock()
        mock_response.message.content = "invalid json"
        mock_client.chat.return_value = mock_response

        # Test
        result = ai_analyzer.analyze_photo("/test/image.jpg", "test_image.jpg")

        # Assertions
        assert result is None

    @patch('ai_analyzer.ollama.Client')
    def test_analyze_photo_general_exception(self, mock_client_class, ai_analyzer):
        """Test photo analysis with general exception"""
        # Setup mock
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        ai_analyzer.client = mock_client
        
        mock_client.chat.side_effect = Exception("Connection error")

        # Test
        result = ai_analyzer.analyze_photo("/test/image.jpg", "test_image.jpg")

        # Assertions
        assert result is None

    @patch('ai_analyzer.ollama.Client')
    def test_test_connection_success_with_model_available(self, mock_client_class, ai_analyzer):
        """Test successful connection with model available"""
        # Setup mock
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        ai_analyzer.client = mock_client
        
        # Mock model list response
        mock_model = Mock()
        mock_model.model = ai_analyzer.model
        mock_response = Mock()
        mock_response.models = [mock_model]
        mock_client.list.return_value = mock_response

        # Test
        result = ai_analyzer.test_connection()

        # Assertions
        assert result is True
        mock_client.list.assert_called_once()

    @patch('ai_analyzer.ollama.Client')
    def test_test_connection_success_model_not_available(self, mock_client_class, ai_analyzer):
        """Test successful connection but model not available"""
        # Setup mock
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        ai_analyzer.client = mock_client
        
        # Mock model list response without our model
        mock_model = Mock()
        mock_model.model = "different-model"
        mock_response = Mock()
        mock_response.models = [mock_model]
        mock_client.list.return_value = mock_response

        # Test
        result = ai_analyzer.test_connection()

        # Assertions
        assert result is False

    @patch('ai_analyzer.ollama.Client')
    def test_test_connection_dict_response_format(self, mock_client_class, ai_analyzer):
        """Test connection with dict response format"""
        # Setup mock
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        ai_analyzer.client = mock_client
        
        # Mock dict response format
        mock_response = {
            'models': [{'model': ai_analyzer.model}]
        }
        mock_client.list.return_value = mock_response

        # Test
        result = ai_analyzer.test_connection()

        # Assertions
        assert result is True

    @patch('ai_analyzer.ollama.Client')
    def test_test_connection_unexpected_format(self, mock_client_class, ai_analyzer):
        """Test connection with unexpected response format"""
        # Setup mock
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        ai_analyzer.client = mock_client
        
        # Mock unexpected response format
        mock_client.list.return_value = "unexpected format"

        # Test
        result = ai_analyzer.test_connection()

        # Assertions
        assert result is False

    @patch('ai_analyzer.ollama.Client')
    def test_test_connection_exception(self, mock_client_class, ai_analyzer):
        """Test connection with exception"""
        # Setup mock
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        ai_analyzer.client = mock_client
        
        mock_client.list.side_effect = Exception("Connection failed")

        # Test
        result = ai_analyzer.test_connection()

        # Assertions
        assert result is False

    @patch('ai_analyzer.ollama.Client')
    def test_ensure_model_available_model_exists(self, mock_client_class, ai_analyzer):
        """Test ensure_model_available when model already exists"""
        # Setup mock
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        ai_analyzer.client = mock_client
        
        # Mock model exists
        mock_model = Mock()
        mock_model.model = ai_analyzer.model
        mock_response = Mock()
        mock_response.models = [mock_model]
        mock_client.list.return_value = mock_response

        # Test
        result = ai_analyzer.ensure_model_available()

        # Assertions
        assert result is True
        mock_client.list.assert_called_once()
        mock_client.pull.assert_not_called()

    @patch('ai_analyzer.ollama.Client')
    def test_ensure_model_available_model_needs_pull(self, mock_client_class, ai_analyzer):
        """Test ensure_model_available when model needs to be pulled"""
        # Setup mock
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        ai_analyzer.client = mock_client
        
        # Mock model doesn't exist
        mock_response = Mock()
        mock_response.models = []
        mock_client.list.return_value = mock_response

        # Test
        result = ai_analyzer.ensure_model_available()

        # Assertions
        assert result is True
        mock_client.list.assert_called_once()
        mock_client.pull.assert_called_once_with(ai_analyzer.model)

    @patch('ai_analyzer.ollama.Client')
    def test_ensure_model_available_exception(self, mock_client_class, ai_analyzer):
        """Test ensure_model_available with exception"""
        # Setup mock
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        ai_analyzer.client = mock_client
        
        mock_client.list.side_effect = Exception("Network error")

        # Test
        result = ai_analyzer.ensure_model_available()

        # Assertions
        assert result is False

    def test_prompt_construction_includes_filename(self, ai_analyzer, mock_ollama_client):
        """Test that the prompt includes the original filename"""
        ai_analyzer.client = mock_ollama_client
        
        mock_response = Mock()
        mock_response.message.content = json.dumps({
            "description": "test",
            "quality": "crisp",
            "primary_subject": "swimmer",
            "primary_subject_box": {"x": 0, "y": 0, "width": 100, "height": 100},
            "recommended_crop": {
                "crop_box": {"x": 0, "y": 0, "width": 100, "height": 100},
                "aspect_ratio": "16:9",
                "composition_rule": "rule_of_thirds",
                "confidence": 0.85
            },
            "color_analysis": {
                "dominant_colors": ["blue"],
                "exposure_assessment": "properly_exposed",
                "white_balance_assessment": "neutral",
                "contrast_level": "good",
                "brightness_adjustment_needed": 0,
                "contrast_adjustment_needed": 0
            },
            "swimming_context": {
                "event_type": "freestyle",
                "pool_type": "indoor",
                "time_of_event": "mid_race",
                "lane_number": 1
            },
            "processing_recommendation": "enhance_only"
        })
        mock_ollama_client.chat.return_value = mock_response

        filename = "DSC09123.ARW"
        ai_analyzer.analyze_photo("/test/path.jpg", filename)

        # Check that filename was included in prompt
        call_args = mock_ollama_client.chat.call_args
        prompt = call_args[1]['messages'][0]['content']
        assert filename in prompt

    def test_analyze_photo_uses_correct_schema(self, ai_analyzer, mock_ollama_client, sample_photo_analysis):
        """Test that analyze_photo uses the correct Pydantic schema"""
        ai_analyzer.client = mock_ollama_client
        
        mock_response = Mock()
        mock_response.message.content = json.dumps(sample_photo_analysis)
        mock_ollama_client.chat.return_value = mock_response

        ai_analyzer.analyze_photo("/test/image.jpg", "test.jpg")

        # Verify schema was passed to chat
        call_args = mock_ollama_client.chat.call_args
        assert 'format' in call_args[1]
        assert call_args[1]['format'] == PhotoAnalysis.model_json_schema()

    def test_analyze_photo_temperature_settings(self, ai_analyzer, mock_ollama_client, sample_photo_analysis):
        """Test that analyze_photo uses correct temperature settings"""
        ai_analyzer.client = mock_ollama_client
        
        mock_response = Mock()
        mock_response.message.content = json.dumps(sample_photo_analysis)
        mock_ollama_client.chat.return_value = mock_response

        ai_analyzer.analyze_photo("/test/image.jpg", "test.jpg")

        # Verify temperature settings
        call_args = mock_ollama_client.chat.call_args
        options = call_args[1]['options']
        assert options['temperature'] == 0.1
        assert options['top_p'] == 0.9

    @patch('ai_analyzer.ollama.Client')
    def test_analyze_photo_timeout_handling(self, mock_client_class, ai_analyzer):
        """Test photo analysis with timeout"""
        from tests.fixtures.mock_data import MockOllamaResponses
        
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        ai_analyzer.client = mock_client
        
        mock_client.chat.side_effect = MockOllamaResponses.timeout_response()
        
        result = ai_analyzer.analyze_photo("/test/image.jpg", "test.jpg")
        
        assert result is None

    @patch('ai_analyzer.ollama.Client')
    def test_analyze_photo_connection_error(self, mock_client_class, ai_analyzer):
        """Test photo analysis with connection error"""
        from tests.fixtures.mock_data import MockOllamaResponses
        
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        ai_analyzer.client = mock_client
        
        mock_client.chat.side_effect = MockOllamaResponses.connection_error_response()
        
        result = ai_analyzer.analyze_photo("/test/image.jpg", "test.jpg")
        
        assert result is None

    @patch('ai_analyzer.ollama.Client')
    def test_analyze_photo_malformed_json(self, mock_client_class, ai_analyzer):
        """Test photo analysis with malformed JSON response"""
        from tests.fixtures.mock_data import MockOllamaResponses
        
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        ai_analyzer.client = mock_client
        
        mock_client.chat.return_value = MockOllamaResponses.malformed_json_response()
        
        result = ai_analyzer.analyze_photo("/test/image.jpg", "test.jpg")
        
        assert result is None

    @patch('ai_analyzer.ollama.Client')
    def test_analyze_photo_rate_limit(self, mock_client_class, ai_analyzer):
        """Test photo analysis with rate limiting"""
        from tests.fixtures.mock_data import MockOllamaResponses
        
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        ai_analyzer.client = mock_client
        
        mock_client.chat.side_effect = MockOllamaResponses.rate_limit_response()
        
        result = ai_analyzer.analyze_photo("/test/image.jpg", "test.jpg")
        
        assert result is None

    @patch('ai_analyzer.ollama.Client')
    def test_test_connection_empty_model_list(self, mock_client_class, ai_analyzer):
        """Test connection test with empty model list"""
        from tests.fixtures.mock_data import MockOllamaResponses
        
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        ai_analyzer.client = mock_client
        
        mock_client.list.return_value = MockOllamaResponses.empty_model_list()
        
        result = ai_analyzer.test_connection()
        
        assert result is False

    @patch('ai_analyzer.ollama.Client')
    def test_test_connection_different_models(self, mock_client_class, ai_analyzer):
        """Test connection test when target model is not available"""
        from tests.fixtures.mock_data import MockOllamaResponses
        
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        ai_analyzer.client = mock_client
        
        mock_client.list.return_value = MockOllamaResponses.model_list_with_different_models()
        
        result = ai_analyzer.test_connection()
        
        assert result is False

    @pytest.mark.slow
    def test_analyze_photo_performance_benchmark(self, ai_analyzer, mock_ollama_client, sample_photo_analysis, performance_timer):
        """Test photo analysis performance benchmark"""
        ai_analyzer.client = mock_ollama_client
        
        mock_response = Mock()
        mock_response.message.content = json.dumps(sample_photo_analysis)
        mock_ollama_client.chat.return_value = mock_response
        
        # Benchmark analysis time
        performance_timer.start("analysis")
        
        for _ in range(10):  # Run multiple times for average
            ai_analyzer.analyze_photo("/test/image.jpg", "test.jpg")
        
        performance_timer.stop("analysis")
        
        # Should complete 10 analyses in under 1 second (mocked)
        performance_timer.assert_faster_than(1.0, "analysis")

    def test_analyze_photo_with_edge_case_data(self, ai_analyzer, mock_ollama_client):
        """Test photo analysis with edge case analysis data"""
        from tests.fixtures.mock_data import MockPhotoAnalysisData
        
        ai_analyzer.client = mock_ollama_client
        
        edge_case_analysis = MockPhotoAnalysisData.edge_case_minimal_analysis()
        mock_response = Mock()
        mock_response.message.content = json.dumps(edge_case_analysis.model_dump())
        mock_ollama_client.chat.return_value = mock_response
        
        result = ai_analyzer.analyze_photo("/test/image.jpg", "test.jpg")
        
        assert result is not None
        assert result.description == ""  # Empty description should be preserved
        assert result.color_analysis.dominant_colors == []  # Empty colors list

    def test_analyze_photo_invalid_image_path(self, ai_analyzer, mock_ollama_client):
        """Test photo analysis with invalid image path"""
        ai_analyzer.client = mock_ollama_client
        
        # Test with None path
        result = ai_analyzer.analyze_photo(None, "test.jpg")
        assert result is None
        
        # Test with empty path
        result = ai_analyzer.analyze_photo("", "test.jpg")
        assert result is None

    def test_client_initialization_custom_host(self):
        """Test AIAnalyzer with custom host configuration"""
        custom_host = "http://custom-ollama:12345"
        analyzer = AIAnalyzer(ollama_host=custom_host, model="custom-model")
        
        assert analyzer.ollama_host == custom_host
        assert analyzer.model == "custom-model"

    @patch('ai_analyzer.ollama.Client')
    def test_ensure_model_pull_failure(self, mock_client_class, ai_analyzer):
        """Test model pull failure handling"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        ai_analyzer.client = mock_client
        
        # Mock model doesn't exist
        mock_response = Mock()
        mock_response.models = []
        mock_client.list.return_value = mock_response
        
        # Mock pull failure
        mock_client.pull.side_effect = Exception("Pull failed")
        
        result = ai_analyzer.ensure_model_available()
        
        assert result is False
#!/usr/bin/env python3
"""
Test the upload functionality with a processed image
"""
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from immich_client import ImmichClient
from schemas import PhotoAnalysis, BoundingBox, CropSuggestion, ColorAnalysis, SwimmingEventContext

def test_upload():
    """Test uploading a processed image to Immich"""
    
    # Initialize client
    client = ImmichClient(
        os.getenv('IMMICH_API_URL', 'http://immich_server:2283'),
        os.getenv('IMMICH_API_KEY')
    )
    
    # Check for processed images
    processed_dir = "/app/processed"
    processed_files = list(os.listdir(processed_dir))
    
    if not processed_files:
        print("No processed files found")
        return False
    
    test_file = os.path.join(processed_dir, processed_files[0])
    print(f"Testing upload with: {test_file}")
    
    # Create mock analysis for testing
    analysis = PhotoAnalysis(
        description="Test swimming photo showing freestyle stroke in lane 4",
        quality="slightly_blurry",
        quality_confidence=0.85,
        primary_subject="swimmer",
        primary_subject_box=BoundingBox(x=30.0, y=25.0, width=40.0, height=50.0),
        subject_confidence=0.9,
        recommended_crop=CropSuggestion(
            crop_box=BoundingBox(x=15.0, y=10.0, width=70.0, height=80.0),
            aspect_ratio="16:9",
            composition_rule="rule_of_thirds",
            confidence=0.8
        ),
        color_analysis=ColorAnalysis(
            dominant_colors=["blue", "white"],
            exposure_assessment="properly_exposed",
            white_balance_assessment="neutral",
            contrast_level="normal",
            brightness_adjustment_needed=5,
            contrast_adjustment_needed=10
        ),
        swimming_context=SwimmingEventContext(
            event_type="freestyle",
            pool_type="indoor",
            time_of_event="mid_race",
            lane_number=4
        ),
        processing_recommendation="enhance_only"
    )
    
    # Get or create album
    album_id = client.get_or_create_album("AI Processed Photos")
    print(f"Album ID: {album_id}")
    
    # Attempt upload with corrected format
    try:
        asset_id = client.upload_photo(
            test_file,
            os.path.basename(test_file),
            analysis,
            album_id
        )
        
        if asset_id:
            print(f"✓ Upload successful! Asset ID: {asset_id}")
            return True
        else:
            print("✗ Upload failed - no asset ID returned")
            return False
            
    except Exception as e:
        print(f"✗ Upload error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_upload()
    sys.exit(0 if success else 1)
"""
Integration example showing how to use the AI pipeline with the existing photo processor.
This demonstrates how to integrate the new AI capabilities into the current system.
"""

import asyncio
from pathlib import Path
from typing import List, Dict, Any
import logging

# Existing photo processor imports
from main_v2 import PhotoProcessorService
from immich_client_v2 import ImmichClient
from recipe_storage import RecipeStorage

# New AI components
from ai_components.orchestrator import (
    PhotoProcessingOrchestrator,
    ProcessingConfig
)
from ai_components.services import (
    CullingService,
    BurstGroupingService,
    SceneAnalysisService,
    RotationDetectionService,
    RAWDevelopmentService
)

logger = logging.getLogger(__name__)


class AIEnhancedPhotoProcessor(PhotoProcessorService):
    """
    Enhanced photo processor that integrates AI capabilities
    while maintaining compatibility with existing infrastructure.
    """
    
    def __init__(self):
        # Initialize parent class
        super().__init__()
        
        # Initialize AI orchestrator
        self.ai_config = ProcessingConfig(
            style_preset="natural",
            cull_aggressively=True,
            quality_threshold=5.0,
            enable_rotation_correction=True,
            enable_smart_crops=True
        )
        
        # Create AI services
        self._init_ai_services()
        
        # Create orchestrator
        self.ai_orchestrator = PhotoProcessingOrchestrator(
            config=self.ai_config,
            output_dir=Path(self.output_folder),
            services={
                'culling': self.culling_service,
                'burst': self.burst_service,
                'scene': self.scene_service,
                'rotation': self.rotation_service,
                'raw': self.raw_service
            }
        )
        
        logger.info("AI-Enhanced Photo Processor initialized")
    
    def _init_ai_services(self):
        """Initialize AI services"""
        device = "cuda"  # Use GPU
        
        self.culling_service = CullingService(
            qwen_host=self.ollama_host,
            device=device
        )
        self.burst_service = BurstGroupingService(device=device)
        self.scene_service = SceneAnalysisService(
            qwen_host=self.ollama_host,
            device=device
        )
        self.rotation_service = RotationDetectionService()
        self.raw_service = RAWDevelopmentService(device=device)
    
    async def process_with_ai(self, image_path: Path) -> Dict[str, Any]:
        """
        Process a single image using AI pipeline.
        This method can be called from the existing process_image method.
        """
        try:
            # Stage 1: Quality check
            cull_decision = await self.culling_service.cull_single(image_path)
            if not cull_decision.keep:
                logger.info(f"Image culled: {cull_decision.reason.value}")
                return {
                    'success': False,
                    'reason': 'culled',
                    'details': cull_decision.reason.value
                }
            
            # Stage 2: Scene analysis
            scene_analysis = await self.scene_service.analyze_scene(
                image_path,
                style_preset=self.ai_config.style_preset
            )
            
            # Stage 3: Rotation detection
            rotation_info = self.rotation_service.detect_rotation(image_path)
            
            # Stage 4: RAW development
            processed_result = await self.raw_service.develop_raw(
                image_path,
                scene_analysis,
                style_preset=self.ai_config.style_preset
            )
            
            # Apply rotation if needed
            if rotation_info['needs_rotation']:
                processed_result.image = self.rotation_service.apply_rotation(
                    processed_result.image,
                    rotation_info['angle']
                )
            
            return {
                'success': True,
                'image': processed_result.image,
                'scene_type': scene_analysis.scene_type.value,
                'processing_parameters': processed_result.parameters_used,
                'quality_score': cull_decision.confidence,
                'rotation_applied': rotation_info.get('angle', 0)
            }
            
        except Exception as e:
            logger.error(f"AI processing failed: {e}")
            return {
                'success': False,
                'reason': 'error',
                'details': str(e)
            }
    
    async def process_image_enhanced(
        self,
        image_path: Path,
        recipe_name: Optional[str] = None
    ) -> bool:
        """
        Enhanced version of process_image that uses AI capabilities.
        Falls back to original processing if AI fails.
        """
        # Try AI processing first
        ai_result = await self.process_with_ai(image_path)
        
        if ai_result['success']:
            # Use AI-processed image
            processed_image = ai_result['image']
            
            # Generate AI description
            ai_description = (
                f"Scene: {ai_result['scene_type']}, "
                f"Quality: {ai_result['quality_score']:.2f}, "
                f"Rotation: {ai_result['rotation_applied']}°"
            )
            
            # Upload to Immich with AI metadata
            upload_result = await self._upload_to_immich(
                original_path=image_path,
                processed_image=processed_image,
                ai_description=ai_description,
                metadata=ai_result
            )
            
            return upload_result
            
        elif ai_result['reason'] == 'culled':
            # Image was culled, mark for deletion or archiving
            logger.info(f"Image culled: {image_path}")
            self._mark_as_culled(image_path)
            return True  # Successfully handled
            
        else:
            # AI processing failed, fall back to original processing
            logger.warning(f"AI processing failed, using fallback: {ai_result['details']}")
            return await super().process_image(image_path, recipe_name)
    
    async def process_burst_sequence(self, photos: List[Path]) -> List[Path]:
        """
        Process a burst sequence of photos.
        Returns the selected best photos.
        """
        # Use burst grouping service
        groups, standalone = await self.burst_service.process_photos(photos)
        
        selected_photos = []
        
        if groups:
            # Select best from each burst
            for group in groups:
                selected = await self.burst_service.select_from_bursts(
                    [group],
                    keep_per_burst=self.ai_config.keep_per_burst
                )
                selected_photos.extend(selected)
        
        # Add standalone photos
        selected_photos.extend(standalone)
        
        return selected_photos
    
    async def _upload_to_immich(
        self,
        original_path: Path,
        processed_image: Any,
        ai_description: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """Upload processed image to Immich with AI metadata"""
        try:
            # Save processed image temporarily
            temp_path = Path(self.output_folder) / f"temp_{original_path.name}"
            
            # Convert and save based on image type
            from PIL import Image
            if hasattr(processed_image, 'save'):
                processed_image.save(temp_path, quality=95)
            else:
                # NumPy array
                img = Image.fromarray(processed_image)
                img.save(temp_path, quality=95)
            
            # Upload both original and processed
            upload_result = self.immich_client.upload_photo_with_original(
                processed_path=temp_path,
                original_path=original_path,
                description=ai_description,
                tags=[
                    "ai-processed",
                    f"scene:{metadata['scene_type']}",
                    f"style:{self.ai_config.style_preset}"
                ]
            )
            
            # Clean up temp file
            temp_path.unlink()
            
            return upload_result
            
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return False
    
    def _mark_as_culled(self, image_path: Path):
        """Mark image as culled (could move to a separate folder)"""
        culled_dir = Path(self.output_folder) / "culled"
        culled_dir.mkdir(exist_ok=True)
        
        # Move to culled directory
        culled_path = culled_dir / image_path.name
        image_path.rename(culled_path)
        
        logger.info(f"Moved culled image to: {culled_path}")


# Example: Integrating with existing watch folder
async def enhanced_watch_folder():
    """Enhanced watch folder with AI processing"""
    processor = AIEnhancedPhotoProcessor()
    
    # Watch for new photos
    watch_folder = Path(processor.watch_folder)
    
    while True:
        # Get new photos
        raw_photos = list(watch_folder.glob("*.NEF")) + \
                    list(watch_folder.glob("*.ARW")) + \
                    list(watch_folder.glob("*.CR2"))
        
        if raw_photos:
            logger.info(f"Found {len(raw_photos)} new photos")
            
            # Group by timestamp to detect bursts
            from collections import defaultdict
            time_groups = defaultdict(list)
            
            for photo in raw_photos:
                # Group by minute
                mtime = photo.stat().st_mtime
                minute_key = int(mtime // 60)
                time_groups[minute_key].append(photo)
            
            # Process each time group
            for minute, group_photos in time_groups.items():
                if len(group_photos) > 3:
                    # Likely a burst sequence
                    logger.info(f"Processing burst sequence of {len(group_photos)} photos")
                    selected = await processor.process_burst_sequence(group_photos)
                    
                    for photo in selected:
                        await processor.process_image_enhanced(photo)
                else:
                    # Process individually
                    for photo in group_photos:
                        await processor.process_image_enhanced(photo)
        
        # Wait before next check
        await asyncio.sleep(30)


# Example: Batch processing with style
async def batch_process_with_style():
    """Batch process photos with specific style"""
    # Configure for dramatic landscape processing
    config = ProcessingConfig(
        style_preset="dramatic",
        processing_intent="artistic",
        cull_aggressively=True,
        quality_threshold=6.0,
        export_formats=["full", "web", "instagram_feed"]
    )
    
    orchestrator = PhotoProcessingOrchestrator(
        config=config,
        output_dir=Path("./output/dramatic_landscapes")
    )
    
    # Find all landscape photos
    landscape_photos = list(Path("./inbox").glob("DSC*.NEF"))
    
    # Process
    result = await orchestrator.process_photo_shoot(
        landscape_photos,
        progress_callback=lambda msg, pct: print(f"{pct}% - {msg}")
    )
    
    print(f"Processed {result.photos_processed} dramatic landscapes!")
    print(f"Culled {result.photos_culled} low-quality shots")
    print(f"Found {result.burst_groups_found} burst sequences")


# Example: Recipe-based processing
async def process_with_recipe():
    """Process photos using a saved recipe"""
    processor = AIEnhancedPhotoProcessor()
    
    # Load recipe
    recipe_storage = RecipeStorage()
    recipe = recipe_storage.get_recipe("wedding_natural_light")
    
    if recipe:
        # Update AI config from recipe
        processor.ai_config.style_preset = recipe.get('style', 'natural')
        processor.ai_config.quality_threshold = recipe.get('quality_threshold', 5.0)
        
        # Process photos
        photos = list(Path("./wedding_photos").glob("*.ARW"))
        
        for photo in photos:
            success = await processor.process_image_enhanced(photo, recipe_name="wedding_natural_light")
            if success:
                print(f"✓ Processed {photo.name}")
            else:
                print(f"✗ Failed {photo.name}")


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "watch":
            # Run enhanced watch folder
            asyncio.run(enhanced_watch_folder())
        elif sys.argv[1] == "batch":
            # Run batch processing
            asyncio.run(batch_process_with_style())
        elif sys.argv[1] == "recipe":
            # Run recipe processing
            asyncio.run(process_with_recipe())
    else:
        print("Usage:")
        print("  python integrate_ai_pipeline.py watch    # Watch folder with AI")
        print("  python integrate_ai_pipeline.py batch    # Batch process")
        print("  python integrate_ai_pipeline.py recipe   # Process with recipe")
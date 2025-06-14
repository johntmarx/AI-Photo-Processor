"""
Main entry point for AI-powered photo processing
Example usage and CLI interface for the photo processing pipeline.
"""

import asyncio
import argparse
from pathlib import Path
from typing import List, Optional
import logging
from datetime import datetime
import json

from ai_components.orchestrator import (
    PhotoProcessingOrchestrator,
    ProcessingConfig,
    ProcessingResult
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_argparser():
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(
        description="AI-Powered RAW Photo Processing Pipeline"
    )
    
    # Input/Output
    parser.add_argument(
        "input",
        type=Path,
        help="Input directory containing RAW photos or single RAW file"
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output directory for processed photos"
    )
    
    # Style and processing
    parser.add_argument(
        "--style",
        choices=["natural", "vibrant", "moody", "dramatic", "soft", "film"],
        default="natural",
        help="Processing style preset"
    )
    parser.add_argument(
        "--intent",
        choices=["quality", "artistic", "documentary", "social"],
        default="quality",
        help="Processing intent"
    )
    
    # Culling options
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=5.0,
        help="Minimum quality score to keep photos (0-10)"
    )
    parser.add_argument(
        "--keep-all",
        action="store_true",
        help="Skip culling, process all photos"
    )
    
    # Burst options
    parser.add_argument(
        "--burst-keep",
        type=int,
        default=2,
        help="Number of photos to keep per burst sequence"
    )
    parser.add_argument(
        "--burst-window",
        type=float,
        default=5.0,
        help="Time window in seconds for burst detection"
    )
    
    # Export options
    parser.add_argument(
        "--export-formats",
        nargs="+",
        default=["full", "web", "social"],
        help="Export formats to generate"
    )
    parser.add_argument(
        "--no-rotation",
        action="store_true",
        help="Disable automatic rotation correction"
    )
    
    # Performance options
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of GPU"
    )
    
    # Recipe options
    parser.add_argument(
        "--recipe",
        type=Path,
        help="Use saved recipe file"
    )
    parser.add_argument(
        "--save-recipe",
        type=Path,
        help="Save processing parameters as recipe"
    )
    
    # Other options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze photos without processing"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    return parser


def find_raw_photos(input_path: Path) -> List[Path]:
    """Find all RAW photos in input path"""
    raw_extensions = {'.nef', '.arw', '.cr2', '.cr3', '.dng', '.raf', '.orf', '.rw2'}
    
    if input_path.is_file():
        if input_path.suffix.lower() in raw_extensions:
            return [input_path]
        else:
            logger.warning(f"Not a RAW file: {input_path}")
            return []
    
    elif input_path.is_dir():
        raw_photos = []
        for ext in raw_extensions:
            raw_photos.extend(input_path.glob(f"*{ext}"))
            raw_photos.extend(input_path.glob(f"*{ext.upper()}"))
        
        return sorted(raw_photos)
    
    else:
        logger.error(f"Invalid input path: {input_path}")
        return []


def progress_callback(message: str, percentage: int):
    """Progress callback for processing updates"""
    bar_length = 40
    filled = int(bar_length * percentage / 100)
    bar = '█' * filled + '░' * (bar_length - filled)
    print(f"\r[{bar}] {percentage:3d}% - {message}", end='', flush=True)
    if percentage >= 100:
        print()  # New line when complete


async def process_photos(args):
    """Main processing function"""
    # Find RAW photos
    raw_photos = find_raw_photos(args.input)
    
    if not raw_photos:
        logger.error("No RAW photos found!")
        return 1
    
    logger.info(f"Found {len(raw_photos)} RAW photos")
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Setup processing config
    config = ProcessingConfig(
        style_preset=args.style,
        processing_intent=args.intent,
        cull_aggressively=not args.keep_all,
        quality_threshold=args.quality_threshold,
        keep_per_burst=args.burst_keep,
        burst_time_window=args.burst_window,
        export_formats=args.export_formats,
        batch_size=args.batch_size,
        max_workers=args.workers,
        use_gpu=not args.cpu,
        enable_rotation_correction=not args.no_rotation
    )
    
    # Create orchestrator
    orchestrator = PhotoProcessingOrchestrator(config, args.output)
    
    # Load recipe if provided
    recipe = None
    if args.recipe:
        with open(args.recipe, 'r') as f:
            recipe = json.load(f)
        logger.info(f"Using recipe: {args.recipe}")
    
    # Process photos
    try:
        if args.dry_run:
            logger.info("DRY RUN - Analyzing photos only")
            # TODO: Implement analysis-only mode
            return 0
        
        result = await orchestrator.process_photo_shoot(
            raw_photos,
            recipe=recipe,
            progress_callback=progress_callback
        )
        
        # Save recipe if requested
        if args.save_recipe and result.processed_photos:
            # Extract common parameters from first processed photo
            sample_params = result.processed_photos[0].get('parameters_used', {})
            orchestrator.save_recipe(
                args.save_recipe,
                sample_params,
                metadata={
                    'style': args.style,
                    'intent': args.intent,
                    'date': datetime.now().isoformat(),
                    'photos_processed': len(result.processed_photos)
                }
            )
            logger.info(f"Recipe saved to: {args.save_recipe}")
        
        # Print results
        print_results(result)
        
        return 0
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def print_results(result: ProcessingResult):
    """Print processing results summary"""
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    
    print(f"\nPhotos processed: {result.photos_processed}/{result.total_input}")
    print(f"Photos culled: {result.photos_culled}")
    print(f"Burst groups found: {result.burst_groups_found}")
    print(f"Processing time: {result.processing_time:.1f} seconds")
    print(f"Average per photo: {result.average_time_per_photo:.1f} seconds")
    
    if result.culling_stats:
        print("\nCulling reasons:")
        for reason, count in result.culling_stats.get('rejection_reasons', {}).items():
            print(f"  - {reason}: {count}")
    
    if result.burst_stats:
        print(f"\nBurst statistics:")
        print(f"  - Groups found: {result.burst_stats.get('total_groups', 0)}")
        print(f"  - Photos in bursts: {result.burst_stats.get('total_photos_in_bursts', 0)}")
        print(f"  - Reduction ratio: {result.burst_stats.get('reduction_ratio', 0):.1%}")
    
    if result.exports:
        print("\nExports generated:")
        for format_name, paths in result.exports.items():
            print(f"  - {format_name}: {len(paths)} files")
    
    if result.processing_errors:
        print(f"\nErrors: {len(result.processing_errors)}")
        for error in result.processing_errors[:5]:  # Show first 5
            print(f"  - {error}")


def main():
    """Main entry point"""
    parser = setup_argparser()
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Print banner
    print("╔══════════════════════════════════════════╗")
    print("║   AI-Powered RAW Photo Processor v2.0    ║")
    print("║   Intelligent • Automated • Beautiful    ║")
    print("╚══════════════════════════════════════════╝")
    print()
    
    # Run async processing
    return asyncio.run(process_photos(args))


if __name__ == "__main__":
    exit(main())
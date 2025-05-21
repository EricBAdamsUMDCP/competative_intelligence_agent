#!/usr/bin/env python
# scripts/train_ner_model.py

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from core.processors.entity_extractor import EntityExtractor
try:
    from core.collectors.sam_gov import SamGovCollector
    from core.collectors.usaspending_gov import USASpendingCollector
    collectors_available = True
except ImportError:
    collectors_available = False
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("train_ner_model")

def load_training_data(file_path: str) -> List[Dict[str, Any]]:
    """Load training data from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

async def collect_data_for_training(days: int = 30, limit: int = 100) -> List[Dict[str, Any]]:
    """Collect real data from APIs for training purposes."""
    logger.info(f"Collecting data from APIs (past {days} days, limit {limit})...")
    
    if not collectors_available:
        logger.warning("Collectors are not available. Make sure sam_gov.py and usaspending_gov.py exist.")
        return []
    
    all_results = []
    
    # Collect data from SAM.gov
    try:
        sam_collector = SamGovCollector()
        sam_results = await sam_collector.run()
        logger.info(f"Collected {len(sam_results)} items from SAM.gov")
        all_results.extend(sam_results)
    except Exception as e:
        logger.error(f"Error collecting from SAM.gov: {str(e)}")
    
    # Collect data from USASpending.gov
    try:
        usa_collector = USASpendingCollector()
        usa_results = await usa_collector.run()
        logger.info(f"Collected {len(usa_results)} items from USASpending.gov")
        all_results.extend(usa_results)
    except Exception as e:
        logger.error(f"Error collecting from USASpending.gov: {str(e)}")
    
    logger.info(f"Collected {len(all_results)} total items from all sources")
    return all_results

def main():
    parser = argparse.ArgumentParser(description="Train a custom NER model for government contracting")
    
    # Input options
    input_group = parser.add_argument_group('Input Options')
    input_group.add_argument('--training-file', type=str, help='Path to training data JSON file')
    input_group.add_argument('--collect-data', action='store_true', help='Collect real data from APIs for training')
    input_group.add_argument('--days', type=int, default=30, help='Number of days of data to collect (default: 30)')
    input_group.add_argument('--limit', type=int, default=100, help='Limit of items to collect per source (default: 100)')
    
    # Training options
    training_group = parser.add_argument_group('Training Options')
    training_group.add_argument('--base-model', type=str, default="en_core_web_lg", help='Base model to start from')
    training_group.add_argument('--iterations', type=int, default=30, help='Number of training iterations')
    training_group.add_argument('--batch-size', type=int, default=16, help='Training batch size')
    training_group.add_argument('--dropout', type=float, default=0.2, help='Dropout rate during training')
    training_group.add_argument('--eval-split', type=float, default=0.2, help='Fraction of data for evaluation')
    
    # Augmentation options
    aug_group = parser.add_argument_group('Data Augmentation Options')
    aug_group.add_argument('--augment', action='store_true', help='Apply data augmentation')
    aug_group.add_argument('--aug-factor', type=int, default=2, help='Augmentation factor')
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--output-dir', type=str, required=True, help='Directory to save the model and metrics')
    output_group.add_argument('--export-training', action='store_true', help='Export processed training data')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.training_file and not args.collect_data:
        logger.error("Either --training-file or --collect-data must be specified")
        parser.print_help()
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize entity extractor
    entity_extractor = EntityExtractor(model_name=args.base_model)
    
    # Get training data
    training_data = []
    
    if args.training_file:
        logger.info(f"Loading training data from {args.training_file}")
        training_data = load_training_data(args.training_file)
        logger.info(f"Loaded {len(training_data)} training examples")
    
    if args.collect_data:
        if not collectors_available:
            logger.error("Cannot collect data: collectors are not available")
            if not training_data:
                logger.error("No training data available. Exiting.")
                sys.exit(1)
        else:
            logger.info("Collecting data from APIs for training")
            collected_data = asyncio.run(collect_data_for_training(args.days, args.limit))
            
            # Process collected data to extract entities
            processed_data = []
            for item in collected_data:
                try:
                    processed_item = entity_extractor.process_document(item)
                    processed_data.append(processed_item)
                except Exception as e:
                    logger.error(f"Error processing item: {str(e)}")
            
            # Export processed data to training format
            if processed_data:
                training_file = os.path.join(args.output_dir, "collected_training_data.json")
                entity_extractor.export_training_data(processed_data, training_file)
                
                # Load the exported training data
                if os.path.exists(training_file):
                    collected_training_data = load_training_data(training_file)
                    logger.info(f"Generated {len(collected_training_data)} training examples from collected data")
                    
                    # Add to existing training data
                    training_data.extend(collected_training_data)
    
    # Data augmentation if enabled
    if args.augment and training_data:
        logger.info(f"Applying data augmentation with factor {args.aug_factor}")
        training_data = entity_extractor.augment_training_data(
            training_data,
            augmentation_factor=args.aug_factor,
            synonym_replacement=True,
            word_insertion=False,  # Disabled to avoid entity boundary issues
            word_deletion=False    # Disabled to avoid entity boundary issues
        )
        
        # Save augmented training data
        augmented_file = os.path.join(args.output_dir, "augmented_training_data.json")
        with open(augmented_file, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        logger.info(f"Saved augmented training data to {augmented_file}")
    
    # Export training data in DocBin format for spaCy CLI
    if args.export_training:
        docbin_file = os.path.join(args.output_dir, "training_data.spacy")
        entity_extractor.create_training_doc_bin(training_data, docbin_file)
    
    # Train the model
    if training_data:
        logger.info(f"Starting model training with {len(training_data)} examples")
        
        # Create timestamped model directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(args.output_dir, f"ner_model_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Train the model
        metrics = entity_extractor.train_custom_model(
            training_data=training_data,
            output_dir=model_dir,
            n_iter=args.iterations,
            batch_size=args.batch_size,
            dropout=args.dropout,
            eval_split=args.eval_split,
            base_model=args.base_model
        )
        
        # Log final metrics
        final_metrics = metrics.get("final_metrics")
        if final_metrics:
            logger.info(f"Training complete. Final metrics:")
            logger.info(f"  Precision: {final_metrics['precision']:.4f}")
            logger.info(f"  Recall: {final_metrics['recall']:.4f}")
            logger.info(f"  F1: {final_metrics['f1']:.4f}")
        
        logger.info(f"Model saved to {model_dir}")
        
        # Create a symbolic link to the latest model
        latest_link = os.path.join(args.output_dir, "latest")
        if os.path.exists(latest_link) or os.path.islink(latest_link):
            try:
                if os.path.islink(latest_link):
                    os.unlink(latest_link)
                else:
                    os.remove(latest_link)
            except Exception as e:
                logger.warning(f"Failed to remove existing latest link: {str(e)}")
        
        try:
            # On Windows, may need directory junction instead of symlink
            if sys.platform == "win32":
                os.system(f'mklink /J "{latest_link}" "{model_dir}"')
            else:
                os.symlink(model_dir, latest_link, target_is_directory=True)
            logger.info(f"Created link to latest model: {latest_link}")
        except Exception as e:
            logger.warning(f"Failed to create link to latest model: {str(e)}")
    else:
        logger.error("No training data available. Model training aborted.")

if __name__ == "__main__":
    main()
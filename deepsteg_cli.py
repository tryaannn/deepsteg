#!/usr/bin/env python
"""
deepsteg_cli.py
Command-line interface untuk DeepSteg pre-trained models dan steganography
"""

import argparse
import logging
import sys
import os
import json
import numpy as np
import cv2
from PIL import Image
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('deepsteg-cli')

def setup_argparse():
    """Setup argument parser"""
    parser = argparse.ArgumentParser(
        description='DeepSteg - Deep Learning Steganography Command Line Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # List available models
  python deepsteg_cli.py list-models
  
  # Download a pre-trained model
  python deepsteg_cli.py download-model --name gan_basic
  
  # Hide message in image
  python deepsteg_cli.py encode --model gan_basic --image cover.jpg --message "Secret message" --output stego.png
  
  # Extract message from stego image
  python deepsteg_cli.py decode --model gan_basic --image stego.png
  
  # Analyze image for steganography
  python deepsteg_cli.py analyze --image suspect.png --detector cnn_basic
  
  # Train custom model with transfer learning
  python deepsteg_cli.py train --base-model gan_basic --dataset ./my_dataset --name my_custom_model --epochs 20
  
  # Evaluate model performance
  python deepsteg_cli.py evaluate --model my_custom_model --dataset ./test_dataset
'''
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # List models command
    list_parser = subparsers.add_parser('list-models', help='List available pre-trained models')
    list_parser.add_argument('--refresh', action='store_true', help='Refresh from remote catalog')
    
    # List detectors command
    list_detector_parser = subparsers.add_parser('list-detectors', help='List available steganalysis detectors')
    list_detector_parser.add_argument('--refresh', action='store_true', help='Refresh from remote catalog')
    
    # Download model command
    download_parser = subparsers.add_parser('download-model', help='Download a pre-trained model')
    download_parser.add_argument('--name', type=str, required=True, help='Name of the model to download')
    
    # Download detector command
    download_detector_parser = subparsers.add_parser('download-detector', help='Download a steganalysis detector')
    download_detector_parser.add_argument('--name', type=str, required=True, help='Name of the detector to download')
    
    # Encode command
    encode_parser = subparsers.add_parser('encode', help='Hide message in image')
    encode_parser.add_argument('--model', type=str, default='gan_basic', help='Model to use for encoding')
    encode_parser.add_argument('--image', type=str, required=True, help='Cover image path')
    encode_parser.add_argument('--message', type=str, required=True, help='Message to hide')
    encode_parser.add_argument('--output', type=str, required=True, help='Output image path')
    encode_parser.add_argument('--visualize', action='store_true', help='Visualize result with matplotlib')
    
    # Decode command
    decode_parser = subparsers.add_parser('decode', help='Extract message from stego image')
    decode_parser.add_argument('--model', type=str, default='gan_basic', help='Model to use for decoding')
    decode_parser.add_argument('--image', type=str, required=True, help='Stego image path')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze image for steganography')
    analyze_parser.add_argument('--image', type=str, required=True, help='Image to analyze')
    analyze_parser.add_argument('--detector', type=str, default=None, help='Detector model to use')
    analyze_parser.add_argument('--visualize', action='store_true', help='Visualize analysis')
    analyze_parser.add_argument('--output', type=str, help='Output image path for visualization')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train custom model with transfer learning')
    train_parser.add_argument('--base-model', type=str, required=True, help='Base model for transfer learning')
    train_parser.add_argument('--dataset', type=str, required=True, help='Dataset directory')
    train_parser.add_argument('--name', type=str, required=True, help='Name for the new model')
    train_parser.add_argument('--description', type=str, help='Description for the new model')
    train_parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    train_parser.add_argument('--learning-rate', type=float, default=1e-4, help='Initial learning rate')
    train_parser.add_argument('--fine-tune-layers', type=str, default='all', 
                             choices=['all', 'last_n', 'none'], help='Layers to fine-tune')
    train_parser.add_argument('--last-n-layers', type=int, default=2, help='Number of layers to fine-tune if using last_n')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model performance')
    eval_parser.add_argument('--model', type=str, required=True, help='Model to evaluate')
    eval_parser.add_argument('--dataset', type=str, required=True, help='Dataset directory')
    eval_parser.add_argument('--max-images', type=int, default=100, help='Maximum number of images to evaluate')
    eval_parser.add_argument('--output', type=str, help='Output JSON file for results')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export model for sharing')
    export_parser.add_argument('--model', type=str, required=True, help='Model to export')
    export_parser.add_argument('--output', type=str, required=True, help='Output zip file path')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show detailed information about a model')
    info_parser.add_argument('--model', type=str, required=True, help='Model name')
    
    # Capacity command
    capacity_parser = subparsers.add_parser('capacity', help='Calculate embedding capacity of an image')
    capacity_parser.add_argument('--image', type=str, required=True, help='Image path')
    capacity_parser.add_argument('--model', type=str, help='Model to use for capacity estimation')
    
    return parser

def load_image(image_path):
    """Load image from path"""
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
            
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img
    except Exception as e:
        logger.error(f"Error loading image: {str(e)}")
        return None

def save_image(image, output_path):
    """Save image to path"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Convert to BGR for OpenCV
        if image.dtype == np.float32 and image.max() <= 1.0:
            img = (image * 255).astype(np.uint8)
        else:
            img = image.astype(np.uint8)
            
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Save image
        cv2.imwrite(output_path, img)
        logger.info(f"Image saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving image: {str(e)}")
        return False

def visualize_comparison(original, stego, metrics=None):
    """Visualize original vs stego comparison"""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(stego)
    plt.title('Stego Image')
    plt.axis('off')
    
    if metrics:
        plt.suptitle(f"PSNR: {metrics.get('psnr', 0):.2f} dB, SSIM: {metrics.get('ssim', 0):.4f}", fontsize=14)
    
    plt.tight_layout()
    plt.show()

def visualize_analysis(image, analysis):
    """Visualize steganalysis results"""
    plt.figure(figsize=(15, 8))
    
    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(image)
    plt.title('Analyzed Image')
    plt.axis('off')
    
    # LSB plane visualization
    if 'visualizations' in analysis and 'lsb_plane' in analysis['visualizations']:
        import base64
        from io import BytesIO
        
        # Decode base64 image
        lsb_data = base64.b64decode(analysis['visualizations']['lsb_plane'])
        lsb_img = Image.open(BytesIO(lsb_data))
        lsb_array = np.array(lsb_img)
        
        plt.subplot(2, 3, 2)
        plt.imshow(lsb_array, cmap='hot')
        plt.title('LSB Plane (Amplified)')
        plt.axis('off')
    
    # LSB Analysis
    plt.subplot(2, 3, 3)
    if 'lsb_analysis' in analysis:
        lsb_data = analysis['lsb_analysis']
        channels = ['red', 'green', 'blue']
        x = np.arange(len(channels))
        ones = [lsb_data[c]['ones_percentage'] for c in channels]
        zeros = [lsb_data[c]['zeros_percentage'] for c in channels]
        
        plt.bar(x - 0.2, ones, 0.4, label='1s')
        plt.bar(x + 0.2, zeros, 0.4, label='0s')
        plt.axhline(y=50, color='r', linestyle='--', alpha=0.3)
        plt.xticks(x, channels)
        plt.ylabel('Percentage (%)')
        plt.title('LSB Distribution')
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'LSB Analysis not available', ha='center', va='center')
        plt.title('LSB Distribution')
    
    # Detection results
    plt.subplot(2, 3, 4)
    detection = analysis['detection_results']
    plt.text(0.5, 0.2, f"Detection Score: {detection['detection_score']:.4f}", ha='center', fontsize=12)
    plt.text(0.5, 0.4, f"Confidence: {detection['confidence']:.2f}", ha='center', fontsize=12)
    plt.text(0.5, 0.6, f"Estimated Payload: {detection['estimated_payload']:.2f}", ha='center', fontsize=12)
    plt.text(0.5, 0.8, f"Interpretation: {detection['interpretation']}", ha='center', fontsize=12)
    plt.axis('off')
    plt.title('Detection Results')
    
    # Recommendations
    plt.subplot(2, 3, (5, 6))
    recommendations = '\n'.join(analysis.get('recommendations', ['No recommendations']))
    plt.text(0.5, 0.5, recommendations, ha='center', va='center', wrap=True)
    plt.axis('off')
    plt.title('Analysis & Recommendations')
    
    plt.tight_layout()
    plt.show()

def list_models(args):
    """List available pre-trained models"""
    from models.pretrained_model_manager import PretrainedModelManager
    
    logger.info("Listing available pre-trained models...")
    
    # Initialize model manager
    model_manager = PretrainedModelManager()
    
    # Refresh catalog if requested
    if args.refresh:
        logger.info("Refreshing model catalog from remote...")
        model_manager.update_catalog_from_remote()
    
    # Get list of models
    models = model_manager.list_available_models()
    
    if not models:
        logger.info("No models available")
        return
    
    # Print models as a table
    print("\nAvailable Pre-trained Models:")
    print("=" * 80)
    print(f"{'Name':<20} {'Status':<10} {'Description':<50}")
    print("-" * 80)
    
    for name, info in models.items():
        status = "✓ Local" if info.get('is_local', False) else "○ Remote"
        desc = info.get('description', 'No description')
        if len(desc) > 50:
            desc = desc[:47] + "..."
        print(f"{name:<20} {status:<10} {desc:<50}")
    
    print("=" * 80)
    print(f"Total: {len(models)} models, {sum(1 for m in models.values() if m.get('is_local', False))} available locally")
    print()

def list_detectors(args):
    """List available steganalysis detectors"""
    from models.pretrained_detector import PretrainedDetector
    
    logger.info("Listing available steganalysis detectors...")
    
    # Initialize detector
    detector = PretrainedDetector()
    
    # Refresh catalog if requested
    if args.refresh:
        logger.info("Refreshing detector catalog from remote...")
        detector.update_catalog_from_remote()
    
    # Get list of detectors
    detectors = detector.list_available_detectors()
    
    if not detectors:
        logger.info("No detectors available")
        return
    
    # Print detectors as a table
    print("\nAvailable Steganalysis Detectors:")
    print("=" * 80)
    print(f"{'Name':<20} {'Status':<10} {'Accuracy':<10} {'Description':<40}")
    print("-" * 80)
    
    for name, info in detectors.items():
        status = "✓ Local" if info.get('is_local', False) else "○ Remote"
        accuracy = f"{info.get('accuracy', 0)*100:.1f}%" if 'accuracy' in info else 'N/A'
        desc = info.get('description', 'No description')
        if len(desc) > 40:
            desc = desc[:37] + "..."
        print(f"{name:<20} {status:<10} {accuracy:<10} {desc:<40}")
    
    print("=" * 80)
    print(f"Total: {len(detectors)} detectors, {sum(1 for d in detectors.values() if d.get('is_local', False))} available locally")
    print()

def download_model(args):
    """Download a pre-trained model"""
    from models.pretrained_model_manager import PretrainedModelManager
    
    # Initialize model manager
    model_manager = PretrainedModelManager()
    
    # Download model
    logger.info(f"Downloading model '{args.name}'...")
    success = model_manager.download_model(args.name)
    
    if success:
        logger.info(f"Model '{args.name}' downloaded successfully")
        
        # Get model info
        model_info = model_manager.get_model_info(args.name)
        if model_info:
            print("\nModel Information:")
            print(f"  Name: {args.name}")
            print(f"  Description: {model_info.get('description', 'No description')}")
            print(f"  Input Shape: {model_info.get('img_shape', 'Unknown')}")
            print(f"  Message Length: {model_info.get('message_length', 'Unknown')} bits")
            print(f"  Capacity Factor: {model_info.get('capacity_factor', 'Unknown')}")
    else:
        logger.error(f"Failed to download model '{args.name}'")

def download_detector(args):
    """Download a steganalysis detector"""
    from models.pretrained_detector import PretrainedDetector
    
    # Initialize detector
    detector = PretrainedDetector()
    
    # Download detector
    logger.info(f"Downloading detector '{args.name}'...")
    success = detector.download_detector(args.name)
    
    if success:
        logger.info(f"Detector '{args.name}' downloaded successfully")
        
        # Get available detectors to show info
        detectors = detector.list_available_detectors()
        if args.name in detectors:
            info = detectors[args.name]
            print("\nDetector Information:")
            print(f"  Name: {args.name}")
            print(f"  Description: {info.get('description', 'No description')}")
            print(f"  Accuracy: {info.get('accuracy', 0)*100:.1f}%")
            print(f"  Target Methods: {', '.join(info.get('target_methods', ['Unknown']))}")
    else:
        logger.error(f"Failed to download detector '{args.name}'")

def encode_message(args):
    """Hide message in image"""
    from models.pretrained_model_manager import PretrainedModelManager
    
    # Load cover image
    cover_image = load_image(args.image)
    if cover_image is None:
        return
    
    # Initialize model manager
    model_manager = PretrainedModelManager()
    
    # Encode message
    logger.info(f"Encoding message using model '{args.model}'...")
    
    start_time = time.time()
    stego_image, metrics = model_manager.encode_with_model(
        args.model, cover_image, args.message
    )
    execution_time = time.time() - start_time
    
    if stego_image is None:
        logger.error(f"Failed to encode message: {metrics.get('error', 'Unknown error')}")
        return
    
    # Save result
    success = save_image(stego_image, args.output)
    if not success:
        return
    
    # Print metrics
    print("\nEncoding Metrics:")
    print(f"  PSNR: {metrics.get('psnr', 0):.2f} dB")
    print(f"  SSIM: {metrics.get('ssim', 0):.4f}")
    print(f"  Capacity Used: {metrics.get('capacity_used', 0):.2f}%")
    print(f"  Execution Time: {execution_time:.2f} seconds")
    
    # Visualize if requested
    if args.visualize:
        visualize_comparison(cover_image, stego_image, metrics)

def decode_message(args):
    """Extract message from stego image"""
    from models.pretrained_model_manager import PretrainedModelManager
    
    # Load stego image
    stego_image = load_image(args.image)
    if stego_image is None:
        return
    
    # Initialize model manager
    model_manager = PretrainedModelManager()
    
    # Decode message
    logger.info(f"Decoding message using model '{args.model}'...")
    
    start_time = time.time()
    message, metadata = model_manager.decode_with_model(
        args.model, stego_image
    )
    execution_time = time.time() - start_time
    
    if message is None:
        logger.error(f"Failed to decode message: {metadata.get('error', 'Unknown error')}")
        return
    
    # Print message
    print("\nDecoded Message:")
    print(f"\"{message}\"")
    
    # Print metadata
    print("\nDecoding Metadata:")
    print(f"  Model: {args.model}")
    print(f"  Message Length: {metadata.get('message_length', 0)} bits")
    print(f"  Execution Time: {execution_time:.2f} seconds")

def analyze_image(args):
    """Analyze image for steganography"""
    from models.pretrained_detector import PretrainedDetector
    
    # Load image
    image = load_image(args.image)
    if image is None:
        return
    
    # Initialize detector
    detector = PretrainedDetector()
    
    # Analyze image
    logger.info(f"Analyzing image for steganography...")
    
    start_time = time.time()
    analysis = detector.analyze_image_for_report(image, args.detector)
    execution_time = time.time() - start_time
    
    if analysis is None:
        logger.error("Failed to analyze image")
        return
    
    # Print analysis results
    detection = analysis['detection_results']
    
    print("\nSteganalysis Results:")
    print(f"  Is Stego: {'Yes' if detection['is_stego'] else 'No'}")
    print(f"  Detection Score: {detection['detection_score']:.4f}")
    print(f"  Confidence: {detection['confidence']:.2f}")
    print(f"  Interpretation: {detection['interpretation']}")
    print(f"  Estimated Payload: {detection['estimated_payload']:.2f}")
    
    if 'statistical_tests' in detection and 'chi_square' in detection['statistical_tests']:
        chi_sq = detection['statistical_tests']['chi_square']
        print(f"\nStatistical Tests:")
        print(f"  Chi-Square p-value: {chi_sq['p_value']:.4f}")
        print(f"  Chi-Square interpretation: {chi_sq['interpretation']}")
    
    print("\nRecommendations:")
    for i, rec in enumerate(analysis.get('recommendations', ['No recommendations']), 1):
        print(f"  {i}. {rec}")
    
    print(f"\nAnalysis completed in {execution_time:.2f} seconds")
    
    # Visualize if requested
    if args.visualize:
        visualize_analysis(image, analysis)
        
    # Save visualization if requested
    if args.output:
        # Create visualization for saving
        fig = plt.figure(figsize=(15, 8))
        
        # Original image
        plt.subplot(2, 3, 1)
        plt.imshow(image)
        plt.title('Analyzed Image')
        plt.axis('off')
        
        # LSB plane visualization
        if 'visualizations' in analysis and 'lsb_plane' in analysis['visualizations']:
            import base64
            from io import BytesIO
            
            # Decode base64 image
            lsb_data = base64.b64decode(analysis['visualizations']['lsb_plane'])
            lsb_img = Image.open(BytesIO(lsb_data))
            lsb_array = np.array(lsb_img)
            
            plt.subplot(2, 3, 2)
            plt.imshow(lsb_array, cmap='hot')
            plt.title('LSB Plane (Amplified)')
            plt.axis('off')
        
        # LSB Analysis
        plt.subplot(2, 3, 3)
        if 'lsb_analysis' in analysis:
            lsb_data = analysis['lsb_analysis']
            channels = ['red', 'green', 'blue']
            x = np.arange(len(channels))
            ones = [lsb_data[c]['ones_percentage'] for c in channels]
            zeros = [lsb_data[c]['zeros_percentage'] for c in channels]
            
            plt.bar(x - 0.2, ones, 0.4, label='1s')
            plt.bar(x + 0.2, zeros, 0.4, label='0s')
            plt.axhline(y=50, color='r', linestyle='--', alpha=0.3)
            plt.xticks(x, channels)
            plt.ylabel('Percentage (%)')
            plt.title('LSB Distribution')
            plt.legend()
        
        # Detection results
        plt.subplot(2, 3, 4)
        plt.text(0.5, 0.2, f"Detection Score: {detection['detection_score']:.4f}", ha='center', fontsize=12)
        plt.text(0.5, 0.4, f"Confidence: {detection['confidence']:.2f}", ha='center', fontsize=12)
        plt.text(0.5, 0.6, f"Estimated Payload: {detection['estimated_payload']:.2f}", ha='center', fontsize=12)
        plt.text(0.5, 0.8, f"Interpretation: {detection['interpretation']}", ha='center', fontsize=12)
        plt.axis('off')
        plt.title('Detection Results')
        
        # Recommendations
        plt.subplot(2, 3, (5, 6))
        recommendations = '\n'.join(analysis.get('recommendations', ['No recommendations']))
        plt.text(0.5, 0.5, recommendations, ha='center', va='center', wrap=True)
        plt.axis('off')
        plt.title('Analysis & Recommendations')
        
        plt.tight_layout()
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved to {args.output}")

def train_model(args):
    """Train custom model with transfer learning"""
    from models.transfer_learning import TransferLearning
    
    # Check if dataset exists
    if not os.path.exists(args.dataset):
        logger.error(f"Dataset directory not found: {args.dataset}")
        return
    
    # Set up configuration
    config = {
        'base_model': args.base_model,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'fine_tune_layers': args.fine_tune_layers,
        'last_n_layers': args.last_n_layers
    }
    
    # Description defaults to auto-generated if not provided
    description = args.description
    if not description:
        description = f"Fine-tuned from {args.base_model} using transfer learning"
    
    # Initialize transfer learning
    transfer = TransferLearning(config)
    
    # Train model
    logger.info(f"Starting transfer learning with base model '{args.base_model}'...")
    logger.info(f"Training for {args.epochs} epochs with batch size {args.batch_size}")
    
    success = transfer.train(
        dataset_path=args.dataset,
        epochs=args.epochs,
        name=args.name,
        description=description
    )
    
    if success:
        logger.info(f"Training completed successfully. Model saved as '{args.name}'")
        
        # Evaluate model
        logger.info("Evaluating model on training dataset...")
        results = transfer.evaluate_model(args.dataset, max_images=50)
        
        if results:
            print("\nModel Performance:")
            print(f"  PSNR: {results.get('psnr', 0):.2f} dB")
            print(f"  SSIM: {results.get('ssim', 0):.4f}")
            print(f"  Bit Accuracy: {results.get('bit_accuracy', 0)*100:.2f}%")
            print(f"  Message Loss: {results.get('message_loss', 0):.4f}")
    else:
        logger.error("Training failed")

def evaluate_model(args):
    """Evaluate model performance"""
    from models.transfer_learning import TransferLearning
    
    # Check if dataset exists
    if not os.path.exists(args.dataset):
        logger.error(f"Dataset directory not found: {args.dataset}")
        return
    
    # Initialize transfer learning
    transfer = TransferLearning()
    
    # Load model
    logger.info(f"Loading model '{args.model}'...")
    success = transfer.load_base_model(args.model)
    
    if not success:
        logger.error(f"Failed to load model '{args.model}'")
        return
    
    # Create target model (same parameters as base)
    transfer.create_target_model()
    
    # Evaluate model
    logger.info(f"Evaluating model on {args.max_images} images...")
    results = transfer.evaluate_model(args.dataset, max_images=args.max_images)
    
    if results is None:
        logger.error("Evaluation failed")
        return
    
    # Print results
    print("\nEvaluation Results:")
    print(f"  PSNR: {results.get('psnr', 0):.2f} dB (±{results.get('psnr_std', 0):.2f})")
    print(f"  SSIM: {results.get('ssim', 0):.4f} (±{results.get('ssim_std', 0):.4f})")
    print(f"  Bit Accuracy: {results.get('bit_accuracy', 0)*100:.2f}% (±{results.get('bit_accuracy_std', 0)*100:.2f}%)")
    print(f"  Message Loss: {results.get('message_loss', 0):.4f} (±{results.get('message_loss_std', 0):.4f})")
    print(f"  Image Loss: {results.get('image_loss', 0):.6f} (±{results.get('image_loss_std', 0):.6f})")
    
    print("\nModel Information:")
    print(f"  Message Length: {results.get('message_length', 0)} bits")
    print(f"  Capacity Factor: {results.get('capacity_factor', 0)}")
    print(f"  Model Size: {results.get('model_size', {}).get('total_params', 0):,} parameters")
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")

def export_model(args):
    """Export model for sharing"""
    from models.utils_pretrained import export_model_for_sharing
    from models.pretrained_model_manager import PretrainedModelManager
    
    # Initialize model manager
    model_manager = PretrainedModelManager()
    
    # Get model info
    model_info = model_manager.get_model_info(args.model)
    if model_info is None:
        logger.error(f"Model '{args.model}' not found")
        return
    
    # Get model path
    model_path = os.path.join(model_manager.config['models_dir'], args.model)
    if not os.path.exists(model_path):
        logger.error(f"Model directory not found: {model_path}")
        return
    
    # Export model
    logger.info(f"Exporting model '{args.model}' to {args.output}...")
    success = export_model_for_sharing(model_path, args.output, model_info)
    
    if success:
        logger.info(f"Model exported successfully to {args.output}")
    else:
        logger.error("Export failed")

def model_info(args):
    """Show detailed information about a model"""
    from models.pretrained_model_manager import PretrainedModelManager
    
    # Initialize model manager
    model_manager = PretrainedModelManager()
    
    # Get model info
    model_info = model_manager.get_model_info(args.model)
    if model_info is None:
        logger.error(f"Model '{args.model}' not found")
        return
    
    # Print model info
    print(f"\nModel: {args.model}")
    print("=" * 60)
    
    # Basic info
    print(f"Description: {model_info.get('description', 'No description')}")
    print(f"Input Shape: {model_info.get('img_shape', model_info.get('input_shape', 'Unknown'))}")
    print(f"Message Length: {model_info.get('message_length', 0)} bits")
    print(f"Capacity Factor: {model_info.get('capacity_factor', 0)}")
    
    # Creation info
    if 'created_date' in model_info:
        print(f"Created: {model_info['created_date']}")
    
    if 'base_model' in model_info:
        print(f"Base Model: {model_info['base_model']}")
    
    # Performance info
    print("\nPerformance:")
    if 'performance' in model_info:
        perf = model_info['performance']
        print(f"  PSNR: {perf.get('final_psnr', 0):.2f} dB")
        print(f"  SSIM: {perf.get('final_ssim', 0):.4f}")
        print(f"  Bit Accuracy: {perf.get('final_bit_accuracy', 0)*100:.2f}%")
    else:
        print("  No performance data available")
    
    # Training info
    if 'training' in model_info:
        train = model_info['training']
        print("\nTraining Details:")
        print(f"  Epochs: {train.get('epochs', 0)}")
        print(f"  Batch Size: {train.get('batch_size', 0)}")
        print(f"  Learning Rate: {train.get('learning_rate', 0)}")
        print(f"  Fine-tuning: {train.get('fine_tune_layers', 'Unknown')}")
        print(f"  Dataset: {train.get('dataset', 'Unknown')}")
    
    # Files info
    if 'files' in model_info:
        files = model_info['files']
        print("\nModel Files:")
        for name, file_info in files.items():
            size_mb = file_info.get('size_bytes', 0) / (1024 * 1024)
            print(f"  {name}: {size_mb:.2f} MB")
        
        total_size = model_info.get('total_size_mb', 0)
        print(f"Total Size: {total_size:.2f} MB")

def calculate_capacity(args):
    """Calculate embedding capacity of an image"""
    from models.utils_pretrained import calculate_embedding_capacity
    from models.pretrained_model_manager import PretrainedModelManager
    
    # Load image
    image = load_image(args.image)
    if image is None:
        return
    
    # Calculate basic capacity
    capacity = calculate_embedding_capacity(image)
    
    # Print capacity
    print(f"\nImage: {args.image}")
    print(f"Dimensions: {image.shape[1]}x{image.shape[0]} pixels")
    print(f"Total Pixels: {capacity['pixels']:,}")
    
    print("\nEmbedding Capacity (LSB method):")
    print(f"  Maximum: {capacity['total_capacity']['chars']:,} characters")
    print(f"  Maximum: {capacity['total_capacity']['bytes']:,} bytes")
    print(f"  Maximum: {capacity['total_capacity']['bits']:,} bits")
    
    print("\nPayload Options:")
    for ratio, payload in capacity['payload_options'].items():
        print(f"  {ratio}: {payload['chars']:,} characters ({payload['bytes']:,} bytes)")
    
    # Get model-specific capacity if model specified
    if args.model:
        # Initialize model manager
        model_manager = PretrainedModelManager()
        
        # Get model info
        model_info = model_manager.get_model_info(args.model)
        if model_info is not None:
            message_length = model_info.get('message_length', 0)
            print(f"\nModel '{args.model}' Capacity:")
            print(f"  Fixed capacity: {message_length} bits ({message_length//8} bytes)")
            
            # Resize might be required
            img_shape = model_info.get('img_shape', model_info.get('input_shape', None))
            if img_shape:
                if image.shape[0] != img_shape[0] or image.shape[1] != img_shape[1]:
                    print(f"  Note: Image will be resized to {img_shape[1]}x{img_shape[0]} for this model")

def main():
    """Main function"""
    parser = setup_argparse()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Execute command
    commands = {
        'list-models': list_models,
        'list-detectors': list_detectors,
        'download-model': download_model,
        'download-detector': download_detector,
        'encode': encode_message,
        'decode': decode_message,
        'analyze': analyze_image,
        'train': train_model,
        'evaluate': evaluate_model,
        'export': export_model,
        'info': model_info,
        'capacity': calculate_capacity
    }
    
    if args.command in commands:
        try:
            commands[args.command](args)
        except Exception as e:
            logger.error(f"Error executing command: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        logger.error(f"Unknown command: {args.command}")
        parser.print_help()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Main training script for recipe generation system."""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any
import yaml
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.training.trainer import RecipeTrainer, setup_deterministic_training, create_training_config
from src.data.recipe_dataset import RecipeDataModule, create_sample_recipe_data
from src.models.recipe_model import RecipeGenerationModel
from transformers import GPT2Tokenizer


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train recipe generation model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/processed/recipes.json",
        help="Path to recipe data file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--create_sample_data",
        action="store_true",
        help="Create sample recipe data if data file doesn't exist",
    )
    parser.add_argument(
        "--num_sample_recipes",
        type=int,
        default=1000,
        help="Number of sample recipes to create",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup deterministic training
    setup_deterministic_training(args.seed)
    
    # Check if data exists, create sample data if needed
    data_path = Path(args.data_path)
    if not data_path.exists():
        if args.create_sample_data:
            print(f"Creating sample recipe data at {data_path}")
            data_path.parent.mkdir(parents=True, exist_ok=True)
            create_sample_recipe_data(data_path, args.num_sample_recipes)
        else:
            print(f"Data file not found at {data_path}")
            print("Use --create_sample_data to create sample data")
            return
    
    # Initialize tokenizer
    model_name = config["model"]["model_name"]
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize data module
    data_module = RecipeDataModule(
        data_path=data_path,
        tokenizer=tokenizer,
        batch_size=config["data"]["batch_size"],
        max_length=config["data"]["max_length"],
        train_split=config["data"]["train_split"],
        val_split=config["data"]["val_split"],
        test_split=config["data"]["test_split"],
        random_seed=args.seed,
    )
    
    # Create training configuration
    training_config = create_training_config(
        model_name=model_name,
        batch_size=config["data"]["batch_size"],
        learning_rate=config["model"]["learning_rate"],
        max_epochs=config["training"]["max_epochs"],
        max_length=config["data"]["max_length"],
        output_dir=args.output_dir,
    )
    
    # Override with config file values
    training_config["model"].update(config["model"])
    training_config["trainer"].update(config["training"])
    training_config["logger"].update(config["logging"])
    
    # Initialize trainer
    trainer = RecipeTrainer(
        config=training_config,
        data_module=data_module,
        output_dir=args.output_dir,
    )
    
    print("Starting training...")
    print(f"Model: {model_name}")
    print(f"Data: {data_path}")
    print(f"Output: {args.output_dir}")
    print(f"Batch size: {config['data']['batch_size']}")
    print(f"Max epochs: {config['training']['max_epochs']}")
    
    # Train the model
    try:
        trainer.train()
        print("Training completed successfully!")
        
        # Test the model
        print("Running evaluation...")
        trainer.test()
        
        # Generate sample recipes
        print("Generating sample recipes...")
        trainer.generate_sample_recipes(num_samples=5)
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()

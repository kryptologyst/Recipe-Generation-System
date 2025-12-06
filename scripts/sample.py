#!/usr/bin/env python3
"""Sampling script for recipe generation."""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any
import json
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models.recipe_model import RecipeGenerationModel
from src.sampling.sampler import RecipeSampler, create_sampling_config, generate_sample_grid


def load_model(model_path: str, model_name: str = "gpt2") -> RecipeGenerationModel:
    """Load a trained recipe generation model.
    
    Args:
        model_path: Path to the model checkpoint
        model_name: Base model name
        
    Returns:
        Loaded model
    """
    model = RecipeGenerationModel(model_name=model_name)
    
    if Path(model_path).exists():
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location="cpu")
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model checkpoint not found at {model_path}, using pre-trained model")
    
    return model


def main():
    """Main sampling function."""
    parser = argparse.ArgumentParser(description="Generate recipes using trained model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./checkpoints/last.ckpt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="Base model name",
    )
    parser.add_argument(
        "--ingredients",
        type=str,
        nargs="+",
        default=["chicken", "garlic", "lemon", "olive oil", "parsley"],
        help="List of ingredients",
    )
    parser.add_argument(
        "--cuisine",
        type=str,
        default="Italian",
        help="Type of cuisine",
    )
    parser.add_argument(
        "--dietary",
        type=str,
        default=None,
        help="Dietary restrictions",
    )
    parser.add_argument(
        "--cooking_time",
        type=str,
        default="30 minutes",
        help="Cooking time",
    )
    parser.add_argument(
        "--servings",
        type=int,
        default=4,
        help="Number of servings",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="medium",
        help="Difficulty level",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of recipes to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling parameter",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=300,
        help="Maximum generation length",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file to save generated recipes",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--interpolate",
        action="store_true",
        help="Interpolate between two ingredient sets",
    )
    parser.add_argument(
        "--ingredients2",
        type=str,
        nargs="+",
        default=["tomatoes", "basil", "mozzarella", "balsamic vinegar"],
        help="Second set of ingredients for interpolation",
    )
    parser.add_argument(
        "--num_interpolation_steps",
        type=int,
        default=5,
        help="Number of interpolation steps",
    )
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model_path, args.model_name)
    
    # Initialize sampler
    sampler = RecipeSampler(model, seed=args.seed)
    
    print("Recipe Generation System")
    print("=" * 50)
    print(f"Model: {args.model_name}")
    print(f"Ingredients: {', '.join(args.ingredients)}")
    print(f"Cuisine: {args.cuisine}")
    if args.dietary:
        print(f"Dietary: {args.dietary}")
    print(f"Cooking Time: {args.cooking_time}")
    print(f"Servings: {args.servings}")
    print(f"Difficulty: {args.difficulty}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Temperature: {args.temperature}")
    print("=" * 50)
    
    if args.interpolate:
        # Interpolate between two ingredient sets
        print("\nGenerating interpolated recipes...")
        samples = sampler.interpolate_recipes(
            ingredients1=args.ingredients,
            ingredients2=args.ingredients2,
            num_steps=args.num_interpolation_steps,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_length=args.max_length,
            cuisine=args.cuisine,
            dietary=args.dietary,
            cooking_time=args.cooking_time,
            servings=args.servings,
            difficulty=args.difficulty,
        )
        
        print(f"\nGenerated {len(samples)} interpolated recipes:")
        for i, sample in enumerate(samples):
            print(f"\n--- Interpolation Step {i+1} (α={sample['interpolation_alpha']:.2f}) ---")
            print(f"Ingredients: {', '.join(sample['ingredients'])}")
            print(f"Recipe:\n{sample['generated_text']}")
    
    else:
        # Generate regular samples
        print("\nGenerating recipes...")
        samples = sampler.sample_recipes(
            ingredients=args.ingredients,
            num_samples=args.num_samples,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_length=args.max_length,
            cuisine=args.cuisine,
            dietary=args.dietary,
            cooking_time=args.cooking_time,
            servings=args.servings,
            difficulty=args.difficulty,
        )
        
        print(f"\nGenerated {len(samples)} recipes:")
        for i, sample in enumerate(samples):
            print(f"\n--- Recipe {i+1} ---")
            print(f"Ingredients: {', '.join(sample['ingredients'])}")
            print(f"Recipe:\n{sample['generated_text']}")
    
    # Save samples if output file specified
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        format = "json" if output_path.suffix == ".json" else "txt"
        sampler.save_samples(samples, output_path, format=format)
        print(f"\nSamples saved to {output_path}")


if __name__ == "__main__":
    main()

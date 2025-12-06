#!/usr/bin/env python3
"""Evaluation script for recipe generation models."""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any
import json
import torch
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models.recipe_model import RecipeGenerationModel
from src.sampling.sampler import RecipeSampler
from src.evaluation.metrics import RecipeEvaluator, create_model_leaderboard


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


def load_test_data(data_path: str) -> List[Dict[str, Any]]:
    """Load test data for evaluation.
    
    Args:
        data_path: Path to test data file
        
    Returns:
        List of test recipes
    """
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def generate_test_recipes(
    model: RecipeGenerationModel,
    test_data: List[Dict[str, Any]],
    num_samples_per_recipe: int = 3,
) -> tuple:
    """Generate recipes for test data.
    
    Args:
        model: Trained model
        test_data: Test dataset
        num_samples_per_recipe: Number of samples to generate per test recipe
        
    Returns:
        Tuple of (generated_recipes, reference_recipes, input_ingredients)
    """
    sampler = RecipeSampler(model)
    
    generated_recipes = []
    reference_recipes = []
    input_ingredients = []
    
    print(f"Generating recipes for {len(test_data)} test samples...")
    
    for i, recipe in enumerate(test_data):
        if i % 10 == 0:
            print(f"Processing recipe {i+1}/{len(test_data)}")
        
        ingredients = recipe.get("ingredients", [])
        if not ingredients:
            continue
        
        # Generate multiple samples for this recipe
        samples = sampler.sample_recipes(
            ingredients=ingredients,
            num_samples=num_samples_per_recipe,
            temperature=0.8,
            top_p=0.9,
            max_length=300,
        )
        
        # Add generated recipes
        for sample in samples:
            generated_recipes.append(sample["generated_text"])
            reference_recipes.append(recipe.get("instructions", ""))
            input_ingredients.append(ingredients)
    
    return generated_recipes, reference_recipes, input_ingredients


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate recipe generation model")
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
        "--test_data_path",
        type=str,
        default="data/processed/test_recipes.json",
        help="Path to test data file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--num_samples_per_recipe",
        type=int,
        default=3,
        help="Number of samples to generate per test recipe",
    )
    parser.add_argument(
        "--compute_bert_score",
        action="store_true",
        help="Compute BERTScore (slower)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = load_model(args.model_path, args.model_name)
    
    # Load test data
    if not Path(args.test_data_path).exists():
        print(f"Test data not found at {args.test_data_path}")
        print("Please provide a valid test data file")
        return
    
    test_data = load_test_data(args.test_data_path)
    print(f"Loaded {len(test_data)} test recipes")
    
    # Generate recipes for evaluation
    generated_recipes, reference_recipes, input_ingredients = generate_test_recipes(
        model, test_data, args.num_samples_per_recipe
    )
    
    print(f"Generated {len(generated_recipes)} recipes for evaluation")
    
    # Initialize evaluator
    evaluator = RecipeEvaluator(tokenizer=model.tokenizer)
    
    # Run evaluation
    print("Running evaluation...")
    evaluation_report = evaluator.create_evaluation_report(
        generated_recipes=generated_recipes,
        reference_recipes=reference_recipes,
        input_ingredients=input_ingredients,
        model=model.model,
        tokenizer=model.tokenizer,
        device=model.device,
    )
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print("\nGeneration Quality Metrics:")
    for metric, value in evaluation_report["generation_quality"].items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nRecipe Structure Metrics:")
    for metric, value in evaluation_report["structure"].items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nIngredient Coherence Metrics:")
    for metric, value in evaluation_report["ingredient_coherence"].items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nDiversity Metrics:")
    for metric, value in evaluation_report["diversity"].items():
        print(f"  {metric}: {value:.4f}")
    
    if "perplexity" in evaluation_report:
        print(f"\nPerplexity: {evaluation_report['perplexity']:.4f}")
    
    # Save results
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {results_file}")
    
    # Create plots
    try:
        plot_file = output_dir / "evaluation_plots.png"
        evaluator.plot_evaluation_metrics(evaluation_report, save_path=str(plot_file))
        print(f"Plots saved to {plot_file}")
    except Exception as e:
        print(f"Failed to create plots: {e}")
    
    # Create leaderboard (if multiple models)
    model_results = {
        f"{args.model_name}_{Path(args.model_path).stem}": evaluation_report
    }
    
    leaderboard_file = output_dir / "model_leaderboard.csv"
    leaderboard = create_model_leaderboard(model_results, save_path=str(leaderboard_file))
    print(f"Leaderboard saved to {leaderboard_file}")
    
    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()

"""Sampling utilities for recipe generation with various decoding strategies."""

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import torch
import numpy as np
from transformers import PreTrainedTokenizer

from ..models.recipe_model import RecipeGenerationModel


class RecipeSampler:
    """Advanced sampling utilities for recipe generation."""
    
    def __init__(
        self,
        model: RecipeGenerationModel,
        seed: Optional[int] = None,
    ):
        """Initialize the recipe sampler.
        
        Args:
            model: Trained recipe generation model
            seed: Random seed for reproducibility
        """
        self.model = model
        self.device = model.device
        
        if seed is not None:
            self.set_seed(seed)
    
    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducible sampling.
        
        Args:
            seed: Random seed
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def sample_recipes(
        self,
        ingredients: List[str],
        num_samples: int = 5,
        sampling_strategy: str = "nucleus",
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        max_length: int = 300,
        cuisine: Optional[str] = None,
        dietary: Optional[str] = None,
        cooking_time: Optional[str] = None,
        servings: Optional[int] = None,
        difficulty: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Sample multiple recipes with the specified parameters.
        
        Args:
            ingredients: List of input ingredients
            num_samples: Number of recipes to generate
            sampling_strategy: Sampling strategy ("greedy", "nucleus", "top_k", "random")
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty
            max_length: Maximum generation length
            cuisine: Type of cuisine
            dietary: Dietary restrictions
            cooking_time: Cooking time
            servings: Number of servings
            difficulty: Difficulty level
            
        Returns:
            List of generated recipe dictionaries
        """
        recipes = []
        
        for i in range(num_samples):
            # Generate recipe
            recipe_text = self.model.generate_recipe(
                ingredients=ingredients,
                cuisine=cuisine,
                dietary=dietary,
                cooking_time=cooking_time,
                servings=servings,
                difficulty=difficulty,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=(sampling_strategy != "greedy"),
            )
            
            # Parse recipe components
            parsed_recipe = self._parse_recipe(recipe_text)
            
            recipe_dict = {
                "id": i,
                "ingredients": ingredients,
                "generated_text": recipe_text,
                "parsed_recipe": parsed_recipe,
                "sampling_params": {
                    "strategy": sampling_strategy,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "repetition_penalty": repetition_penalty,
                    "max_length": max_length,
                },
                "metadata": {
                    "cuisine": cuisine,
                    "dietary": dietary,
                    "cooking_time": cooking_time,
                    "servings": servings,
                    "difficulty": difficulty,
                },
            }
            
            recipes.append(recipe_dict)
        
        return recipes
    
    def _parse_recipe(self, recipe_text: str) -> Dict[str, Any]:
        """Parse generated recipe text into structured components.
        
        Args:
            recipe_text: Generated recipe text
            
        Returns:
            Dictionary with parsed recipe components
        """
        parsed = {
            "title": "",
            "ingredients": [],
            "instructions": [],
            "cooking_time": "",
            "servings": "",
            "difficulty": "",
            "cuisine": "",
            "dietary": "",
        }
        
        # Extract title (first line or first sentence)
        lines = recipe_text.strip().split('\n')
        if lines:
            parsed["title"] = lines[0].strip()
        
        # Extract ingredients (look for ingredient-related keywords)
        ingredient_keywords = ["ingredients", "ingredient", "you'll need", "need:"]
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in ingredient_keywords):
                # Extract ingredients from this line
                ingredients_text = line.split(":", 1)[-1] if ":" in line else line
                ingredients = [ing.strip() for ing in ingredients_text.split(",")]
                parsed["ingredients"].extend(ingredients)
        
        # Extract instructions (look for numbered steps or instruction keywords)
        instruction_keywords = ["instructions", "steps", "directions", "method", "how to"]
        instructions = []
        in_instructions = False
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in instruction_keywords):
                in_instructions = True
                continue
            
            if in_instructions and line.strip():
                # Check if it's a numbered step
                if re.match(r'^\d+\.', line.strip()):
                    instructions.append(line.strip())
                elif line.strip() and not any(keyword in line_lower for keyword in ingredient_keywords):
                    instructions.append(line.strip())
        
        parsed["instructions"] = instructions
        
        # Extract metadata (cooking time, servings, etc.)
        for line in lines:
            line_lower = line.lower()
            if "time" in line_lower and ("minute" in line_lower or "hour" in line_lower):
                parsed["cooking_time"] = line.strip()
            elif "serves" in line_lower or "servings" in line_lower:
                parsed["servings"] = line.strip()
            elif "difficulty" in line_lower:
                parsed["difficulty"] = line.strip()
            elif "cuisine" in line_lower:
                parsed["cuisine"] = line.strip()
            elif "dietary" in line_lower:
                parsed["dietary"] = line.strip()
        
        return parsed
    
    def interpolate_recipes(
        self,
        ingredients1: List[str],
        ingredients2: List[str],
        num_steps: int = 5,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Interpolate between two sets of ingredients.
        
        Args:
            ingredients1: First set of ingredients
            ingredients2: Second set of ingredients
            num_steps: Number of interpolation steps
            **kwargs: Additional generation parameters
            
        Returns:
            List of interpolated recipes
        """
        interpolated_recipes = []
        
        for i in range(num_steps):
            # Linear interpolation of ingredients
            alpha = i / (num_steps - 1) if num_steps > 1 else 0
            
            # Simple interpolation: randomly sample ingredients from both sets
            interpolated_ingredients = []
            
            # Add ingredients from first set
            num_from_first = int(len(ingredients1) * (1 - alpha))
            if num_from_first > 0:
                interpolated_ingredients.extend(
                    random.sample(ingredients1, min(num_from_first, len(ingredients1)))
                )
            
            # Add ingredients from second set
            num_from_second = int(len(ingredients2) * alpha)
            if num_from_second > 0:
                interpolated_ingredients.extend(
                    random.sample(ingredients2, min(num_from_second, len(ingredients2)))
                )
            
            # Remove duplicates while preserving order
            seen = set()
            unique_ingredients = []
            for ing in interpolated_ingredients:
                if ing not in seen:
                    seen.add(ing)
                    unique_ingredients.append(ing)
            
            # Generate recipe with interpolated ingredients
            recipe = self.sample_recipes(
                ingredients=unique_ingredients,
                num_samples=1,
                **kwargs,
            )[0]
            
            recipe["interpolation_step"] = i
            recipe["interpolation_alpha"] = alpha
            interpolated_recipes.append(recipe)
        
        return interpolated_recipes
    
    def save_samples(
        self,
        samples: List[Dict[str, Any]],
        output_path: Union[str, Path],
        format: str = "json",
    ) -> None:
        """Save generated samples to file.
        
        Args:
            samples: List of generated samples
            output_path: Path to save the samples
            format: Output format ("json", "jsonl", "txt")
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(samples, f, indent=2, ensure_ascii=False)
        
        elif format == "jsonl":
            with open(output_path, "w", encoding="utf-8") as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        
        elif format == "txt":
            with open(output_path, "w", encoding="utf-8") as f:
                for i, sample in enumerate(samples):
                    f.write(f"=== Recipe {i+1} ===\n")
                    f.write(f"Ingredients: {', '.join(sample['ingredients'])}\n")
                    f.write(f"Generated Text:\n{sample['generated_text']}\n\n")
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def load_samples(self, input_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Load samples from file.
        
        Args:
            input_path: Path to the samples file
            
        Returns:
            List of loaded samples
        """
        input_path = Path(input_path)
        
        if input_path.suffix == ".json":
            with open(input_path, "r", encoding="utf-8") as f:
                return json.load(f)
        
        elif input_path.suffix == ".jsonl":
            samples = []
            with open(input_path, "r", encoding="utf-8") as f:
                for line in f:
                    samples.append(json.loads(line.strip()))
            return samples
        
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")


def create_sampling_config(
    sampling_strategy: str = "nucleus",
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    max_length: int = 300,
    num_samples: int = 5,
) -> Dict[str, Any]:
    """Create a sampling configuration dictionary.
    
    Args:
        sampling_strategy: Sampling strategy
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        repetition_penalty: Repetition penalty
        max_length: Maximum generation length
        num_samples: Number of samples to generate
        
    Returns:
        Sampling configuration dictionary
    """
    return {
        "sampling_strategy": sampling_strategy,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "max_length": max_length,
        "num_samples": num_samples,
    }


def generate_sample_grid(
    sampler: RecipeSampler,
    ingredients_list: List[List[str]],
    output_dir: Union[str, Path],
    **sampling_kwargs,
) -> None:
    """Generate a grid of samples for different ingredient combinations.
    
    Args:
        sampler: Recipe sampler instance
        ingredients_list: List of ingredient combinations
        output_dir: Output directory for samples
        **sampling_kwargs: Additional sampling parameters
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_samples = []
    
    for i, ingredients in enumerate(ingredients_list):
        print(f"Generating samples for ingredient set {i+1}/{len(ingredients_list)}")
        
        samples = sampler.sample_recipes(
            ingredients=ingredients,
            **sampling_kwargs,
        )
        
        # Add ingredient set info
        for sample in samples:
            sample["ingredient_set_id"] = i
            sample["ingredient_set"] = ingredients
        
        all_samples.extend(samples)
    
    # Save all samples
    sampler.save_samples(all_samples, output_dir / "sample_grid.json")
    
    # Save individual files for each ingredient set
    for i, ingredients in enumerate(ingredients_list):
        set_samples = [s for s in all_samples if s["ingredient_set_id"] == i]
        sampler.save_samples(
            set_samples,
            output_dir / f"ingredient_set_{i+1}.json",
        )
    
    print(f"Generated {len(all_samples)} samples saved to {output_dir}")

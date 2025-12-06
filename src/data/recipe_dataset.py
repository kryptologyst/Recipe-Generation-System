"""Data loading and preprocessing utilities for recipe generation."""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
import numpy as np


class RecipeDataset(Dataset):
    """Dataset for recipe generation tasks."""
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        include_metadata: bool = True,
    ):
        """Initialize the recipe dataset.
        
        Args:
            data: List of recipe dictionaries
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
            include_metadata: Whether to include recipe metadata
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_metadata = include_metadata
        
    def __len__(self) -> int:
        """Return the number of recipes in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a recipe sample.
        
        Args:
            idx: Index of the recipe
            
        Returns:
            Dictionary containing tokenized recipe data
        """
        recipe = self.data[idx]
        
        # Create the recipe text
        recipe_text = self._format_recipe(recipe)
        
        # Tokenize
        encoding = self.tokenizer(
            recipe_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # Prepare input and labels
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        # For language modeling, labels are the same as input_ids
        labels = input_ids.clone()
        
        # Mask the prompt part (everything before <INSTRUCTIONS>)
        prompt_end = self._find_prompt_end(input_ids)
        if prompt_end > 0:
            labels[:prompt_end] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    def _format_recipe(self, recipe: Dict) -> str:
        """Format a recipe dictionary into text.
        
        Args:
            recipe: Recipe dictionary
            
        Returns:
            Formatted recipe text
        """
        parts = []
        
        # Add metadata if requested
        if self.include_metadata:
            if "cuisine" in recipe:
                parts.append(f"<CUISINE> {recipe['cuisine']}")
            if "dietary" in recipe:
                parts.append(f"<DIETARY> {recipe['dietary']}")
            if "cooking_time" in recipe:
                parts.append(f"<COOKING_TIME> {recipe['cooking_time']}")
            if "servings" in recipe:
                parts.append(f"<SERVINGS> {recipe['servings']}")
            if "difficulty" in recipe:
                parts.append(f"<DIFFICULTY> {recipe['difficulty']}")
        
        # Add ingredients
        ingredients = recipe.get("ingredients", [])
        if isinstance(ingredients, list):
            ingredients_str = ", ".join(ingredients)
        else:
            ingredients_str = str(ingredients)
        parts.append(f"<INGREDIENTS> {ingredients_str}")
        
        # Add instructions
        instructions = recipe.get("instructions", "")
        parts.append(f"<INSTRUCTIONS> {instructions}")
        
        return " ".join(parts)
    
    def _find_prompt_end(self, input_ids: torch.Tensor) -> int:
        """Find the end of the prompt (before <INSTRUCTIONS>).
        
        Args:
            input_ids: Token IDs
            
        Returns:
            Index where instructions start
        """
        instructions_token_id = self.tokenizer.encode("<INSTRUCTIONS>", add_special_tokens=False)[0]
        
        # Find the first occurrence of <INSTRUCTIONS>
        for i, token_id in enumerate(input_ids):
            if token_id == instructions_token_id:
                return i + 1  # Include the <INSTRUCTIONS> token
        
        return 0


class RecipeDataModule:
    """Data module for recipe generation with train/val/test splits."""
    
    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 8,
        max_length: int = 512,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        random_seed: int = 42,
    ):
        """Initialize the data module.
        
        Args:
            data_path: Path to the recipe data file
            tokenizer: Tokenizer for text processing
            batch_size: Batch size for data loaders
            max_length: Maximum sequence length
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            test_split: Fraction of data for testing
            random_seed: Random seed for reproducibility
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.random_seed = random_seed
        
        # Load and split data
        self.data = self._load_data()
        self.train_data, self.val_data, self.test_data = self._split_data()
        
    def _load_data(self) -> List[Dict]:
        """Load recipe data from file.
        
        Returns:
            List of recipe dictionaries
        """
        if self.data_path.suffix == ".json":
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif self.data_path.suffix == ".jsonl":
            data = []
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    data.append(json.loads(line.strip()))
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
        
        return data
    
    def _split_data(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split data into train/val/test sets.
        
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        random.seed(self.random_seed)
        data = self.data.copy()
        random.shuffle(data)
        
        n_total = len(data)
        n_train = int(n_total * self.train_split)
        n_val = int(n_total * self.val_split)
        
        train_data = data[:n_train]
        val_data = data[n_train:n_train + n_val]
        test_data = data[n_train + n_val:]
        
        return train_data, val_data, test_data
    
    def get_train_dataloader(self) -> DataLoader:
        """Get training data loader."""
        dataset = RecipeDataset(
            self.train_data,
            self.tokenizer,
            self.max_length,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
    
    def get_val_dataloader(self) -> DataLoader:
        """Get validation data loader."""
        dataset = RecipeDataset(
            self.val_data,
            self.tokenizer,
            self.max_length,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
    
    def get_test_dataloader(self) -> DataLoader:
        """Get test data loader."""
        dataset = RecipeDataset(
            self.test_data,
            self.tokenizer,
            self.max_length,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )


def create_sample_recipe_data(output_path: Union[str, Path], num_recipes: int = 100) -> None:
    """Create sample recipe data for testing.
    
    Args:
        output_path: Path to save the sample data
        num_recipes: Number of sample recipes to generate
    """
    cuisines = ["Italian", "Mexican", "Chinese", "Indian", "French", "Thai", "Japanese", "American"]
    dietary_options = ["vegetarian", "vegan", "gluten-free", "dairy-free", "keto", "paleo", None]
    difficulties = ["easy", "medium", "hard"]
    cooking_times = ["15 minutes", "30 minutes", "45 minutes", "1 hour", "1.5 hours", "2 hours"]
    
    # Sample ingredients by cuisine
    cuisine_ingredients = {
        "Italian": ["tomatoes", "basil", "garlic", "olive oil", "parmesan", "pasta", "mozzarella"],
        "Mexican": ["beans", "rice", "cilantro", "lime", "jalapeno", "cumin", "avocado"],
        "Chinese": ["soy sauce", "ginger", "garlic", "sesame oil", "rice", "bok choy", "scallions"],
        "Indian": ["curry powder", "cumin", "garlic", "ginger", "rice", "yogurt", "cilantro"],
        "French": ["butter", "wine", "herbs", "garlic", "cream", "mushrooms", "thyme"],
        "Thai": ["coconut milk", "lemongrass", "lime", "fish sauce", "chili", "basil", "rice"],
        "Japanese": ["soy sauce", "miso", "rice", "seaweed", "ginger", "sesame", "mirin"],
        "American": ["cheese", "bacon", "potatoes", "onions", "garlic", "butter", "flour"],
    }
    
    recipes = []
    
    for i in range(num_recipes):
        cuisine = random.choice(cuisines)
        dietary = random.choice(dietary_options)
        difficulty = random.choice(difficulties)
        cooking_time = random.choice(cooking_times)
        servings = random.randint(2, 8)
        
        # Select ingredients based on cuisine
        base_ingredients = cuisine_ingredients[cuisine]
        num_ingredients = random.randint(4, 8)
        ingredients = random.sample(base_ingredients, min(num_ingredients, len(base_ingredients)))
        
        # Add some random ingredients
        all_ingredients = [
            "chicken", "beef", "fish", "shrimp", "eggs", "milk", "cheese", "onions",
            "carrots", "potatoes", "spinach", "broccoli", "peppers", "mushrooms",
            "salt", "pepper", "sugar", "flour", "oil", "vinegar", "honey", "nuts"
        ]
        additional_ingredients = random.sample(
            [ing for ing in all_ingredients if ing not in ingredients],
            random.randint(1, 3)
        )
        ingredients.extend(additional_ingredients)
        
        # Generate simple instructions
        instructions = f"1. Prepare all ingredients. 2. Heat oil in a pan. 3. Add main ingredients and cook for {cooking_time.lower()}. 4. Season with salt and pepper. 5. Serve hot."
        
        recipe = {
            "id": i,
            "title": f"{cuisine} Recipe {i+1}",
            "cuisine": cuisine,
            "ingredients": ingredients,
            "instructions": instructions,
            "cooking_time": cooking_time,
            "servings": servings,
            "difficulty": difficulty,
        }
        
        if dietary:
            recipe["dietary"] = dietary
        
        recipes.append(recipe)
    
    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(recipes, f, indent=2, ensure_ascii=False)
    
    print(f"Created {num_recipes} sample recipes at {output_path}")


def load_recipe_data(data_path: Union[str, Path]) -> List[Dict]:
    """Load recipe data from various formats.
    
    Args:
        data_path: Path to the recipe data file
        
    Returns:
        List of recipe dictionaries
    """
    data_path = Path(data_path)
    
    if data_path.suffix == ".json":
        with open(data_path, "r", encoding="utf-8") as f:
            return json.load(f)
    elif data_path.suffix == ".jsonl":
        data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    elif data_path.suffix == ".csv":
        df = pd.read_csv(data_path)
        return df.to_dict("records")
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")

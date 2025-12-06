"""Unit tests for recipe generation system."""

import pytest
import torch
import tempfile
from pathlib import Path
import json
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models.recipe_model import RecipeGenerationModel, FineTunedRecipeModel
from src.data.recipe_dataset import RecipeDataset, RecipeDataModule, create_sample_recipe_data
from src.sampling.sampler import RecipeSampler, create_sampling_config
from src.evaluation.metrics import RecipeEvaluator
from src.training.trainer import setup_deterministic_training, create_training_config


class TestRecipeModel:
    """Test cases for recipe generation model."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = RecipeGenerationModel(model_name="gpt2")
        assert model.model_name == "gpt2"
        assert model.device in ["cpu", "cuda", "mps"]
        assert model.tokenizer is not None
    
    def test_create_recipe_prompt(self):
        """Test recipe prompt creation."""
        model = RecipeGenerationModel(model_name="gpt2")
        
        ingredients = ["chicken", "garlic", "lemon"]
        prompt = model.create_recipe_prompt(
            ingredients=ingredients,
            cuisine="Italian",
            dietary="gluten-free",
            cooking_time="30 minutes",
            servings=4,
            difficulty="medium"
        )
        
        assert "<CUISINE> Italian" in prompt
        assert "<DIETARY> gluten-free" in prompt
        assert "<COOKING_TIME> 30 minutes" in prompt
        assert "<SERVINGS> 4" in prompt
        assert "<DIFFICULTY> medium" in prompt
        assert "<INGREDIENTS> chicken, garlic, lemon" in prompt
        assert "<INSTRUCTIONS>" in prompt
    
    def test_generate_recipe(self):
        """Test recipe generation."""
        model = RecipeGenerationModel(model_name="gpt2")
        
        ingredients = ["chicken", "garlic", "lemon"]
        recipe = model.generate_recipe(
            ingredients=ingredients,
            cuisine="Italian",
            max_length=100,
            temperature=0.8
        )
        
        assert isinstance(recipe, str)
        assert len(recipe) > 0
    
    def test_fine_tuned_model(self):
        """Test fine-tuned model initialization."""
        model = FineTunedRecipeModel(model_name="gpt2")
        assert model.learning_rate == 5e-5
        assert model.weight_decay == 0.01
        assert model.warmup_steps == 100
        
        optimizer = model.get_optimizer()
        assert isinstance(optimizer, torch.optim.Optimizer)


class TestRecipeDataset:
    """Test cases for recipe dataset."""
    
    def test_sample_data_creation(self):
        """Test sample data creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "sample_recipes.json"
            create_sample_recipe_data(output_path, num_recipes=10)
            
            assert output_path.exists()
            
            with open(output_path, "r") as f:
                data = json.load(f)
            
            assert len(data) == 10
            assert all("ingredients" in recipe for recipe in data)
            assert all("instructions" in recipe for recipe in data)
    
    def test_recipe_dataset(self):
        """Test recipe dataset functionality."""
        # Create sample data
        sample_data = [
            {
                "ingredients": ["chicken", "garlic", "lemon"],
                "instructions": "Cook chicken with garlic and lemon.",
                "cuisine": "Italian",
                "cooking_time": "30 minutes",
                "servings": 4,
                "difficulty": "medium"
            }
        ]
        
        model = RecipeGenerationModel(model_name="gpt2")
        dataset = RecipeDataset(sample_data, model.tokenizer, max_length=128)
        
        assert len(dataset) == 1
        
        sample = dataset[0]
        assert "input_ids" in sample
        assert "attention_mask" in sample
        assert "labels" in sample
        
        assert sample["input_ids"].shape[0] == 128
        assert sample["attention_mask"].shape[0] == 128
        assert sample["labels"].shape[0] == 128


class TestRecipeSampler:
    """Test cases for recipe sampler."""
    
    def test_sampler_initialization(self):
        """Test sampler initialization."""
        model = RecipeGenerationModel(model_name="gpt2")
        sampler = RecipeSampler(model, seed=42)
        
        assert sampler.model == model
        assert sampler.device == model.device
    
    def test_sampling_config(self):
        """Test sampling configuration creation."""
        config = create_sampling_config(
            sampling_strategy="nucleus",
            temperature=0.8,
            top_p=0.9,
            num_samples=5
        )
        
        assert config["sampling_strategy"] == "nucleus"
        assert config["temperature"] == 0.8
        assert config["top_p"] == 0.9
        assert config["num_samples"] == 5
    
    def test_recipe_sampling(self):
        """Test recipe sampling."""
        model = RecipeGenerationModel(model_name="gpt2")
        sampler = RecipeSampler(model, seed=42)
        
        ingredients = ["chicken", "garlic", "lemon"]
        samples = sampler.sample_recipes(
            ingredients=ingredients,
            num_samples=2,
            max_length=100
        )
        
        assert len(samples) == 2
        assert all("ingredients" in sample for sample in samples)
        assert all("generated_text" in sample for sample in samples)
        assert all(sample["ingredients"] == ingredients for sample in samples)


class TestRecipeEvaluator:
    """Test cases for recipe evaluator."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = RecipeEvaluator()
        assert evaluator.rouge_scorer is not None
        assert evaluator.bleu_scorer is not None
    
    def test_generation_quality_evaluation(self):
        """Test generation quality evaluation."""
        evaluator = RecipeEvaluator()
        
        generated_recipes = [
            "Cook chicken with garlic and lemon. Season with salt and pepper.",
            "Heat oil in a pan. Add chicken and cook until golden."
        ]
        reference_recipes = [
            "Season chicken with garlic and lemon. Cook until done.",
            "Heat oil. Add chicken and cook until golden brown."
        ]
        
        metrics = evaluator.evaluate_generation_quality(
            generated_recipes, reference_recipes, compute_bert_score=False
        )
        
        assert "bleu" in metrics
        assert "rouge_rouge1" in metrics
        assert "rouge_rouge2" in metrics
        assert "rouge_rougeL" in metrics
        
        assert all(0 <= value <= 1 for value in metrics.values())
    
    def test_recipe_structure_evaluation(self):
        """Test recipe structure evaluation."""
        evaluator = RecipeEvaluator()
        
        recipes = [
            "Ingredients: chicken, garlic. Instructions: Cook chicken with garlic.",
            "You need: tomatoes, basil. Steps: Mix tomatoes with basil."
        ]
        
        metrics = evaluator.evaluate_recipe_structure(recipes)
        
        assert "has_ingredients" in metrics
        assert "has_instructions" in metrics
        assert "avg_recipe_length" in metrics
        
        assert 0 <= metrics["has_ingredients"] <= 1
        assert 0 <= metrics["has_instructions"] <= 1
        assert metrics["avg_recipe_length"] > 0
    
    def test_ingredient_coherence_evaluation(self):
        """Test ingredient coherence evaluation."""
        evaluator = RecipeEvaluator()
        
        generated_recipes = [
            "Cook chicken with garlic and lemon.",
            "Make pasta with tomatoes and basil."
        ]
        input_ingredients = [
            ["chicken", "garlic", "lemon"],
            ["pasta", "tomatoes", "basil"]
        ]
        
        metrics = evaluator.evaluate_ingredient_coherence(
            generated_recipes, input_ingredients
        )
        
        assert "ingredient_coverage" in metrics
        assert "ingredient_relevance" in metrics
        
        assert 0 <= metrics["ingredient_coverage"] <= 1
        assert 0 <= metrics["ingredient_relevance"] <= 1
    
    def test_diversity_evaluation(self):
        """Test diversity evaluation."""
        evaluator = RecipeEvaluator()
        
        recipes = [
            "Cook chicken with garlic and lemon.",
            "Make pasta with tomatoes and basil.",
            "Bake fish with herbs and butter."
        ]
        
        metrics = evaluator.evaluate_diversity(recipes)
        
        assert "token_diversity" in metrics
        assert "recipe_diversity" in metrics
        
        assert 0 <= metrics["token_diversity"] <= 1
        assert 0 <= metrics["recipe_diversity"] <= 1


class TestTrainingUtilities:
    """Test cases for training utilities."""
    
    def test_deterministic_training_setup(self):
        """Test deterministic training setup."""
        setup_deterministic_training(42)
        # This function doesn't return anything, just sets seeds
        # We can't easily test the side effects, but we can ensure it doesn't crash
        assert True
    
    def test_training_config_creation(self):
        """Test training configuration creation."""
        config = create_training_config(
            model_name="gpt2",
            batch_size=8,
            learning_rate=5e-5,
            max_epochs=10
        )
        
        assert config["model"]["model_name"] == "gpt2"
        assert config["model"]["learning_rate"] == 5e-5
        assert config["trainer"]["max_epochs"] == 10
        assert config["data"]["batch_size"] == 8


if __name__ == "__main__":
    pytest.main([__file__])

#!/usr/bin/env python3
"""Simple test script to verify the recipe generation system works end-to-end."""

import sys
from pathlib import Path
import tempfile
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.recipe_model import RecipeGenerationModel
from src.data.recipe_dataset import create_sample_recipe_data
from src.sampling.sampler import RecipeSampler
from src.evaluation.metrics import RecipeEvaluator


def test_basic_functionality():
    """Test basic functionality of the recipe generation system."""
    print("Testing Recipe Generation System")
    print("=" * 50)
    
    # Test 1: Model initialization
    print("1. Testing model initialization...")
    model = RecipeGenerationModel(model_name="gpt2")
    print(f"   ✓ Model loaded: {model.model_name}")
    print(f"   ✓ Device: {model.device}")
    
    # Test 2: Sample data creation
    print("\n2. Testing sample data creation...")
    with tempfile.TemporaryDirectory() as temp_dir:
        data_path = Path(temp_dir) / "test_recipes.json"
        create_sample_recipe_data(data_path, num_recipes=10)
        
        with open(data_path, "r") as f:
            data = json.load(f)
        
        print(f"   ✓ Created {len(data)} sample recipes")
        print(f"   ✓ Sample recipe keys: {list(data[0].keys())}")
    
    # Test 3: Recipe generation
    print("\n3. Testing recipe generation...")
    ingredients = ["chicken", "garlic", "lemon", "olive oil", "parsley"]
    recipe = model.generate_recipe(
        ingredients=ingredients,
        cuisine="Italian",
        max_length=100,
        temperature=0.8
    )
    
    print(f"   ✓ Generated recipe length: {len(recipe)} characters")
    print(f"   ✓ Recipe preview: {recipe[:100]}...")
    
    # Test 4: Sampling utilities
    print("\n4. Testing sampling utilities...")
    sampler = RecipeSampler(model, seed=42)
    samples = sampler.sample_recipes(
        ingredients=ingredients,
        num_samples=2,
        max_length=100
    )
    
    print(f"   ✓ Generated {len(samples)} samples")
    print(f"   ✓ Sample structure: {list(samples[0].keys())}")
    
    # Test 5: Evaluation metrics
    print("\n5. Testing evaluation metrics...")
    evaluator = RecipeEvaluator()
    
    generated_recipes = [sample["generated_text"] for sample in samples]
    reference_recipes = ["Cook chicken with garlic and lemon. Season with salt and pepper."] * len(samples)
    
    metrics = evaluator.evaluate_generation_quality(
        generated_recipes, reference_recipes, compute_bert_score=False
    )
    
    print(f"   ✓ Evaluation metrics: {list(metrics.keys())}")
    print(f"   ✓ BLEU score: {metrics['bleu']:.4f}")
    print(f"   ✓ ROUGE-1 score: {metrics['rouge_rouge1']:.4f}")
    
    # Test 6: Recipe structure evaluation
    print("\n6. Testing recipe structure evaluation...")
    structure_metrics = evaluator.evaluate_recipe_structure(generated_recipes)
    print(f"   ✓ Structure metrics: {list(structure_metrics.keys())}")
    print(f"   ✓ Has ingredients: {structure_metrics['has_ingredients']:.2f}")
    print(f"   ✓ Has instructions: {structure_metrics['has_instructions']:.2f}")
    
    print("\n" + "=" * 50)
    print("All tests passed! The recipe generation system is working correctly.")
    print("=" * 50)


def test_advanced_features():
    """Test advanced features of the system."""
    print("\nTesting Advanced Features")
    print("=" * 50)
    
    model = RecipeGenerationModel(model_name="gpt2")
    sampler = RecipeSampler(model, seed=42)
    
    # Test recipe interpolation
    print("1. Testing recipe interpolation...")
    ingredients1 = ["chicken", "garlic", "lemon"]
    ingredients2 = ["tomatoes", "basil", "mozzarella"]
    
    interpolated = sampler.interpolate_recipes(
        ingredients1=ingredients1,
        ingredients2=ingredients2,
        num_steps=3,
        max_length=100
    )
    
    print(f"   ✓ Generated {len(interpolated)} interpolated recipes")
    print(f"   ✓ Interpolation alphas: {[s['interpolation_alpha'] for s in interpolated]}")
    
    # Test different sampling strategies
    print("\n2. Testing different sampling strategies...")
    strategies = ["nucleus", "top_k", "greedy"]
    
    for strategy in strategies:
        samples = sampler.sample_recipes(
            ingredients=["rice", "soy sauce", "ginger"],
            num_samples=1,
            sampling_strategy=strategy,
            max_length=100
        )
        print(f"   ✓ {strategy} sampling: {len(samples[0]['generated_text'])} chars")
    
    print("\nAdvanced features test completed!")


def main():
    """Main test function."""
    try:
        test_basic_functionality()
        test_advanced_features()
        
        print("\n🎉 All tests completed successfully!")
        print("\nThe recipe generation system is ready to use!")
        print("\nNext steps:")
        print("1. Run 'python scripts/train.py --create_sample_data' to create training data")
        print("2. Run 'python scripts/sample.py' to generate recipes")
        print("3. Run 'streamlit run demo/streamlit_app.py' to launch the web demo")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

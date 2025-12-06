"""Streamlit demo application for recipe generation."""

import streamlit as st
import sys
from pathlib import Path
import torch
import json
from typing import List, Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models.recipe_model import RecipeGenerationModel
from src.sampling.sampler import RecipeSampler


@st.cache_resource
def load_model(model_path: str, model_name: str = "gpt2"):
    """Load the recipe generation model with caching."""
    model = RecipeGenerationModel(model_name=model_name)
    
    if Path(model_path).exists():
        try:
            checkpoint = torch.load(model_path, map_location="cpu")
            if "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
            st.success(f"Loaded trained model from {model_path}")
        except Exception as e:
            st.warning(f"Failed to load checkpoint: {e}. Using pre-trained model.")
    else:
        st.info(f"Model checkpoint not found at {model_path}. Using pre-trained model.")
    
    return model


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Recipe Generation System",
        page_icon="🍳",
        layout="wide",
    )
    
    st.title("🍳 Recipe Generation System")
    st.markdown("Generate creative recipes from your available ingredients!")
    
    # Sidebar for model configuration
    st.sidebar.header("Model Configuration")
    
    model_name = st.sidebar.selectbox(
        "Model",
        ["gpt2", "gpt2-medium", "gpt2-large"],
        index=0,
        help="Choose the base model for recipe generation"
    )
    
    model_path = st.sidebar.text_input(
        "Model Path",
        value="./checkpoints/last.ckpt",
        help="Path to trained model checkpoint (optional)"
    )
    
    # Load model
    with st.spinner("Loading model..."):
        model = load_model(model_path, model_name)
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Input Ingredients")
        
        # Ingredient input
        ingredients_text = st.text_area(
            "Enter ingredients (comma-separated):",
            value="chicken, garlic, lemon, olive oil, parsley",
            height=100,
            help="Enter the ingredients you have available, separated by commas"
        )
        
        ingredients = [ing.strip() for ing in ingredients_text.split(",") if ing.strip()]
        
        # Recipe metadata
        st.subheader("Recipe Preferences")
        
        col1a, col1b = st.columns(2)
        with col1a:
            cuisine = st.selectbox(
                "Cuisine",
                ["Any", "Italian", "Mexican", "Chinese", "Indian", "French", "Thai", "Japanese", "American"],
                index=1
            )
            
            dietary = st.selectbox(
                "Dietary Restrictions",
                ["None", "vegetarian", "vegan", "gluten-free", "dairy-free", "keto", "paleo"],
                index=0
            )
        
        with col1b:
            cooking_time = st.selectbox(
                "Cooking Time",
                ["Any", "15 minutes", "30 minutes", "45 minutes", "1 hour", "1.5 hours", "2 hours"],
                index=2
            )
            
            servings = st.slider("Servings", 1, 8, 4)
        
        difficulty = st.select_slider(
            "Difficulty",
            options=["easy", "medium", "hard"],
            value="medium"
        )
    
    with col2:
        st.header("Generation Settings")
        
        # Sampling parameters
        num_samples = st.slider("Number of Recipes", 1, 10, 3)
        
        col2a, col2b = st.columns(2)
        with col2a:
            temperature = st.slider("Temperature", 0.1, 2.0, 0.8, 0.1)
            top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)
        
        with col2b:
            top_k = st.slider("Top-k", 1, 100, 50)
            max_length = st.slider("Max Length", 100, 500, 300)
        
        repetition_penalty = st.slider("Repetition Penalty", 1.0, 2.0, 1.1, 0.1)
        
        # Advanced options
        with st.expander("Advanced Options"):
            sampling_strategy = st.selectbox(
                "Sampling Strategy",
                ["nucleus", "top_k", "greedy", "random"],
                index=0
            )
            
            seed = st.number_input("Random Seed", value=42, min_value=0, max_value=1000000)
    
    # Generate button
    if st.button("🍳 Generate Recipes", type="primary"):
        if not ingredients:
            st.error("Please enter at least one ingredient!")
        else:
            with st.spinner("Generating recipes..."):
                # Initialize sampler
                sampler = RecipeSampler(model, seed=seed)
                
                # Prepare parameters
                cuisine_param = None if cuisine == "Any" else cuisine
                dietary_param = None if dietary == "None" else dietary
                cooking_time_param = None if cooking_time == "Any" else cooking_time
                
                # Generate recipes
                samples = sampler.sample_recipes(
                    ingredients=ingredients,
                    num_samples=num_samples,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    max_length=max_length,
                    cuisine=cuisine_param,
                    dietary=dietary_param,
                    cooking_time=cooking_time_param,
                    servings=servings,
                    difficulty=difficulty,
                )
                
                # Display results
                st.header("Generated Recipes")
                
                for i, sample in enumerate(samples):
                    with st.expander(f"Recipe {i+1}", expanded=True):
                        st.markdown(f"**Ingredients:** {', '.join(sample['ingredients'])}")
                        
                        # Display parsed recipe if available
                        if sample.get("parsed_recipe"):
                            parsed = sample["parsed_recipe"]
                            if parsed.get("title"):
                                st.markdown(f"**Title:** {parsed['title']}")
                            if parsed.get("instructions"):
                                st.markdown("**Instructions:**")
                                for j, instruction in enumerate(parsed["instructions"], 1):
                                    st.markdown(f"{j}. {instruction}")
                        
                        # Display full generated text
                        st.markdown("**Full Recipe:**")
                        st.text_area(
                            f"Recipe {i+1} Text",
                            value=sample["generated_text"],
                            height=200,
                            key=f"recipe_{i}",
                            label_visibility="collapsed"
                        )
                        
                        # Download button
                        recipe_json = json.dumps(sample, indent=2, ensure_ascii=False)
                        st.download_button(
                            label=f"Download Recipe {i+1}",
                            data=recipe_json,
                            file_name=f"recipe_{i+1}.json",
                            mime="application/json",
                            key=f"download_{i}"
                        )
    
    # Sample ingredient suggestions
    st.header("Sample Ingredient Combinations")
    
    sample_combinations = [
        ["chicken", "garlic", "lemon", "olive oil", "parsley"],
        ["tomatoes", "basil", "mozzarella", "balsamic vinegar"],
        ["rice", "soy sauce", "ginger", "garlic", "vegetables"],
        ["pasta", "tomatoes", "onions", "garlic", "cheese"],
        ["salmon", "dill", "lemon", "butter", "potatoes"],
    ]
    
    cols = st.columns(len(sample_combinations))
    for i, (col, ingredients) in enumerate(zip(cols, sample_combinations)):
        with col:
            if st.button(f"Use Sample {i+1}", key=f"sample_{i}"):
                st.session_state.ingredients_text = ", ".join(ingredients)
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        **Recipe Generation System** - Powered by transformer-based language models
        
        This system generates creative recipes based on your available ingredients and preferences.
        The model uses advanced natural language processing to create coherent and useful cooking instructions.
        """
    )


if __name__ == "__main__":
    main()

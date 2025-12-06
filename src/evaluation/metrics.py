"""Evaluation metrics and utilities for recipe generation."""

import json
import re
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import pandas as pd
from torchmetrics import BLEUScore
import nltk
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


class RecipeEvaluator:
    """Comprehensive evaluator for recipe generation models."""
    
    def __init__(self, tokenizer=None):
        """Initialize the evaluator.
        
        Args:
            tokenizer: Tokenizer for text processing
        """
        self.tokenizer = tokenizer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Initialize BLEU scorer
        self.bleu_scorer = BLEUScore(n_gram=4)
        
    def evaluate_generation_quality(
        self,
        generated_recipes: List[str],
        reference_recipes: List[str],
        compute_bert_score: bool = True,
    ) -> Dict[str, float]:
        """Evaluate the quality of generated recipes.
        
        Args:
            generated_recipes: List of generated recipe texts
            reference_recipes: List of reference recipe texts
            compute_bert_score: Whether to compute BERTScore (slower)
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # BLEU Score
        bleu_scores = []
        for gen, ref in zip(generated_recipes, reference_recipes):
            gen_tokens = self._tokenize_text(gen)
            ref_tokens = self._tokenize_text(ref)
            bleu_score = self.bleu_scorer([gen_tokens], [[ref_tokens]])
            bleu_scores.append(bleu_score.item())
        
        metrics["bleu"] = np.mean(bleu_scores)
        
        # ROUGE Scores
        rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        for gen, ref in zip(generated_recipes, reference_recipes):
            scores = self.rouge_scorer.score(ref, gen)
            for metric in rouge_scores:
                rouge_scores[metric].append(scores[metric].fmeasure)
        
        for metric in rouge_scores:
            metrics[f"rouge_{metric}"] = np.mean(rouge_scores[metric])
        
        # BERTScore (optional, slower)
        if compute_bert_score:
            try:
                P, R, F1 = bert_score(generated_recipes, reference_recipes, lang="en", verbose=False)
                metrics["bert_score_precision"] = P.mean().item()
                metrics["bert_score_recall"] = R.mean().item()
                metrics["bert_score_f1"] = F1.mean().item()
            except Exception as e:
                print(f"BERTScore computation failed: {e}")
                metrics["bert_score_precision"] = 0.0
                metrics["bert_score_recall"] = 0.0
                metrics["bert_score_f1"] = 0.0
        
        return metrics
    
    def evaluate_recipe_structure(self, recipes: List[str]) -> Dict[str, float]:
        """Evaluate the structural quality of recipes.
        
        Args:
            recipes: List of recipe texts
            
        Returns:
            Dictionary of structural metrics
        """
        metrics = {}
        
        # Check for common recipe components
        has_ingredients = []
        has_instructions = []
        has_cooking_time = []
        has_servings = []
        
        # Recipe length metrics
        recipe_lengths = []
        instruction_lengths = []
        
        for recipe in recipes:
            recipe_lower = recipe.lower()
            
            # Check for components
            has_ingredients.append(any(word in recipe_lower for word in ["ingredients", "ingredient"]))
            has_instructions.append(any(word in recipe_lower for word in ["instructions", "steps", "directions"]))
            has_cooking_time.append(any(word in recipe_lower for word in ["minutes", "hours", "time"]))
            has_servings.append(any(word in recipe_lower for word in ["serves", "servings", "people"]))
            
            # Length metrics
            recipe_lengths.append(len(recipe.split()))
            
            # Extract instructions part
            instructions_match = re.search(r"<instructions>(.*)", recipe.lower())
            if instructions_match:
                instructions_text = instructions_match.group(1)
                instruction_lengths.append(len(instructions_text.split()))
            else:
                instruction_lengths.append(0)
        
        metrics["has_ingredients"] = np.mean(has_ingredients)
        metrics["has_instructions"] = np.mean(has_instructions)
        metrics["has_cooking_time"] = np.mean(has_cooking_time)
        metrics["has_servings"] = np.mean(has_servings)
        metrics["avg_recipe_length"] = np.mean(recipe_lengths)
        metrics["avg_instruction_length"] = np.mean(instruction_lengths)
        
        return metrics
    
    def evaluate_ingredient_coherence(
        self,
        generated_recipes: List[str],
        input_ingredients: List[List[str]],
    ) -> Dict[str, float]:
        """Evaluate how well generated recipes use the input ingredients.
        
        Args:
            generated_recipes: List of generated recipe texts
            input_ingredients: List of input ingredient lists
            
        Returns:
            Dictionary of coherence metrics
        """
        metrics = {}
        
        ingredient_coverage = []
        ingredient_relevance = []
        
        for recipe, ingredients in zip(generated_recipes, input_ingredients):
            recipe_lower = recipe.lower()
            
            # Check ingredient coverage
            covered_ingredients = 0
            for ingredient in ingredients:
                if ingredient.lower() in recipe_lower:
                    covered_ingredients += 1
            
            coverage = covered_ingredients / len(ingredients) if ingredients else 0
            ingredient_coverage.append(coverage)
            
            # Check for irrelevant ingredients (simple heuristic)
            # Count ingredients mentioned that weren't in input
            recipe_words = set(recipe_lower.split())
            input_words = set([ing.lower() for ing in ingredients])
            
            # Simple relevance check (can be improved)
            extra_ingredients = len(recipe_words - input_words)
            relevance = 1.0 / (1.0 + extra_ingredients * 0.1)  # Penalty for extra ingredients
            ingredient_relevance.append(relevance)
        
        metrics["ingredient_coverage"] = np.mean(ingredient_coverage)
        metrics["ingredient_relevance"] = np.mean(ingredient_relevance)
        
        return metrics
    
    def evaluate_diversity(self, recipes: List[str]) -> Dict[str, float]:
        """Evaluate the diversity of generated recipes.
        
        Args:
            recipes: List of recipe texts
            
        Returns:
            Dictionary of diversity metrics
        """
        metrics = {}
        
        # Token-level diversity
        all_tokens = []
        for recipe in recipes:
            tokens = self._tokenize_text(recipe)
            all_tokens.extend(tokens)
        
        unique_tokens = set(all_tokens)
        metrics["token_diversity"] = len(unique_tokens) / len(all_tokens) if all_tokens else 0
        
        # Recipe-level diversity (using simple similarity)
        recipe_similarities = []
        for i in range(len(recipes)):
            for j in range(i + 1, len(recipes)):
                similarity = self._compute_text_similarity(recipes[i], recipes[j])
                recipe_similarities.append(similarity)
        
        metrics["avg_recipe_similarity"] = np.mean(recipe_similarities) if recipe_similarities else 0
        metrics["recipe_diversity"] = 1.0 - metrics["avg_recipe_similarity"]
        
        return metrics
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text for evaluation.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        if self.tokenizer:
            return self.tokenizer.tokenize(text)
        else:
            return text.lower().split()
    
    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute simple text similarity using Jaccard similarity.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        return intersection / union if union > 0 else 0
    
    def compute_perplexity(
        self,
        model,
        tokenizer,
        texts: List[str],
        device: str = "cpu",
    ) -> float:
        """Compute perplexity of the model on given texts.
        
        Args:
            model: Trained model
            tokenizer: Tokenizer
            texts: List of texts to evaluate
            device: Device to run evaluation on
            
        Returns:
            Average perplexity
        """
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for text in texts:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                
                total_loss += loss.item() * inputs["input_ids"].size(1)
                total_tokens += inputs["input_ids"].size(1)
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return perplexity
    
    def create_evaluation_report(
        self,
        generated_recipes: List[str],
        reference_recipes: List[str],
        input_ingredients: List[List[str]],
        model=None,
        tokenizer=None,
        device: str = "cpu",
    ) -> Dict[str, Union[float, Dict]]:
        """Create a comprehensive evaluation report.
        
        Args:
            generated_recipes: List of generated recipes
            reference_recipes: List of reference recipes
            input_ingredients: List of input ingredient lists
            model: Trained model (optional)
            tokenizer: Tokenizer (optional)
            device: Device for evaluation
            
        Returns:
            Comprehensive evaluation report
        """
        report = {}
        
        # Generation quality metrics
        report["generation_quality"] = self.evaluate_generation_quality(
            generated_recipes, reference_recipes
        )
        
        # Structural metrics
        report["structure"] = self.evaluate_recipe_structure(generated_recipes)
        
        # Ingredient coherence
        report["ingredient_coherence"] = self.evaluate_ingredient_coherence(
            generated_recipes, input_ingredients
        )
        
        # Diversity metrics
        report["diversity"] = self.evaluate_diversity(generated_recipes)
        
        # Perplexity (if model provided)
        if model and tokenizer:
            report["perplexity"] = self.compute_perplexity(
                model, tokenizer, generated_recipes, device
            )
        
        return report
    
    def plot_evaluation_metrics(self, report: Dict, save_path: Optional[str] = None) -> None:
        """Plot evaluation metrics.
        
        Args:
            report: Evaluation report dictionary
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Recipe Generation Evaluation Metrics", fontsize=16)
        
        # Generation quality metrics
        quality_metrics = report["generation_quality"]
        ax1 = axes[0, 0]
        metrics_names = list(quality_metrics.keys())
        metrics_values = list(quality_metrics.values())
        ax1.bar(metrics_names, metrics_values)
        ax1.set_title("Generation Quality Metrics")
        ax1.set_ylabel("Score")
        ax1.tick_params(axis='x', rotation=45)
        
        # Structure metrics
        structure_metrics = report["structure"]
        ax2 = axes[0, 1]
        structure_names = list(structure_metrics.keys())
        structure_values = list(structure_metrics.values())
        ax2.bar(structure_names, structure_values)
        ax2.set_title("Recipe Structure Metrics")
        ax2.set_ylabel("Score")
        ax2.tick_params(axis='x', rotation=45)
        
        # Ingredient coherence
        coherence_metrics = report["ingredient_coherence"]
        ax3 = axes[1, 0]
        coherence_names = list(coherence_metrics.keys())
        coherence_values = list(coherence_metrics.values())
        ax3.bar(coherence_names, coherence_values)
        ax3.set_title("Ingredient Coherence Metrics")
        ax3.set_ylabel("Score")
        
        # Diversity metrics
        diversity_metrics = report["diversity"]
        ax4 = axes[1, 1]
        diversity_names = list(diversity_metrics.keys())
        diversity_values = list(diversity_metrics.values())
        ax4.bar(diversity_names, diversity_values)
        ax4.set_title("Diversity Metrics")
        ax4.set_ylabel("Score")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        plt.show()


def create_model_leaderboard(
    model_results: Dict[str, Dict],
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """Create a model leaderboard from evaluation results.
    
    Args:
        model_results: Dictionary mapping model names to evaluation results
        save_path: Path to save the leaderboard
        
    Returns:
        DataFrame with model rankings
    """
    leaderboard_data = []
    
    for model_name, results in model_results.items():
        row = {"Model": model_name}
        
        # Extract key metrics
        if "generation_quality" in results:
            quality = results["generation_quality"]
            row.update({
                "BLEU": quality.get("bleu", 0),
                "ROUGE-1": quality.get("rouge_rouge1", 0),
                "ROUGE-2": quality.get("rouge_rouge2", 0),
                "ROUGE-L": quality.get("rouge_rougeL", 0),
                "BERTScore-F1": quality.get("bert_score_f1", 0),
            })
        
        if "structure" in results:
            structure = results["structure"]
            row.update({
                "Has Ingredients": structure.get("has_ingredients", 0),
                "Has Instructions": structure.get("has_instructions", 0),
                "Avg Length": structure.get("avg_recipe_length", 0),
            })
        
        if "ingredient_coherence" in results:
            coherence = results["ingredient_coherence"]
            row.update({
                "Ingredient Coverage": coherence.get("ingredient_coverage", 0),
                "Ingredient Relevance": coherence.get("ingredient_relevance", 0),
            })
        
        if "diversity" in results:
            diversity = results["diversity"]
            row.update({
                "Token Diversity": diversity.get("token_diversity", 0),
                "Recipe Diversity": diversity.get("recipe_diversity", 0),
            })
        
        if "perplexity" in results:
            row["Perplexity"] = results["perplexity"]
        
        leaderboard_data.append(row)
    
    df = pd.DataFrame(leaderboard_data)
    
    # Sort by BLEU score (or another primary metric)
    if "BLEU" in df.columns:
        df = df.sort_values("BLEU", ascending=False)
    
    if save_path:
        df.to_csv(save_path, index=False)
    
    return df

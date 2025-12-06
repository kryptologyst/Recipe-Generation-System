"""Training utilities and PyTorch Lightning module for recipe generation."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from transformers import PreTrainedTokenizer

from ..models.recipe_model import FineTunedRecipeModel
from ..data.recipe_dataset import RecipeDataModule


class RecipeGenerationModule(pl.LightningModule):
    """PyTorch Lightning module for recipe generation training."""
    
    def __init__(
        self,
        model_name: str = "gpt2",
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        max_length: int = 512,
        **kwargs,
    ):
        """Initialize the training module.
        
        Args:
            model_name: HuggingFace model name
            learning_rate: Learning rate for training
            weight_decay: Weight decay for regularization
            warmup_steps: Number of warmup steps
            max_length: Maximum sequence length
            **kwargs: Additional arguments
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.model = FineTunedRecipeModel(
            model_name=model_name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            max_length=max_length,
            **kwargs,
        )
        
        # Training metrics
        self.train_loss = []
        self.val_loss = []
        
    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass."""
        return self.model(input_ids=input_ids, **kwargs)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        
        loss = self.model.compute_loss(input_ids, labels)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_loss.append(loss.item())
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        
        loss = self.model.compute_loss(input_ids, labels)
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_loss.append(loss.item())
        
        return loss
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        
        loss = self.model.compute_loss(input_ids, labels)
        
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and schedulers."""
        optimizer = self.model.get_optimizer()
        
        # Calculate total training steps
        if hasattr(self.trainer, "datamodule") and self.trainer.datamodule:
            num_training_steps = len(self.trainer.datamodule.get_train_dataloader()) * self.trainer.max_epochs
        else:
            num_training_steps = 1000  # Default fallback
        
        scheduler = self.model.get_scheduler(optimizer, num_training_steps)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
    
    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch."""
        avg_train_loss = torch.tensor(self.train_loss).mean()
        self.log("avg_train_loss", avg_train_loss, on_epoch=True)
        self.train_loss.clear()
    
    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch."""
        avg_val_loss = torch.tensor(self.val_loss).mean()
        self.log("avg_val_loss", avg_val_loss, on_epoch=True)
        self.val_loss.clear()


class RecipeTrainer:
    """High-level trainer for recipe generation models."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        data_module: RecipeDataModule,
        output_dir: Union[str, os.PathLike] = "./outputs",
    ):
        """Initialize the trainer.
        
        Args:
            config: Training configuration
            data_module: Data module for training
            output_dir: Output directory for checkpoints and logs
        """
        self.config = config
        self.data_module = data_module
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = RecipeGenerationModule(**config["model"])
        
        # Setup callbacks
        self.callbacks = self._setup_callbacks()
        
        # Setup logger
        self.logger = self._setup_logger()
        
        # Initialize trainer
        self.trainer = pl.Trainer(
            **config["trainer"],
            callbacks=self.callbacks,
            logger=self.logger,
        )
    
    def _setup_callbacks(self) -> list:
        """Setup training callbacks."""
        callbacks = []
        
        # Model checkpointing
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.output_dir / "checkpoints",
            filename="recipe-model-{epoch:02d}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=5,
            verbose=True,
        )
        callbacks.append(early_stop_callback)
        
        return callbacks
    
    def _setup_logger(self) -> pl.loggers.Logger:
        """Setup logging."""
        logger_type = self.config.get("logger", {}).get("type", "tensorboard")
        
        if logger_type == "wandb":
            return WandbLogger(
                project=self.config.get("logger", {}).get("project", "recipe-generation"),
                name=self.config.get("logger", {}).get("name", "recipe-model"),
                save_dir=self.output_dir / "logs",
            )
        else:
            return TensorBoardLogger(
                save_dir=self.output_dir / "logs",
                name="recipe-generation",
            )
    
    def train(self) -> None:
        """Start training."""
        self.trainer.fit(self.model, self.data_module)
    
    def test(self) -> None:
        """Run testing."""
        self.trainer.test(self.model, self.data_module)
    
    def generate_sample_recipes(self, num_samples: int = 5) -> None:
        """Generate sample recipes after training."""
        self.model.eval()
        
        sample_ingredients = [
            ["chicken", "garlic", "lemon", "olive oil", "parsley"],
            ["tomatoes", "basil", "mozzarella", "balsamic vinegar"],
            ["rice", "soy sauce", "ginger", "garlic", "vegetables"],
        ]
        
        print("\n" + "="*50)
        print("SAMPLE RECIPE GENERATIONS")
        print("="*50)
        
        for i, ingredients in enumerate(sample_ingredients[:num_samples]):
            print(f"\nRecipe {i+1}:")
            print(f"Ingredients: {', '.join(ingredients)}")
            print("-" * 30)
            
            recipe = self.model.model.generate_recipe(
                ingredients=ingredients,
                cuisine="Italian" if i == 0 else "Asian" if i == 2 else "Mediterranean",
                max_length=200,
                temperature=0.8,
            )
            print(recipe)
            print()


def setup_deterministic_training(seed: int = 42) -> None:
    """Setup deterministic training for reproducibility.
    
    Args:
        seed: Random seed
    """
    import random
    import numpy as np
    
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Set deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def create_training_config(
    model_name: str = "gpt2",
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    max_epochs: int = 10,
    max_length: int = 512,
    output_dir: str = "./outputs",
) -> Dict[str, Any]:
    """Create a training configuration dictionary.
    
    Args:
        model_name: HuggingFace model name
        batch_size: Training batch size
        learning_rate: Learning rate
        max_epochs: Maximum number of epochs
        max_length: Maximum sequence length
        output_dir: Output directory
        
    Returns:
        Training configuration dictionary
    """
    return {
        "model": {
            "model_name": model_name,
            "learning_rate": learning_rate,
            "weight_decay": 0.01,
            "warmup_steps": 100,
            "max_length": max_length,
        },
        "trainer": {
            "max_epochs": max_epochs,
            "accelerator": "auto",
            "devices": 1,
            "precision": "16-mixed" if torch.cuda.is_available() else "32",
            "gradient_clip_val": 1.0,
            "accumulate_grad_batches": 2,
        },
        "data": {
            "batch_size": batch_size,
            "max_length": max_length,
        },
        "logger": {
            "type": "tensorboard",
            "project": "recipe-generation",
            "name": f"recipe-model-{model_name}",
        },
        "output_dir": output_dir,
    }

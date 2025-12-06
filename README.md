# Recipe Generation System

A production-ready recipe generation system using transformer-based language models. This system can generate creative and coherent recipes based on available ingredients, cuisine preferences, dietary restrictions, and cooking constraints.

## Features

- **Advanced Text Generation**: Uses GPT-2 and other transformer models for high-quality recipe generation
- **Conditional Generation**: Supports conditioning on ingredients, cuisine type, dietary preferences, cooking time, servings, and difficulty
- **Multiple Sampling Strategies**: Nucleus sampling, top-k sampling, greedy decoding, and random sampling
- **Comprehensive Evaluation**: BLEU, ROUGE, BERTScore, perplexity, and custom recipe-specific metrics
- **Interactive Demo**: Streamlit web application for easy recipe generation
- **Production Ready**: Proper training framework, checkpointing, logging, and reproducible results
- **Extensible Architecture**: Modular design for easy customization and extension

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Recipe-Generation-System.git
cd Recipe-Generation-System
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create sample data (optional):
```bash
python scripts/train.py --create_sample_data --num_sample_recipes 1000
```

### Basic Usage

#### Generate Recipes from Command Line

```bash
python scripts/sample.py \
    --ingredients chicken garlic lemon olive_oil parsley \
    --cuisine Italian \
    --num_samples 3 \
    --temperature 0.8
```

#### Train a Model

```bash
python scripts/train.py \
    --config configs/training.yaml \
    --data_path data/processed/recipes.json \
    --output_dir ./outputs
```

#### Run Evaluation

```bash
python scripts/evaluate.py \
    --model_path ./checkpoints/last.ckpt \
    --test_data_path data/processed/test_recipes.json \
    --output_dir ./evaluation_results
```

#### Launch Interactive Demo

```bash
streamlit run demo/streamlit_app.py
```

## Project Structure

```
recipe-generation-system/
├── src/                          # Source code
│   ├── models/                   # Model definitions
│   │   └── recipe_model.py       # Recipe generation models
│   ├── data/                     # Data handling
│   │   └── recipe_dataset.py     # Dataset and data loaders
│   ├── training/                 # Training utilities
│   │   └── trainer.py           # PyTorch Lightning trainer
│   ├── evaluation/               # Evaluation metrics
│   │   └── metrics.py           # Recipe evaluation metrics
│   └── sampling/                 # Sampling utilities
│       └── sampler.py           # Advanced sampling strategies
├── configs/                      # Configuration files
│   ├── default.yaml             # Default configuration
│   └── training.yaml            # Training configuration
├── scripts/                      # Command-line scripts
│   ├── train.py                 # Training script
│   ├── sample.py                # Sampling script
│   └── evaluate.py              # Evaluation script
├── demo/                         # Demo applications
│   └── streamlit_app.py         # Streamlit web app
├── tests/                        # Unit tests
│   └── test_recipe_system.py    # Test cases
├── data/                         # Data directory
│   ├── raw/                     # Raw data
│   └── processed/               # Processed data
├── assets/                       # Generated assets
│   ├── samples/                 # Generated samples
│   └── plots/                   # Evaluation plots
├── checkpoints/                  # Model checkpoints
├── logs/                         # Training logs
└── outputs/                      # Training outputs
```

## Configuration

The system uses YAML configuration files for easy customization. Key configuration options:

### Model Configuration
- `model_name`: Base transformer model (gpt2, gpt2-medium, gpt2-large)
- `learning_rate`: Learning rate for fine-tuning
- `max_length`: Maximum sequence length
- `temperature`, `top_p`, `top_k`: Sampling parameters

### Training Configuration
- `max_epochs`: Number of training epochs
- `batch_size`: Training batch size
- `precision`: Mixed precision training (16-mixed, 32)
- `accelerator`: Training accelerator (auto, cpu, gpu)

### Data Configuration
- `train_split`, `val_split`, `test_split`: Data split ratios
- `batch_size`: Data loader batch size
- `max_length`: Maximum sequence length for data

## Data Format

The system expects recipe data in JSON format with the following structure:

```json
[
  {
    "id": 1,
    "title": "Italian Chicken with Lemon",
    "cuisine": "Italian",
    "ingredients": ["chicken", "garlic", "lemon", "olive oil", "parsley"],
    "instructions": "1. Season chicken with garlic and lemon. 2. Heat oil in pan. 3. Cook chicken until golden.",
    "cooking_time": "30 minutes",
    "servings": 4,
    "difficulty": "medium",
    "dietary": "gluten-free"
  }
]
```

## Model Architecture

### Base Model
- **GPT-2**: Pre-trained transformer language model
- **Fine-tuning**: Custom training on recipe data
- **Conditioning**: Structured prompts with special tokens

### Special Tokens
- `<INGREDIENTS>`: Marks ingredient list
- `<INSTRUCTIONS>`: Marks cooking instructions
- `<CUISINE>`: Cuisine type
- `<DIETARY>`: Dietary restrictions
- `<COOKING_TIME>`: Cooking time
- `<SERVINGS>`: Number of servings
- `<DIFFICULTY>`: Difficulty level

### Training Process
1. **Data Preprocessing**: Tokenization and formatting
2. **Fine-tuning**: Language modeling objective with recipe-specific conditioning
3. **Validation**: Perplexity and recipe quality metrics
4. **Checkpointing**: Best model saving based on validation loss

## Evaluation Metrics

### Generation Quality
- **BLEU**: N-gram overlap with reference recipes
- **ROUGE**: Recall-oriented evaluation metrics
- **BERTScore**: Semantic similarity using BERT embeddings
- **Perplexity**: Model confidence on test data

### Recipe Structure
- **Component Coverage**: Presence of ingredients, instructions, timing
- **Length Metrics**: Average recipe and instruction lengths
- **Structural Completeness**: Recipe component analysis

### Ingredient Coherence
- **Coverage**: How well input ingredients are used
- **Relevance**: Appropriateness of generated ingredients
- **Consistency**: Coherence between ingredients and instructions

### Diversity
- **Token Diversity**: Vocabulary richness
- **Recipe Diversity**: Uniqueness of generated recipes
- **Content Variation**: Diversity across different ingredient sets

## Sampling Strategies

### Nucleus Sampling (Recommended)
- Uses top-p parameter for dynamic vocabulary selection
- Balances quality and diversity
- Good for creative recipe generation

### Top-k Sampling
- Samples from top-k most likely tokens
- More conservative than nucleus sampling
- Good for consistent, high-quality recipes

### Greedy Decoding
- Always selects most likely token
- Deterministic but potentially repetitive
- Good for baseline comparisons

### Random Sampling
- Pure random sampling with temperature
- High diversity but potentially incoherent
- Good for exploring model capabilities

## Advanced Features

### Recipe Interpolation
Generate intermediate recipes between two ingredient sets:

```python
from src.sampling.sampler import RecipeSampler

sampler = RecipeSampler(model)
interpolated = sampler.interpolate_recipes(
    ingredients1=["chicken", "garlic", "lemon"],
    ingredients2=["tomatoes", "basil", "mozzarella"],
    num_steps=5
)
```

### Custom Evaluation
Create custom evaluation metrics:

```python
from src.evaluation.metrics import RecipeEvaluator

evaluator = RecipeEvaluator()
report = evaluator.create_evaluation_report(
    generated_recipes=generated,
    reference_recipes=references,
    input_ingredients=ingredients
)
```

### Batch Generation
Generate multiple recipes efficiently:

```python
ingredient_sets = [
    ["chicken", "garlic", "lemon"],
    ["pasta", "tomatoes", "basil"],
    ["salmon", "dill", "butter"]
]

all_samples = []
for ingredients in ingredient_sets:
    samples = sampler.sample_recipes(ingredients, num_samples=3)
    all_samples.extend(samples)
```

## Performance Optimization

### Mixed Precision Training
- Use `precision: "16-mixed"` for faster training
- Reduces memory usage and training time
- Maintains numerical stability

### Gradient Accumulation
- Use `accumulate_grad_batches` for larger effective batch sizes
- Useful when GPU memory is limited
- Improves training stability

### Model Parallelism
- Support for multi-GPU training
- Automatic device detection (CUDA, MPS, CPU)
- Efficient memory usage

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use gradient accumulation
   - Enable mixed precision training

2. **Poor Recipe Quality**
   - Increase training epochs
   - Use larger model (gpt2-medium, gpt2-large)
   - Adjust sampling parameters

3. **Repetitive Generation**
   - Increase repetition penalty
   - Use nucleus sampling with lower top-p
   - Add diversity to training data

### Debug Mode
Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install -e .

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v

# Format code
black src/ tests/ scripts/
ruff check src/ tests/ scripts/
```

## Model Card

### Model Information
- **Model Type**: Transformer-based language model (GPT-2)
- **Task**: Conditional text generation for recipes
- **Training Data**: Recipe datasets with structured formatting
- **Languages**: English

### Intended Use
- **Primary**: Generate cooking recipes from ingredient lists
- **Secondary**: Recipe inspiration and meal planning assistance
- **Research**: Text generation evaluation and recipe analysis

### Limitations
- Generated recipes may not be safe for consumption without verification
- Model may generate unrealistic or unsafe cooking instructions
- Quality depends on training data and model size
- May not handle dietary restrictions perfectly

### Bias and Fairness
- Model may reflect biases in training data
- Cuisine representation may be uneven
- Dietary preferences may not be fully respected
- Cultural cooking practices may be oversimplified

### Safety Considerations
- Always verify generated recipes before cooking
- Check for allergen information and dietary restrictions
- Ensure proper food safety practices
- Consider individual dietary needs and restrictions

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{recipe_generation_system,
  title={Recipe Generation System: A Modern Approach to AI-Powered Recipe Creation},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Recipe-Generation-System}
}
```

## Acknowledgments

- Hugging Face Transformers library
- PyTorch Lightning framework
- Streamlit for web interface
- The open-source community for various tools and libraries
# Recipe-Generation-System

"""Claude pricing data for different models."""

# Pricing per million tokens
MODELS = {
    "sonnet": {
        "name": "Claude Sonnet 5",
        "input": 3.00,  # $3 per 1M input tokens
        "output": 15.00,  # $15 per 1M output tokens
        "reasoning": 15.00,  # $15 per 1M reasoning tokens (extended thinking)
        "cache_creation": 3.75,  # $3.75 per 1M (1.25x base input)
        "cache_read": 0.30,  # $0.30 per 1M (0.1x base input)
    },
    "opus": {
        "name": "Claude Opus 4.8",
        "input": 5.00,  # $5 per 1M input tokens
        "output": 25.00,  # $25 per 1M output tokens
        "reasoning": 25.00,  # $25 per 1M reasoning tokens
        "cache_creation": 6.25,  # $6.25 per 1M (1.25x base input)
        "cache_read": 0.50,  # $0.50 per 1M (0.1x base input)
    },
    "haiku": {
        "name": "Claude Haiku 4.5",
        "input": 1.00,  # $1 per 1M input tokens
        "output": 5.00,  # $5 per 1M output tokens
        "reasoning": 5.00,  # $5 per 1M reasoning tokens
        "cache_creation": 1.25,  # $1.25 per 1M (1.25x base input)
        "cache_read": 0.10,  # $0.10 per 1M (0.1x base input)
    },
}


def get_model(model_name: str) -> dict:
    """Get pricing info for a model."""
    if model_name not in MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(MODELS.keys())}"
        )
    return MODELS[model_name]


def get_cost(tokens: int, price_per_million: float) -> float:
    """Calculate cost for given token count.

    Args:
        tokens: Number of tokens
        price_per_million: Price per 1 million tokens

    Returns:
        Cost in dollars
    """
    return (tokens / 1_000_000) * price_per_million

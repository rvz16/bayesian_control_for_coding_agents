"""Configuration profiles for SAGE Agent v4.

v4 uses hybrid approach:
- Natural code generation (no tool call format)
- Tool calling for DECISIONS (validation strategy, recovery strategy)
- SAGE uncertainty guides decision-making

Profiles:
- FAST: Minimal decision overhead, trust generation
- BALANCED: Default, good balance of speed and accuracy
- THOROUGH: Maximum validation, resampling on all decisions
- CAUTIOUS: Conservative, comprehensive testing
"""

# Fast profile: Minimize decision overhead
FAST = {
    # Core thresholds
    "confidence_threshold": 0.6,
    "uncertainty_threshold": 0.4,
    "max_attempts": 2,
    "max_validation_retries": 1,

    # Decision-making features
    "enable_sgr_for_decisions": False,  # Skip SGR
    "enable_resampling_decisions": False,  # No resampling
    "decision_samples": 1,
    "decision_agreement_threshold": 0.5,

    # Testing strategy
    "default_test_strategy": "quick",  # Always quick test
    "trust_threshold": 0.8,  # Lower bar to skip testing

    # Recovery strategy
    "max_recovery_attempts": 1,
    "enable_reflexion": False,  # No reflexion

    # Generation parameters
    "temperature": 0.5,
    "max_tokens": 2048,
}

# Balanced profile: Default configuration
BALANCED = {
    # Core thresholds
    "confidence_threshold": 0.7,
    "uncertainty_threshold": 0.3,
    "max_attempts": 3,
    "max_validation_retries": 2,

    # Decision-making features
    "enable_sgr_for_decisions": True,
    "enable_resampling_decisions": True,
    "decision_samples": 3,
    "decision_agreement_threshold": 0.7,

    # Testing strategy
    "default_test_strategy": "adaptive",
    "trust_threshold": 0.9,

    # Recovery strategy
    "max_recovery_attempts": 2,
    "enable_reflexion": True,

    # Generation parameters
    "temperature": 0.7,
    "max_tokens": 2048,
}

# Thorough profile: Maximum validation
THOROUGH = {
    # Core thresholds
    "confidence_threshold": 0.8,
    "uncertainty_threshold": 0.2,
    "max_attempts": 4,
    "max_validation_retries": 3,

    # Decision-making features
    "enable_sgr_for_decisions": True,
    "enable_resampling_decisions": True,
    "decision_samples": 5,  # More samples
    "decision_agreement_threshold": 0.8,

    # Testing strategy
    "default_test_strategy": "comprehensive",  # Always thorough
    "trust_threshold": 0.95,  # Very high bar

    # Recovery strategy
    "max_recovery_attempts": 3,
    "enable_reflexion": True,

    # Generation parameters
    "temperature": 0.6,
    "max_tokens": 4096,
}

# Cautious profile: Conservative, never trust
CAUTIOUS = {
    # Core thresholds
    "confidence_threshold": 0.9,
    "uncertainty_threshold": 0.1,
    "max_attempts": 5,
    "max_validation_retries": 3,

    # Decision-making features
    "enable_sgr_for_decisions": True,
    "enable_resampling_decisions": True,
    "decision_samples": 5,
    "decision_agreement_threshold": 0.9,

    # Testing strategy
    "default_test_strategy": "comprehensive",
    "trust_threshold": 1.0,  # Never trust, always test

    # Recovery strategy
    "max_recovery_attempts": 3,
    "enable_reflexion": True,

    # Generation parameters
    "temperature": 0.5,
    "max_tokens": 4096,
}

# Default: Use balanced
DEFAULT = BALANCED


def get_config(profile: str = "balanced") -> dict:
    """Get configuration by profile name.

    Args:
        profile: One of "fast", "balanced", "thorough", "cautious"

    Returns:
        Configuration dictionary
    """
    profiles = {
        "fast": FAST,
        "balanced": BALANCED,
        "thorough": THOROUGH,
        "cautious": CAUTIOUS,
    }

    profile_lower = profile.lower()
    if profile_lower not in profiles:
        raise ValueError(
            f"Unknown profile '{profile}'. "
            f"Choose from: {', '.join(profiles.keys())}"
        )

    return profiles[profile_lower].copy()


def print_config_comparison():
    """Print comparison of all configs."""
    profiles = ["fast", "balanced", "thorough", "cautious"]
    keys = list(BALANCED.keys())

    print(f"{'Key':<40} | " + " | ".join(f"{p:^12}" for p in profiles))
    print("-" * (40 + 4 + len(profiles) * 15))

    for key in keys:
        values = [str(get_config(p).get(key, "N/A"))[:10] for p in profiles]
        print(f"{key:<40} | " + " | ".join(f"{v:^12}" for v in values))


if __name__ == "__main__":
    print_config_comparison()

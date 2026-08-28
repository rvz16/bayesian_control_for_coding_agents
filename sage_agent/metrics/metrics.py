from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

from ..core.types import ToolCall


@dataclass(frozen=True)
class MetricResult:
    """Standard metrics for SAGE-Agent evaluation."""
    coverage_rate: float
    tool_match_rate: float
    parameter_match_rate: float
    avg_questions: float


@dataclass(frozen=True)
class CalibrationResult:
    """Uncertainty calibration metrics.
    
    These metrics measure how well the agent's uncertainty estimates
    correspond to actual error rates. Well-calibrated uncertainty is
    critical for deciding when to ask clarifying questions.
    """
    # Expected Calibration Error (ECE) - lower is better
    ece: float
    # Maximum Calibration Error (MCE) - worst bin error
    mce: float
    # Brier score - measures probabilistic accuracy
    brier_score: float
    # Per-bin statistics for reliability diagram
    bin_accuracies: Tuple[float, ...]
    bin_confidences: Tuple[float, ...]
    bin_counts: Tuple[int, ...]


@dataclass(frozen=True)
class ExtendedMetricResult:
    """Extended metrics including calibration for comprehensive evaluation."""
    coverage_rate: float
    tool_match_rate: float
    parameter_match_rate: float
    avg_questions: float
    # Calibration metrics (optional, requires confidence scores)
    calibration: Optional[CalibrationResult] = None


def evaluate_metrics(
    predictions: Sequence[ToolCall],
    ground_truths: Sequence[ToolCall],
    question_counts: Sequence[int],
) -> MetricResult:
    """Evaluate standard SAGE-Agent metrics.
    
    Args:
        predictions: Predicted tool calls
        ground_truths: Ground truth tool calls
        question_counts: Number of clarifying questions asked per example
        
    Returns:
        MetricResult with coverage, match rates, and question efficiency
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have the same length")
    if len(predictions) != len(question_counts):
        raise ValueError("Question counts must align with predictions")
    if not predictions:
        raise ValueError("At least one prediction is required for metric computation")

    total_tool_match = 0
    total_param_match = 0.0
    total_coverage = 0
    total_questions = 0.0

    for idx, (pred, truth, questions) in enumerate(
        zip(predictions, ground_truths, question_counts)
    ):
        if questions < 0:
            raise ValueError(f"Question count must be non-negative at index {idx}")
        tool_match = pred.tool_name == truth.tool_name
        total_tool_match += 1 if tool_match else 0
        total_questions += float(questions)

        if tool_match:
            match_ratio = _parameter_match_ratio(pred.arguments, truth.arguments)
        else:
            match_ratio = 0.0
        total_param_match += match_ratio
        total_coverage += 1 if tool_match and match_ratio == 1.0 else 0

    count = float(len(predictions))
    return MetricResult(
        coverage_rate=total_coverage / count,
        tool_match_rate=total_tool_match / count,
        parameter_match_rate=total_param_match / count,
        avg_questions=total_questions / count,
    )


def evaluate_extended_metrics(
    predictions: Sequence[ToolCall],
    ground_truths: Sequence[ToolCall],
    question_counts: Sequence[int],
    confidence_scores: Optional[Sequence[float]] = None,
    num_bins: int = 10,
) -> ExtendedMetricResult:
    """Evaluate extended metrics including uncertainty calibration.
    
    This computes standard metrics plus Expected Calibration Error (ECE)
    and related calibration metrics when confidence scores are provided.
    
    Args:
        predictions: Predicted tool calls
        ground_truths: Ground truth tool calls
        question_counts: Number of clarifying questions per example
        confidence_scores: Agent's confidence (1 - uncertainty) for each prediction
        num_bins: Number of bins for calibration computation
        
    Returns:
        ExtendedMetricResult with all metrics including calibration
    """
    base_metrics = evaluate_metrics(predictions, ground_truths, question_counts)
    
    calibration = None
    if confidence_scores is not None:
        # Compute correctness for each prediction
        correctness = []
        for pred, truth in zip(predictions, ground_truths):
            tool_match = pred.tool_name == truth.tool_name
            if tool_match:
                param_match = _parameter_match_ratio(pred.arguments, truth.arguments)
                # Consider correct if tool matches and all params match
                correctness.append(1.0 if param_match == 1.0 else 0.0)
            else:
                correctness.append(0.0)
        
        calibration = compute_calibration(
            confidence_scores=list(confidence_scores),
            correctness=correctness,
            num_bins=num_bins,
        )
    
    return ExtendedMetricResult(
        coverage_rate=base_metrics.coverage_rate,
        tool_match_rate=base_metrics.tool_match_rate,
        parameter_match_rate=base_metrics.parameter_match_rate,
        avg_questions=base_metrics.avg_questions,
        calibration=calibration,
    )


def compute_calibration(
    confidence_scores: List[float],
    correctness: List[float],
    num_bins: int = 10,
) -> CalibrationResult:
    """Compute Expected Calibration Error (ECE) and related metrics.
    
    ECE measures how well predicted confidence aligns with actual accuracy.
    A well-calibrated model should have confidence ≈ accuracy for each bin.
    
    For SAGE-Agent, this helps assess whether the structured uncertainty
    properly reflects actual prediction quality.
    
    Args:
        confidence_scores: Agent's confidence (1 - uncertainty) per prediction
        correctness: 1.0 if prediction was correct, 0.0 otherwise
        num_bins: Number of bins for computing ECE
        
    Returns:
        CalibrationResult with ECE, MCE, Brier score, and bin statistics
    """
    if len(confidence_scores) != len(correctness):
        raise ValueError("Confidence scores and correctness must have same length")
    
    n = len(confidence_scores)
    if n == 0:
        return CalibrationResult(
            ece=0.0,
            mce=0.0,
            brier_score=0.0,
            bin_accuracies=(),
            bin_confidences=(),
            bin_counts=(),
        )
    
    # Initialize bins
    bin_boundaries = [i / num_bins for i in range(num_bins + 1)]
    bin_correct_sum = [0.0] * num_bins
    bin_conf_sum = [0.0] * num_bins
    bin_counts_list = [0] * num_bins
    
    # Assign samples to bins
    for conf, correct in zip(confidence_scores, correctness):
        # Clamp confidence to [0, 1]
        conf = max(0.0, min(1.0, conf))
        # Find bin index
        bin_idx = min(int(conf * num_bins), num_bins - 1)
        bin_correct_sum[bin_idx] += correct
        bin_conf_sum[bin_idx] += conf
        bin_counts_list[bin_idx] += 1
    
    # Compute per-bin accuracy and confidence
    bin_accuracies = []
    bin_confidences = []
    ece = 0.0
    mce = 0.0
    
    for i in range(num_bins):
        if bin_counts_list[i] > 0:
            bin_acc = bin_correct_sum[i] / bin_counts_list[i]
            bin_conf = bin_conf_sum[i] / bin_counts_list[i]
            bin_accuracies.append(bin_acc)
            bin_confidences.append(bin_conf)
            
            # Calibration error for this bin
            bin_error = abs(bin_acc - bin_conf)
            ece += (bin_counts_list[i] / n) * bin_error
            mce = max(mce, bin_error)
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)
    
    # Compute Brier score
    brier_score = sum(
        (conf - correct) ** 2
        for conf, correct in zip(confidence_scores, correctness)
    ) / n
    
    return CalibrationResult(
        ece=ece,
        mce=mce,
        brier_score=brier_score,
        bin_accuracies=tuple(bin_accuracies),
        bin_confidences=tuple(bin_confidences),
        bin_counts=tuple(bin_counts_list),
    )


def compute_uncertainty_aware_accuracy(
    predictions: Sequence[ToolCall],
    ground_truths: Sequence[ToolCall],
    uncertainties: Sequence[float],
    threshold: float = 0.5,
) -> Tuple[float, float, float]:
    """Compute accuracy metrics with uncertainty-based filtering.
    
    This evaluates the agent's ability to "know what it doesn't know":
    - Accuracy on low-uncertainty (confident) predictions
    - Abstention rate (fraction rejected due to high uncertainty)
    - Coverage (fraction of correct predictions among all)
    
    Args:
        predictions: Predicted tool calls
        ground_truths: Ground truth tool calls
        uncertainties: Uncertainty scores (0 = certain, 1 = uncertain)
        threshold: Uncertainty threshold for execution
        
    Returns:
        Tuple of (confident_accuracy, abstention_rate, selective_coverage)
    """
    if len(predictions) != len(ground_truths) or len(predictions) != len(uncertainties):
        raise ValueError("All sequences must have the same length")
    
    n = len(predictions)
    if n == 0:
        return (0.0, 0.0, 0.0)
    
    confident_correct = 0
    confident_total = 0
    abstained = 0
    
    for pred, truth, unc in zip(predictions, ground_truths, uncertainties):
        if unc > threshold:
            abstained += 1
            continue
        
        confident_total += 1
        tool_match = pred.tool_name == truth.tool_name
        if tool_match:
            param_match = _parameter_match_ratio(pred.arguments, truth.arguments)
            if param_match == 1.0:
                confident_correct += 1
    
    confident_accuracy = confident_correct / confident_total if confident_total > 0 else 0.0
    abstention_rate = abstained / n
    selective_coverage = confident_correct / n
    
    return (confident_accuracy, abstention_rate, selective_coverage)


def _parameter_match_ratio(
    predicted: Mapping[str, object],
    ground_truth: Mapping[str, object],
) -> float:
    if not ground_truth:
        return 1.0
    matched = 0
    for key, value in ground_truth.items():
        if key in predicted and predicted[key] == value:
            matched += 1
    return matched / float(len(ground_truth))

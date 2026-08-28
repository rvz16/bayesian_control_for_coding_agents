"""Advanced reasoning techniques for uncertainty-guided agents.

This module implements techniques from multiple research papers:

1. SAUP (Semantic-Aware Uncertainty Propagation):
   - Uncertainty decomposition (aleatoric vs epistemic)
   - Semantic similarity-based propagation

2. Chain-of-Thought Verification (Wei et al.):
   - Step-by-step verification of reasoning chains
   - Early error detection

3. Reflexion (Shinn et al.):
   - Self-reflection on failures
   - Iterative improvement with memory

Usage:
    from sage_agent.core.advanced_reasoning import (
        UncertaintyDecomposer,
        ChainOfThoughtVerifier,
        ReflexionAgent,
    )
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import Counter


# =============================================================================
# SAUP: Uncertainty Decomposition
# =============================================================================

@dataclass
class DecomposedUncertainty:
    """Decomposed uncertainty into aleatoric and epistemic components.
    
    Aleatoric: Irreducible uncertainty from data ambiguity
    Epistemic: Reducible uncertainty from model limitations
    """
    aleatoric: float  # Data uncertainty (irreducible)
    epistemic: float  # Model uncertainty (reducible with more data/knowledge)
    total: float      # Combined uncertainty
    
    @property
    def is_epistemic_dominant(self) -> bool:
        """Check if epistemic uncertainty dominates (could benefit from clarification)."""
        return self.epistemic > self.aleatoric
    
    @property
    def confidence(self) -> float:
        """Convert to confidence score."""
        return 1.0 - self.total


class UncertaintyDecomposer:
    """Decomposes uncertainty into aleatoric and epistemic components.
    
    Based on SAUP paper's approach:
    - Epistemic: Variance between different model samples (disagreement)
    - Aleatoric: Average entropy/uncertainty within each sample
    """
    
    def __init__(self, num_samples: int = 5):
        self.num_samples = num_samples
    
    def decompose_from_samples(
        self,
        samples: List[str],
        extract_answer: Callable[[str], Any] = None,
    ) -> DecomposedUncertainty:
        """Decompose uncertainty from multiple model samples.
        
        Args:
            samples: Multiple responses from the model
            extract_answer: Function to extract answer from response
            
        Returns:
            DecomposedUncertainty with aleatoric and epistemic components
        """
        if not samples:
            return DecomposedUncertainty(0.5, 0.5, 1.0)
        
        # Extract answers if extractor provided
        if extract_answer:
            answers = [extract_answer(s) for s in samples]
        else:
            answers = samples
        
        # Epistemic uncertainty: disagreement between samples
        epistemic = self._compute_disagreement(answers)
        
        # Aleatoric uncertainty: average "hedging" in responses
        aleatoric = self._compute_hedging(samples)
        
        # Combine (not simply additive - use noisy-or)
        total = 1.0 - (1.0 - epistemic) * (1.0 - aleatoric)
        
        return DecomposedUncertainty(
            aleatoric=aleatoric,
            epistemic=epistemic,
            total=min(1.0, total),
        )
    
    def _compute_disagreement(self, answers: List[Any]) -> float:
        """Compute disagreement between samples (epistemic uncertainty)."""
        if len(answers) <= 1:
            return 0.0
        
        # Count unique answers
        counter = Counter(str(a) for a in answers if a is not None)
        if not counter:
            return 1.0
        
        # If all agree, low epistemic uncertainty
        most_common_count = counter.most_common(1)[0][1]
        agreement_rate = most_common_count / len(answers)
        
        # Convert to uncertainty (1 - agreement)
        return 1.0 - agreement_rate
    
    def _compute_hedging(self, samples: List[str]) -> float:
        """Compute hedging/uncertainty language in responses (aleatoric proxy)."""
        hedging_phrases = [
            r'\bmight\b', r'\bmaybe\b', r'\bperhaps\b', r'\bpossibly\b',
            r'\bcould be\b', r'\bnot sure\b', r'\buncertain\b',
            r'\bapproximately\b', r'\babout\b', r'\baround\b',
            r'\bi think\b', r'\bi believe\b', r'\bit seems\b',
            r'\bor\b.*\bor\b',  # Multiple alternatives
        ]
        
        total_hedging = 0
        for sample in samples:
            sample_lower = sample.lower()
            hedging_count = sum(
                1 for phrase in hedging_phrases 
                if re.search(phrase, sample_lower)
            )
            # Normalize by response length
            words = len(sample.split())
            if words > 0:
                total_hedging += min(hedging_count / 10, 1.0)  # Cap at 1.0
        
        return total_hedging / len(samples) if samples else 0.5


# =============================================================================
# Chain-of-Thought Verification
# =============================================================================

@dataclass
class VerificationResult:
    """Result of verifying a reasoning step."""
    step_index: int
    step_content: str
    is_valid: bool
    confidence: float
    error_type: Optional[str] = None
    suggested_fix: Optional[str] = None


@dataclass
class ChainVerificationResult:
    """Result of verifying entire reasoning chain."""
    steps: List[VerificationResult]
    overall_valid: bool
    first_error_index: int  # -1 if no errors
    chain_confidence: float
    
    @property
    def error_step(self) -> Optional[VerificationResult]:
        """Get the first erroneous step if any."""
        if self.first_error_index >= 0:
            return self.steps[self.first_error_index]
        return None


class ChainOfThoughtVerifier:
    """Verifies chain-of-thought reasoning step by step.
    
    Based on self-consistency and step verification research.
    Identifies errors early in reasoning chains before they propagate.
    """
    
    def __init__(self, llm_verify: Callable[[str], Tuple[bool, float, str]] = None):
        """Initialize verifier.
        
        Args:
            llm_verify: Function that takes a verification prompt and returns
                       (is_valid, confidence, explanation)
        """
        self._llm_verify = llm_verify
    
    def extract_steps(self, reasoning: str) -> List[str]:
        """Extract individual reasoning steps from a response."""
        # Try numbered steps first
        numbered_pattern = r'(?:^|\n)\s*(?:\d+[\.\)]\s*|Step\s*\d+[:\.]?\s*)(.*?)(?=\n\s*(?:\d+[\.\)]|Step\s*\d+)|$)'
        matches = re.findall(numbered_pattern, reasoning, re.IGNORECASE | re.DOTALL)
        
        if matches and len(matches) > 1:
            return [m.strip() for m in matches if m.strip()]
        
        # Try sentence-based splitting
        sentences = re.split(r'(?<=[.!?])\s+', reasoning)
        
        # Filter out very short sentences
        steps = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        return steps if steps else [reasoning]
    
    def verify_step(
        self,
        step: str,
        context: str,
        step_index: int,
    ) -> VerificationResult:
        """Verify a single reasoning step.
        
        Args:
            step: The step to verify
            context: Previous steps and problem context
            step_index: Index of this step
            
        Returns:
            VerificationResult
        """
        # Rule-based checks first
        rule_result = self._rule_based_check(step)
        if rule_result is not None:
            return VerificationResult(
                step_index=step_index,
                step_content=step,
                is_valid=rule_result[0],
                confidence=rule_result[1],
                error_type=rule_result[2] if not rule_result[0] else None,
            )
        
        # LLM-based verification if available
        if self._llm_verify:
            prompt = f"""Verify if this reasoning step is logically correct.

Context:
{context}

Step to verify:
{step}

Is this step correct? Answer with:
- VALID: if the step is logically sound
- INVALID: if there's an error

Your judgment:"""
            
            is_valid, confidence, explanation = self._llm_verify(prompt)
            
            return VerificationResult(
                step_index=step_index,
                step_content=step,
                is_valid=is_valid,
                confidence=confidence,
                error_type="logical_error" if not is_valid else None,
                suggested_fix=explanation if not is_valid else None,
            )
        
        # Default: assume valid with moderate confidence
        return VerificationResult(
            step_index=step_index,
            step_content=step,
            is_valid=True,
            confidence=0.7,
        )
    
    def _rule_based_check(self, step: str) -> Optional[Tuple[bool, float, str]]:
        """Apply rule-based checks for common errors."""
        step_lower = step.lower()
        
        # Check for mathematical contradictions
        if re.search(r'(\d+)\s*[+]\s*(\d+)\s*=\s*(\d+)', step):
            match = re.search(r'(\d+)\s*[+]\s*(\d+)\s*=\s*(\d+)', step)
            a, b, c = int(match.group(1)), int(match.group(2)), int(match.group(3))
            if a + b != c:
                return (False, 0.95, f"arithmetic_error: {a}+{b}≠{c}")
        
        if re.search(r'(\d+)\s*[-]\s*(\d+)\s*=\s*(\d+)', step):
            match = re.search(r'(\d+)\s*[-]\s*(\d+)\s*=\s*(\d+)', step)
            a, b, c = int(match.group(1)), int(match.group(2)), int(match.group(3))
            if a - b != c:
                return (False, 0.95, f"arithmetic_error: {a}-{b}≠{c}")
        
        if re.search(r'(\d+)\s*[*×x]\s*(\d+)\s*=\s*(\d+)', step):
            match = re.search(r'(\d+)\s*[*×x]\s*(\d+)\s*=\s*(\d+)', step)
            a, b, c = int(match.group(1)), int(match.group(2)), int(match.group(3))
            if a * b != c:
                return (False, 0.95, f"arithmetic_error: {a}×{b}≠{c}")
        
        # Check for division by zero
        if re.search(r'/\s*0(?:\s|$|[^\d])', step):
            return (False, 0.99, "division_by_zero")
        
        # Check for nonsensical conclusions
        contradiction_patterns = [
            (r'therefore.*impossible', "logical_contradiction"),
            (r'negative.*(?:count|number of|quantity)', "negative_quantity"),
        ]
        
        for pattern, error_type in contradiction_patterns:
            if re.search(pattern, step_lower):
                return (False, 0.8, error_type)
        
        return None
    
    def verify_chain(
        self,
        reasoning: str,
        problem: str = "",
    ) -> ChainVerificationResult:
        """Verify an entire reasoning chain.
        
        Args:
            reasoning: Full reasoning response
            problem: Original problem statement
            
        Returns:
            ChainVerificationResult with per-step and overall results
        """
        steps = self.extract_steps(reasoning)
        
        results = []
        context = f"Problem: {problem}\n\n" if problem else ""
        first_error = -1
        
        for i, step in enumerate(steps):
            result = self.verify_step(step, context, i)
            results.append(result)
            
            if not result.is_valid and first_error == -1:
                first_error = i
            
            context += f"Step {i+1}: {step}\n"
        
        # Compute chain confidence (multiplicative)
        if results:
            chain_confidence = 1.0
            for r in results:
                chain_confidence *= r.confidence
        else:
            chain_confidence = 0.5
        
        return ChainVerificationResult(
            steps=results,
            overall_valid=(first_error == -1),
            first_error_index=first_error,
            chain_confidence=chain_confidence,
        )


# =============================================================================
# Reflexion: Self-Reflection and Iterative Improvement
# =============================================================================

@dataclass
class ReflectionMemory:
    """Memory of past attempts and reflections."""
    problem: str
    attempt: str
    was_correct: bool
    reflection: str
    error_type: Optional[str] = None
    improvement_hint: Optional[str] = None


@dataclass
class ReflexionResult:
    """Result of reflexion loop."""
    final_answer: str
    num_attempts: int
    reflections: List[ReflectionMemory]
    final_confidence: float
    improved: bool  # Whether later attempts were better


class ReflexionAgent:
    """Agent that learns from its mistakes through self-reflection.
    
    Based on Reflexion paper (Shinn et al. 2023):
    1. Generate initial response
    2. Evaluate/verify the response
    3. If wrong, reflect on what went wrong
    4. Use reflection to improve next attempt
    5. Repeat until correct or max attempts
    """
    
    def __init__(
        self,
        generate_fn: Callable[[str], str],
        evaluate_fn: Callable[[str, str], Tuple[bool, str]],
        reflect_fn: Optional[Callable[[str, str, str], str]] = None,
        max_attempts: int = 3,
    ):
        """Initialize Reflexion agent.
        
        Args:
            generate_fn: Function to generate response given prompt
            evaluate_fn: Function to evaluate response, returns (is_correct, feedback)
            reflect_fn: Function to generate reflection given problem, attempt, feedback
            max_attempts: Maximum number of attempts
        """
        self._generate = generate_fn
        self._evaluate = evaluate_fn
        self._reflect = reflect_fn
        self.max_attempts = max_attempts
        self._memory: List[ReflectionMemory] = []
    
    def _default_reflect(self, problem: str, attempt: str, feedback: str) -> str:
        """Default reflection generation using the generate function."""
        reflect_prompt = f"""Analyze why this solution might be wrong and how to improve it.

Problem: {problem}

My attempt: {attempt}

Feedback: {feedback}

What went wrong and how should I approach this differently?
Reflection:"""
        
        return self._generate(reflect_prompt)
    
    def solve(
        self,
        problem: str,
        ground_truth: Optional[str] = None,
    ) -> ReflexionResult:
        """Solve a problem using reflexion loop.
        
        Args:
            problem: The problem to solve
            ground_truth: Optional ground truth for evaluation
            
        Returns:
            ReflexionResult with final answer and reflection history
        """
        reflections: List[ReflectionMemory] = []
        best_attempt = ""
        best_confidence = 0.0
        
        for attempt_num in range(self.max_attempts):
            # Build prompt with previous reflections
            prompt = self._build_prompt(problem, reflections)
            
            # Generate attempt
            attempt = self._generate(prompt)
            
            # Evaluate
            is_correct, feedback = self._evaluate(attempt, ground_truth or "")
            
            # Track best attempt
            # (In real scenario, might use a scoring function)
            confidence = 0.9 if is_correct else 0.5 - (attempt_num * 0.1)
            if confidence > best_confidence:
                best_confidence = confidence
                best_attempt = attempt
            
            if is_correct:
                return ReflexionResult(
                    final_answer=attempt,
                    num_attempts=attempt_num + 1,
                    reflections=reflections,
                    final_confidence=0.9,
                    improved=attempt_num > 0,
                )
            
            # Generate reflection
            reflect_fn = self._reflect or self._default_reflect
            reflection = reflect_fn(problem, attempt, feedback)
            
            # Store in memory
            memory = ReflectionMemory(
                problem=problem,
                attempt=attempt,
                was_correct=False,
                reflection=reflection,
                error_type=self._classify_error(feedback),
                improvement_hint=self._extract_improvement(reflection),
            )
            reflections.append(memory)
            self._memory.append(memory)
        
        # Return best attempt after all tries
        return ReflexionResult(
            final_answer=best_attempt,
            num_attempts=self.max_attempts,
            reflections=reflections,
            final_confidence=best_confidence,
            improved=len(reflections) > 1,
        )
    
    def _build_prompt(
        self,
        problem: str,
        reflections: List[ReflectionMemory],
    ) -> str:
        """Build prompt incorporating previous reflections."""
        prompt = f"Problem: {problem}\n\n"
        
        if reflections:
            prompt += "Previous attempts and lessons learned:\n"
            for i, ref in enumerate(reflections[-3:]):  # Last 3 reflections
                prompt += f"\nAttempt {i+1} (incorrect):\n"
                prompt += f"  What I tried: {ref.attempt[:200]}...\n"
                prompt += f"  What went wrong: {ref.reflection[:200]}...\n"
            prompt += "\nUsing these lessons, solve the problem correctly:\n"
        else:
            prompt += "Solve this problem step by step:\n"
        
        return prompt
    
    def _classify_error(self, feedback: str) -> Optional[str]:
        """Classify the type of error from feedback."""
        feedback_lower = feedback.lower()
        
        if any(word in feedback_lower for word in ['arithmetic', 'calculation', 'math']):
            return "arithmetic_error"
        if any(word in feedback_lower for word in ['logic', 'reasoning', 'contradiction']):
            return "logical_error"
        if any(word in feedback_lower for word in ['misread', 'misunderstand', 'wrong assumption']):
            return "comprehension_error"
        if any(word in feedback_lower for word in ['incomplete', 'missing', 'forgot']):
            return "incomplete_solution"
        
        return "unknown_error"
    
    def _extract_improvement(self, reflection: str) -> Optional[str]:
        """Extract actionable improvement from reflection."""
        # Look for improvement patterns
        patterns = [
            r'(?:should|need to|must)\s+(.{20,100})',
            r'(?:instead|rather)\s+(.{20,100})',
            r'(?:correct approach|better way)\s*[:is]*\s*(.{20,100})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, reflection.lower())
            if match:
                return match.group(1).strip()
        
        return None
    
    def get_error_patterns(self) -> Dict[str, int]:
        """Analyze error patterns from memory."""
        patterns: Dict[str, int] = {}
        for mem in self._memory:
            if mem.error_type:
                patterns[mem.error_type] = patterns.get(mem.error_type, 0) + 1
        return patterns


# =============================================================================
# Integration Helper
# =============================================================================

def create_enhanced_uncertainty_pipeline(
    llm_fn: Callable[[str], str],
    num_samples: int = 4,
) -> Dict[str, Any]:
    """Create a complete uncertainty-aware reasoning pipeline.
    
    Integrates:
    - SAUP uncertainty decomposition
    - Chain-of-thought verification
    - Reflexion for improvement
    
    Args:
        llm_fn: Function to call LLM
        num_samples: Number of samples for uncertainty estimation
        
    Returns:
        Dict with pipeline components
    """
    # Uncertainty decomposer
    decomposer = UncertaintyDecomposer(num_samples=num_samples)
    
    # CoT Verifier with LLM
    def llm_verify(prompt: str) -> Tuple[bool, float, str]:
        response = llm_fn(prompt)
        is_valid = "VALID" in response.upper() and "INVALID" not in response.upper()
        confidence = 0.8 if is_valid else 0.3
        return is_valid, confidence, response
    
    verifier = ChainOfThoughtVerifier(llm_verify=llm_verify)
    
    # Simple evaluator for reflexion
    def evaluate(attempt: str, ground_truth: str) -> Tuple[bool, str]:
        if not ground_truth:
            return True, "No ground truth provided"
        # Simple check - could be enhanced
        is_correct = ground_truth.strip().lower() in attempt.lower()
        feedback = "Correct!" if is_correct else f"Expected: {ground_truth}"
        return is_correct, feedback
    
    reflexion = ReflexionAgent(
        generate_fn=llm_fn,
        evaluate_fn=evaluate,
        max_attempts=3,
    )
    
    return {
        "decomposer": decomposer,
        "verifier": verifier,
        "reflexion": reflexion,
        "sample_and_decompose": lambda prompt: decomposer.decompose_from_samples(
            [llm_fn(prompt) for _ in range(num_samples)]
        ),
    }


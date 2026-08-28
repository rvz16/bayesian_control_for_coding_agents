from __future__ import annotations

import json
import re
from typing import Iterable, List, Optional, Protocol, Set

from .domains import ParameterDomain
from .types import ConstraintExtractor


class LLMClient(Protocol):
    """Protocol for LLM clients used by constraint extractor."""
    def complete(self, prompt: str) -> str:
        ...


class SimpleConstraintExtractor(ConstraintExtractor):
    """Simple rule-based constraint extractor using pattern matching."""
    
    def __init__(self, negative_patterns: Optional[Iterable[str]] = None) -> None:
        self._negative_patterns = tuple(negative_patterns or ("not {value}", "no {value}"))

    def update_domain(self, domain: ParameterDomain, response: str) -> ParameterDomain:
        if domain.is_finite():
            return self._update_finite(domain, response)
        return domain

    def _update_finite(self, domain: ParameterDomain, response: str) -> ParameterDomain:
        response_lower = response.lower()
        values = list(domain.values or [])
        matched = {v for v in values if str(v).lower() in response_lower}
        excluded = set()
        for v in values:
            v_lower = str(v).lower()
            for pattern in self._negative_patterns:
                if pattern.format(value=v_lower) in response_lower:
                    excluded.add(v)
                    break
        if matched:
            new_values = matched.difference(excluded)
            if not new_values:
                # If all matched values were excluded, fall back to excluding only.
                return domain.exclude_values(excluded)
            return domain.intersect_values(new_values)
        if excluded:
            return domain.exclude_values(excluded)
        return domain


class LLMConstraintExtractor(ConstraintExtractor):
    """LLM-backed constraint extractor for semantic domain refinement.
    
    This implements a more sophisticated constraint extraction that can understand
    semantic intent from user responses, not just literal string matches.
    
    For example, if the domain is ["morning", "afternoon", "evening"] and the user
    says "I prefer earlier times", this extractor can infer that "morning" is preferred.
    
    This addresses the paper's emphasis on leveraging structured tool schemas
    for better disambiguation (Section 3.2).
    """
    
    EXTRACTION_PROMPT = """You are analyzing a user response to extract parameter constraints.

Given a user response and a set of valid domain values, determine:
1. Which values the user explicitly or implicitly prefers (include)
2. Which values the user explicitly or implicitly rejects (exclude)
3. If the response is unclear or doesn't constrain the domain

User response: "{response}"

Valid domain values: {values}

Respond with a JSON object:
{{
    "include": ["list of preferred values from the domain"],
    "exclude": ["list of rejected values from the domain"],
    "unclear": true/false
}}

Only include values that are in the valid domain values list.
If the response doesn't clearly constrain values, set "unclear": true and leave include/exclude empty.
JSON only:"""

    NUMERIC_CONSTRAINT_PROMPT = """You are analyzing a user response to extract numeric constraints.

Given a user response, extract any numeric constraints mentioned.

User response: "{response}"

Respond with a JSON object:
{{
    "min": null or number (minimum value mentioned),
    "max": null or number (maximum value mentioned),
    "exact": null or number (exact value if specified),
    "unclear": true/false
}}

JSON only:"""

    def __init__(
        self,
        llm: LLMClient,
        fallback_extractor: Optional[ConstraintExtractor] = None,
        max_domain_size: int = 50,
    ) -> None:
        """Initialize LLM constraint extractor.
        
        Args:
            llm: LLM client for semantic parsing
            fallback_extractor: Fallback for when LLM fails (defaults to SimpleConstraintExtractor)
            max_domain_size: Maximum domain size to send to LLM (larger domains use fallback)
        """
        self.llm = llm
        self.fallback = fallback_extractor or SimpleConstraintExtractor()
        self.max_domain_size = max_domain_size

    def update_domain(self, domain: ParameterDomain, response: str) -> ParameterDomain:
        """Update domain based on user response using LLM semantic understanding."""
        if not response.strip():
            return domain
        
        if domain.is_finite():
            return self._update_finite_domain(domain, response)
        else:
            return self._update_continuous_domain(domain, response)

    def _update_finite_domain(self, domain: ParameterDomain, response: str) -> ParameterDomain:
        """Update finite domain using LLM to understand user preferences."""
        values = list(domain.values or [])
        
        # Fall back for very large domains (too expensive/slow for LLM)
        if len(values) > self.max_domain_size:
            return self.fallback.update_domain(domain, response)
        
        # Format values for prompt
        values_str = json.dumps([str(v) for v in values])
        prompt = self.EXTRACTION_PROMPT.format(response=response, values=values_str)
        
        try:
            llm_response = self.llm.complete(prompt)
            parsed = self._parse_json_response(llm_response)
            
            if parsed.get("unclear", True):
                # LLM couldn't determine constraints, try fallback
                return self.fallback.update_domain(domain, response)
            
            include_strs: List[str] = parsed.get("include", [])
            exclude_strs: List[str] = parsed.get("exclude", [])
            
            # Map string responses back to actual domain values
            include_values = self._match_values_to_domain(include_strs, values)
            exclude_values = self._match_values_to_domain(exclude_strs, values)
            
            # Apply constraints
            if include_values:
                # Intersection with included values (minus any excluded)
                final_include = include_values - exclude_values
                if final_include:
                    return domain.intersect_values(final_include)
            
            if exclude_values:
                return domain.exclude_values(exclude_values)
            
            return domain
            
        except Exception:
            # On any error, fall back to simple extractor
            return self.fallback.update_domain(domain, response)

    def _update_continuous_domain(self, domain: ParameterDomain, response: str) -> ParameterDomain:
        """Update continuous domain by extracting numeric constraints."""
        prompt = self.NUMERIC_CONSTRAINT_PROMPT.format(response=response)
        
        try:
            llm_response = self.llm.complete(prompt)
            parsed = self._parse_json_response(llm_response)
            
            if parsed.get("unclear", True):
                return domain
            
            constraints = []
            
            if parsed.get("exact") is not None:
                exact = float(parsed["exact"])
                constraints.append(lambda v, e=exact: v == e)
            else:
                if parsed.get("min") is not None:
                    min_val = float(parsed["min"])
                    constraints.append(lambda v, m=min_val: v >= m)
                if parsed.get("max") is not None:
                    max_val = float(parsed["max"])
                    constraints.append(lambda v, m=max_val: v <= m)
            
            if constraints:
                return domain.with_constraints(*constraints)
            
            return domain
            
        except Exception:
            return domain

    def _parse_json_response(self, text: str) -> dict:
        """Parse JSON from LLM response, handling common formatting issues."""
        cleaned = text.strip()
        
        # Try direct parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON object from response
        match = re.search(r'\{[\s\S]*\}', cleaned)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        
        return {"unclear": True}

    def _match_values_to_domain(self, str_values: List[str], domain_values: List) -> Set:
        """Match string values from LLM back to actual domain values."""
        matched = set()
        str_lower_map = {str(v).lower(): v for v in domain_values}
        
        for s in str_values:
            s_lower = str(s).lower().strip()
            if s_lower in str_lower_map:
                matched.add(str_lower_map[s_lower])
            else:
                # Fuzzy match: check if domain value is contained in response
                for domain_str, domain_val in str_lower_map.items():
                    if domain_str in s_lower or s_lower in domain_str:
                        matched.add(domain_val)
                        break
        
        return matched


class HybridConstraintExtractor(ConstraintExtractor):
    """Hybrid extractor that combines rule-based and LLM approaches.
    
    Uses simple rule-based extraction first for efficiency, then falls back
    to LLM for ambiguous cases. This balances speed and accuracy.
    """
    
    def __init__(
        self,
        llm: LLMClient,
        ambiguity_threshold: float = 0.5,
    ) -> None:
        """Initialize hybrid extractor.
        
        Args:
            llm: LLM client for complex cases
            ambiguity_threshold: If simple extraction doesn't reduce domain by
                                 at least this fraction, use LLM
        """
        self.simple = SimpleConstraintExtractor()
        self.llm_extractor = LLMConstraintExtractor(llm)
        self.ambiguity_threshold = ambiguity_threshold

    def update_domain(self, domain: ParameterDomain, response: str) -> ParameterDomain:
        """Update domain using hybrid approach."""
        if not domain.is_finite():
            # For continuous domains, always use LLM (rules don't work well)
            return self.llm_extractor.update_domain(domain, response)
        
        original_size = domain.size() or 1
        
        # Try simple extraction first
        simple_result = self.simple.update_domain(domain, response)
        simple_size = simple_result.size() or original_size
        
        # Calculate reduction ratio
        reduction = 1.0 - (simple_size / original_size)
        
        if reduction >= self.ambiguity_threshold:
            # Simple extraction was effective
            return simple_result
        
        # Simple extraction wasn't effective enough, try LLM
        return self.llm_extractor.update_domain(domain, response)

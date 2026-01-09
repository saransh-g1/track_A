"""
Global Consistency Aggregator
Aggregates evidence from all claims, resolves conflicts, and makes final decision.
"""
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import statistics


@dataclass
class ClaimVerification:
    """Result of verifying a single claim (forward reference)"""
    claim: "AtomicClaim"
    is_satisfied: bool
    confidence: float
    evidence_chunks: List
    reasoning: str
    contradiction_signals: List[str]
    symbolic_violations: List[str]


@dataclass
class FinalDecision:
    """Final decision on narrative coherence"""
    label: int  # 1 = coherent, 0 = not coherent
    confidence: float  # 0.0 to 1.0
    rationale: str
    satisfied_claims: int
    violated_claims: int
    total_claims: int
    claim_details: List[Dict]
    conflict_summary: Dict


class GlobalConsistencyAggregator:
    """
    Aggregates verification results from all claims and makes final decision.
    """
    
    def __init__(
        self,
        consistency_threshold: float = 0.6,
        contradiction_threshold: float = 0.7
    ):
        self.consistency_threshold = consistency_threshold
        self.contradiction_threshold = contradiction_threshold
    
    def aggregate(
        self,
        claim_verifications: List[ClaimVerification],
        generate_rationale: bool = True,
        max_rationale_length: int = 500
    ) -> FinalDecision:
        """
        Aggregate all claim verifications into final decision.
        
        Args:
            claim_verifications: List of claim verification results
            generate_rationale: Whether to generate detailed rationale
            max_rationale_length: Maximum length of rationale text
            
        Returns:
            FinalDecision object
        """
        if not claim_verifications:
            return FinalDecision(
                label=0,
                confidence=0.0,
                rationale="No claims to verify",
                satisfied_claims=0,
                violated_claims=0,
                total_claims=0,
                claim_details=[],
                conflict_summary={}
            )
        
        # Count satisfied vs violated claims
        satisfied = [cv for cv in claim_verifications if cv.is_satisfied]
        violated = [cv for cv in claim_verifications if not cv.is_satisfied]
        
        satisfied_count = len(satisfied)
        violated_count = len(violated)
        total_count = len(claim_verifications)
        
        # Calculate aggregate confidence
        all_confidences = [cv.confidence for cv in claim_verifications]
        avg_confidence = statistics.mean(all_confidences) if all_confidences else 0.0
        
        # Weight by claim importance (claims with violations reduce confidence more)
        weighted_confidence = self._calculate_weighted_confidence(claim_verifications)
        
        # Analyze conflicts
        conflict_summary = self._analyze_conflicts(claim_verifications)
        
        # Determine final label
        # If majority of claims are satisfied AND no critical violations, label = 1
        satisfaction_ratio = satisfied_count / total_count if total_count > 0 else 0.0
        
        # Check for critical violations (high confidence violations)
        critical_violations = [
            cv for cv in violated
            if cv.confidence > self.contradiction_threshold
        ]
        
        # Decision logic:
        # - If satisfaction ratio >= threshold AND no critical violations -> coherent (1)
        # - If critical violations exist -> not coherent (0)
        # - If satisfaction ratio < threshold -> not coherent (0)
        if critical_violations:
            label = 0
            final_confidence = max(0.0, weighted_confidence - len(critical_violations) * 0.2)
        elif satisfaction_ratio >= self.consistency_threshold:
            label = 1
            final_confidence = weighted_confidence
        else:
            label = 0
            final_confidence = weighted_confidence
        
        # Generate rationale
        rationale = ""
        if generate_rationale:
            rationale = self._generate_rationale(
                claim_verifications,
                satisfied,
                violated,
                critical_violations,
                satisfaction_ratio,
                max_rationale_length
            )
        
        # Prepare claim details
        claim_details = [
            {
                "claim": cv.claim.claim_text,
                "type": cv.claim.claim_type,
                "satisfied": cv.is_satisfied,
                "confidence": cv.confidence,
                "violations": cv.symbolic_violations,
                "contradictions": cv.contradiction_signals
            }
            for cv in claim_verifications
        ]
        
        return FinalDecision(
            label=label,
            confidence=final_confidence,
            rationale=rationale,
            satisfied_claims=satisfied_count,
            violated_claims=violated_count,
            total_claims=total_count,
            claim_details=claim_details,
            conflict_summary=conflict_summary
        )
    
    def _calculate_weighted_confidence(
        self,
        claim_verifications: List[ClaimVerification]
    ) -> float:
        """
        Calculate weighted confidence, giving more weight to violated claims.
        """
        if not claim_verifications:
            return 0.0
        
        # Violated claims reduce confidence more
        total_weight = 0.0
        weighted_sum = 0.0
        
        for cv in claim_verifications:
            if cv.is_satisfied:
                weight = 1.0
                weighted_sum += cv.confidence * weight
            else:
                # Violated claims have higher weight (they matter more)
                weight = 2.0
                # Invert confidence for violations (high confidence violation = low overall confidence)
                weighted_sum += (1.0 - cv.confidence) * weight
            
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _analyze_conflicts(
        self,
        claim_verifications: List[ClaimVerification]
    ) -> Dict:
        """
        Analyze conflicts and contradictions across all claims.
        """
        conflict_summary = {
            "total_conflicts": 0,
            "symbolic_violations": [],
            "contradiction_types": defaultdict(int),
            "conflicting_entities": defaultdict(list)
        }
        
        for cv in claim_verifications:
            # Count symbolic violations
            if cv.symbolic_violations:
                conflict_summary["total_conflicts"] += len(cv.symbolic_violations)
                conflict_summary["symbolic_violations"].extend(cv.symbolic_violations)
                
                # Categorize violations
                for violation in cv.symbolic_violations:
                    if ":" in violation:
                        violation_type = violation.split(":")[0]
                        conflict_summary["contradiction_types"][violation_type] += 1
            
            # Track conflicting entities
            if not cv.is_satisfied and cv.claim.entities:
                for entity in cv.claim.entities:
                    conflict_summary["conflicting_entities"][entity].append(
                        cv.claim.claim_text
                    )
        
        # Convert defaultdicts to regular dicts
        conflict_summary["contradiction_types"] = dict(conflict_summary["contradiction_types"])
        conflict_summary["conflicting_entities"] = {
            k: v for k, v in conflict_summary["conflicting_entities"].items()
        }
        
        return conflict_summary
    
    def _generate_rationale(
        self,
        claim_verifications: List[ClaimVerification],
        satisfied: List[ClaimVerification],
        violated: List[ClaimVerification],
        critical_violations: List[ClaimVerification],
        satisfaction_ratio: float,
        max_length: int
    ) -> str:
        """
        Generate human-readable rationale for the decision.
        """
        rationale_parts = []
        
        # Summary
        rationale_parts.append(
            f"Analyzed {len(claim_verifications)} claims from the backstory. "
            f"{len(satisfied)} claims are satisfied ({satisfaction_ratio:.1%}), "
            f"{len(violated)} claims are violated."
        )
        
        # Critical violations
        if critical_violations:
            rationale_parts.append(
                f"\nCritical violations detected ({len(critical_violations)}):"
            )
            for cv in critical_violations[:3]:  # Top 3
                rationale_parts.append(
                    f"- {cv.claim.claim_text[:100]}... "
                    f"(confidence: {cv.confidence:.2f})"
                )
        
        # Symbolic violations
        all_symbolic = []
        for cv in violated:
            all_symbolic.extend(cv.symbolic_violations)
        
        if all_symbolic:
            unique_violations = list(set(all_symbolic))[:5]
            rationale_parts.append(
                f"\nSymbolic rule violations: {', '.join(unique_violations[:3])}"
            )
        
        # Key satisfied claims (if coherent)
        if len(satisfied) > len(violated):
            rationale_parts.append(
                f"\nKey satisfied constraints include character traits, "
                f"temporal consistency, and world rules."
            )
        
        rationale = " ".join(rationale_parts)
        
        # Truncate if too long
        if len(rationale) > max_length:
            rationale = rationale[:max_length-3] + "..."
        
        return rationale
    
    def resolve_conflicts(
        self,
        claim_verifications: List[ClaimVerification]
    ) -> List[ClaimVerification]:
        """
        Resolve conflicts between claims (optional advanced feature).
        For example, if two claims contradict each other, determine which is more reliable.
        """
        # Simple implementation: prioritize claims with higher confidence
        # More sophisticated: use entity co-occurrence, temporal ordering, etc.
        
        resolved = []
        seen_claims = set()
        
        # Sort by confidence (higher first)
        sorted_claims = sorted(
            claim_verifications,
            key=lambda x: x.confidence,
            reverse=True
        )
        
        for cv in sorted_claims:
            claim_key = cv.claim.claim_text.lower().strip()
            if claim_key not in seen_claims:
                resolved.append(cv)
                seen_claims.add(claim_key)
        
        return resolved
    
    def get_evidence_summary(
        self,
        claim_verifications: List[ClaimVerification]
    ) -> Dict:
        """
        Generate summary of evidence used across all claims.
        """
        all_evidence_chunks = []
        evidence_by_claim = {}
        
        for cv in claim_verifications:
            claim_evidence = []
            for chunk in cv.evidence_chunks:
                all_evidence_chunks.append(chunk)
                claim_evidence.append({
                    "chunk_id": chunk.chunk_id,
                    "position": chunk.position_ratio,
                    "relevance": chunk.relevance_score
                })
            evidence_by_claim[cv.claim.claim_text] = claim_evidence
        
        # Analyze evidence distribution
        positions = [chunk.position_ratio for chunk in all_evidence_chunks]
        avg_position = statistics.mean(positions) if positions else 0.0
        
        return {
            "total_evidence_chunks": len(set(chunk.chunk_id for chunk in all_evidence_chunks)),
            "average_position": avg_position,
            "evidence_by_claim": evidence_by_claim,
            "position_distribution": {
                "early": len([p for p in positions if p < 0.33]),
                "middle": len([p for p in positions if 0.33 <= p < 0.67]),
                "late": len([p for p in positions if p >= 0.67])
            }
        }



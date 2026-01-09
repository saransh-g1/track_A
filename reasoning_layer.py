"""
Hybrid Reasoning Layer
Combines LLM causal checks with symbolic contradiction rules.
"""
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from collections import defaultdict


@dataclass
class EvidenceChunk:
    """Evidence chunk with metadata"""
    chunk_text: str
    chunk_id: str
    position_ratio: float
    relevance_score: float
    document_id: str


@dataclass
class ClaimVerification:
    """Result of verifying a single claim"""
    claim: "AtomicClaim"  # Forward reference
    is_satisfied: bool  # True if claim is satisfied, False if violated
    confidence: float  # 0.0 to 1.0
    evidence_chunks: List[EvidenceChunk]
    reasoning: str
    contradiction_signals: List[str]  # List of detected contradictions
    symbolic_violations: List[str]  # Symbolic rule violations


@dataclass
class SymbolicRule:
    """Symbolic rule for contradiction detection"""
    rule_name: str
    pattern: str  # Regex or keyword pattern
    rule_type: str  # "contradiction", "consistency", "temporal"
    description: str


class SymbolicRuleEngine:
    """
    Symbolic rule engine for detecting contradictions.
    """
    
    def __init__(self):
        self.rules = self._initialize_rules()
    
    def _initialize_rules(self) -> List[SymbolicRule]:
        """Initialize built-in symbolic rules"""
        rules = [
            # Moral/ethical contradictions
            SymbolicRule(
                rule_name="moral_code_violation",
                pattern=r"(never|must not|forbidden|prohibited).*?(revenge|betrayal|harm)",
                rule_type="contradiction",
                description="Violation of stated moral code"
            ),
            SymbolicRule(
                rule_name="trust_contradiction",
                pattern=r"(never.*trust|distrust|suspicious).*?(trust|believe|rely)",
                rule_type="contradiction",
                description="Contradiction in trust behavior"
            ),
            # Temporal contradictions
            SymbolicRule(
                rule_name="age_contradiction",
                pattern=r"age.*?(\d+).*?(\d+)",
                rule_type="temporal",
                description="Age inconsistency"
            ),
            SymbolicRule(
                rule_name="temporal_order",
                pattern=r"(before|after|then|later).*?(before|after|then|earlier)",
                rule_type="temporal",
                description="Temporal ordering violation"
            ),
            # Character trait contradictions
            SymbolicRule(
                rule_name="fear_contradiction",
                pattern=r"(fear|afraid|terrified).*?(confront|face|brave)",
                rule_type="contradiction",
                description="Fear vs bravery contradiction"
            ),
            SymbolicRule(
                rule_name="authority_contradiction",
                pattern=r"(never.*authority|rebel|resist).*?(obey|submit|authority)",
                rule_type="contradiction",
                description="Authority relationship contradiction"
            ),
        ]
        return rules
    
    def check_contradictions(
        self,
        claim_text: str,
        evidence_texts: List[str]
    ) -> List[str]:
        """
        Check for contradictions using symbolic rules.
        
        Args:
            claim_text: The claim being verified
            evidence_texts: List of evidence chunk texts
            
        Returns:
            List of detected violations
        """
        violations = []
        
        # Combine all evidence
        combined_evidence = " ".join(evidence_texts)
        
        # Check each rule
        for rule in self.rules:
            # Check if claim matches rule pattern
            claim_match = re.search(rule.pattern, claim_text, re.IGNORECASE)
            
            if claim_match:
                # Check if evidence contradicts
                evidence_match = re.search(rule.pattern, combined_evidence, re.IGNORECASE)
                
                if evidence_match and rule.rule_type == "contradiction":
                    # Check for negation patterns
                    claim_negated = self._is_negated(claim_text, claim_match)
                    evidence_negated = self._is_negated(combined_evidence, evidence_match)
                    
                    if claim_negated != evidence_negated:
                        violations.append(f"{rule.rule_name}: {rule.description}")
        
        return violations
    
    def _is_negated(self, text: str, match) -> bool:
        """Check if matched pattern is negated"""
        start = max(0, match.start() - 20)
        context = text[start:match.start()]
        negations = ["not", "never", "no", "cannot", "won't", "doesn't"]
        return any(neg in context.lower() for neg in negations)
    
    def check_temporal_consistency(
        self,
        claim_text: str,
        evidence_texts: List[str]
    ) -> List[str]:
        """Check temporal consistency"""
        violations = []
        
        # Extract temporal markers
        temporal_pattern = r"(before|after|then|later|earlier|when|while|during|age|year|old)"
        claim_temporals = re.findall(temporal_pattern, claim_text, re.IGNORECASE)
        
        if claim_temporals:
            combined_evidence = " ".join(evidence_texts)
            evidence_temporals = re.findall(temporal_pattern, combined_evidence, re.IGNORECASE)
            
            # Simple check: if claim has temporal markers but evidence doesn't align
            # This is a simplified check - can be enhanced
            if not evidence_temporals:
                violations.append("temporal_marker_missing: Evidence lacks temporal context")
        
        return violations


class LLMReasoningEngine:
    """
    LLM-based causal reasoning engine using LLaMA-3.1-8B.
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        use_quantization: bool = True,
        device: str = "cuda",
        hf_token: Optional[str] = None
    ):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        
        print(f"  Loading tokenizer for {model_name}...")
        start_time = time.time()
        # Load tokenizer
        tokenizer_kwargs = {}
        if self.hf_token:
            tokenizer_kwargs["token"] = self.hf_token
        tokenizer_kwargs["progress"] = True
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"  ✓ Tokenizer loaded in {time.time() - start_time:.2f}s")
        
        print(f"  Loading model (quantization={use_quantization}, device={self.device})...")
        start_time = time.time()
        # Load model (reuse from claim decomposer if available, or load separately)
        if use_quantization and self.device == "cuda":
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_kwargs = {
                "quantization_config": quantization_config,
                "device_map": "auto",
                "torch_dtype": torch.float16
            }
            if self.hf_token:
                model_kwargs["token"] = self.hf_token
            model_kwargs["progress"] = True
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        else:
            model_kwargs = {
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "device_map": "auto" if self.device == "cuda" else None
            }
            if self.hf_token:
                model_kwargs["token"] = self.hf_token
            model_kwargs["progress"] = True
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        
        self.model.eval()
        print(f"  ✓ Model loaded in {time.time() - start_time:.2f}s")
    
    def verify_claim(
        self,
        claim: "AtomicClaim",
        evidence_chunks: List[EvidenceChunk]
    ) -> Tuple[bool, float, str]:
        """
        Verify a claim against evidence using LLM reasoning.
        
        Args:
            claim: The claim to verify
            evidence_chunks: Retrieved evidence chunks
            
        Returns:
            Tuple of (is_satisfied, confidence, reasoning)
        """
        # Prepare evidence text
        evidence_text = "\n\n".join([
            f"[Evidence {i+1} (position: {chunk.position_ratio:.2f})]: {chunk.chunk_text}"
            for i, chunk in enumerate(evidence_chunks)
        ])
        
        prompt = self._create_verification_prompt(claim, evidence_text)
        
        # Generate reasoning
        reasoning_text = self._generate_reasoning(prompt)
        
        # Parse result
        is_satisfied, confidence = self._parse_verification_result(reasoning_text)
        
        return is_satisfied, confidence, reasoning_text
    
    def _create_verification_prompt(
        self,
        claim: "AtomicClaim",
        evidence_text: str
    ) -> str:
        """Create prompt for claim verification"""
        prompt = f"""You are verifying whether a narrative claim is satisfied by evidence from the full story.

Claim to verify:
Type: {claim.claim_type}
Claim: {claim.claim_text}
Entities: {', '.join(claim.entities) if claim.entities else 'None'}

Evidence from the narrative:
{evidence_text}

Determine:
1. Does the evidence support or contradict this claim?
2. Is there a causal/logical connection between the backstory claim and the narrative events?
3. What is your confidence level (0.0 to 1.0)?

Respond in JSON format:
{{
  "is_satisfied": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "Detailed explanation of why the claim is satisfied or violated",
  "contradiction_signals": ["list of specific contradictions if any"]
}}

Return only valid JSON:"""
        return prompt
    
    def _generate_reasoning(self, prompt: str) -> str:
        """Generate reasoning using LLM"""
        messages = [
            {"role": "system", "content": "You are a narrative coherence expert. Verify claims against evidence."},
            {"role": "user", "content": prompt}
        ]
        
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.device)
        
        # Show progress during generation
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.2,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text
    
    def _parse_verification_result(self, reasoning_text: str) -> Tuple[bool, float]:
        """Parse verification result from LLM output"""
        import json
        
        try:
            json_start = reasoning_text.find('{')
            json_end = reasoning_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = reasoning_text[json_start:json_end]
                data = json.loads(json_str)
                
                is_satisfied = data.get("is_satisfied", False)
                confidence = float(data.get("confidence", 0.5))
                
                return is_satisfied, confidence
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
        
        # Fallback: try to infer from text
        text_lower = reasoning_text.lower()
        is_satisfied = any(word in text_lower for word in ["satisfied", "supported", "consistent", "true"])
        if not is_satisfied:
            is_satisfied = not any(word in text_lower for word in ["contradict", "violate", "inconsistent", "false"])
        
        # Try to extract confidence number
        import re
        confidence_match = re.search(r"confidence[:\s]+([0-9.]+)", reasoning_text, re.IGNORECASE)
        confidence = float(confidence_match.group(1)) if confidence_match else 0.5
        
        return is_satisfied, confidence


class HybridReasoningLayer:
    """
    Main hybrid reasoning layer combining LLM and symbolic approaches.
    """
    
    def __init__(
        self,
        llm_model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        use_quantization: bool = True,
        device: str = "cuda",
        hf_token: Optional[str] = None
    ):
        self.symbolic_engine = SymbolicRuleEngine()
        self.llm_engine = LLMReasoningEngine(
            llm_model_name,
            use_quantization,
            device,
            hf_token
        )
    
    def verify_claim(
        self,
        claim: "AtomicClaim",
        evidence_chunks: List[EvidenceChunk],
        use_reranker: bool = False
    ) -> ClaimVerification:
        """
        Verify a claim using hybrid reasoning.
        
        Args:
            claim: The claim to verify
            evidence_chunks: Retrieved evidence chunks
            use_reranker: Whether to rerank evidence (optional)
            
        Returns:
            ClaimVerification object
        """
        # Rerank evidence if requested
        if use_reranker:
            evidence_chunks = self._rerank_evidence(claim, evidence_chunks)
        
        # Symbolic checks
        evidence_texts = [chunk.chunk_text for chunk in evidence_chunks]
        symbolic_violations = self.symbolic_engine.check_contradictions(
            claim.claim_text,
            evidence_texts
        )
        temporal_violations = self.symbolic_engine.check_temporal_consistency(
            claim.claim_text,
            evidence_texts
        )
        all_symbolic_violations = symbolic_violations + temporal_violations
        
        # LLM reasoning
        is_satisfied, confidence, reasoning = self.llm_engine.verify_claim(
            claim,
            evidence_chunks
        )
        
        # Extract contradiction signals from reasoning
        contradiction_signals = self._extract_contradiction_signals(reasoning)
        
        # Combine symbolic and LLM signals
        if all_symbolic_violations:
            # Symbolic violations reduce confidence
            confidence = max(0.0, confidence - len(all_symbolic_violations) * 0.1)
            if not is_satisfied:  # If LLM also says not satisfied, strengthen
                confidence = min(1.0, confidence + 0.2)
        
        # Final decision
        final_satisfied = is_satisfied and len(all_symbolic_violations) == 0
        
        return ClaimVerification(
            claim=claim,
            is_satisfied=final_satisfied,
            confidence=confidence,
            evidence_chunks=evidence_chunks,
            reasoning=reasoning,
            contradiction_signals=contradiction_signals,
            symbolic_violations=all_symbolic_violations
        )
    
    def _rerank_evidence(
        self,
        claim: "AtomicClaim",
        evidence_chunks: List[EvidenceChunk]
    ) -> List[EvidenceChunk]:
        """
        Rerank evidence chunks using cross-encoder (optional).
        For now, use simple position-based reranking.
        """
        # Weight by position (later = more important)
        weighted_chunks = []
        for chunk in evidence_chunks:
            weighted_score = chunk.relevance_score * (1.0 + chunk.position_ratio * 0.2)
            weighted_chunks.append((chunk, weighted_score))
        
        # Sort by weighted score
        weighted_chunks.sort(key=lambda x: x[1], reverse=True)
        
        return [chunk for chunk, _ in weighted_chunks]
    
    def _extract_contradiction_signals(self, reasoning: str) -> List[str]:
        """Extract contradiction signals from LLM reasoning"""
        signals = []
        
        # Look for contradiction keywords
        contradiction_keywords = [
            "contradict", "violate", "inconsistent", "conflict",
            "opposite", "against", "breaks", "disobeys"
        ]
        
        reasoning_lower = reasoning.lower()
        for keyword in contradiction_keywords:
            if keyword in reasoning_lower:
                # Extract sentence containing keyword
                sentences = reasoning.split('.')
                for sentence in sentences:
                    if keyword in sentence.lower():
                        signals.append(sentence.strip())
                        break
        
        return signals


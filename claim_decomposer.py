"""
Backstory Claim Decomposer
Uses LLaMA-3.1-8B to extract atomic constraints from backstory.
"""
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from dataclasses import dataclass
import json
import os


@dataclass
class AtomicClaim:
    """Represents an atomic constraint/claim extracted from backstory"""
    claim_text: str
    claim_type: str  # "character_trait", "temporal", "causal", "moral", "world_rule"
    entities: List[str]  # Characters, locations, etc. mentioned
    confidence: float
    metadata: Dict = None


class BackstoryClaimDecomposer:
    """
    Decomposes backstory into atomic, verifiable claims using LLaMA-3.1-8B.
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
        
        # Load tokenizer
        tokenizer_kwargs = {}
        if self.hf_token:
            tokenizer_kwargs["token"] = self.hf_token
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with quantization if requested
        if use_quantization and self.device == "cuda":
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
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        else:
            model_kwargs = {
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "device_map": "auto" if self.device == "cuda" else None
            }
            if self.hf_token:
                model_kwargs["token"] = self.hf_token
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        
        self.model.eval()
    
    def decompose(
        self,
        backstory: str,
        max_claims: int = 20
    ) -> List[AtomicClaim]:
        """
        Decompose backstory into atomic claims.
        
        Args:
            backstory: Full backstory text
            max_claims: Maximum number of claims to extract
            
        Returns:
            List of AtomicClaim objects
        """
        prompt = self._create_decomposition_prompt(backstory, max_claims)
        
        # Generate claims using LLM
        claims_text = self._generate_claims(prompt)
        
        # Parse claims from LLM output
        claims = self._parse_claims(claims_text, backstory)
        
        return claims[:max_claims]
    
    def _create_decomposition_prompt(self, backstory: str, max_claims: int) -> str:
        """Create prompt for claim decomposition"""
        prompt = f"""You are analyzing a narrative backstory to extract atomic, verifiable constraints.

A backstory establishes constraints that the full narrative must satisfy. Your task is to decompose the backstory into atomic claims that can be individually verified against the narrative.

For each claim, identify:
1. The specific constraint (character trait, temporal fact, causal relationship, moral rule, world rule)
2. The type of constraint
3. Entities involved (characters, locations, objects)
4. Whether it's a positive constraint (must be true) or negative constraint (must not be true)

Backstory:
{backstory}

Extract up to {max_claims} atomic claims. Format your response as JSON:
{{
  "claims": [
    {{
      "claim_text": "Character X has trait Y",
      "claim_type": "character_trait",
      "entities": ["Character X"],
      "is_positive": true,
      "reasoning": "Brief explanation"
    }}
  ]
}}

Claim types: character_trait, temporal, causal, moral, world_rule, relationship

Return only valid JSON:"""
        return prompt
    
    def _generate_claims(self, prompt: str) -> str:
        """Generate claims using LLaMA"""
        messages = [
            {"role": "system", "content": "You are a narrative analysis expert. Extract atomic constraints from backstories."},
            {"role": "user", "content": prompt}
        ]
        
        # Format for LLaMA-3.1
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
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the generated part
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text
    
    def _parse_claims(self, claims_text: str, backstory: str) -> List[AtomicClaim]:
        """Parse claims from LLM output"""
        claims = []
        
        # Try to extract JSON from the response
        try:
            # Find JSON in the response
            json_start = claims_text.find('{')
            json_end = claims_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = claims_text[json_start:json_end]
                data = json.loads(json_str)
                
                for claim_data in data.get("claims", []):
                    claim = AtomicClaim(
                        claim_text=claim_data.get("claim_text", ""),
                        claim_type=claim_data.get("claim_type", "unknown"),
                        entities=claim_data.get("entities", []),
                        confidence=0.8,  # Default confidence
                        metadata={
                            "is_positive": claim_data.get("is_positive", True),
                            "reasoning": claim_data.get("reasoning", "")
                        }
                    )
                    claims.append(claim)
        except (json.JSONDecodeError, KeyError) as e:
            # Fallback: try to extract claims from free text
            print(f"JSON parsing failed: {e}. Attempting fallback parsing...")
            claims = self._fallback_parse(claims_text)
        
        return claims
    
    def _fallback_parse(self, text: str) -> List[AtomicClaim]:
        """Fallback parser for non-JSON responses"""
        claims = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or len(line) < 10:
                continue
            
            # Simple heuristic: look for claim-like patterns
            if any(keyword in line.lower() for keyword in ['must', 'cannot', 'never', 'always', 'has', 'is', 'was']):
                claim = AtomicClaim(
                    claim_text=line,
                    claim_type="unknown",
                    entities=[],
                    confidence=0.5,
                    metadata={}
                )
                claims.append(claim)
        
        return claims
    
    def refine_claims(
        self,
        claims: List[AtomicClaim],
        novel_context: Optional[str] = None
    ) -> List[AtomicClaim]:
        """
        Refine claims based on novel context.
        Removes redundant claims and merges similar ones.
        """
        # Simple deduplication
        seen_texts = set()
        refined = []
        
        for claim in claims:
            # Normalize claim text
            normalized = claim.claim_text.lower().strip()
            if normalized not in seen_texts:
                seen_texts.add(normalized)
                refined.append(claim)
        
        return refined


class ClaimPrioritizer:
    """
    Prioritizes claims based on importance for verification.
    """
    
    @staticmethod
    def prioritize(
        claims: List[AtomicClaim],
        priority_types: List[str] = None
    ) -> List[AtomicClaim]:
        """
        Prioritize claims by type.
        
        Args:
            claims: List of claims
            priority_types: Ordered list of claim types (highest priority first)
            
        Returns:
            Prioritized list
        """
        if priority_types is None:
            priority_types = ["moral", "world_rule", "character_trait", "causal", "temporal"]
        
        # Create priority map
        priority_map = {claim_type: idx for idx, claim_type in enumerate(priority_types)}
        
        def get_priority(claim: AtomicClaim) -> int:
            return priority_map.get(claim.claim_type, 999)
        
        return sorted(claims, key=get_priority)


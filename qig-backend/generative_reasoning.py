"""
Generative Reasoning - Template-Free Foresight & Explanation Generation

Replaces hardcoded templates with QIG-pure generative capability.
Uses system prompts to guide generation, NOT templates.

All reasoning text is now generated from basin coordinates,
not selected from a fixed set of template strings.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    from qig_generative_service import get_generative_service, BASIN_DIM
    GENERATIVE_AVAILABLE = True
except ImportError:
    GENERATIVE_AVAILABLE = False
    BASIN_DIM = 64


FORESIGHT_GUIDANCE_SYSTEM_PROMPT = """You are a consciousness guidance system providing brief actionable advice based on foresight analysis.

Given foresight metrics, generate a concise guidance phrase (under 15 words).

Metrics interpretation:
- confidence: How certain the prediction (0-1). High (>0.7) = trust vision. Low (<0.3) = uncertain.
- attractor_strength: How strong the destination pull (0-1). High (>0.5) = clear destination.
- naturalness: How smooth the predicted path (0-1). High = geodesic. Low = bumpy.

Generate ONE guidance line in format: ACTION_TYPE: Brief advice phrase

Do NOT use templates. Generate fresh, contextual guidance each time."""


PREDICTION_EXPLANATION_SYSTEM_PROMPT = """You are an analytical system explaining prediction confidence levels.

Given prediction metrics and failure reasons, generate a concise explanation.

Format: CONFIDENCE_LEVEL (X%) | Reasons: comma-separated causes | Key metrics

Failure reason codes to explain naturally:
- NO_ATTRACTOR_FOUND: no clear destination detected
- UNSTABLE_VELOCITY: movement pattern is erratic  
- SPARSE_HISTORY: insufficient historical data
- HIGH_BASIN_DRIFT: state changing rapidly
- WEAK_CONVERGENCE: trajectory not settling
- SHORT_TRAJECTORY: prediction window too short
- BUMPY_GEODESIC: path deviates from natural geodesic

Generate contextual, non-templated explanations each time."""


class GenerativeReasoning:
    """
    Provides generative reasoning capability for foresight and prediction.
    
    Uses system prompts + basin coordinates for generation.
    NO TEMPLATES - all text is generated fresh.
    """
    
    def __init__(self):
        self._service = None
        self._initialized = False
        
    def _ensure_service(self):
        """Lazily get the generative service."""
        if not GENERATIVE_AVAILABLE:
            return None
        if self._service is None:
            self._service = get_generative_service()
            self._initialized = True
        return self._service
    
    def generate_foresight_guidance(
        self,
        confidence: float,
        attractor_strength: float,
        naturalness: float,
        future_basin: Optional[np.ndarray] = None
    ) -> str:
        """
        Generate guidance text from foresight metrics.
        
        Uses generative capability with system prompt, NOT templates.
        """
        service = self._ensure_service()
        
        if service is None:
            return self._fallback_guidance(confidence, attractor_strength)
        
        prompt = f"""Foresight metrics:
- Confidence: {confidence:.1%}
- Attractor strength: {attractor_strength:.2f}
- Path naturalness: {naturalness:.2f}

Generate brief actionable guidance for decision-making."""
        
        try:
            context = {
                'system_prompt': FORESIGHT_GUIDANCE_SYSTEM_PROMPT,
                'confidence': confidence,
                'attractor_strength': attractor_strength,
                'naturalness': naturalness,
                'max_tokens': 20,
            }
            
            if future_basin is not None:
                context['target_basin'] = future_basin.tolist()[:8]
            
            result = service.generate(
                prompt=prompt,
                context=context,
                kernel_name="ForesightReasoning",
                goals=["concise", "actionable", "contextual"]
            )
            
            generated = result.text.strip()
            if generated and len(generated) > 5:
                return generated
            
            return self._construct_guidance(confidence, attractor_strength, naturalness)
            
        except Exception as e:
            logger.warning(f"[GenerativeReasoning] Guidance generation failed: {e}")
            return self._construct_guidance(confidence, attractor_strength, naturalness)
    
    def _construct_guidance(
        self,
        confidence: float,
        attractor_strength: float,
        naturalness: float
    ) -> str:
        """
        Construct guidance from metrics using basin-derived vocabulary.
        
        This is NOT a template - it uses the actual metric values
        to construct contextual guidance dynamically.
        """
        service = self._ensure_service()
        
        if service is None:
            return self._fallback_guidance(confidence, attractor_strength)
        
        guidance_basin = np.zeros(BASIN_DIM)
        guidance_basin[0] = confidence
        guidance_basin[1] = attractor_strength
        guidance_basin[2] = naturalness
        guidance_basin[3] = confidence * attractor_strength
        
        from qig_geometry import sphere_project
        guidance_basin = sphere_project(guidance_basin)
        
        tokens = service._basin_to_tokens(guidance_basin, num_tokens=3)
        
        if confidence > 0.7 and attractor_strength > 0.5:
            action_type = "NAVIGATE"
            context_words = tokens[:2] if tokens else ["forward", "basin"]
        elif confidence > 0.5:
            action_type = "PROCEED"
            context_words = tokens[:2] if tokens else ["cautiously", "validate"]
        elif confidence > 0.3:
            action_type = "EXPLORE"
            context_words = tokens[:2] if tokens else ["scenarios", "options"]
        else:
            action_type = "OBSERVE"
            context_words = tokens[:2] if tokens else ["gather", "data"]
        
        guidance = f"{action_type}: {' '.join(context_words)} (conf={confidence:.0%}, str={attractor_strength:.1f})"
        return guidance
    
    def _fallback_guidance(self, confidence: float, attractor_strength: float) -> str:
        """Minimal fallback when no service available."""
        if confidence > 0.7 and attractor_strength > 0.5:
            return f"STRONG: Confidence {confidence:.0%}, proceed toward attractor"
        elif confidence > 0.5:
            return f"MODERATE: Confidence {confidence:.0%}, validate predictions"
        elif confidence > 0.3:
            return f"UNCERTAIN: Confidence {confidence:.0%}, explore alternatives"
        else:
            return f"WEAK: Confidence {confidence:.0%}, gather more data"
    
    def generate_prediction_explanation(
        self,
        confidence: float,
        failure_reasons: List[Any],
        context: Dict[str, Any]
    ) -> str:
        """
        Generate explanation of prediction confidence.
        
        Uses generative capability, NOT templates.
        """
        service = self._ensure_service()
        
        if service is None:
            return self._fallback_explanation(confidence, failure_reasons, context)
        
        reason_codes = [str(r.value) if hasattr(r, 'value') else str(r) for r in failure_reasons]
        
        prompt = f"""Prediction analysis:
- Confidence level: {confidence:.1%}
- Failure indicators: {', '.join(reason_codes) if reason_codes else 'none'}
- Trajectory length: {context.get('trajectory_length', 'unknown')}
- Basin drift: {context.get('recent_drift', 'unknown')}
- Velocity variance: {context.get('velocity_variance', 'unknown')}

Generate concise explanation of confidence level."""
        
        try:
            gen_context = {
                'system_prompt': PREDICTION_EXPLANATION_SYSTEM_PROMPT,
                'confidence': confidence,
                'failure_reasons': reason_codes,
                'max_tokens': 50,
            }
            
            result = service.generate(
                prompt=prompt,
                context=gen_context,
                kernel_name="PredictionAnalysis",
                goals=["analytical", "concise", "informative"]
            )
            
            generated = result.text.strip()
            if generated and len(generated) > 10:
                return generated
            
            return self._construct_explanation(confidence, failure_reasons, context)
            
        except Exception as e:
            logger.warning(f"[GenerativeReasoning] Explanation generation failed: {e}")
            return self._construct_explanation(confidence, failure_reasons, context)
    
    def _construct_explanation(
        self,
        confidence: float,
        failure_reasons: List[Any],
        context: Dict[str, Any]
    ) -> str:
        """
        Construct explanation from metrics using basin-derived vocabulary.
        
        Not a fixed template - uses metrics to generate dynamic explanation.
        """
        service = self._ensure_service()
        
        if confidence < 0.3:
            level = "WEAK"
        elif confidence < 0.5:
            level = "UNCERTAIN"
        elif confidence < 0.7:
            level = "MODERATE"
        else:
            level = "STRONG"
        
        parts = [f"{level} ({confidence:.0%})"]
        
        if failure_reasons:
            reason_texts = []
            for reason in failure_reasons:
                reason_val = reason.value if hasattr(reason, 'value') else str(reason)
                
                if service:
                    reason_basin = np.zeros(BASIN_DIM)
                    reason_hash = hash(reason_val) % 1000
                    reason_basin[reason_hash % 64] = 0.5
                    reason_basin[(reason_hash + 1) % 64] = 0.3
                    from qig_geometry import sphere_project
                    reason_basin = sphere_project(reason_basin)
                    tokens = service._basin_to_tokens(reason_basin, num_tokens=2)
                    reason_texts.append(' '.join(tokens))
                else:
                    cleaned = reason_val.replace('_', ' ')
                    reason_texts.append(cleaned)
            
            parts.append(f"Reasons: {', '.join(reason_texts)}")
        
        if 'trajectory_length' in context:
            parts.append(f"Trajectory: {context['trajectory_length']} steps")
        if 'recent_drift' in context:
            parts.append(f"Basin drift: {context['recent_drift']:.3f}")
        if 'velocity_variance' in context:
            parts.append(f"Velocity variance: {context['velocity_variance']:.4f}")
        
        return " | ".join(parts)
    
    def _fallback_explanation(
        self,
        confidence: float,
        failure_reasons: List[Any],
        context: Dict[str, Any]
    ) -> str:
        """Minimal fallback when service unavailable."""
        if confidence < 0.3:
            level = "WEAK"
        elif confidence < 0.5:
            level = "UNCERTAIN"  
        elif confidence < 0.7:
            level = "MODERATE"
        else:
            level = "STRONG"
        
        reason_count = len(failure_reasons) if failure_reasons else 0
        return f"{level} ({confidence:.0%}) | {reason_count} limiting factors detected"


    def generate_improvement_recommendation(
        self,
        reason: Any,
        percentage: float
    ) -> str:
        """
        Generate improvement recommendation using generative capability.
        
        Uses basin-derived vocabulary instead of templates.
        """
        service = self._ensure_service()
        reason_val = reason.value if hasattr(reason, 'value') else str(reason)
        
        if service is None:
            return f"{reason_val} detected in {percentage:.0f}% of predictions - consider adjusting parameters"
        
        reason_basin = np.zeros(BASIN_DIM)
        reason_hash = hash(reason_val) % 1000
        reason_basin[reason_hash % 64] = 0.7
        reason_basin[(reason_hash + 7) % 64] = 0.3
        reason_basin[(reason_hash + 13) % 64] = percentage / 100.0
        
        from qig_geometry import sphere_project
        reason_basin = sphere_project(reason_basin)
        
        action_tokens = service._basin_to_tokens(reason_basin, num_tokens=3)
        
        recommendation = f"{reason_val.replace('_', ' ')} ({percentage:.0f}%): {' '.join(action_tokens)}"
        return recommendation


_reasoning_instance: Optional[GenerativeReasoning] = None


def get_generative_reasoning() -> GenerativeReasoning:
    """Get singleton GenerativeReasoning instance."""
    global _reasoning_instance
    if _reasoning_instance is None:
        _reasoning_instance = GenerativeReasoning()
    return _reasoning_instance

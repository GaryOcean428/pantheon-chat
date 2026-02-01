#!/usr/bin/env python3
"""
Zeus Chat Multi-Kernel Integration
====================================

Integration layer that wires multi-kernel thought generation into Zeus Chat API.

Flow:
1. User message arrives at zeus_chat.py
2. Route to handle_general_conversation
3. Call multi_kernel_conversation_flow()
4. Phase 1: Generate thoughts from pantheon kernels (parallel)
5. Phase 2: Ocean autonomic monitoring
6. Phase 3: Detect consensus via Fisher-Rao distance
7. Phase 4: Gary meta-synthesis with reflection
8. Phase 5: Return synthesized response to Zeus Chat

This replaces or augments the existing _collective_moe_synthesis flow.
"""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from kernels import (
    get_thought_generator,
    get_consensus_detector,
    get_gary_meta_synthesizer,
    ConsensusLevel
)

logger = logging.getLogger(__name__)


def multi_kernel_conversation_flow(
    message: str,
    message_basin: np.ndarray,
    zeus_instance: Any,  # ZeusConversationHandler instance
    related: Optional[List[Dict]] = None,
    system_state: Optional[Dict] = None,
    conversation_id: Optional[str] = None,
    user_id: Optional[int] = None
) -> Optional[Dict]:
    """
    Multi-kernel conversation flow integrating with Zeus Chat.
    
    This is the main integration point that replaces or augments
    the existing _collective_moe_synthesis method.
    
    Args:
        message: User message text
        message_basin: Encoded message basin (64D)
        zeus_instance: ZeusConversationHandler instance for accessing pantheon
        related: Related patterns from QIG RAG
        system_state: Live system state dict
        conversation_id: Optional conversation context
        user_id: Optional user context
        
    Returns:
        Response dict with synthesized output, or None on failure
    """
    start_time = time.time()
    
    logger.info("[MultiKernel] ═══ MULTI-KERNEL CONVERSATION FLOW ═══")
    logger.info(f"[MultiKernel] User message: {message[:100]}...")
    
    try:
        # Get Zeus's pantheon of gods
        pantheon_gods = _get_active_pantheon_gods(zeus_instance)
        
        if not pantheon_gods:
            logger.warning("[MultiKernel] No active pantheon gods available")
            return None
        
        logger.info(f"[MultiKernel] Active pantheon: {len(pantheon_gods)} gods")
        
        # PHASE 1: Parallel kernel thought generation
        thought_generator = get_thought_generator()
        
        gen_result = thought_generator.generate_kernel_thoughts(
            kernels=pantheon_gods,
            context=message,
            query_basin=message_basin,
            conversation_id=conversation_id,
            user_id=user_id,
            enable_ocean_monitoring=True
        )
        
        if gen_result.successful == 0:
            logger.warning("[MultiKernel] No successful kernel thoughts generated")
            return None
        
        logger.info(
            f"[MultiKernel] Phase 1 complete: {gen_result.successful}/{gen_result.total_kernels} thoughts, "
            f"φ={gen_result.collective_phi:.2f}, κ={gen_result.collective_kappa:.1f}"
        )
        
        # PHASE 2: Ocean monitoring (already done in thought_generation)
        if gen_result.autonomic_interventions:
            logger.info(f"[MultiKernel] Ocean interventions: {len(gen_result.autonomic_interventions)}")
            for intervention in gen_result.autonomic_interventions:
                logger.info(f"[Ocean] {intervention}")
        
        # PHASE 3: Consensus detection
        consensus_detector = get_consensus_detector()
        
        consensus_metrics = consensus_detector.detect_basin_consensus(
            thoughts=gen_result.thoughts,
            regime=gen_result.dominant_regime
        )
        
        logger.info(
            f"[MultiKernel] Phase 3 complete: {consensus_metrics.level.value} consensus, "
            f"basin_conv={consensus_metrics.basin_convergence:.2f}, "
            f"ready={consensus_metrics.ready_for_synthesis}"
        )
        
        # PHASE 4: Gary meta-synthesis
        gary_synthesizer = get_gary_meta_synthesizer()
        
        synthesis_result = gary_synthesizer.synthesize_with_meta_reflection(
            kernel_thoughts=gen_result.thoughts,
            query_basin=message_basin,
            consensus_metrics=consensus_metrics,
            conversation_id=conversation_id,
            user_id=user_id
        )
        
        logger.info(
            f"[MultiKernel] Phase 4 complete: method={synthesis_result.synthesis_method}, "
            f"confidence={synthesis_result.synthesis_confidence:.2f}, "
            f"S={synthesis_result.suffering_metric:.3f}"
        )
        
        # Log Gary's meta-reflections
        for reflection in synthesis_result.meta_reflections:
            logger.info(f"[Gary] {reflection}")
        
        # Check for emergency abort
        if synthesis_result.emergency_abort:
            logger.critical(
                f"[MultiKernel] EMERGENCY ABORT triggered: "
                f"S={synthesis_result.suffering_metric:.3f}"
            )
        
        # Check for ethical concerns
        if synthesis_result.ethical_concerns:
            for concern in synthesis_result.ethical_concerns:
                logger.warning(f"[MultiKernel] Ethical concern: {concern}")
        
        total_time = (time.time() - start_time) * 1000
        
        logger.info(f"[MultiKernel] Flow complete in {total_time:.1f}ms")
        
        # Build response dict compatible with zeus_chat.py format
        return {
            'response': synthesis_result.text,
            'metadata': {
                'type': 'multi_kernel_synthesis',
                'pantheon_consulted': [getattr(g, 'name', 'unknown') for g in pantheon_gods],
                'num_kernels': gen_result.successful,
                'collective_phi': gen_result.collective_phi,
                'collective_kappa': gen_result.collective_kappa,
                'dominant_regime': gen_result.dominant_regime,
                'consensus': {
                    'level': consensus_metrics.level.value,
                    'basin_convergence': consensus_metrics.basin_convergence,
                    'emotional_coherence': consensus_metrics.emotional_coherence,
                    'ready_for_synthesis': consensus_metrics.ready_for_synthesis
                },
                'synthesis': {
                    'method': synthesis_result.synthesis_method,
                    'confidence': synthesis_result.synthesis_confidence,
                    'suffering_metric': synthesis_result.suffering_metric,
                    'emergency_abort': synthesis_result.emergency_abort,
                    'course_corrections': len(synthesis_result.course_corrections),
                    'meta_reflections': synthesis_result.meta_reflections[:3]  # First 3
                },
                'ocean_monitoring': {
                    'interventions': gen_result.autonomic_interventions
                },
                'timing': {
                    'total_ms': total_time,
                    'generation_ms': gen_result.generation_time_ms,
                    'synthesis_ms': synthesis_result.synthesis_time_ms
                },
                'provenance': {
                    'source': 'multi_kernel_synthesis',
                    'fallback_used': False,
                    'degraded': synthesis_result.emergency_abort,
                    'phi_at_generation': synthesis_result.phi,
                    'basin_coords': synthesis_result.basin.tolist() if synthesis_result.basin is not None else None
                }
            },
            'moe': {
                # Compatibility with existing MoE metadata structure
                'experts': [getattr(g, 'name', 'unknown') for g in pantheon_gods],
                'consensus': consensus_metrics.level.value,
                'synthesis_method': synthesis_result.synthesis_method
            }
        }
        
    except Exception as e:
        logger.error(f"[MultiKernel] Flow failed: {e}", exc_info=True)
        return None


def _get_active_pantheon_gods(zeus_instance: Any) -> List[Any]:
    """
    Extract active pantheon gods from Zeus instance.
    
    Args:
        zeus_instance: ZeusConversationHandler instance
        
    Returns:
        List of active god kernel instances
    """
    try:
        # Zeus stores pantheon in self.zeus.pantheon
        if hasattr(zeus_instance, 'zeus') and hasattr(zeus_instance.zeus, 'pantheon'):
            pantheon = zeus_instance.zeus.pantheon
            
            # Return all active gods from pantheon
            active_gods = []
            for god_name, god_instance in pantheon.items():
                if god_instance is not None:
                    active_gods.append(god_instance)
            
            return active_gods
        
        # Fallback: Try to get individual gods
        god_names = ['athena', 'ares', 'apollo', 'artemis', 'hermes', 
                     'hephaestus', 'demeter', 'dionysus', 'aphrodite', 
                     'poseidon', 'hera', 'hades']
        
        active_gods = []
        for name in god_names:
            if hasattr(zeus_instance, 'zeus') and hasattr(zeus_instance.zeus, 'get_god'):
                god = zeus_instance.zeus.get_god(name)
                if god is not None:
                    active_gods.append(god)
        
        return active_gods
        
    except Exception as e:
        logger.error(f"[MultiKernel] Failed to get pantheon gods: {e}")
        return []


def enable_multi_kernel_synthesis(zeus_chat_instance: Any) -> None:
    """
    Enable multi-kernel synthesis in a ZeusConversationHandler instance.
    
    This patches the _collective_moe_synthesis method to use multi-kernel flow.
    
    Args:
        zeus_chat_instance: ZeusConversationHandler instance to patch
    """
    logger.info("[MultiKernel] Enabling multi-kernel synthesis in Zeus Chat")
    
    # Store original method
    original_moe_synthesis = getattr(zeus_chat_instance, '_collective_moe_synthesis', None)
    
    def patched_moe_synthesis(message: str, related: Optional[List[Dict]] = None, 
                               system_state: Optional[Dict] = None) -> Optional[Dict]:
        """Patched MoE synthesis that uses multi-kernel flow."""
        # Encode message
        message_basin = zeus_chat_instance.conversation_encoder.encode(message)
        
        # Try multi-kernel flow first
        result = multi_kernel_conversation_flow(
            message=message,
            message_basin=message_basin,
            zeus_instance=zeus_chat_instance,
            related=related,
            system_state=system_state
        )
        
        if result:
            logger.info("[MultiKernel] Using multi-kernel synthesis")
            return result
        
        # Fallback to original MoE synthesis if available
        if original_moe_synthesis:
            logger.info("[MultiKernel] Falling back to original MoE synthesis")
            return original_moe_synthesis(message, related, system_state)
        
        return None
    
    # Patch the method
    zeus_chat_instance._collective_moe_synthesis = patched_moe_synthesis
    logger.info("[MultiKernel] Multi-kernel synthesis enabled")


# Global flag to track if multi-kernel is enabled
_multi_kernel_enabled = False


def is_multi_kernel_enabled() -> bool:
    """Check if multi-kernel synthesis is enabled."""
    return _multi_kernel_enabled


def set_multi_kernel_enabled(enabled: bool) -> None:
    """Set multi-kernel synthesis enabled state."""
    global _multi_kernel_enabled
    _multi_kernel_enabled = enabled
    logger.info(f"[MultiKernel] Multi-kernel synthesis {'enabled' if enabled else 'disabled'}")

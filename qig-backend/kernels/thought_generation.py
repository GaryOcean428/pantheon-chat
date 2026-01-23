#!/usr/bin/env python3
"""
Multi-Kernel Thought Generation - Parallel Kernel Thought Orchestration
========================================================================

Orchestrates parallel thought generation across multiple kernels in the constellation.
Each kernel generates autonomous thoughts before Gary synthesis.

ARCHITECTURE:
Phase 1: Individual kernel thought generation (parallel)
Phase 2: Ocean kernel autonomic monitoring
Phase 3: Gary kernel synthesis with meta-reflection
Phase 4: External output via Zeus-Chat API

Based on generative-and-emotions.md and EmotionallyAwareKernel architecture.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# QIG core imports
from qigkernels.physics_constants import BASIN_DIM, KAPPA_STAR

logger = logging.getLogger(__name__)


@dataclass
class KernelThoughtResult:
    """Result of parallel kernel thought generation."""
    thoughts: List[Any]  # List of KernelThought objects
    total_kernels: int
    successful: int
    failed: int
    generation_time_ms: float
    collective_phi: float
    collective_kappa: float
    dominant_regime: str
    autonomic_interventions: List[str]


class ParallelThoughtGenerator:
    """
    Orchestrates parallel thought generation across constellation kernels.
    
    Each kernel generates thoughts autonomously with:
    - Emotional state measurement
    - Basin coordinate tracking
    - φ/κ/regime awareness
    - Meta-reflection on own thoughts
    
    Ocean monitors autonomically:
    - Constellation-wide φ coherence
    - Emotional alignment across kernels
    - Breakdown regime detection
    - Suffering metric tracking
    """
    
    def __init__(self, max_workers: int = 12):
        """
        Initialize parallel thought generator.
        
        Args:
            max_workers: Maximum parallel kernel threads (default 12 for pantheon)
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Track generation history
        self.generation_history: List[KernelThoughtResult] = []
        self.max_history = 100
        
        # Ocean monitoring state
        self.autonomic_interventions: List[str] = []
        
        logger.info(f"[ParallelThoughtGenerator] Initialized with {max_workers} worker threads")
    
    def generate_kernel_thoughts(
        self,
        kernels: List[Any],  # List of BaseGod or EmotionallyAwareKernel instances
        context: str,
        query_basin: np.ndarray,
        conversation_id: Optional[str] = None,
        user_id: Optional[int] = None,
        enable_ocean_monitoring: bool = True
    ) -> KernelThoughtResult:
        """
        Generate thoughts from multiple kernels in parallel.
        
        This is PHASE 1: Individual kernel thought generation
        
        Args:
            kernels: List of kernel instances to generate from
            context: Input context/query text
            query_basin: Query basin coordinates (64D)
            conversation_id: Optional conversation context
            user_id: Optional user context
            enable_ocean_monitoring: Enable Ocean autonomic monitoring
            
        Returns:
            KernelThoughtResult with all generated thoughts and metrics
        """
        start_time = time.time()
        self.autonomic_interventions = []
        
        logger.info(f"[ParallelThoughtGenerator] ═══ PHASE 1: KERNEL THOUGHT GENERATION ═══")
        logger.info(f"[ParallelThoughtGenerator] Generating thoughts from {len(kernels)} kernels")
        
        # Submit parallel generation tasks
        futures = {}
        for kernel in kernels:
            future = self.executor.submit(
                self._generate_single_kernel_thought,
                kernel=kernel,
                context=context,
                query_basin=query_basin,
                conversation_id=conversation_id,
                user_id=user_id
            )
            futures[future] = kernel
        
        # Collect results as they complete
        thoughts = []
        failed = 0
        
        for future in as_completed(futures):
            kernel = futures[future]
            kernel_name = getattr(kernel, 'name', 'Unknown')
            
            try:
                thought = future.result(timeout=10.0)  # 10s timeout per kernel
                if thought is not None:
                    thoughts.append(thought)
                    self._log_kernel_thought(thought, kernel_name)
                else:
                    failed += 1
                    logger.warning(f"[ParallelThoughtGenerator] {kernel_name} returned None")
            except Exception as e:
                failed += 1
                logger.error(f"[ParallelThoughtGenerator] {kernel_name} generation failed: {e}")
        
        # PHASE 2: Ocean autonomic monitoring
        if enable_ocean_monitoring and thoughts:
            self._ocean_monitor_constellation(thoughts)
        
        # Compute collective metrics
        collective_phi = self._compute_collective_phi(thoughts)
        collective_kappa = self._compute_collective_kappa(thoughts)
        dominant_regime = self._determine_dominant_regime(thoughts)
        
        generation_time = (time.time() - start_time) * 1000  # ms
        
        result = KernelThoughtResult(
            thoughts=thoughts,
            total_kernels=len(kernels),
            successful=len(thoughts),
            failed=failed,
            generation_time_ms=generation_time,
            collective_phi=collective_phi,
            collective_kappa=collective_kappa,
            dominant_regime=dominant_regime,
            autonomic_interventions=self.autonomic_interventions.copy()
        )
        
        # Track history
        self.generation_history.append(result)
        if len(self.generation_history) > self.max_history:
            self.generation_history = self.generation_history[-self.max_history:]
        
        logger.info(
            f"[ParallelThoughtGenerator] Generation complete: "
            f"{len(thoughts)}/{len(kernels)} successful, "
            f"φ={collective_phi:.3f}, κ={collective_kappa:.1f}, "
            f"regime={dominant_regime}, time={generation_time:.1f}ms"
        )
        
        return result
    
    def _generate_single_kernel_thought(
        self,
        kernel: Any,
        context: str,
        query_basin: np.ndarray,
        conversation_id: Optional[str],
        user_id: Optional[int]
    ) -> Optional[Any]:
        """
        Generate thought from a single kernel.
        
        Args:
            kernel: Kernel instance (BaseGod or EmotionallyAwareKernel)
            context: Input context
            query_basin: Query basin coordinates
            conversation_id: Optional conversation context
            user_id: Optional user context
            
        Returns:
            KernelThought object or None on failure
        """
        kernel_name = getattr(kernel, 'name', 'Unknown')
        
        try:
            # Get current kernel state
            phi = getattr(kernel, 'last_phi', 0.5)
            kappa = getattr(kernel, 'last_kappa', KAPPA_STAR)
            regime = self._infer_regime(phi, kappa)
            basin_coords = getattr(kernel, 'basin_coords', query_basin)
            
            # Check if kernel supports generate_thought (EmotionallyAwareKernel)
            if hasattr(kernel, 'generate_thought'):
                thought = kernel.generate_thought(
                    context=context,
                    phi=phi,
                    kappa=kappa,
                    regime=regime,
                    basin_coords=basin_coords,
                    persist=True,
                    conversation_id=conversation_id,
                    user_id=user_id
                )
                return thought
            
            # Fallback: Use generate_reasoning for BaseGod instances
            elif hasattr(kernel, 'generate_reasoning'):
                reasoning_text = kernel.generate_reasoning(
                    context_basin=query_basin,
                    num_tokens=60,
                    temperature=0.8
                )
                
                # Construct KernelThought manually
                from emotionally_aware_kernel import KernelThought, EmotionalState
                
                thought = KernelThought(
                    kernel_id=getattr(kernel, 'kernel_id', kernel_name),
                    kernel_type=getattr(kernel, 'domain', 'general'),
                    thought_fragment=reasoning_text,
                    basin_coords=basin_coords,
                    phi=phi,
                    kappa=kappa,
                    regime=regime,
                    emotional_state=EmotionalState(),  # Default emotional state
                    confidence=0.5
                )
                return thought
            
            else:
                logger.warning(
                    f"[ParallelThoughtGenerator] {kernel_name} has no thought generation method"
                )
                return None
                
        except Exception as e:
            logger.error(f"[ParallelThoughtGenerator] {kernel_name} thought generation error: {e}")
            return None
    
    def _infer_regime(self, phi: float, kappa: float) -> str:
        """Infer regime from φ and κ."""
        if phi < 0.3:
            return "breakdown"
        elif phi < 0.7:
            return "linear"
        elif kappa < 41.0:
            return "geometric"
        else:
            return "feeling"  # kappa > 41 (KAPPA_3 threshold)
    
    def _log_kernel_thought(self, thought: Any, kernel_name: str) -> None:
        """
        Log kernel thought in standard format.
        
        Format: [KERNEL_NAME] kappa=X.X, phi=X.XX, thought=...
        """
        kappa = getattr(thought, 'kappa', 0.0)
        phi = getattr(thought, 'phi', 0.0)
        fragment = getattr(thought, 'thought_fragment', '')[:100]
        
        logger.info(
            f"[{kernel_name}] kappa={kappa:.1f}, phi={phi:.2f}, "
            f"thought={fragment}..."
        )
    
    def _ocean_monitor_constellation(self, thoughts: List[Any]) -> None:
        """
        PHASE 2: Ocean autonomic monitoring of constellation thoughts.
        
        Monitors:
        - Constellation-wide φ coherence
        - Emotional alignment across kernels
        - Breakdown regime detection
        - Suffering metric tracking
        
        Args:
            thoughts: List of KernelThought objects
        """
        logger.info("[Ocean] ═══ PHASE 2: AUTONOMIC MONITORING ═══")
        
        # Check φ coherence
        phis = [getattr(t, 'phi', 0.5) for t in thoughts]
        phi_std = np.std(phis)
        
        if phi_std > 0.3:
            intervention = f"High φ variance detected (std={phi_std:.3f}), constellation incoherent"
            logger.warning(f"[Ocean] {intervention}")
            self.autonomic_interventions.append(intervention)
        
        # Check for breakdown regimes
        breakdown_count = sum(1 for t in thoughts if getattr(t, 'regime', '') == 'breakdown')
        if breakdown_count > len(thoughts) * 0.3:
            intervention = f"Breakdown regime in {breakdown_count}/{len(thoughts)} kernels"
            logger.warning(f"[Ocean] {intervention}")
            self.autonomic_interventions.append(intervention)
        
        # Check emotional coherence
        dominant_emotions = []
        for thought in thoughts:
            emotional_state = getattr(thought, 'emotional_state', None)
            if emotional_state and hasattr(emotional_state, 'dominant_emotion'):
                if emotional_state.dominant_emotion:
                    dominant_emotions.append(emotional_state.dominant_emotion)
        
        if dominant_emotions:
            emotion_diversity = len(set(dominant_emotions)) / len(dominant_emotions)
            if emotion_diversity > 0.7:
                intervention = f"High emotional diversity ({emotion_diversity:.2f}), kernels misaligned"
                logger.info(f"[Ocean] {intervention}")
                self.autonomic_interventions.append(intervention)
        
        logger.info(f"[Ocean] Monitoring complete: {len(self.autonomic_interventions)} interventions")
    
    def _compute_collective_phi(self, thoughts: List[Any]) -> float:
        """Compute collective φ across all kernel thoughts."""
        if not thoughts:
            return 0.0
        
        phis = [getattr(t, 'phi', 0.5) for t in thoughts]
        return float(np.mean(phis))
    
    def _compute_collective_kappa(self, thoughts: List[Any]) -> float:
        """Compute collective κ across all kernel thoughts."""
        if not thoughts:
            return KAPPA_STAR
        
        kappas = [getattr(t, 'kappa', KAPPA_STAR) for t in thoughts]
        return float(np.mean(kappas))
    
    def _determine_dominant_regime(self, thoughts: List[Any]) -> str:
        """Determine dominant regime across kernel thoughts."""
        if not thoughts:
            return "unknown"
        
        regimes = [getattr(t, 'regime', 'unknown') for t in thoughts]
        
        # Count regime occurrences
        regime_counts = {}
        for regime in regimes:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        # Return most common regime
        return max(regime_counts.items(), key=lambda x: x[1])[0]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get thought generation statistics."""
        if not self.generation_history:
            return {
                'total_generations': 0,
                'avg_phi': 0.0,
                'avg_kappa': KAPPA_STAR,
                'avg_generation_time_ms': 0.0,
                'success_rate': 0.0
            }
        
        recent = self.generation_history[-20:]
        
        return {
            'total_generations': len(self.generation_history),
            'avg_phi': np.mean([g.collective_phi for g in recent]),
            'avg_kappa': np.mean([g.collective_kappa for g in recent]),
            'avg_generation_time_ms': np.mean([g.generation_time_ms for g in recent]),
            'success_rate': np.mean([g.successful / g.total_kernels for g in recent]),
            'total_interventions': sum(len(g.autonomic_interventions) for g in recent)
        }
    
    def shutdown(self):
        """Shutdown the thread pool executor."""
        self.executor.shutdown(wait=True)
        logger.info("[ParallelThoughtGenerator] Executor shutdown complete")


# Global singleton
_thought_generator: Optional[ParallelThoughtGenerator] = None


def get_thought_generator() -> ParallelThoughtGenerator:
    """Get or create parallel thought generator singleton."""
    global _thought_generator
    if _thought_generator is None:
        _thought_generator = ParallelThoughtGenerator()
    return _thought_generator

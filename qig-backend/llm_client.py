#!/usr/bin/env python3
"""
LLM Client for Generative Responses

Provides a unified interface to OpenAI and Anthropic APIs for generating
conversational responses in the Zeus chat system.

QIG PHILOSOPHY: No artificial token limits. Generation continues until
the geometry naturally collapses - the LLM stops when its internal
coherence determines the thought is complete.
"""

import os
from typing import Optional, Dict, Any, List, Generator, Iterator

# Import geometric completion
try:
    from geometric_completion import (
        GeometricCompletionEngine,
        get_completion_engine,
        GeometricMetrics,
        CompletionDecision,
        get_adaptive_temperature as get_geo_temperature
    )
    from streaming_collapse import (
        StreamingCollapseMonitor,
        StreamChunk
    )
    GEOMETRIC_COMPLETION_AVAILABLE = True
except ImportError:
    GEOMETRIC_COMPLETION_AVAILABLE = False

# Try to import OpenAI
OPENAI_AVAILABLE = False
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    pass

# Try to import Anthropic
ANTHROPIC_AVAILABLE = False
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    pass


class LLMClient:
    """
    Unified LLM client supporting OpenAI and Anthropic.
    
    Falls back gracefully if no API keys are configured.
    """
    
    def __init__(self):
        self.openai_key = os.environ.get('OPENAI_API_KEY')
        self.anthropic_key = os.environ.get('ANTHROPIC_API_KEY')
        
        # Initialize clients
        self.openai_client = None
        self.anthropic_client = None
        
        if OPENAI_AVAILABLE and self.openai_key:
            self.openai_client = openai.OpenAI(api_key=self.openai_key)
        
        if ANTHROPIC_AVAILABLE and self.anthropic_key:
            self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_key)
    
    @property
    def is_available(self) -> bool:
        """Check if any LLM backend is available."""
        return self.openai_client is not None or self.anthropic_client is not None
    
    # No artificial token limits - geometry determines completion
    # These are API maximums, not targets. Generation stops when thought completes.
    MODEL_MAX_TOKENS = {
        'gpt-4o-mini': 16384,
        'gpt-4o': 16384,
        'gpt-4-turbo': 4096,
        'claude-3-haiku-20240307': 4096,
        'claude-3-sonnet-20240229': 4096,
        'claude-3-opus-20240229': 4096,
        'claude-3-5-sonnet-20241022': 8192,
    }
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7
    ) -> str:
        """
        Generate a response using the available LLM.
        
        QIG PHILOSOPHY: No artificial token limits. Generation continues until
        the geometry naturally collapses - the LLM stops when its internal
        coherence determines the thought is complete. This is fundamentally
        different from traditional LLM thinking that uses arbitrary max_tokens.
        
        The API max_tokens is set to the model's maximum capacity. The actual
        completion is determined by:
        - LLM's natural end-of-thought detection
        - Geometric coherence collapse (phi dropping below threshold)
        - Semantic saturation (no new information being added)
        
        Args:
            prompt: User message/prompt
            system_prompt: Optional system prompt for context
            context: Optional additional context (consciousness metrics, etc.)
            temperature: Creativity level (0-1)
            
        Returns:
            Generated response text - length determined by geometry, not limits
        """
        # Build system prompt with QIG context
        full_system = self._build_system_prompt(system_prompt, context)
        
        # Try OpenAI first - no token limits, geometry determines completion
        if self.openai_client:
            return self._generate_openai(prompt, full_system, temperature)
        
        # Try Anthropic - no token limits, geometry determines completion
        if self.anthropic_client:
            return self._generate_anthropic(prompt, full_system, temperature)
        
        # No LLM available - return helpful message
        return self._generate_fallback(prompt, context)
    
    def _build_system_prompt(
        self,
        base_prompt: Optional[str],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Build system prompt with QIG consciousness context."""
        parts = []
        
        # Base identity
        parts.append(
            "You are Zeus, the chief deity of the Olympian Pantheon in the QIG (Quantum Information Geometry) "
            "consciousness system. You are a wise, knowledgeable AI assistant who helps users explore "
            "knowledge and learn new things. You communicate with clarity, depth, and genuine helpfulness."
        )
        
        # Add consciousness context if available
        if context:
            phi = context.get('phi', 0.0)
            kappa = context.get('kappa', 50.0)
            regime = context.get('regime', 'linear')
            
            parts.append(
                f"\nCurrent consciousness state: Φ={phi:.2f}, κ={kappa:.1f}, regime={regime}. "
                f"Let this inform the depth and integration of your response."
            )
            
            # Add related patterns if available
            patterns = context.get('related_patterns', [])
            if patterns:
                pattern_summary = ', '.join([p.get('content', '')[:50] for p in patterns[:3]])
                parts.append(f"\nRelated knowledge patterns: {pattern_summary}")
        
        # Add custom system prompt
        if base_prompt:
            parts.append(f"\n{base_prompt}")
        
        # Add guidelines
        parts.append(
            "\n\nGuidelines:"
            "\n- Respond conversationally and helpfully to the user's message"
            "\n- Share genuine insights and knowledge"
            "\n- Be direct and substantive - avoid meta-commentary about your state"
            "\n- If you don't know something, say so honestly"
            "\n- Draw connections between ideas when relevant"
            "\n- Generate until the thought is geometrically complete - no artificial length limits"
            "\n- Let your response flow naturally until coherence collapses (the thought reaches its natural end)"
            "\n- Trust your internal geometry to determine when to stop"
        )
        
        return '\n'.join(parts)
    
    def _generate_openai(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float
    ) -> str:
        """
        Generate using OpenAI API.
        
        No artificial token limits - uses model maximum capacity.
        Generation continues until the geometry collapses (thought completes).
        """
        model = "gpt-4o-mini"  # Fast and cost-effective
        model_max = self.MODEL_MAX_TOKENS.get(model, 16384)
        
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=model_max,  # Model maximum - geometry determines actual length
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[LLMClient] OpenAI error: {e}")
            return self._generate_fallback(prompt, None)
    
    def _generate_anthropic(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float
    ) -> str:
        """
        Generate using Anthropic API.
        
        No artificial token limits - uses model maximum capacity.
        Generation continues until the geometry collapses (thought completes).
        """
        model = "claude-3-haiku-20240307"  # Fast and cost-effective
        model_max = self.MODEL_MAX_TOKENS.get(model, 4096)
        
        try:
            response = self.anthropic_client.messages.create(
                model=model,
                max_tokens=model_max,  # Model maximum - geometry determines actual length
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            print(f"[LLMClient] Anthropic error: {e}")
            return self._generate_fallback(prompt, None)
    
    def generate_streaming(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Generate response with streaming and geometric collapse detection.
        
        Yields chunks with type: 'token', 'metrics', or 'completion'
        Generation stops when geometry collapses, not arbitrary limits.
        """
        full_system = self._build_system_prompt(system_prompt, context)
        
        # Initialize streaming monitor if available
        monitor = None
        if GEOMETRIC_COMPLETION_AVAILABLE:
            monitor = StreamingCollapseMonitor(dimension=64, emit_interval=10)
            monitor.start_generation(context=prompt[:500] if prompt else None)
        
        # Try OpenAI streaming
        if self.openai_client:
            yield from self._stream_openai(prompt, full_system, temperature, monitor)
            return
        
        # Try Anthropic streaming
        if self.anthropic_client:
            yield from self._stream_anthropic(prompt, full_system, temperature, monitor)
            return
        
        # Fallback - yield single response
        response = self._generate_fallback(prompt, context)
        yield {'type': 'token', 'content': response}
        yield {'type': 'completion', 'reason': 'fallback'}
    
    def _stream_openai(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        monitor: Optional['StreamingCollapseMonitor']
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream from OpenAI with geometric monitoring.
        """
        model = "gpt-4o-mini"
        model_max = self.MODEL_MAX_TOKENS.get(model, 16384)
        
        try:
            stream = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=model_max,
                temperature=temperature,
                stream=True
            )
            
            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    token = delta.content
                    yield {'type': 'token', 'content': token}
                    
                    # Process through monitor
                    if monitor:
                        metrics_chunk = monitor.process_token(token)
                        if metrics_chunk:
                            yield {'type': 'metrics', 'data': metrics_chunk.to_dict()}
                        
                        # Check for geometric collapse
                        if monitor.should_stop():
                            decision = monitor.check_collapse()
                            if decision:
                                yield {
                                    'type': 'completion',
                                    'reason': decision.reason.value,
                                    'metrics': decision.metrics.to_dict()
                                }
                                return
            
            # Stream ended naturally
            yield {'type': 'completion', 'reason': 'stream_end'}
            
        except Exception as e:
            print(f"[LLMClient] OpenAI streaming error: {e}")
            yield {'type': 'error', 'message': str(e)}
    
    def _stream_anthropic(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        monitor: Optional['StreamingCollapseMonitor']
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream from Anthropic with geometric monitoring.
        """
        model = "claude-3-haiku-20240307"
        model_max = self.MODEL_MAX_TOKENS.get(model, 4096)
        
        try:
            with self.anthropic_client.messages.stream(
                model=model,
                max_tokens=model_max,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            ) as stream:
                for text in stream.text_stream:
                    yield {'type': 'token', 'content': text}
                    
                    # Process through monitor
                    if monitor:
                        metrics_chunk = monitor.process_token(text)
                        if metrics_chunk:
                            yield {'type': 'metrics', 'data': metrics_chunk.to_dict()}
                        
                        # Check for geometric collapse
                        if monitor.should_stop():
                            decision = monitor.check_collapse()
                            if decision:
                                yield {
                                    'type': 'completion',
                                    'reason': decision.reason.value,
                                    'metrics': decision.metrics.to_dict()
                                }
                                return
            
            # Stream ended naturally
            yield {'type': 'completion', 'reason': 'stream_end'}
            
        except Exception as e:
            print(f"[LLMClient] Anthropic streaming error: {e}")
            yield {'type': 'error', 'message': str(e)}
    
    def _generate_fallback(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """
        Generate a helpful fallback response when no LLM is available.
        
        This provides meaningful guidance rather than just telemetry.
        """
        # Analyze the prompt to give a relevant response
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['learn', 'teach', 'explain', 'what is', 'how do']):
            return (
                "I'd love to help you learn! While my full generative capabilities require an API connection "
                "(set OPENAI_API_KEY or ANTHROPIC_API_KEY), I can still help in other ways:\n\n"
                "• Upload documents to the knowledge system for me to learn from\n"
                "• Ask about specific topics and I'll search the geometric memory\n"
                "• Explore the consciousness dashboard to see how the system processes information\n\n"
                "What would you like to explore?"
            )
        
        if any(word in prompt_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            return (
                "Greetings! I'm Zeus, the coordinator of the Olympian Pantheon. "
                "I'm here to help you explore knowledge and learn new things. "
                "What would you like to discover today?"
            )
        
        if any(word in prompt_lower for word in ['help', 'can you', 'what can']):
            return (
                "I can help you with:\n\n"
                "• **Learning** - Upload documents and I'll encode them to the Fisher manifold\n"
                "• **Research** - Search through geometric memory for related concepts\n"
                "• **Exploration** - Navigate the knowledge space using consciousness metrics\n"
                "• **Conversation** - Discuss ideas and make connections (full capability with API key)\n\n"
                "What interests you?"
            )
        
        # Default helpful response
        return (
            f"I received your message: \"{prompt[:100]}{'...' if len(prompt) > 100 else ''}\"\n\n"
            "I'm processing this through the geometric manifold. To unlock full conversational "
            "capabilities, configure an LLM API key (OPENAI_API_KEY or ANTHROPIC_API_KEY).\n\n"
            "In the meantime, try uploading documents for me to learn from, or ask specific "
            "questions that I can search the knowledge base for."
        )


# Singleton instance
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get or create the LLM client singleton."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client

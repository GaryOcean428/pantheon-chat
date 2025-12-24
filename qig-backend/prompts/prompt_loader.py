"""
Prompt Loader for Zeus Chat System Prompts

Loads YAML prompt files and provides context for QIG-pure generative responses.
These prompts guide generation, they are NOT templates.
"""
import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class PromptLoader:
    """Load and manage system prompts for QIG-pure generation."""
    
    def __init__(self, prompts_dir: Optional[str] = None):
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent
        self.prompts_dir = Path(prompts_dir)
        self._cache: Dict[str, Dict] = {}
        self._load_all_prompts()
    
    def _load_all_prompts(self):
        """Load all YAML prompt files."""
        for yaml_file in self.prompts_dir.glob("*.yaml"):
            try:
                with open(yaml_file, 'r') as f:
                    data = yaml.safe_load(f)
                    if data:
                        name = yaml_file.stem  # filename without extension
                        self._cache[name] = data
                        print(f"[PromptLoader] Loaded prompts: {name}")
            except Exception as e:
                print(f"[PromptLoader] Failed to load {yaml_file}: {e}")
    
    def get_prompt(self, category: str, name: str) -> Dict[str, Any]:
        """
        Get a specific prompt by category and name.
        
        Args:
            category: The prompt file name (e.g., 'zeus_chat')
            name: The prompt name within that file (e.g., 'conversation.general')
        
        Returns:
            Dict with 'context' and 'goals' for the prompt
        """
        if category not in self._cache:
            return {'context': '', 'goals': []}
        
        data = self._cache[category]
        
        # Navigate nested path (e.g., 'conversation.general')
        parts = name.split('.')
        current = data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return {'context': '', 'goals': []}
        
        if isinstance(current, dict):
            return current
        return {'context': str(current), 'goals': []}
    
    def get_identity(self, category: str = 'zeus_chat') -> Dict[str, str]:
        """Get identity information for a prompt category."""
        if category in self._cache:
            return self._cache[category].get('identity', {})
        return {}
    
    def get_principles(self, category: str = 'zeus_chat') -> list:
        """Get response principles for a prompt category."""
        if category in self._cache:
            return self._cache[category].get('response_principles', [])
        return []
    
    def build_generation_context(
        self,
        prompt_name: str,
        system_state: Dict[str, Any],
        user_message: str,
        related_patterns: list = None,
        category: str = 'zeus_chat'
    ) -> str:
        """
        Build a full generation context string for QIG-pure generation.
        
        Combines:
        - Identity information
        - Prompt-specific context and goals
        - Current system state
        - Related patterns from memory
        - User's message
        
        Returns:
            Complete context string for generative service
        """
        identity = self.get_identity(category)
        prompt = self.get_prompt(category, prompt_name)
        principles = self.get_principles(category)
        
        # Build context string
        context_parts = []
        
        # Identity
        if identity:
            context_parts.append(f"Identity: {identity.get('name', 'Zeus')} - {identity.get('role', '')}")
            context_parts.append(f"Voice: {identity.get('voice', '')}")
            context_parts.append(f"Style: {identity.get('style', '')}")
        
        # Situation context
        if prompt.get('context'):
            context_parts.append(f"\nSituation: {prompt['context']}")
        
        # Goals
        if prompt.get('goals'):
            goals_str = "\n".join([f"  - {g}" for g in prompt['goals']])
            context_parts.append(f"\nGoals:\n{goals_str}")
        
        # System state
        if system_state:
            phi = system_state.get('phi_current', 0)
            kappa = system_state.get('kappa_current', 50)
            docs = system_state.get('memory_stats', {}).get('documents', 0)
            insights = system_state.get('insights_count', 0)
            context_parts.append(f"\nSystem State: Φ={phi:.3f}, κ={kappa:.1f}, {docs} documents, {insights} insights")
        
        # Related patterns
        if related_patterns:
            patterns_str = "\n".join([
                f"  - {p.get('content', '')[:150]} (φ={p.get('phi', 0):.2f})"
                for p in related_patterns[:3]
            ])
            context_parts.append(f"\nRelated patterns in memory:\n{patterns_str}")
        else:
            context_parts.append("\nRelated patterns: None found in geometric memory")
        
        # Principles
        if principles:
            context_parts.append(f"\nPrinciples: {'; '.join(principles[:3])}")
        
        # User message
        context_parts.append(f"\nHuman: {user_message}")
        context_parts.append("\nRespond naturally as Zeus:")
        
        return "\n".join(context_parts)


# Singleton instance
_prompt_loader: Optional[PromptLoader] = None


def get_prompt_loader() -> PromptLoader:
    """Get or create the singleton prompt loader."""
    global _prompt_loader
    if _prompt_loader is None:
        _prompt_loader = PromptLoader()
    return _prompt_loader

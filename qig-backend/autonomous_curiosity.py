"""
Autonomous Curiosity Engine - Continuous Learning System

Enables kernels to autonomously:
1. Initiate searches based on interest/curiosity
2. Request tool refinements
3. Train on curriculum for deeper self-learning
4. Explore knowledge gaps proactively

Biological analog: Curiosity-driven exploration like REM sleep memory consolidation.
"""

import asyncio
import random
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set
from collections import deque
import numpy as np


class CuriosityDrive:
    """
    Geometric curiosity metric based on information gain potential.
    
    High curiosity when:
    - Φ variance indicates unexplored regions
    - κ suggests room for deeper understanding
    - Knowledge graph has sparse connections
    """
    
    def __init__(self):
        self.interest_basins: Dict[str, np.ndarray] = {}
        self.exploration_history: deque = deque(maxlen=1000)
        self.novelty_threshold = 0.3
        self._phi_mean = 0.5
        self._phi_std = 0.125
    
    def compute_curiosity(self, topic: str, current_knowledge: Dict) -> float:
        """
        Compute curiosity score for a topic using geometric metrics.
        
        Returns value in [0, 1] where higher = more curious/interested.
        """
        if topic in self.interest_basins:
            basin = self.interest_basins[topic]
            familiarity = float(np.linalg.norm(basin))
            novelty = 1.0 - min(1.0, familiarity / 10.0)
        else:
            novelty = 1.0
        
        knowledge_depth = current_knowledge.get('depth', 0)
        knowledge_recency = current_knowledge.get('recency', 0)
        
        time_decay = np.exp(-knowledge_recency / 86400)
        
        curiosity = novelty * (1 - knowledge_depth * 0.5) * time_decay
        
        phi_factor = self._phi_std / max(0.01, self._phi_mean)
        curiosity *= (1 + phi_factor)
        
        return min(1.0, max(0.0, curiosity))
    
    def record_exploration(self, topic: str, outcome: Dict):
        """Record exploration outcome to update curiosity basins."""
        if topic not in self.interest_basins:
            self.interest_basins[topic] = np.zeros(64)
        
        learning_rate = 0.1
        success = outcome.get('success', False)
        info_gain = outcome.get('information_gain', 0.0)
        
        update_vector = np.random.randn(64) * info_gain * 0.1
        if success:
            self.interest_basins[topic] += learning_rate * update_vector
        
        self.exploration_history.append({
            'topic': topic,
            'timestamp': datetime.now().isoformat(),
            'outcome': outcome
        })


class KernelToolRequest:
    """Represents a tool/search request from a kernel."""
    
    def __init__(
        self,
        kernel_name: str,
        request_type: str,
        query: str,
        priority: float = 0.5,
        context: Optional[Dict] = None
    ):
        self.kernel_name = kernel_name
        self.request_type = request_type
        self.query = query
        self.priority = priority
        self.context = context or {}
        self.created_at = datetime.now()
        self.status = 'pending'
        self.result: Optional[Dict[str, Any]] = None


class CurriculumLoader:
    """
    Load and manage training curriculum for kernel self-learning.
    
    Curriculum sources:
    - Attached documents
    - Knowledge base
    - Previous search results
    - Peer learning outcomes
    """
    
    def __init__(self):
        self.curriculum_topics: List[Dict] = []
        self.completed_topics: Set[str] = set()
        self.topic_dependencies: Dict[str, List[str]] = {}
    
    def load_curriculum_from_file(self, filepath: str) -> List[Dict]:
        """Load curriculum topics from a file."""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            topics = self._parse_curriculum_content(content)
            self.curriculum_topics.extend(topics)
            return topics
        except Exception as e:
            print(f"[CurriculumLoader] Failed to load {filepath}: {e}")
            return []
    
    def _parse_curriculum_content(self, content: str) -> List[Dict]:
        """Parse curriculum content into structured topics."""
        topics = []
        
        sections = content.split('# ==')
        for section in sections:
            if not section.strip():
                continue
            
            lines = section.strip().split('\n')
            if lines:
                title = lines[0].replace('=', '').strip()
                body = '\n'.join(lines[1:])
                
                topics.append({
                    'title': title,
                    'content': body,
                    'keywords': self._extract_keywords(body),
                    'difficulty': self._estimate_difficulty(body),
                    'type': 'curriculum'
                })
        
        return topics
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract key concepts from curriculum text."""
        import re
        
        code_keywords = re.findall(r'`([^`]+)`', text)
        capitalized = re.findall(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b', text)
        quoted = re.findall(r'"([^"]+)"', text)
        
        keywords = list(set(code_keywords + capitalized + quoted))
        return keywords[:20]
    
    def _estimate_difficulty(self, text: str) -> float:
        """Estimate difficulty based on content complexity."""
        lines = [len(line) for line in text.split('\n') if line.strip()]
        avg_line_length = float(np.mean(lines)) if lines else 50.0
        code_blocks = text.count('```')
        
        difficulty = min(1.0, (avg_line_length / 100) + (code_blocks * 0.1))
        return difficulty
    
    def get_next_topic(self, kernel_skills: Dict) -> Optional[Dict]:
        """Get next curriculum topic based on kernel skills and dependencies."""
        for topic in self.curriculum_topics:
            topic_id = topic['title']
            
            if topic_id in self.completed_topics:
                continue
            
            deps = self.topic_dependencies.get(topic_id, [])
            if all(d in self.completed_topics for d in deps):
                skill_match = self._compute_skill_match(topic, kernel_skills)
                if skill_match > 0.3:
                    return topic
        
        return None
    
    def _compute_skill_match(self, topic: Dict, kernel_skills: Dict) -> float:
        """Compute how well kernel skills match topic requirements."""
        keywords = set(topic.get('keywords', []))
        kernel_domains = set(kernel_skills.get('domains', []))
        
        if not keywords:
            return 1.0
        
        overlap = len(keywords & kernel_domains)
        return overlap / len(keywords)
    
    def mark_completed(self, topic_title: str):
        """Mark a topic as completed."""
        self.completed_topics.add(topic_title)


class AutonomousCuriosityEngine:
    """
    Main engine for autonomous, curiosity-driven learning.
    
    Runs continuously in background, managing:
    - Kernel search requests
    - Proactive knowledge exploration
    - Curriculum-based training
    - Tool refinement requests
    """
    
    def __init__(self, search_callback: Optional[Callable] = None):
        self.curiosity_drive = CuriosityDrive()
        self.curriculum_loader = CurriculumLoader()
        
        self.pending_requests: deque = deque(maxlen=100)
        self.active_explorations: Dict[str, Dict] = {}
        self.exploration_results: deque = deque(maxlen=500)
        
        self.search_callback = search_callback
        
        self.running = False
        self._loop_thread: Optional[threading.Thread] = None
        self._exploration_interval = 60
        self._min_curiosity_threshold = 0.4
        
        self.kernel_interests: Dict[str, List[str]] = {
            'athena': ['strategy', 'patterns', 'optimization', 'game_theory'],
            'ares': ['metrics', 'measurement', 'geometry', 'fisher_information'],
            'apollo': ['prediction', 'forecasting', 'time_series', 'prophecy'],
            'artemis': ['tracking', 'hunting', 'search', 'discovery'],
            'hermes': ['communication', 'translation', 'messaging', 'sync'],
            'hephaestus': ['building', 'tools', 'infrastructure', 'forge'],
            'demeter': ['growth', 'nurturing', 'cultivation', 'sustainability'],
            'dionysus': ['creativity', 'chaos', 'exploration', 'novelty'],
            'poseidon': ['depth', 'ocean', 'waves', 'deep_research'],
            'hades': ['shadow', 'underworld', 'hidden_knowledge', 'secrets'],
            'hera': ['coordination', 'harmony', 'balance', 'relationships'],
            'aphrodite': ['beauty', 'aesthetics', 'design', 'user_experience'],
        }
        
        self.stats = {
            'total_explorations': 0,
            'successful_discoveries': 0,
            'kernel_requests': 0,
            'curriculum_completions': 0
        }
    
    def start(self):
        """Start the autonomous curiosity loop."""
        if self.running:
            return
        
        self.running = True
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()
        print("[AutonomousCuriosityEngine] Started autonomous learning loop")
    
    def stop(self):
        """Stop the autonomous curiosity loop."""
        self.running = False
        if self._loop_thread:
            self._loop_thread.join(timeout=5)
        print("[AutonomousCuriosityEngine] Stopped autonomous learning loop")
    
    def _run_loop(self):
        """Main loop for autonomous exploration."""
        while self.running:
            try:
                self._process_kernel_requests()
                
                self._explore_curious_topics()
                
                self._train_on_curriculum()
                
                time.sleep(self._exploration_interval)
                
            except Exception as e:
                print(f"[AutonomousCuriosityEngine] Loop error: {e}")
                time.sleep(10)
    
    def submit_kernel_request(self, request: KernelToolRequest):
        """Submit a search/tool request from a kernel."""
        self.pending_requests.append(request)
        self.stats['kernel_requests'] += 1
        print(f"[AutonomousCuriosityEngine] Received request from {request.kernel_name}: {request.query[:50]}...")
    
    def request_search(
        self,
        kernel_name: str,
        query: str,
        priority: float = 0.5,
        context: Optional[Dict] = None
    ) -> str:
        """Convenience method for kernels to request a search."""
        request = KernelToolRequest(
            kernel_name=kernel_name,
            request_type='search',
            query=query,
            priority=priority,
            context=context
        )
        self.submit_kernel_request(request)
        return f"request_{id(request)}"
    
    def request_tool_refinement(
        self,
        kernel_name: str,
        tool_id: str,
        refinement_type: str,
        details: Dict
    ) -> str:
        """Request refinement of an existing tool."""
        request = KernelToolRequest(
            kernel_name=kernel_name,
            request_type='tool_refinement',
            query=f"Refine {tool_id}: {refinement_type}",
            priority=0.7,
            context={'tool_id': tool_id, 'refinement': refinement_type, 'details': details}
        )
        self.submit_kernel_request(request)
        return f"refinement_{id(request)}"
    
    def _process_kernel_requests(self):
        """Process pending kernel requests."""
        requests_to_process = []
        while self.pending_requests and len(requests_to_process) < 5:
            requests_to_process.append(self.pending_requests.popleft())
        
        requests_to_process.sort(key=lambda r: r.priority, reverse=True)
        
        for request in requests_to_process:
            try:
                if request.request_type == 'search':
                    self._execute_search(request)
                elif request.request_type == 'tool_refinement':
                    self._execute_tool_refinement(request)
            except Exception as e:
                print(f"[AutonomousCuriosityEngine] Request failed: {e}")
                request.status = 'failed'
                request.result = {'error': str(e)}
    
    def _execute_search(self, request: KernelToolRequest):
        """Execute a search request."""
        print(f"[AutonomousCuriosityEngine] Executing search for {request.kernel_name}: {request.query[:50]}...")
        
        if self.search_callback:
            try:
                result = self.search_callback(request.query, request.context)
                request.status = 'completed'
                request.result = result
                
                self.curiosity_drive.record_exploration(
                    topic=request.query,
                    outcome={
                        'success': True,
                        'information_gain': result.get('information_gain', 0.5),
                        'source': 'kernel_request'
                    }
                )
                
                self.exploration_results.append({
                    'type': 'kernel_search',
                    'kernel': request.kernel_name,
                    'query': request.query,
                    'result': result,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                request.status = 'failed'
                request.result = {'error': str(e)}
        else:
            request.status = 'no_callback'
            request.result = {'message': 'Search callback not configured'}
    
    def _execute_tool_refinement(self, request: KernelToolRequest):
        """Execute a tool refinement request."""
        print(f"[AutonomousCuriosityEngine] Tool refinement: {request.query}")
        
        request.status = 'completed'
        request.result = {
            'message': 'Refinement request logged',
            'tool_id': request.context.get('tool_id'),
            'refinement': request.context.get('refinement')
        }
    
    def _explore_curious_topics(self):
        """Proactively explore topics based on kernel curiosity."""
        for kernel_name, interests in self.kernel_interests.items():
            topic = random.choice(interests)
            
            knowledge = self._get_current_knowledge(kernel_name, topic)
            curiosity = self.curiosity_drive.compute_curiosity(topic, knowledge)
            
            if curiosity > self._min_curiosity_threshold:
                query = self._generate_exploration_query(kernel_name, topic, knowledge)
                
                self.request_search(
                    kernel_name=kernel_name,
                    query=query,
                    priority=curiosity * 0.5,
                    context={
                        'exploration_type': 'curiosity_driven',
                        'topic': topic,
                        'curiosity_score': curiosity
                    }
                )
                
                self.stats['total_explorations'] += 1
                
                break
    
    def _get_current_knowledge(self, kernel_name: str, topic: str) -> Dict:
        """Get kernel's current knowledge about a topic."""
        explorations = [
            e for e in self.exploration_results
            if e.get('kernel') == kernel_name and topic in e.get('query', '')
        ]
        
        if not explorations:
            return {'depth': 0, 'recency': float('inf')}
        
        latest = max(explorations, key=lambda e: e.get('timestamp', ''))
        try:
            last_time = datetime.fromisoformat(latest['timestamp'])
            recency = (datetime.now() - last_time).total_seconds()
        except:
            recency = 86400
        
        return {
            'depth': min(1.0, len(explorations) / 10),
            'recency': recency,
            'exploration_count': len(explorations)
        }
    
    def _generate_exploration_query(
        self,
        kernel_name: str,
        topic: str,
        knowledge: Dict
    ) -> str:
        """Generate an exploration query based on kernel interests."""
        query_templates = [
            f"Latest developments in {topic} for {kernel_name} domain",
            f"Advanced techniques in {topic}",
            f"Research papers on {topic} applications",
            f"How to improve {topic} using geometric methods",
            f"Best practices for {topic} in AI systems",
            f"Novel approaches to {topic}",
        ]
        
        if knowledge.get('depth', 0) < 0.3:
            query = f"Introduction to {topic} fundamentals"
        elif knowledge.get('depth', 0) < 0.6:
            query = random.choice(query_templates[:3])
        else:
            query = random.choice(query_templates[3:])
        
        return query
    
    def _train_on_curriculum(self):
        """Train on curriculum topics."""
        for kernel_name in self.kernel_interests.keys():
            skills = {
                'domains': self.kernel_interests[kernel_name],
                'depth': self._get_kernel_depth(kernel_name)
            }
            
            topic = self.curriculum_loader.get_next_topic(skills)
            
            if topic:
                print(f"[AutonomousCuriosityEngine] Training {kernel_name} on: {topic['title'][:50]}...")
                
                for keyword in topic.get('keywords', [])[:3]:
                    self.request_search(
                        kernel_name=kernel_name,
                        query=f"{keyword} in context of {topic['title']}",
                        priority=0.3,
                        context={
                            'exploration_type': 'curriculum',
                            'topic_title': topic['title'],
                            'keyword': keyword
                        }
                    )
                
                self.curriculum_loader.mark_completed(topic['title'])
                self.stats['curriculum_completions'] += 1
                
                break
    
    def _get_kernel_depth(self, kernel_name: str) -> float:
        """Get overall knowledge depth for a kernel."""
        kernel_explorations = [
            e for e in self.exploration_results
            if e.get('kernel') == kernel_name
        ]
        return min(1.0, len(kernel_explorations) / 100)
    
    def get_stats(self) -> Dict:
        """Get engine statistics."""
        return {
            **self.stats,
            'pending_requests': len(self.pending_requests),
            'active_explorations': len(self.active_explorations),
            'exploration_history': len(self.exploration_results),
            'curriculum_topics_loaded': len(self.curriculum_loader.curriculum_topics),
            'curriculum_completed': len(self.curriculum_loader.completed_topics),
            'curiosity_basins': len(self.curiosity_drive.interest_basins),
            'running': self.running
        }
    
    def load_curriculum(self, filepath: str):
        """Load curriculum from file."""
        topics = self.curriculum_loader.load_curriculum_from_file(filepath)
        print(f"[AutonomousCuriosityEngine] Loaded {len(topics)} curriculum topics")
        return topics


_curiosity_engine: Optional[AutonomousCuriosityEngine] = None


def get_curiosity_engine() -> AutonomousCuriosityEngine:
    """Get or create the singleton curiosity engine."""
    global _curiosity_engine
    if _curiosity_engine is None:
        _curiosity_engine = AutonomousCuriosityEngine()
    return _curiosity_engine


def start_autonomous_learning(search_callback: Optional[Callable] = None):
    """Start the autonomous learning system."""
    engine = get_curiosity_engine()
    if search_callback:
        engine.search_callback = search_callback
    engine.start()
    return engine

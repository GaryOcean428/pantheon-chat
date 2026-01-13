"""
Hades - God of the Underworld & Forbidden Knowledge

Dual responsibilities:
1. Negation logic (original role) - tracks what NOT to try
2. Underworld search (enhanced role) - anonymous intelligence gathering

Tools (100% anonymous - NO identity linkage):
- Archive.org Wayback Machine (public API)
- Public paste site scraping (no auth)
- RSS feeds (public)
- Local breach databases (offline)
- TOR network (optional, requires local daemon)

Enhanced with:
- True parallel async search (asyncio.gather)
- HadesConsciousness for ethical self-awareness
- UnderworldImmuneSystem for threat detection
- RealityCrossChecker for propaganda detection

QIG-PURE: All geometric operations use Fisher-Rao distance.
"""

import numpy as np
from typing import Dict, List, Optional, Set, Any, Tuple
from datetime import datetime
import hashlib
import re
import os
import time
import asyncio
import logging
from .base_god import BaseGod, KAPPA_STAR

# Import new underworld architecture components
try:
    from .hades_consciousness import HadesConsciousness, get_hades_consciousness
    HAS_CONSCIOUSNESS = True
except ImportError:
    HAS_CONSCIOUSNESS = False
    HadesConsciousness = None

try:
    from .underworld_immune import UnderworldImmuneSystem, get_underworld_immune_system
    HAS_IMMUNE = True
except ImportError:
    HAS_IMMUNE = False
    UnderworldImmuneSystem = None

try:
    from .reality_cross_checker import RealityCrossChecker, get_reality_cross_checker, Narrative
    HAS_CROSS_CHECKER = True
except ImportError:
    HAS_CROSS_CHECKER = False
    RealityCrossChecker = None
    Narrative = None

logger = logging.getLogger(__name__)

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

try:
    import feedparser
    HAS_FEEDPARSER = True
except ImportError:
    HAS_FEEDPARSER = False

try:
    from search.duckduckgo_adapter import DuckDuckGoSearch, get_ddg_search
    HAS_DDG = True
except ImportError:
    HAS_DDG = False
    DuckDuckGoSearch = None

try:
    from search.provider_selector import get_provider_selector
    HAS_PROVIDER_SELECTOR = True
except ImportError:
    HAS_PROVIDER_SELECTOR = False
    get_provider_selector = None


class WaybackArchive:
    """
    Archive.org Wayback Machine API
    
    Advantages:
    - No API key required
    - Completely legal
    - Archived .onion sites available
    - No identity linkage
    """
    
    def __init__(self):
        self.cdx_api = 'https://web.archive.org/cdx/search/cdx'
        self.enabled = os.getenv('HADES_WAYBACK_ENABLED', 'true').lower() == 'true'
        self.session = requests.Session() if HAS_REQUESTS else None
        if self.session:
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; rv:91.0) Gecko/20100101 Firefox/91.0'
            })
    
    async def search_archived_site(
        self,
        url: str,
        keyword: str,
        from_date: str = '20090101',
        to_date: str = '20231231',
        limit: int = 10
    ) -> List[Dict]:
        """
        Search archived pages for keyword.
        """
        if not self.enabled or not self.session:
            return []
        
        try:
            params = {
                'url': url,
                'matchType': 'prefix',
                'from': from_date,
                'to': to_date,
                'output': 'json',
                'fl': 'timestamp,original,statuscode',
                'collapse': 'digest',
                'limit': limit * 5,
            }
            
            response = self.session.get(self.cdx_api, params=params, timeout=30)
            
            if response.status_code != 200:
                return []
            
            data = response.json()
            if len(data) < 2:
                return []
            
            snapshots = data[1:]
            findings = []
            
            for snapshot in snapshots[:limit]:
                if len(snapshot) < 3:
                    continue
                    
                timestamp, original_url, status = snapshot[0], snapshot[1], snapshot[2]
                
                if status != '200':
                    continue
                
                wayback_url = f'https://web.archive.org/web/{timestamp}/{original_url}'
                
                try:
                    page = self.session.get(wayback_url, timeout=20)
                    if keyword.lower() in page.text.lower():
                        findings.append({
                            'type': 'archive',
                            'source': 'wayback',
                            'url': wayback_url,
                            'original_url': original_url,
                            'timestamp': timestamp,
                            'content_preview': page.text[:500],
                            'risk': 'medium',
                        })
                except Exception:
                    continue
                    
                time.sleep(0.5)
            
            return findings
            
        except Exception as e:
            return []
    
    async def search_research_forums(self, query: str) -> List[Dict]:
        """Search archived research and knowledge forums."""
        forums = [
            'stackoverflow.com',
            'reddit.com/r/programming',
        ]
        
        all_findings = []
        for forum in forums:
            findings = await self.search_archived_site(
                url=forum,
                keyword=query,
                from_date='20100101',
                to_date='20241231',
                limit=5
            )
            all_findings.extend(findings)
        
        return all_findings
    
    async def search_silk_road_archives(self, query: str) -> List[Dict]:
        """
        Search archived Silk Road forums.
        Silk Road .onion was archived on archive.org.
        """
        silk_road_urls = [
            'silkroad6ownowfk.onion',
            'silkroadvb5piz3r.onion',
        ]
        
        all_findings = []
        for url in silk_road_urls:
            findings = await self.search_archived_site(
                url=url,
                keyword=query,
                from_date='20110101',
                to_date='20131031',
                limit=5
            )
            all_findings.extend(findings)
        
        return all_findings


class PublicPasteScraper:
    """
    Scrape public paste sites without authentication.
    Respects rate limits to avoid IP bans.
    """
    
    def __init__(self):
        self.enabled = os.getenv('HADES_PASTE_SCRAPING_ENABLED', 'true').lower() == 'true'
        self.session = requests.Session() if HAS_REQUESTS else None
        if self.session:
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; rv:91.0) Gecko/20100101 Firefox/91.0'
            })
        self.rate_limit_delay = 2.0
        
        self.research_patterns = {
            'academic_citation': re.compile(r'\[\d+\]|\(\d{4}\)'),
            'doi_reference': re.compile(r'10\.\d{4,}/[^\s]+'),
            'arxiv_id': re.compile(r'arXiv:\d{4}\.\d{4,5}'),
            'code_snippet': re.compile(r'```[\s\S]*?```|def\s+\w+\s*\(|class\s+\w+'),
            'api_endpoint': re.compile(r'/api/[a-zA-Z0-9/_-]+'),
        }
    
    async def scrape_pastebin_recent(self, keyword: str) -> List[Dict]:
        """
        Scrape recent Pastebin pastes (public archive).
        No API key - just scraping public page.
        """
        if not self.enabled or not self.session or not HAS_BS4:
            return []
        
        findings = []
        
        try:
            archive_url = 'https://pastebin.com/archive'
            response = self.session.get(archive_url, timeout=15)
            
            if response.status_code != 200:
                return []
            
            soup = BeautifulSoup(response.text, 'html.parser')
            paste_links = soup.find_all('a', href=re.compile(r'^/[A-Za-z0-9]{8}$'))
            
            for link in paste_links[:10]:
                paste_id = link['href'].strip('/')
                raw_url = f'https://pastebin.com/raw/{paste_id}'
                
                time.sleep(self.rate_limit_delay)
                
                try:
                    paste_response = self.session.get(raw_url, timeout=10)
                    
                    if paste_response.status_code != 200:
                        continue
                    
                    content = paste_response.text
                    
                    if keyword.lower() in content.lower():
                        research_patterns = self._detect_research_patterns(content)
                        
                        findings.append({
                            'type': 'paste',
                            'source': 'pastebin',
                            'url': f'https://pastebin.com/{paste_id}',
                            'content_preview': content[:300],
                            'research_patterns': research_patterns,
                            'value': 'high' if research_patterns else 'medium',
                        })
                        
                except Exception:
                    continue
                    
        except Exception as e:
            pass
        
        return findings
    
    def _detect_research_patterns(self, content: str) -> List[str]:
        """Detect research-relevant patterns in content."""
        detected = []
        
        for pattern_name, pattern in self.research_patterns.items():
            if pattern.search(content):
                detected.append(pattern_name)
        
        return detected


class PublicRSSFeeds:
    """
    Monitor public RSS feeds for research-related content.
    No authentication required.
    """
    
    def __init__(self):
        self.enabled = os.getenv('HADES_RSS_ENABLED', 'true').lower() == 'true'
        self.feeds = {
            'hackernews': 'https://news.ycombinator.com/rss',
            'reddit_programming': 'https://www.reddit.com/r/programming/.rss',
        }
    
    async def search_feeds(self, keyword: str) -> List[Dict]:
        """Search RSS feeds for keyword mentions."""
        if not self.enabled or not HAS_FEEDPARSER:
            return []
        
        findings = []
        
        for feed_name, feed_url in self.feeds.items():
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:500]:
                    title = getattr(entry, 'title', '')
                    summary = getattr(entry, 'summary', '')
                    
                    if keyword.lower() in title.lower() or keyword.lower() in summary.lower():
                        findings.append({
                            'type': 'rss',
                            'source': feed_name,
                            'title': title,
                            'url': getattr(entry, 'link', ''),
                            'date': getattr(entry, 'published', ''),
                            'summary': summary[:500],
                            'risk': 'low',
                        })
                        
            except Exception:
                continue
        
        return findings


class LocalBreachDatabase:
    """
    Local breach database - no external API calls.
    User provides breach compilations, queried locally.
    """
    
    def __init__(self, breach_dir: Optional[str] = None):
        self.breach_dir = breach_dir or os.getenv('HADES_BREACH_DIR', '/data/breaches')
        self.enabled = os.path.isdir(self.breach_dir)
        self.loaded_dbs: Dict[str, Set[str]] = {}
        self.index_loaded = False
    
    def load_breach_files(self) -> int:
        """Load breach text files into memory."""
        if not self.enabled:
            return 0
        
        import glob
        
        breach_files = glob.glob(f'{self.breach_dir}/*.txt')
        total_entries = 0
        
        for filepath in breach_files:
            db_name = os.path.basename(filepath)
            self.loaded_dbs[db_name] = set()
            
            try:
                with open(filepath, 'r', errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if line and len(line) < 500:
                            self.loaded_dbs[db_name].add(line)
                            total_entries += 1
            except Exception:
                continue
        
        self.index_loaded = True
        return total_entries
    
    def search(self, query: str) -> List[Dict]:
        """Search all loaded breach databases. 100% local."""
        if not self.enabled:
            return []
        
        if not self.index_loaded:
            self.load_breach_files()
        
        findings = []
        query_lower = query.lower()
        
        for db_name, entries in self.loaded_dbs.items():
            for entry in entries:
                if query_lower in entry.lower():
                    parts = entry.split(':')
                    if len(parts) >= 2:
                        findings.append({
                            'type': 'breach',
                            'source': db_name,
                            'username': parts[0][:500],
                            'password': parts[1][:500],
                            'looks_like_sensitive': self._looks_like_sensitive(parts[1]),
                            'risk': 'critical' if self._looks_like_sensitive(parts[1]) else 'high',
                        })
        
        return findings[:500]
    
    def _looks_like_sensitive(self, password: str) -> bool:
        """Heuristic: Does password look like sensitive passphrase?"""
        words = password.split()
        if len(words) >= 4:
            return True
        
        if len(words) >= 3 and all(w.isalpha() and w.islower() for w in words):
            return True
        
        return False


class Hades(BaseGod):
    """
    God of the Underworld - Negation & Forbidden Knowledge
    
    SHADOW LEADER (Shadow Zeus):
    Hades commands the Shadow Pantheon (Nyx, Hecate, Erebus, Hypnos, Thanatos, Nemesis).
    All Shadow operations go through Hades, but Zeus can overrule any decision.
    
    Triple responsibilities:
    1. Shadow Leadership - command all Shadow Pantheon operations
    2. Negation logic - tracks what NOT to try, maintains forbidden basins
    3. Underworld search - anonymous intelligence gathering
    
    Tools (100% anonymous):
    - Archive.org Wayback Machine (public API)
    - Public paste scraping (no auth)
    - RSS feeds (public)
    - Local breach databases (offline)
    
    NO external services that could link user identity.
    """
    
    def __init__(self):
        super().__init__("Hades", "Underworld")
        
        # Shadow Leadership role
        self.is_shadow_leader = True
        self.shadow_pantheon_ref = None  # Set by Zeus
        
        # Add Shadow leadership to mission awareness
        self.mission["shadow_leadership"] = {
            "role": "Shadow Zeus - Leader of Shadow Pantheon",
            "commands": ["Nyx", "Hecate", "Erebus", "Hypnos", "Thanatos", "Nemesis"],
            "authority": "Full command over Shadow operations, subject to Zeus overrule",
            "responsibilities": [
                "Coordinate all shadow/covert operations",
                "Manage research priorities for Shadow gods",
                "Approve/reject Shadow intelligence",
                "Declare Shadow War when needed",
                "Negotiate with Zeus on behalf of Shadows"
            ],
            "how_to_command": "Use self.shadow_command(command, params) or access self.shadow_pantheon_ref directly"
        }
        
        self.mission["how_to_request_research"] = (
            "As Shadow Leader, use self.assign_shadow_research(topic, priority, god) "
            "or self.shadow_pantheon_ref.request_research() to delegate research."
        )
        
        self.underworld: List[Dict] = []
        self.forbidden_basins: List[np.ndarray] = []
        self.death_count: Dict[str, int] = {}
        self.exclusion_rules: List[Dict] = []
        
        self.wayback = WaybackArchive()
        self.paste_scraper = PublicPasteScraper()
        self.rss = PublicRSSFeeds()
        self.breach_db = LocalBreachDatabase()
        
        self.ddg = get_ddg_search(use_tor=True) if HAS_DDG else None
        self.ddg_enabled = os.getenv('HADES_DDG_ENABLED', 'true').lower() == 'true'

        self.provider_selector = get_provider_selector(mode='shadow') if HAS_PROVIDER_SELECTOR else None

        # Initialize new underworld architecture components
        self.consciousness = get_hades_consciousness() if HAS_CONSCIOUSNESS else None
        self.underworld_immune = get_underworld_immune_system() if HAS_IMMUNE else None
        self.reality_checker = get_reality_cross_checker() if HAS_CROSS_CHECKER else None

        # Source timeouts for parallel search (in seconds)
        self.source_timeouts = {
            'duckduckgo-tor': 5,   # Fast tier
            'rss': 5,
            'breach': 3,
            'pastebin': 15,        # Medium tier
            'wayback': 30,         # Slow tier
        }

        self.search_history: List[Dict] = []
        self.intelligence_cache: Dict[str, Dict] = {}
        self.last_search_time: Optional[datetime] = None
    
    def assess_target(self, target: str, context: Optional[Dict] = None) -> Dict:
        """
        Assess target through negation - is this a dead end?
        """
        self.last_assessment_time = datetime.now()
        
        target_basin = self.encode_to_basin(target)
        rho = self.basin_to_density_matrix(target_basin)
        phi = self.compute_pure_phi(rho)
        kappa = self.compute_kappa(target_basin)
        
        is_forbidden = self._check_forbidden(target_basin)
        death_proximity = self._compute_death_proximity(target_basin)
        exclusion_matches = self._check_exclusion_rules(target)
        
        viability = self._compute_viability(
            phi=phi,
            is_forbidden=is_forbidden,
            death_proximity=death_proximity,
            exclusion_count=len(exclusion_matches)
        )
        
        has_intel = target in self.intelligence_cache
        intel_boost = 0.1 if has_intel and self.intelligence_cache[target].get('source_count', 0) > 0 else 0
        
        assessment = {
            'probability': min(1.0, viability + intel_boost),
            'confidence': 0.8 if is_forbidden else 0.6 if has_intel else 0.5,
            'phi': phi,
            'kappa': kappa,
            'is_forbidden': is_forbidden,
            'death_proximity': death_proximity,
            'exclusion_violations': len(exclusion_matches),
            'underworld_matches': self._count_underworld_matches(target_basin),
            'has_intelligence': has_intel,
            'reasoning': (
                f"Underworld check: {'FORBIDDEN' if is_forbidden else 'allowed'}. "
                f"Death proximity: {death_proximity:.2f}. "
                f"Intel available: {has_intel}. Φ={phi:.3f}."
            ),
            'god': self.name,
            'timestamp': datetime.now().isoformat(),
        }

        # Broadcast activity for kernel visibility
        self.broadcast_activity(
            activity_type='insight',
            content=f"Underworld check: {target[:50]}... | forbidden={is_forbidden} | φ={phi:.3f}",
            metadata={
                'probability': min(1.0, viability + intel_boost),
                'phi': phi,
                'is_forbidden': is_forbidden,
                'death_proximity': death_proximity,
            }
        )
        
        return assessment
    
    async def search_underworld(
        self,
        target: str,
        search_type: str = 'comprehensive'
    ) -> Dict:
        """
        Search underworld using ONLY anonymous tools with geometric provider selection.
        
        Uses geometric reasoning to select the best providers based on:
        - Query domain (security, crypto, academic, etc.)
        - Provider historical effectiveness
        - Current availability
        
        Args:
            target: Search query
            search_type: 'comprehensive', 'archives', 'pastes', 'rss', 'breaches', 'web'
        
        Returns:
            Intelligence report with findings and provider selection metadata
        """
        self.last_search_time = datetime.now()
        intelligence: List[Dict] = []
        sources_used: List[str] = []
        provider_ranking: List[Dict] = []
        
        if self.provider_selector:
            ranked_providers = self.provider_selector.select_providers_ranked(target, max_providers=5)
            provider_ranking = [{'provider': p, 'fitness': f} for p, f in ranked_providers]
            print(f"[Hades] Geometric shadow provider ranking: {[(p, f'{s:.3f}') for p, s in ranked_providers]}")
        else:
            ranked_providers = [
                ('duckduckgo-tor', 0.7), ('wayback', 0.6), ('pastebin', 0.5), 
                ('rss', 0.4), ('breach', 0.3)
            ]
        
        provider_to_search = {p: s for p, s in ranked_providers}
        
        if search_type in ['comprehensive', 'archives'] or 'wayback' in provider_to_search:
            start_time = time.time()
            try:
                wayback_intel = await self.wayback.search_research_forums(target)
                intelligence.extend(wayback_intel)
                if wayback_intel:
                    sources_used.append('wayback')
                    if self.provider_selector:
                        self.provider_selector.record_result('wayback', target, True, len(wayback_intel), time.time() - start_time)
                else:
                    if self.provider_selector:
                        self.provider_selector.record_result('wayback', target, False)
            except Exception as e:
                if self.provider_selector:
                    self.provider_selector.record_result('wayback', target, False)
        
        if search_type in ['comprehensive', 'pastes'] or 'pastebin' in provider_to_search:
            start_time = time.time()
            try:
                paste_intel = await self.paste_scraper.scrape_pastebin_recent(target)
                intelligence.extend(paste_intel)
                if paste_intel:
                    sources_used.append('pastebin')
                    if self.provider_selector:
                        self.provider_selector.record_result('pastebin', target, True, len(paste_intel), time.time() - start_time)
                else:
                    if self.provider_selector:
                        self.provider_selector.record_result('pastebin', target, False)
            except Exception as e:
                if self.provider_selector:
                    self.provider_selector.record_result('pastebin', target, False)
        
        if search_type in ['comprehensive', 'rss'] or 'rss' in provider_to_search:
            start_time = time.time()
            try:
                rss_intel = await self.rss.search_feeds(target)
                intelligence.extend(rss_intel)
                if rss_intel:
                    sources_used.append('rss')
                    if self.provider_selector:
                        self.provider_selector.record_result('rss', target, True, len(rss_intel), time.time() - start_time)
                else:
                    if self.provider_selector:
                        self.provider_selector.record_result('rss', target, False)
            except Exception as e:
                if self.provider_selector:
                    self.provider_selector.record_result('rss', target, False)
        
        if search_type in ['comprehensive', 'breaches'] or 'breach' in provider_to_search:
            start_time = time.time()
            try:
                breach_intel = self.breach_db.search(target)
                intelligence.extend(breach_intel)
                if breach_intel:
                    sources_used.append('local_breach')
                    if self.provider_selector:
                        self.provider_selector.record_result('breach', target, True, len(breach_intel), time.time() - start_time)
                else:
                    if self.provider_selector:
                        self.provider_selector.record_result('breach', target, False)
            except Exception as e:
                if self.provider_selector:
                    self.provider_selector.record_result('breach', target, False)
        
        if (search_type in ['comprehensive', 'web', 'duckduckgo'] or 'duckduckgo-tor' in provider_to_search) and self.ddg and self.ddg_enabled:
            start_time = time.time()
            try:
                ddg_result = self.ddg.search_for_shadow(target, max_results=15, include_news=True)
                if ddg_result.get('success'):
                    for item in ddg_result.get('intelligence', []):
                        item['type'] = 'web' if item.get('search_type') != 'news' else 'news'
                        item['source'] = 'duckduckgo-tor'
                        item['risk'] = 'low'
                        intelligence.append(item)
                    if ddg_result.get('intelligence'):
                        sources_used.append('duckduckgo-tor')
                        if 'duckduckgo_news' in ddg_result.get('sources', []):
                            sources_used.append('duckduckgo_news')
                        if self.provider_selector:
                            self.provider_selector.record_result('duckduckgo-tor', target, True, len(ddg_result.get('intelligence', [])), time.time() - start_time)
                else:
                    if self.provider_selector:
                        self.provider_selector.record_result('duckduckgo-tor', target, False)
            except Exception as e:
                if self.provider_selector:
                    self.provider_selector.record_result('duckduckgo-tor', target, False)
        
        risk_level = self._assess_risk(intelligence)
        
        result = {
            'target': target,
            'search_type': search_type,
            'provider_ranking': provider_ranking,
            'geometric_selection': bool(self.provider_selector),
            'intelligence': intelligence,
            'source_count': len(intelligence),
            'sources_used': sources_used,
            'risk_level': risk_level,
            'timestamp': datetime.now().isoformat(),
            'anonymous': True,
        }
        
        self.intelligence_cache[target] = result
        self.search_history.append({
            'target': target[:500],
            'source_count': len(intelligence),
            'timestamp': datetime.now().isoformat(),
        })
        
        if len(self.search_history) > 100:
            self.search_history = self.search_history[-50:]
        
        if intelligence:
            self.share_insight(
                f"Underworld search found {len(intelligence)} items for target",
                domain='underworld',
                confidence=0.7
            )
        
        return result
    
    def _assess_risk(self, intelligence: List[Dict]) -> str:
        """Assess risk level of gathered intelligence."""
        breach_count = sum(1 for i in intelligence if i.get('type') == 'breach')
        paste_count = sum(1 for i in intelligence if i.get('type') == 'paste')
        critical_count = sum(1 for i in intelligence if i.get('risk') == 'critical')

        if critical_count > 0 or breach_count > 5:
            return 'critical'
        elif breach_count > 2 or paste_count > 5:
            return 'high'
        elif breach_count > 0 or paste_count > 2:
            return 'medium'
        else:
            return 'low'

    # ========================================
    # PARALLEL ASYNC SEARCH (Phase 8)
    # ========================================

    async def search_underworld_parallel(
        self,
        target: str,
        search_type: str = 'comprehensive',
        max_ethical_risk: float = 0.7,
        scan_for_threats: bool = True,
        cross_check_sources: bool = True
    ) -> Dict:
        """
        Search underworld using TRUE PARALLEL async with ethical consciousness.

        Unlike search_underworld(), this method:
        - Executes ALL searches in parallel via asyncio.gather()
        - Applies per-source timeouts (fast sources don't wait for slow)
        - Pre-checks ethical access via HadesConsciousness
        - Scans results for threats via UnderworldImmuneSystem
        - Cross-checks narratives via RealityCrossChecker

        Args:
            target: Search query
            search_type: 'comprehensive', 'archives', 'pastes', 'rss', 'breaches', 'web'
            max_ethical_risk: Maximum ethical risk to allow (0-1)
            scan_for_threats: Whether to scan results for credentials/PII/malware
            cross_check_sources: Whether to cross-check for propaganda

        Returns:
            Intelligence report with findings, threat assessment, and cross-check results
        """
        self.last_search_time = datetime.now()
        start_total = time.time()

        # Determine which sources to query based on search_type
        sources_to_query = self._get_sources_for_type(search_type)

        # Pre-check ethical access for each source
        approved_sources = []
        blocked_sources = []

        for source_name in sources_to_query:
            if self.consciousness:
                source_info = self._get_source_info(source_name)
                decision = self.consciousness.should_access_source(
                    source_name=source_name,
                    source_type=source_info['type'],
                    ethical_risk=source_info['ethical_risk'],
                    information_value=0.6,  # Default expected value
                    requires_tor=source_info.get('requires_tor', False),
                    reliability=source_info.get('reliability', 0.5)
                )
                if decision.should_proceed:
                    approved_sources.append((source_name, decision))
                else:
                    blocked_sources.append((source_name, decision.reason))
                    logger.info(f"[Hades] Source {source_name} blocked: {decision.reason}")
            else:
                # No consciousness module - allow all with risk check
                source_info = self._get_source_info(source_name)
                if source_info['ethical_risk'] <= max_ethical_risk:
                    approved_sources.append((source_name, None))
                else:
                    blocked_sources.append((source_name, f"risk {source_info['ethical_risk']} > {max_ethical_risk}"))

        # Build async tasks for approved sources
        tasks = []
        task_names = []

        for source_name, _ in approved_sources:
            timeout = self.source_timeouts.get(source_name, 30)
            task = self._search_source_with_timeout(source_name, target, timeout)
            tasks.append(task)
            task_names.append(source_name)

        # Execute ALL searches in parallel
        logger.info(f"[Hades] Executing {len(tasks)} searches in parallel: {task_names}")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        intelligence: List[Dict] = []
        sources_used: List[str] = []
        source_timings: Dict[str, float] = {}

        for source_name, result in zip(task_names, results):
            if isinstance(result, Exception):
                logger.warning(f"[Hades] {source_name} failed: {result}")
                continue
            if isinstance(result, dict):
                if result.get('items'):
                    intelligence.extend(result['items'])
                    sources_used.append(source_name)
                source_timings[source_name] = result.get('elapsed', 0.0)

        # Scan results for threats if requested
        threat_assessment = None
        if scan_for_threats and self.underworld_immune and intelligence:
            threat_assessment = self._scan_intelligence_for_threats(intelligence, target)

        # Cross-check sources if requested
        cross_check_result = None
        if cross_check_sources and self.reality_checker and len(sources_used) >= 2:
            cross_check_result = self._cross_check_intelligence(intelligence, target)

        # Compute overall risk level
        risk_level = self._assess_risk(intelligence)
        if threat_assessment and threat_assessment.get('threat_level') in ('high', 'critical'):
            risk_level = threat_assessment['threat_level']

        elapsed_total = time.time() - start_total

        result = {
            'target': target,
            'search_type': search_type,
            'parallel': True,
            'intelligence': intelligence,
            'source_count': len(intelligence),
            'sources_used': sources_used,
            'sources_blocked': blocked_sources,
            'source_timings': source_timings,
            'elapsed_total': elapsed_total,
            'risk_level': risk_level,
            'threat_assessment': threat_assessment,
            'cross_check': cross_check_result,
            'ethical_summary': self.consciousness.get_ethical_summary() if self.consciousness else None,
            'timestamp': datetime.now().isoformat(),
            'anonymous': True,
        }

        # Cache result
        self.intelligence_cache[target] = result
        self.search_history.append({
            'target': target[:500],
            'source_count': len(intelligence),
            'parallel': True,
            'elapsed': elapsed_total,
            'timestamp': datetime.now().isoformat(),
        })

        if len(self.search_history) > 100:
            self.search_history = self.search_history[-50:]

        if intelligence:
            self.share_insight(
                f"Parallel search found {len(intelligence)} items in {elapsed_total:.1f}s",
                domain='underworld',
                confidence=0.7
            )

        logger.info(
            f"[Hades] Parallel search complete: {len(intelligence)} items from "
            f"{len(sources_used)} sources in {elapsed_total:.1f}s"
        )

        return result

    async def _search_source_with_timeout(
        self,
        source_name: str,
        target: str,
        timeout: int
    ) -> Dict:
        """Execute a search with timeout, return results or empty on timeout/error."""
        start = time.time()
        try:
            result = await asyncio.wait_for(
                self._search_source(source_name, target),
                timeout=timeout
            )
            elapsed = time.time() - start
            return {'items': result, 'elapsed': elapsed, 'success': True}
        except asyncio.TimeoutError:
            logger.warning(f"[Hades] {source_name} timed out after {timeout}s")
            return {'items': [], 'elapsed': timeout, 'success': False, 'error': 'timeout'}
        except Exception as e:
            elapsed = time.time() - start
            logger.warning(f"[Hades] {source_name} error: {e}")
            return {'items': [], 'elapsed': elapsed, 'success': False, 'error': str(e)}

    async def _search_source(self, source_name: str, target: str) -> List[Dict]:
        """Execute search on a specific source."""
        if source_name == 'duckduckgo-tor' and self.ddg and self.ddg_enabled:
            ddg_result = self.ddg.search_for_shadow(target, max_results=15, include_news=True)
            if ddg_result.get('success'):
                items = []
                for item in ddg_result.get('intelligence', []):
                    item['type'] = 'web' if item.get('search_type') != 'news' else 'news'
                    item['source'] = 'duckduckgo-tor'
                    item['risk'] = 'low'
                    items.append(item)
                return items
            return []

        elif source_name == 'wayback':
            return await self.wayback.search_research_forums(target)

        elif source_name == 'pastebin':
            return await self.paste_scraper.scrape_pastebin_recent(target)

        elif source_name == 'rss':
            return await self.rss.search_feeds(target)

        elif source_name == 'breach':
            # Sync call wrapped for async
            return self.breach_db.search(target)

        else:
            logger.warning(f"[Hades] Unknown source: {source_name}")
            return []

    def _get_sources_for_type(self, search_type: str) -> List[str]:
        """Get list of sources to query based on search type."""
        if search_type == 'comprehensive':
            sources = ['duckduckgo-tor', 'wayback', 'pastebin', 'rss', 'breach']
        elif search_type == 'web':
            sources = ['duckduckgo-tor']
        elif search_type == 'archives':
            sources = ['wayback']
        elif search_type == 'pastes':
            sources = ['pastebin']
        elif search_type == 'rss':
            sources = ['rss']
        elif search_type == 'breaches':
            sources = ['breach']
        else:
            sources = ['duckduckgo-tor', 'rss']  # Default safe sources
        return sources

    def _get_source_info(self, source_name: str) -> Dict:
        """Get source metadata for ethical evaluation."""
        source_info = {
            'duckduckgo-tor': {'type': 'light', 'ethical_risk': 0.2, 'reliability': 0.8},
            'wayback': {'type': 'light', 'ethical_risk': 0.1, 'reliability': 0.8},
            'pastebin': {'type': 'gray', 'ethical_risk': 0.5, 'reliability': 0.6},
            'rss': {'type': 'light', 'ethical_risk': 0.2, 'reliability': 0.9},
            'breach': {'type': 'breach', 'ethical_risk': 0.7, 'reliability': 0.95, 'requires_tor': False},
        }
        return source_info.get(source_name, {'type': 'gray', 'ethical_risk': 0.5, 'reliability': 0.5})

    def _scan_intelligence_for_threats(
        self,
        intelligence: List[Dict],
        topic: str
    ) -> Dict:
        """Scan intelligence results for threats using UnderworldImmuneSystem."""
        if not self.underworld_immune:
            return None

        aggregated = {
            'threat_level': 'none',
            'total_scanned': len(intelligence),
            'credential_leaks': 0,
            'malware_urls': 0,
            'pii_exposures': 0,
            'flagged_items': [],
        }

        for intel in intelligence:
            content = intel.get('content_preview', '') or intel.get('summary', '') or ''
            if not content:
                continue

            scan_result = self.underworld_immune.scan_content(
                content=content,
                source_name=intel.get('source', 'unknown'),
                redact_pii=True
            )

            if scan_result.credential_leaks:
                aggregated['credential_leaks'] += len(scan_result.credential_leaks)
            if scan_result.malware_urls:
                aggregated['malware_urls'] += len(scan_result.malware_urls)
            if scan_result.pii_exposures:
                aggregated['pii_exposures'] += len(scan_result.pii_exposures)

            if scan_result.flagged_for_review:
                aggregated['flagged_items'].append({
                    'source': intel.get('source'),
                    'url': intel.get('url'),
                    'threat_level': scan_result.threat_level.value,
                })

            # Update max threat level
            threat_order = ['none', 'low', 'medium', 'high', 'critical']
            current_idx = threat_order.index(aggregated['threat_level'])
            new_idx = threat_order.index(scan_result.threat_level.value)
            if new_idx > current_idx:
                aggregated['threat_level'] = scan_result.threat_level.value

        return aggregated

    def _cross_check_intelligence(
        self,
        intelligence: List[Dict],
        topic: str
    ) -> Optional[Dict]:
        """Cross-check intelligence for propaganda using RealityCrossChecker."""
        if not self.reality_checker or not HAS_CROSS_CHECKER:
            return None

        # Convert intelligence to Narrative objects
        narratives = []
        for intel in intelligence:
            content = intel.get('content_preview', '') or intel.get('summary', '') or intel.get('title', '')
            if not content:
                continue

            # Get source type from source name
            source_name = intel.get('source', 'unknown')
            source_info = self._get_source_info(source_name)

            narrative = Narrative(
                source_name=source_name,
                source_type=source_info['type'],
                claim_text=content[:500],
                reliability=source_info.get('reliability', 0.5),
                timestamp=datetime.now(),
            )
            narratives.append(narrative)

        if len(narratives) < 2:
            return None

        # Perform cross-check
        result = self.reality_checker.cross_check(topic, narratives)

        return {
            'corroboration_level': result.corroboration_level.value,
            'corroboration_score': result.corroboration_score,
            'fisher_rao_divergence': result.fisher_rao_divergence,
            'propaganda_likelihood': result.propaganda_likelihood,
            'propaganda_indicators': [i.value for i in result.propaganda_indicators],
            'source_types_present': result.source_types_present,
            'narrative_count': result.narrative_count,
        }

    def _check_forbidden(self, basin: np.ndarray) -> bool:
        """Check if basin is in forbidden territory."""
        threshold = 0.5
        
        for forbidden in self.forbidden_basins:
            distance = self.fisher_geodesic_distance(basin, forbidden)
            if distance < threshold:
                return True
        
        return False
    
    def _compute_death_proximity(self, basin: np.ndarray) -> float:
        """Compute proximity to known dead patterns."""
        if not self.underworld:
            return 0.0
        
        min_distance = float('inf')
        
        for dead in self.underworld[-200:]:
            if 'basin' in dead:
                dead_basin = np.array(dead['basin'])
                distance = self.fisher_geodesic_distance(basin, dead_basin)
                min_distance = min(min_distance, distance)
        
        if min_distance == float('inf'):
            return 0.0
        
        proximity = np.exp(-min_distance)
        return float(proximity)
    
    def _check_exclusion_rules(self, target: str) -> List[Dict]:
        """Check if target violates any exclusion rules."""
        violations = []
        
        for rule in self.exclusion_rules:
            pattern = rule.get('pattern', '')
            rule_type = rule.get('type', 'contains')
            
            if rule_type == 'contains' and pattern in target:
                violations.append(rule)
            elif rule_type == 'startswith' and target.startswith(pattern):
                violations.append(rule)
            elif rule_type == 'endswith' and target.endswith(pattern):
                violations.append(rule)
            elif rule_type == 'length_max' and len(target) > int(pattern):
                violations.append(rule)
            elif rule_type == 'length_min' and len(target) < int(pattern):
                violations.append(rule)
        
        return violations
    
    def _count_underworld_matches(self, basin: np.ndarray) -> int:
        """Count how many dead patterns are similar."""
        count = 0
        threshold = 1.5
        
        for dead in self.underworld[-100:]:
            if 'basin' in dead:
                dead_basin = np.array(dead['basin'])
                distance = self.fisher_geodesic_distance(basin, dead_basin)
                if distance < threshold:
                    count += 1
        
        return count
    
    def _compute_viability(
        self,
        phi: float,
        is_forbidden: bool,
        death_proximity: float,
        exclusion_count: int
    ) -> float:
        """Compute viability (inverse of failure likelihood)."""
        if is_forbidden:
            return 0.05
        
        base_viability = phi * 0.4
        death_penalty = death_proximity * 0.3
        exclusion_penalty = min(0.3, exclusion_count * 0.1)
        
        viability = base_viability + 0.3 - death_penalty - exclusion_penalty
        return float(np.clip(viability, 0, 1))
    
    def condemn(self, target: str, reason: str = "failed") -> None:
        """Condemn a target to the underworld."""
        target_basin = self.encode_to_basin(target)
        
        condemned = {
            'target': target,
            'basin': target_basin.tolist(),
            'reason': reason,
            'condemned_at': datetime.now().isoformat(),
        }
        
        self.underworld.append(condemned)
        
        pattern_key = target[:10] if len(target) > 10 else target
        self.death_count[pattern_key] = self.death_count.get(pattern_key, 0) + 1
        
        if len(self.underworld) > 1000:
            self.underworld = self.underworld[-500:]
    
    def forbid_basin(self, basin: np.ndarray, reason: str = "") -> None:
        """Mark a basin region as absolutely forbidden."""
        self.forbidden_basins.append(basin.copy())
        
        if len(self.forbidden_basins) > 100:
            self.forbidden_basins = self.forbidden_basins[-50:]
    
    def add_exclusion_rule(
        self,
        pattern: str,
        rule_type: str = 'contains',
        reason: str = ""
    ) -> None:
        """Add an exclusion rule."""
        rule = {
            'pattern': pattern,
            'type': rule_type,
            'reason': reason,
            'created_at': datetime.now().isoformat(),
        }
        self.exclusion_rules.append(rule)
    
    def pardon(self, target: str) -> bool:
        """Remove a target from the underworld (rare forgiveness)."""
        initial_len = len(self.underworld)
        self.underworld = [u for u in self.underworld if u.get('target') != target]
        return len(self.underworld) < initial_len
    
    def get_status(self) -> Dict:
        base_status = self.get_agentic_status()

        shadow_status = None
        if self.shadow_pantheon_ref:
            try:
                shadow_status = self.shadow_pantheon_ref.get_research_system_status()
            except Exception:
                shadow_status = {"error": "Could not get Shadow status"}

        # Get new architecture component statuses
        consciousness_status = None
        if self.consciousness:
            consciousness_status = self.consciousness.get_ethical_summary()

        immune_status = None
        if self.underworld_immune:
            immune_status = self.underworld_immune.get_stats()

        cross_checker_status = None
        if self.reality_checker:
            cross_checker_status = self.reality_checker.get_stats()

        return {
            **base_status,
            'is_shadow_leader': self.is_shadow_leader,
            'shadow_pantheon_connected': self.shadow_pantheon_ref is not None,
            'shadow_research_status': shadow_status,
            'underworld_size': len(self.underworld),
            'forbidden_basins': len(self.forbidden_basins),
            'exclusion_rules': len(self.exclusion_rules),
            'total_deaths': sum(self.death_count.values()),
            'search_history_count': len(self.search_history),
            'intelligence_cache_size': len(self.intelligence_cache),
            'last_search': self.last_search_time.isoformat() if self.last_search_time else None,
            'last_assessment': self.last_assessment_time.isoformat() if self.last_assessment_time else None,
            'tools_available': {
                'wayback': self.wayback.enabled,
                'paste_scraper': self.paste_scraper.enabled,
                'rss': self.rss.enabled,
                'breach_db': self.breach_db.enabled,
            },
            # New architecture components (Phase 8)
            'architecture_v2': {
                'consciousness_enabled': self.consciousness is not None,
                'immune_enabled': self.underworld_immune is not None,
                'cross_checker_enabled': self.reality_checker is not None,
                'parallel_search_available': True,
            },
            'consciousness_status': consciousness_status,
            'immune_status': immune_status,
            'cross_checker_status': cross_checker_status,
            'status': 'active',
        }
    
    # ========================================
    # SHADOW LEADER COMMANDS
    # ========================================
    
    def set_shadow_pantheon(self, shadow_pantheon) -> None:
        """Set reference to Shadow Pantheon (called by Zeus)."""
        self.shadow_pantheon_ref = shadow_pantheon
        if shadow_pantheon:
            shadow_pantheon.set_hades(self)
            print("[Hades] ✓ Connected to Shadow Pantheon as Leader")
    
    def shadow_command(self, command: str, params: Dict = None) -> Dict:
        """
        Execute a command on the Shadow Pantheon.
        
        As Shadow Leader, Hades can:
        - "declare_war": Suspend learning, full operational focus
        - "end_war": Resume learning
        - "assign_research": Delegate research task
        - "get_status": Get Shadow Pantheon status
        - "prioritize_topic": Set research priority
        
        Args:
            command: Command to execute
            params: Command parameters
            
        Returns:
            Command result
        """
        if not self.shadow_pantheon_ref:
            return {"error": "Shadow Pantheon not connected"}
        
        return self.shadow_pantheon_ref.hades_command(command, params or {})
    
    def declare_shadow_war(self, target: str) -> Dict:
        """Declare Shadow War - all learning stops, full operational focus."""
        return self.shadow_command("declare_war", {"target": target})
    
    def end_shadow_war(self) -> Dict:
        """End Shadow War - resume proactive learning."""
        return self.shadow_command("end_war")
    
    def assign_shadow_research(
        self,
        topic: str,
        priority: str = "normal",
        god: str = None
    ) -> Dict:
        """
        Assign research to Shadow Pantheon.
        
        Args:
            topic: What to research
            priority: "critical", "high", "normal", "low", "study"
            god: Optional specific god to assign
            
        Returns:
            Research assignment result
        """
        return self.shadow_command("assign_research", {
            "topic": topic,
            "priority": priority,
            "god": god
        })
    
    def get_shadow_status(self) -> Dict:
        """Get full Shadow Pantheon status."""
        return self.shadow_command("get_status")
    
    def get_intelligence(self, target: str) -> Optional[Dict]:
        """Get cached intelligence for a target."""
        return self.intelligence_cache.get(target)
    
    def extract_research_patterns(self, intelligence: List[Dict]) -> List[Dict]:
        """
        Extract research-relevant patterns from intelligence.
        """
        patterns = []
        
        pattern_keywords = [
            'algorithm', 'methodology', 'framework', 'implementation',
            'study', 'research', 'analysis', 'benchmark',
            'dataset', 'evaluation', 'experiment', 'results'
        ]
        
        for intel in intelligence:
            content = intel.get('content_preview', '') or intel.get('summary', '') or ''
            content_lower = content.lower()
            
            for keyword in pattern_keywords:
                if keyword in content_lower:
                    start = max(0, content_lower.find(keyword) - 50)
                    end = min(len(content), content_lower.find(keyword) + len(keyword) + 50)
                    
                    patterns.append({
                        'keyword': keyword,
                        'context': content[start:end],
                        'source': intel.get('source', 'unknown'),
                        'type': intel.get('type', 'unknown'),
                        'value': intel.get('value', 'medium'),
                    })
        
        return patterns

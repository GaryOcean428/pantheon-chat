/**
 * GOOGLE WEB SEARCH ADAPTER
 * 
 * FREE web search using Google search results (NO API KEYS REQUIRED)
 * Based on https://github.com/pskill9/web-search
 * 
 * PARADIGM: Same geometric discovery interface as SearXNG/Tavily
 * We MEASURE what exists at specific coordinates in the external block universe.
 */

import axios from 'axios';
import * as cheerio from 'cheerio';
import { fisherCoordDistance } from '../qig-universal';
import { tps, TemporalPositioningSystem } from './temporal-positioning-system';
import {
  type BlockUniverseMap,
  type GeometricDiscovery,
  type GeometricQuery,
  type RawDiscovery,
} from './types';
import { isCurriculumOnlyEnabled } from '../lib/curriculum-mode';

interface GoogleSearchResult {
  title: string;
  url: string;
  description: string;
}

interface SearchResponse {
  results: GoogleSearchResult[];
  status: 'success' | 'error' | 'rate_limited';
  error?: string;
}

/**
 * Google Web Search Geometric Adapter
 * 
 * Free web search by scraping Google results - no API keys required
 * Drops in alongside SearXNG with same interface
 */
export class GoogleWebSearchAdapter {
  private tps: TemporalPositioningSystem;
  private timeout: number = 15000;
  private userAgent: string = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36';
  private lastRequestTime: number = 0;
  private minRequestInterval: number = 2000;
  
  constructor() {
    this.tps = tps;
    
    // CURRICULUM-ONLY MODE: Skip initialization to prevent external connections
    if (isCurriculumOnlyEnabled()) {
      console.log('[GoogleWebSearch] Skipped initialization (curriculum-only mode)');
      return;
    }
    
    console.log('[GoogleWebSearch] Initialized FREE Google web search adapter');
    console.log('[GoogleWebSearch] NO API KEYS REQUIRED');
  }
  
  /**
   * Respect rate limiting to avoid being blocked
   */
  private async respectRateLimit(): Promise<void> {
    const now = Date.now();
    const elapsed = now - this.lastRequestTime;
    if (elapsed < this.minRequestInterval) {
      const delay = this.minRequestInterval - elapsed;
      await new Promise(resolve => setTimeout(resolve, delay));
    }
    this.lastRequestTime = Date.now();
  }
  
  /**
   * Perform raw Google search and extract results
   */
  private async performSearch(query: string, limit: number = 10): Promise<SearchResponse> {
    await this.respectRateLimit();
    
    try {
      const response = await axios.get('https://www.google.com/search', {
        params: { q: query },
        headers: {
          'User-Agent': this.userAgent,
          'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
          'Accept-Language': 'en-US,en;q=0.5',
          'Accept-Encoding': 'gzip, deflate',
          'Connection': 'keep-alive',
          'Upgrade-Insecure-Requests': '1',
        },
        timeout: this.timeout,
      });

      const $ = cheerio.load(response.data);
      const results: GoogleSearchResult[] = [];

      $('div.g').each((i, element) => {
        if (results.length >= limit) return false;

        const titleElement = $(element).find('h3');
        const linkElement = $(element).find('a');
        const snippetElement = $(element).find('.VwiC3b');

        if (titleElement.length && linkElement.length) {
          const url = linkElement.attr('href');
          if (url && url.startsWith('http')) {
            results.push({
              title: titleElement.text().trim(),
              url: url,
              description: snippetElement.text().trim() || '',
            });
          }
        }
      });

      return { results, status: 'success' };
    } catch (error: any) {
      console.error('[GoogleWebSearch] Search error:', error.message);
      const status = error.response?.status === 429 ? 'rate_limited' : 'error';
      return { results: [], status, error: error.message };
    }
  }
  
  /**
   * Discover content at specific 68D coordinates
   * Same interface as SearXNGGeometricAdapter
   */
  async discoverAtCoordinates(
    targetCoords: BlockUniverseMap,
    radius: number = 2.0
  ): Promise<GeometricDiscovery[]> {
    const query = this.coordsToQuery(targetCoords);
    
    console.log(`[GoogleWebSearch] Discovering at coordinates:`);
    console.log(`  Era: ${this.tps.classifyEra(targetCoords.spacetime.t)}`);
    console.log(`  Query: "${query.text}"`);
    
    const rawResults = await this.search(query);
    
    if (rawResults.length === 0) {
      console.log(`[GoogleWebSearch] No discoveries found`);
      return [];
    }
    
    const discoveries: GeometricDiscovery[] = [];
    
    for (const result of rawResults) {
      const resultCoords = this.tps.locateInBlockUniverse(
        result.content,
        result.url
      );
      
      const distance = fisherCoordDistance(
        targetCoords.cultural,
        resultCoords.cultural
      );
      
      if (distance < radius) {
        const patterns = this.extractPatterns(result.content);
        const pastLightCone = this.tps.getPastLightCone(resultCoords);
        
        discoveries.push({
          content: result.content,
          url: result.url,
          source: 'google-web-search',
          coords: resultCoords,
          distance,
          phi: resultCoords.phi,
          patterns,
          causalChain: pastLightCone,
          entropyReduction: this.computeEntropyReduction(distance, patterns.length)
        });
      }
    }
    
    discoveries.sort((a, b) => a.distance - b.distance);
    
    console.log(`[GoogleWebSearch] Found ${discoveries.length} geometric discoveries`);
    
    return discoveries;
  }
  
  /**
   * Search with geometric query
   */
  async search(query: GeometricQuery): Promise<RawDiscovery[]> {
    // CURRICULUM-ONLY MODE: Block external web searches
    if (isCurriculumOnlyEnabled()) {
      console.log(`[GoogleWebSearch] Geometric search blocked by curriculum-only mode`);
      return [];
    }
    
    const searchQuery = this.buildSearchQuery(query);
    const response = await this.performSearch(searchQuery, query.maxResults || 10);
    
    return response.results.map(result => ({
      url: result.url,
      title: result.title,
      content: result.description,
      score: 0.5,
      publishedDate: undefined,
    }));
  }
  
  /**
   * Simple web search - returns raw results with status
   */
  async simpleSearch(query: string, limit: number = 5): Promise<{ results: GoogleSearchResult[]; status: string; error?: string }> {
    // CURRICULUM-ONLY MODE: Block external web searches
    if (isCurriculumOnlyEnabled()) {
      console.log(`[GoogleWebSearch] Blocked by curriculum-only mode: "${query}"`);
      return {
        results: [],
        status: 'curriculum_only_blocked',
        error: 'External web search blocked in curriculum-only mode'
      };
    }
    
    console.log(`[GoogleWebSearch] Simple search: "${query}" (limit: ${limit})`);
    const response = await this.performSearch(query, Math.min(limit, 10));
    console.log(`[GoogleWebSearch] Found ${response.results.length} results (status: ${response.status})`);
    return response;
  }
  
  /**
   * Convert block universe coordinates to search query
   */
  private coordsToQuery(coords: BlockUniverseMap): GeometricQuery {
    const era = this.tps.classifyEra(coords.spacetime.t);
    let queryText = '';
    
    switch (era) {
      case 'genesis':
        queryText = 'early cryptocurrency technology concepts';
        break;
      case 'early_adoption':
        queryText = 'decentralized ledger technology innovation';
        break;
      case 'modern':
      default:
        queryText = 'blockchain research knowledge discovery';
        break;
    }
    
    return {
      text: queryText,
      targetCoords: coords,
      maxResults: 10
    };
  }
  
  /**
   * Build search query string from geometric query
   */
  private buildSearchQuery(query: GeometricQuery): string {
    let searchQuery = query.text;
    
    if (query.targetCoords?.spacetime?.t) {
      const era = this.tps.classifyEra(query.targetCoords.spacetime.t);
      const eraContext: Record<string, string> = {
        'genesis': 'cryptography distributed systems',
        'early_adoption': 'peer-to-peer networks',
        'modern': 'machine learning knowledge graphs',
      };
      const context = eraContext[era] || eraContext['modern'];
      searchQuery = `${searchQuery} ${context}`;
    }
    
    return searchQuery;
  }
  
  /**
   * Extract research patterns from content
   */
  private extractPatterns(content: string): string[] {
    const patterns: string[] = [];
    const lower = content.toLowerCase();
    
    const researchPatterns = [
      /arxiv\.org\/abs\/\d+\.\d+/gi,
      /doi:\s*[\d.\/\-a-z]+/gi,
      /\[[\d,\s]+\]/g,
      /et al\./gi,
      /github\.com\/[\w\-]+\/[\w\-]+/gi,
    ];
    
    for (const pattern of researchPatterns) {
      const matches = content.match(pattern);
      if (matches) {
        patterns.push(...matches.slice(0, 3));
      }
    }
    
    const conceptKeywords = [
      'neural network', 'transformer', 'attention mechanism',
      'knowledge graph', 'semantic', 'basin coordinates',
      'geometric', 'manifold', 'information theory',
    ];
    
    for (const keyword of conceptKeywords) {
      if (lower.includes(keyword)) {
        patterns.push(keyword);
      }
    }
    
    return [...new Set(patterns)].slice(0, 10);
  }
  
  /**
   * Compute entropy reduction based on geometric relevance
   */
  private computeEntropyReduction(distance: number, patternCount: number): number {
    const distanceFactor = Math.exp(-distance);
    const patternFactor = 1 + (patternCount * 0.1);
    return distanceFactor * patternFactor * 0.5;
  }
}

export const googleWebSearchAdapter = new GoogleWebSearchAdapter();

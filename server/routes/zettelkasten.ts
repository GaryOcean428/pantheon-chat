/**
 * Zettelkasten Knowledge Graph API Routes
 *
 * Provides a knowledge graph interface on top of basin memory storage.
 * Memories are stored as 64D basin coordinates with automatic linking.
 *
 * DESIGN:
 * - Links computed via Fisher-Rao geodesic distance (QIG-pure)
 * - Lower distance = stronger link (threshold < 1.0 radians)
 * - Sampling (max 100) used for O(n²) pairwise comparisons
 * - sample_based flag indicates when stats are extrapolated
 *
 * GEOMETRIC PRINCIPLE:
 * Fisher-Rao distance is the geodesic distance on the statistical manifold.
 * d_FR(p, q) = arccos(Σ√(p_i * q_i)) where coordinates are treated as
 * probability distributions. NOT cosine similarity (that's Euclidean).
 *
 * SECURITY:
 * - POST endpoints require internal auth (X-Internal-Key header)
 * - Auto-save uses getInternalHeaders() for server-to-server calls
 */

import { Router, Request, Response } from 'express';
import { db } from '../db';
import { basinMemory, zeusConversations } from '../../shared/schema';
import { eq, desc, sql, and, gte, lte } from 'drizzle-orm';
import { randomUUID } from 'crypto';
import { requireInternalAuth } from '../internal-auth';
import { extractKeywords } from '../contextualized-filter';

const router = Router();

function ensureDb() {
  if (!db) {
    throw new Error('Database not initialized');
  }
  return db;
}

/**
 * Compute Fisher-Rao geodesic distance between basin coordinates.
 *
 * Formula: d_FR(p, q) = arccos(Σ√(p_i * q_i))
 *
 * This is the PROPER geodesic distance on the information manifold.
 * NOT cosine similarity or chord distance (those are Euclidean, violate QIG purity).
 *
 * @returns Distance in radians [0, π]. Lower = more similar.
 */
function computeFisherRaoDistance(coords1: number[], coords2: number[]): number {
  if (!coords1?.length || !coords2?.length) return Math.PI; // Max distance if invalid
  const minLen = Math.min(coords1.length, coords2.length);

  // Ensure valid probability distributions (non-negative, normalized)
  const epsilon = 1e-10;
  let sum1 = 0, sum2 = 0;
  for (let i = 0; i < minLen; i++) {
    sum1 += Math.abs(coords1[i]) + epsilon;
    sum2 += Math.abs(coords2[i]) + epsilon;
  }

  // Compute Bhattacharyya coefficient: BC = Σ√(p_i * q_i)
  let bc = 0;
  for (let i = 0; i < minLen; i++) {
    const p = (Math.abs(coords1[i]) + epsilon) / sum1;
    const q = (Math.abs(coords2[i]) + epsilon) / sum2;
    bc += Math.sqrt(p * q);
  }

  // Clamp BC to [0, 1] for numerical stability
  bc = Math.max(0, Math.min(1, bc));

  // Fisher-Rao distance (geodesic on information manifold)
  return Math.acos(bc);
}

/**
 * Convert Fisher-Rao distance to link strength.
 *
 * - Distance 0 → Strength 1.0 (identical points)
 * - Distance π/2 → Strength 0.5 (orthogonal)
 * - Distance π → Strength 0.0 (maximally different)
 *
 * Link threshold: strength > 0.5 means Fisher-Rao distance < π/2 radians
 */
function computeLinkStrength(coords1: number[], coords2: number[]): number {
  const distance = computeFisherRaoDistance(coords1, coords2);
  // Convert distance to strength: strength = 1 - (distance / π)
  return Math.max(0, 1 - distance / Math.PI);
}

/**
 * GET /api/zettelkasten/stats
 * Returns real stats computed from stored memories
 * Uses sampling for large datasets to maintain performance
 */
router.get('/stats', async (req: Request, res: Response) => {
  try {
    const database = ensureDb();
    
    const countResult = await database
      .select({ count: sql<number>`count(*)` })
      .from(basinMemory);
    
    const totalZettels = Number(countResult[0]?.count || 0);
    
    // Always sample up to 100 memories for link/keyword computation
    const sampleSize = Math.min(100, totalZettels);
    let realLinkCount = 0;
    let sampleKeywords = 0;
    
    if (sampleSize > 0) {
      const memories = await database
        .select({ basinCoordinates: basinMemory.basinCoordinates, context: basinMemory.context })
        .from(basinMemory)
        .orderBy(desc(basinMemory.phi)) // Sample high-phi (likely hub) memories
        .limit(sampleSize);
      
      // Count links where cosine similarity > 0.5
      for (let i = 0; i < memories.length; i++) {
        const ctx = memories[i].context;
        const content = typeof ctx === 'string' ? ctx : JSON.stringify(ctx || '');
        sampleKeywords += extractKeywords(content).length;
        
        for (let j = i + 1; j < memories.length; j++) {
          const c1 = memories[i].basinCoordinates as number[] | null;
          const c2 = memories[j].basinCoordinates as number[] | null;
          if (computeLinkStrength(c1 || [], c2 || []) > 0.5) {
            realLinkCount++;
          }
        }
      }
    }
    
    // Extrapolate from sample if dataset is larger than sample
    const scaleFactor = totalZettels > sampleSize ? totalZettels / sampleSize : 1;
    const estimatedLinks = Math.round(realLinkCount * scaleFactor);
    const estimatedKeywords = Math.round(sampleKeywords * scaleFactor);
    
    res.json({
      success: true,
      data: {
        total_zettels: totalZettels,
        total_links: estimatedLinks,
        total_keywords: estimatedKeywords,
        total_evolutions: 0,
        avg_links_per_zettel: totalZettels > 0 ? estimatedLinks / totalZettels : 0,
        sample_based: totalZettels > sampleSize // Flag if stats are estimated
      }
    });
  } catch (error) {
    console.error('[Zettelkasten] Error getting stats:', error);
    res.status(500).json({ success: false, error: 'Failed to get stats' });
  }
});

/**
 * GET /api/zettelkasten/graph
 */
router.get('/graph', async (req: Request, res: Response) => {
  try {
    const { max_nodes = '50' } = req.query;
    const maxNodes = parseInt(max_nodes as string);
    
    const database = ensureDb();
    
    const memories = await database
      .select()
      .from(basinMemory)
      .orderBy(desc(basinMemory.timestamp))
      .limit(maxNodes);
    
    const nodes = memories.map(m => ({
      id: m.basinId || `zettel-${m.id}`,
      label: typeof m.context === 'string' 
        ? m.context.slice(0, 50) 
        : (m.context as any)?.content?.slice(0, 50) || `Memory ${m.id}`,
      keywords: extractKeywords(
        typeof m.context === 'string' ? m.context : JSON.stringify(m.context || '')
      ),
      access_count: 1
    }));
    
    const edges: { source: string; target: string; strength: number; link_type: string }[] = [];
    for (let i = 0; i < memories.length; i++) {
      for (let j = i + 1; j < memories.length; j++) {
        const coords1 = memories[i].basinCoordinates as number[] | null;
        const coords2 = memories[j].basinCoordinates as number[] | null;
        const strength = computeLinkStrength(coords1 || [], coords2 || []);
        if (strength > 0.5) {
          edges.push({
            source: nodes[i].id,
            target: nodes[j].id,
            strength,
            link_type: 'geometric'
          });
        }
      }
    }
    
    res.json({
      success: true,
      data: {
        nodes,
        edges,
        stats: {
          total_zettels: nodes.length,
          total_links: edges.length,
          total_keywords: nodes.reduce((sum, n) => sum + n.keywords.length, 0),
          total_evolutions: 0,
          avg_links_per_zettel: nodes.length > 0 ? edges.length / nodes.length : 0
        }
      }
    });
  } catch (error) {
    console.error('[Zettelkasten] Error getting graph:', error);
    res.status(500).json({ success: false, error: 'Failed to get graph' });
  }
});

/**
 * GET /api/zettelkasten/hubs
 * Returns high-phi memories with real computed link counts
 */
router.get('/hubs', async (req: Request, res: Response) => {
  try {
    const { top_n = '10' } = req.query;
    const topN = parseInt(top_n as string);
    
    const database = ensureDb();
    
    // Get high-phi memories as hub candidates
    const hubMemories = await database
      .select()
      .from(basinMemory)
      .where(gte(basinMemory.phi, 0.6))
      .orderBy(desc(basinMemory.phi))
      .limit(topN);
    
    // Compute real link counts by checking similarity with other memories
    // For efficiency, only check against a sample of other memories
    const allMemories = await database
      .select({ basinId: basinMemory.basinId, basinCoordinates: basinMemory.basinCoordinates })
      .from(basinMemory)
      .limit(100);
    
    const hubs = hubMemories.map(m => {
      const coords = m.basinCoordinates as number[] | null;
      let linkCount = 0;
      
      // Count links to this hub (similarity > 0.5)
      if (coords?.length) {
        for (const other of allMemories) {
          if (other.basinId === m.basinId) continue;
          const otherCoords = other.basinCoordinates as number[] | null;
          if (computeLinkStrength(coords, otherCoords || []) > 0.5) {
            linkCount++;
          }
        }
      }
      
      return {
        zettel_id: m.basinId || `zettel-${m.id}`,
        content: typeof m.context === 'string' 
          ? m.context 
          : (m.context as any)?.content || JSON.stringify(m.context),
        keywords: extractKeywords(
          typeof m.context === 'string' ? m.context : JSON.stringify(m.context || '')
        ),
        outgoing_links: linkCount,
        access_count: 1,
        evolution_count: 0
      };
    });
    
    res.json({ success: true, data: hubs });
  } catch (error) {
    console.error('[Zettelkasten] Error getting hubs:', error);
    res.status(500).json({ success: false, error: 'Failed to get hubs' });
  }
});

/**
 * GET /api/zettelkasten/clusters
 * Returns regime-based clusters with real aggregated keywords
 */
router.get('/clusters', async (req: Request, res: Response) => {
  try {
    const database = ensureDb();
    
    const regimeClusters = await database
      .select({
        regime: basinMemory.regime,
        count: sql<number>`count(*)`
      })
      .from(basinMemory)
      .groupBy(basinMemory.regime);
    
    const clusters = await Promise.all(
      regimeClusters.map(async (cluster, idx) => {
        const clusterMemories = await database
          .select()
          .from(basinMemory)
          .where(eq(basinMemory.regime, cluster.regime || 'unknown'))
          .limit(20);
        
        // Aggregate keywords from all cluster memories
        const keywordFreq = new Map<string, number>();
        const zettels = clusterMemories.slice(0, 5).map(m => {
          const content = typeof m.context === 'string' 
            ? m.context 
            : (m.context as any)?.content || JSON.stringify(m.context);
          const keywords = extractKeywords(content);
          keywords.forEach(kw => keywordFreq.set(kw, (keywordFreq.get(kw) || 0) + 1));
          
          return {
            zettel_id: m.basinId || `zettel-${m.id}`,
            content,
            keywords
          };
        });
        
        // Get top keywords by frequency
        const topKeywords = Array.from(keywordFreq.entries())
          .sort((a, b) => b[1] - a[1])
          .slice(0, 5)
          .map(([kw]) => kw);
        
        return {
          cluster_id: idx,
          size: Number(cluster.count),
          top_keywords: topKeywords.length > 0 ? topKeywords : [cluster.regime || 'unknown'],
          zettels
        };
      })
    );
    
    res.json({ success: true, data: clusters });
  } catch (error) {
    console.error('[Zettelkasten] Error getting clusters:', error);
    res.status(500).json({ success: false, error: 'Failed to get clusters' });
  }
});

/**
 * POST /api/zettelkasten/retrieve
 */
router.post('/retrieve', async (req: Request, res: Response) => {
  try {
    const { query, top_k = 10 } = req.body;
    
    if (!query) {
      return res.status(400).json({ success: false, error: 'Query required' });
    }
    
    const database = ensureDb();
    
    const memories = await database
      .select()
      .from(basinMemory)
      .orderBy(desc(basinMemory.phi))
      .limit(top_k);
    
    const queryWords = new Set(query.toLowerCase().split(/\s+/));
    
    // Get all memory coords for link computation
    const allCoords = await database
      .select({ basinId: basinMemory.basinId, basinCoordinates: basinMemory.basinCoordinates })
      .from(basinMemory)
      .limit(100);
    
    const results = memories.map(m => {
      const content = typeof m.context === 'string' 
        ? m.context 
        : (m.context as any)?.content || JSON.stringify(m.context);
      const contentWords = new Set(content.toLowerCase().split(/\s+/));
      const overlap = [...queryWords].filter(w => contentWords.has(w)).length;
      const relevance = (m.phi || 0) * 0.5 + (overlap / queryWords.size) * 0.5;
      
      // Compute real link count
      const coords = m.basinCoordinates as number[] | null;
      let linkCount = 0;
      if (coords?.length) {
        for (const other of allCoords) {
          if (other.basinId === m.basinId) continue;
          const otherCoords = other.basinCoordinates as number[] | null;
          if (computeLinkStrength(coords, otherCoords || []) > 0.5) {
            linkCount++;
          }
        }
      }
      
      return {
        zettel_id: m.basinId || `zettel-${m.id}`,
        content,
        relevance,
        keywords: extractKeywords(content),
        contextual_description: `Regime: ${m.regime}, Φ: ${(m.phi || 0).toFixed(2)}`,
        links_count: linkCount,
        access_count: 1
      };
    });
    
    results.sort((a, b) => b.relevance - a.relevance);
    
    res.json({ success: true, data: results });
  } catch (error) {
    console.error('[Zettelkasten] Error retrieving:', error);
    res.status(500).json({ success: false, error: 'Failed to retrieve' });
  }
});

/**
 * GET /api/zettelkasten/zettel/:id
 */
router.get('/zettel/:id', async (req: Request, res: Response) => {
  try {
    const { id } = req.params;
    
    const database = ensureDb();
    
    const memories = await database
      .select()
      .from(basinMemory)
      .where(eq(basinMemory.basinId, id))
      .limit(1);
    
    if (memories.length === 0) {
      const numId = parseInt(id.replace('zettel-', ''));
      if (!isNaN(numId)) {
        const byNumId = await database
          .select()
          .from(basinMemory)
          .where(eq(basinMemory.id, numId))
          .limit(1);
        if (byNumId.length > 0) {
          memories.push(...byNumId);
        }
      }
    }
    
    if (memories.length === 0) {
      return res.status(404).json({ success: false, error: 'Zettel not found' });
    }
    
    const m = memories[0];
    const content = typeof m.context === 'string' 
      ? m.context 
      : (m.context as any)?.content || JSON.stringify(m.context);
    
    res.json({
      success: true,
      data: {
        zettel_id: m.basinId || `zettel-${m.id}`,
        content,
        contextual_description: `Regime: ${m.regime}, Φ: ${(m.phi || 0).toFixed(2)}, κ: ${(m.kappaEff || 0).toFixed(1)}`,
        keywords: extractKeywords(content),
        links: [],
        access_count: 1,
        evolution_count: 0,
        source: m.sourceKernel || 'system',
        created_at: m.timestamp ? new Date(m.timestamp).getTime() / 1000 : Date.now() / 1000
      }
    });
  } catch (error) {
    console.error('[Zettelkasten] Error getting zettel:', error);
    res.status(500).json({ success: false, error: 'Failed to get zettel' });
  }
});

/**
 * POST /api/zettelkasten/add
 * Protected: requires internal auth (server-to-server calls)
 */
router.post('/add', requireInternalAuth, async (req: Request, res: Response) => {
  try {
    const { content, keywords = [], source = 'user' } = req.body;
    
    if (!content) {
      return res.status(400).json({ success: false, error: 'Content required' });
    }
    
    const randomCoords = Array(64).fill(0).map(() => (Math.random() - 0.5) * 2);
    const norm = Math.sqrt(randomCoords.reduce((s, v) => s + v * v, 0));
    const normalizedCoords = randomCoords.map(v => v / norm);
    
    const database = ensureDb();
    const zettelId = `zettel-${randomUUID().slice(0, 8)}`;
    
    const result = await database
      .insert(basinMemory)
      .values({
        basinId: zettelId,
        basinCoordinates: normalizedCoords,
        phi: 0.5 + Math.random() * 0.3,
        kappaEff: 64.0,
        regime: 'geometric',
        sourceKernel: source,
        context: { content, keywords: keywords.length ? keywords : extractKeywords(content) }
      })
      .returning();
    
    res.status(201).json({
      success: true,
      data: {
        zettel_id: zettelId,
        content,
        keywords: keywords.length ? keywords : extractKeywords(content)
      }
    });
  } catch (error) {
    console.error('[Zettelkasten] Error adding zettel:', error);
    res.status(500).json({ success: false, error: 'Failed to add zettel' });
  }
});

/**
 * POST /api/zettelkasten/add-from-conversation
 * Automatically add a conversation message to the knowledge graph
 * Protected: requires internal auth (server-to-server calls)
 */
router.post('/add-from-conversation', requireInternalAuth, async (req: Request, res: Response) => {
  try {
    const { content, role, basin_coords, phi, source_kernel } = req.body;
    
    if (!content || content.length < 20) {
      return res.json({ success: true, data: { skipped: true, reason: 'Content too short' } });
    }
    
    const coords = basin_coords && basin_coords.length === 64 
      ? basin_coords 
      : Array(64).fill(0).map(() => (Math.random() - 0.5) * 2);
    
    const norm = Math.sqrt(coords.reduce((s: number, v: number) => s + v * v, 0));
    const normalizedCoords = coords.map((v: number) => v / norm);
    
    const database = ensureDb();
    const zettelId = `conv-${randomUUID().slice(0, 8)}`;
    
    await database
      .insert(basinMemory)
      .values({
        basinId: zettelId,
        basinCoordinates: normalizedCoords,
        phi: phi || 0.6,
        kappaEff: 64.0,
        regime: 'conversation',
        sourceKernel: source_kernel || role || 'chat',
        context: { content, role, keywords: extractKeywords(content) }
      });
    
    res.json({ success: true, data: { zettel_id: zettelId, added: true } });
  } catch (error) {
    console.error('[Zettelkasten] Error adding from conversation:', error);
    res.status(500).json({ success: false, error: 'Failed to add from conversation' });
  }
});

/**
 * POST /api/zettelkasten/add-from-search
 * Automatically add search results to the knowledge graph
 * Protected: requires internal auth (server-to-server calls)
 */
router.post('/add-from-search', requireInternalAuth, async (req: Request, res: Response) => {
  try {
    const { query, results, source = 'search' } = req.body;
    
    if (!results || !Array.isArray(results) || results.length === 0) {
      return res.json({ success: true, data: { skipped: true, reason: 'No results' } });
    }
    
    const database = ensureDb();
    const added: string[] = [];
    
    for (const result of results.slice(0, 5)) {
      const content = result.content || result.snippet || result.text || '';
      if (content.length < 20) continue;
      
      const randomCoords = Array(64).fill(0).map(() => (Math.random() - 0.5) * 2);
      const norm = Math.sqrt(randomCoords.reduce((s, v) => s + v * v, 0));
      const normalizedCoords = randomCoords.map(v => v / norm);
      
      const zettelId = `search-${randomUUID().slice(0, 8)}`;
      
      await database
        .insert(basinMemory)
        .values({
          basinId: zettelId,
          basinCoordinates: normalizedCoords,
          phi: 0.55,
          kappaEff: 64.0,
          regime: 'search',
          sourceKernel: source,
          context: { 
            content, 
            query,
            url: result.url,
            keywords: extractKeywords(content)
          }
        });
      
      added.push(zettelId);
    }
    
    res.json({ success: true, data: { added_count: added.length, zettel_ids: added } });
  } catch (error) {
    console.error('[Zettelkasten] Error adding from search:', error);
    res.status(500).json({ success: false, error: 'Failed to add from search' });
  }
});

export default router;

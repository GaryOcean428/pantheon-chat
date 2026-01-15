/**
 * Database Sync API for Multi-Instance Deployments (Replit ↔ Railway)
 * 
 * Provides endpoints for:
 * 1. Exporting database state (vocabulary, patterns, conversations)
 * 2. Importing database state from another instance
 * 3. Delta sync for incremental updates
 * 
 * Security: All endpoints require SYNC_API_KEY header
 */

import { Router, Request, Response, NextFunction } from 'express';
import { db, withDbRetry } from './db';
import { logger } from './lib/logger';
import * as schema from '@shared/schema';
import { sql, desc } from 'drizzle-orm';
import { upsertToken } from './persistence/vocabulary';

const router = Router();

// Middleware to validate sync API key
function requireSyncKey(req: Request, res: Response, next: NextFunction) {
  const syncKey = process.env.SYNC_API_KEY;
  const providedKey = req.headers['x-sync-api-key'] as string;
  
  if (!syncKey) {
    return res.status(503).json({ error: 'Sync API not configured (SYNC_API_KEY not set)' });
  }
  
  if (!providedKey || providedKey !== syncKey) {
    return res.status(401).json({ error: 'Invalid or missing sync API key' });
  }
  
  next();
}

router.use(requireSyncKey);

/**
 * GET /api/sync/status
 * Returns sync status and last sync timestamps
 */
router.get('/status', async (req: Request, res: Response) => {
  try {
    const instanceId = process.env.RAILWAY_DEPLOYMENT_ID || process.env.REPLIT_DEPLOYMENT_ID || 'unknown';
    const instanceType = process.env.RAILWAY_DEPLOYMENT_ID ? 'railway' : 
                         process.env.REPLIT_DEPLOYMENT_ID ? 'replit' : 'local';
    
    // Get table counts
    const counts: Record<string, number> = {};
    
    if (db) {
      const vocabCount = await withDbRetry(
        async () => db!.select({ count: sql<number>`count(*)` }).from(schema.coordizerVocabulary),
        'sync-status-vocab'
      );
      counts.vocabulary = vocabCount?.[0]?.count || 0;
      
      const conversationCount = await withDbRetry(
        async () => db!.select({ count: sql<number>`count(*)` }).from(schema.hermesConversations),
        'sync-status-conversations'
      );
      counts.conversations = conversationCount?.[0]?.count || 0;
    }
    
    res.json({
      instanceId,
      instanceType,
      databaseAvailable: !!db,
      counts,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    logger.error({ error }, '[Sync] Status check failed');
    res.status(500).json({ error: 'Status check failed' });
  }
});

/**
 * GET /api/sync/export/vocabulary
 * Export vocabulary data for sync
 */
router.get('/export/vocabulary', async (req: Request, res: Response) => {
  try {
    if (!db) {
      return res.status(503).json({ error: 'Database not available' });
    }
    
    const since = req.query.since ? new Date(req.query.since as string) : null;
    const limit = Math.min(parseInt(req.query.limit as string) || 10000, 50000);
    
    let query = db.select().from(schema.coordizerVocabulary).limit(limit);
    
    // Note: If there's an updatedAt column, filter by it
    // For now, export all within limit
    
    const vocabulary = await withDbRetry(
      async () => query,
      'sync-export-vocabulary'
    );
    
    res.json({
      type: 'vocabulary',
      count: vocabulary?.length || 0,
      data: vocabulary || [],
      exportedAt: new Date().toISOString()
    });
  } catch (error) {
    logger.error({ error }, '[Sync] Vocabulary export failed');
    res.status(500).json({ error: 'Export failed' });
  }
});

/**
 * POST /api/sync/import/vocabulary
 * Import vocabulary data from another instance
 */
router.post('/import/vocabulary', async (req: Request, res: Response) => {
  try {
    if (!db) {
      return res.status(503).json({ error: 'Database not available' });
    }
    
    const { data } = req.body;
    
    if (!Array.isArray(data)) {
      return res.status(400).json({ error: 'Invalid data format - expected array' });
    }
    
    let imported = 0;
    let skipped = 0;
    let errors = 0;
    
    // Process in batches
    const batchSize = 100;
    for (let i = 0; i < data.length; i += batchSize) {
      const batch = data.slice(i, i + batchSize);
      
      for (const item of batch) {
        try {
          // Upsert - insert or update on conflict
          await upsertToken({
            token: item.token,
            tokenId: item.tokenId ?? item.token_id ?? 0,
            weight: item.weight ?? 1,
            frequency: item.frequency ?? 1,
            phiScore: item.phiScore ?? item.phi_score ?? 0,
            basinEmbedding: item.basinEmbedding ?? item.basin_embedding ?? null,
            sourceType: item.sourceType ?? item.source_type ?? 'sync',
            tokenRole: item.tokenRole ?? item.token_role ?? 'encoding',
            phraseCategory: item.phraseCategory ?? item.phrase_category ?? 'unknown',
            isRealWord: item.isRealWord ?? item.is_real_word ?? false,
            tokenStatus: item.tokenStatus ?? item.token_status ?? 'active',
            source: 'sync'
          });
          imported++;
        } catch (err) {
          errors++;
        }
      }
    }
    
    res.json({
      success: true,
      imported,
      skipped,
      errors,
      importedAt: new Date().toISOString()
    });
  } catch (error) {
    logger.error({ error }, '[Sync] Vocabulary import failed');
    res.status(500).json({ error: 'Import failed' });
  }
});

/**
 * GET /api/sync/export/conversations
 * Export recent conversations for sync
 */
router.get('/export/conversations', async (req: Request, res: Response) => {
  try {
    if (!db) {
      return res.status(503).json({ error: 'Database not available' });
    }
    
    const limit = Math.min(parseInt(req.query.limit as string) || 100, 1000);
    
    const conversations = await withDbRetry(
      async () => db!.select()
        .from(schema.hermesConversations)
        .orderBy(desc(schema.hermesConversations.createdAt))
        .limit(limit),
      'sync-export-conversations'
    );
    
    res.json({
      type: 'conversations',
      count: conversations?.length || 0,
      data: conversations || [],
      exportedAt: new Date().toISOString()
    });
  } catch (error) {
    logger.error({ error }, '[Sync] Conversations export failed');
    res.status(500).json({ error: 'Export failed' });
  }
});

/**
 * POST /api/sync/trigger
 * Trigger a sync pull from another instance
 */
router.post('/trigger', async (req: Request, res: Response) => {
  try {
    const { sourceUrl, tables = ['vocabulary'] } = req.body;
    
    if (!sourceUrl) {
      return res.status(400).json({ error: 'sourceUrl required' });
    }
    
    const syncKey = process.env.SYNC_API_KEY;
    const results: Record<string, { error?: string; success?: boolean; imported?: number; skipped?: number; errors?: number; importedAt?: string }> = {};
    
    for (const table of tables) {
      try {
        // Fetch from source
        const exportResponse = await fetch(`${sourceUrl}/api/sync/export/${table}`, {
          headers: { 'x-sync-api-key': syncKey || '' }
        });
        
        if (!exportResponse.ok) {
          results[table] = { error: `Export failed: ${exportResponse.status}` };
          continue;
        }
        
        const exportData = await exportResponse.json();
        
        // Import locally
        const importResponse = await fetch(`http://localhost:${process.env.PORT || 5000}/api/sync/import/${table}`, {
          method: 'POST',
          headers: { 
            'Content-Type': 'application/json',
            'x-sync-api-key': syncKey || ''
          },
          body: JSON.stringify({ data: exportData.data })
        });
        
        results[table] = await importResponse.json();
      } catch (err) {
        results[table] = { error: String(err) };
      }
    }
    
    res.json({
      success: true,
      results,
      syncedAt: new Date().toISOString()
    });
  } catch (error) {
    logger.error({ error }, '[Sync] Trigger sync failed');
    res.status(500).json({ error: 'Sync trigger failed' });
  }
});

export default router;

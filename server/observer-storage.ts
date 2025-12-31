/**
 * Observer Storage - Recovery Workflow Storage Interface
 *
 * Provides storage operations for recovery workflows linked to search jobs.
 * Used by search-coordinator to sync workflow progress.
 */

import { db } from './db';
import { sql } from 'drizzle-orm';
import { logger } from './lib/logger';

export interface RecoveryWorkflow {
  id: string;
  status: string;
  progress: Record<string, unknown> | null;
  completedAt?: Date | null;
  notes?: string;
}

/**
 * Find a recovery workflow by its associated search job ID
 */
async function findWorkflowBySearchJobId(searchJobId: string): Promise<RecoveryWorkflow | null> {
  if (!db) {
    logger.warn('[ObserverStorage] Database not available');
    return null;
  }

  try {
    // Look for workflow with matching search job ID in progress
    const result = await db.execute(sql`
      SELECT id, status, progress, completed_at as "completedAt"
      FROM recovery_workflows
      WHERE progress->>'constrainedSearchProgress'->>'searchJobId' = ${searchJobId}
      LIMIT 1
    `);

    const rows = result.rows as unknown as RecoveryWorkflow[];
    return rows?.[0] || null;
  } catch {
    // Table may not exist - this is a legacy feature
    logger.debug(`[ObserverStorage] Could not find workflow by search job ID: ${searchJobId}`);
    return null;
  }
}

/**
 * Update a recovery workflow
 */
async function updateRecoveryWorkflow(
  workflowId: string,
  updates: Partial<RecoveryWorkflow>
): Promise<void> {
  if (!db) {
    logger.warn('[ObserverStorage] Database not available');
    return;
  }

  try {
    // Build dynamic update - simplified approach
    if (updates.status !== undefined) {
      await db.execute(sql`
        UPDATE recovery_workflows
        SET status = ${updates.status}
        WHERE id = ${workflowId}
      `);
    }

    if (updates.progress !== undefined) {
      await db.execute(sql`
        UPDATE recovery_workflows
        SET progress = ${JSON.stringify(updates.progress)}::jsonb
        WHERE id = ${workflowId}
      `);
    }

    if (updates.completedAt !== undefined) {
      await db.execute(sql`
        UPDATE recovery_workflows
        SET completed_at = ${updates.completedAt}
        WHERE id = ${workflowId}
      `);
    }
  } catch {
    // Table may not exist - this is a legacy feature
    logger.debug(`[ObserverStorage] Could not update recovery workflow: ${workflowId}`);
  }
}

export const observerStorage = {
  findWorkflowBySearchJobId,
  updateRecoveryWorkflow,
};

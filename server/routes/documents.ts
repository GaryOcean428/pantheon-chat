/**
 * Document Upload API
 * 
 * Upload documents (markdown, text, PDFs) that sync into the Ocean knowledge system.
 * 
 * Supported formats:
 * - Markdown (.md)
 * - Plain text (.txt)
 * - PDF (.pdf)
 * 
 * All uploaded documents are:
 * 1. Parsed and extracted
 * 2. Coordized (converted to basin coordinates)
 * 3. Synced to Ocean knowledge system
 * 4. Made available for Zeus chat context
 */

import { Router, Request, Response } from 'express';
import { isAuthenticated } from '../replitAuth';
import { z } from 'zod';
import { randomUUID } from 'crypto';
import multer from 'multer';

const router = Router();
const BACKEND_URL = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';

// Configure multer for file uploads
const storage = multer.memoryStorage();
const upload = multer({
  storage,
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB max
    files: 10, // Max 10 files per request
  },
  fileFilter: (_req, file, cb) => {
    const allowedTypes = [
      'text/markdown',
      'text/plain',
      'text/x-markdown',
      'application/pdf',
      'application/octet-stream', // Some systems report .md as this
    ];
    const allowedExtensions = ['.md', '.txt', '.pdf', '.markdown'];
    const ext = file.originalname.toLowerCase().slice(file.originalname.lastIndexOf('.'));
    
    if (allowedTypes.includes(file.mimetype) || allowedExtensions.includes(ext)) {
      cb(null, true);
    } else {
      cb(new Error(`Unsupported file type: ${file.mimetype}. Allowed: markdown, text, PDF`));
    }
  },
});

// Document metadata schema
const documentMetadataSchema = z.object({
  title: z.string().optional(),
  domain: z.string().optional(),
  tags: z.array(z.string()).optional(),
  source: z.string().optional(),
});

export interface UploadedDocument {
  id: string;
  filename: string;
  contentType: string;
  size: number;
  extractedText?: string;
  basinCoordinates?: number[];
  syncedToOcean: boolean;
  metadata?: {
    title?: string;
    domain?: string;
    tags?: string[];
    source?: string;
  };
  createdAt: string;
}

export interface DocumentUploadResponse {
  success: boolean;
  documents: UploadedDocument[];
  oceanSyncStatus: {
    synced: number;
    failed: number;
    errors?: string[];
  };
}

/**
 * POST /api/documents/upload
 * 
 * Upload one or more documents to sync with Ocean knowledge system.
 * 
 * Request:
 *   Content-Type: multipart/form-data
 *   files: File[] - The document files to upload
 *   metadata: JSON string - Optional metadata for the documents
 * 
 * Response:
 *   {
 *     success: boolean,
 *     documents: UploadedDocument[],
 *     oceanSyncStatus: { synced: number, failed: number, errors?: string[] }
 *   }
 */
router.post('/upload', isAuthenticated, upload.array('files', 10), async (req: Request, res: Response) => {
  try {
    const files = req.files as Express.Multer.File[];
    
    if (!files || files.length === 0) {
      res.status(400).json({ error: 'No files provided' });
      return;
    }
    
    // Parse optional metadata
    let metadata: z.infer<typeof documentMetadataSchema> = {};
    if (req.body.metadata) {
      try {
        metadata = documentMetadataSchema.parse(JSON.parse(req.body.metadata));
      } catch {
        // Ignore invalid metadata
      }
    }
    
    const documents: UploadedDocument[] = [];
    const syncErrors: string[] = [];
    let syncedCount = 0;
    
    for (const file of files) {
      const docId = randomUUID();
      const ext = file.originalname.toLowerCase().slice(file.originalname.lastIndexOf('.'));
      
      // Prepare document for Python backend processing
      const formData = new FormData();
      formData.append('file', new Blob([file.buffer]), file.originalname);
      formData.append('doc_id', docId);
      formData.append('content_type', file.mimetype);
      if (metadata.title) formData.append('title', metadata.title);
      if (metadata.domain) formData.append('domain', metadata.domain);
      if (metadata.tags) formData.append('tags', JSON.stringify(metadata.tags));
      if (metadata.source) formData.append('source', metadata.source);
      
      try {
        // Send to Python backend for processing and Ocean sync
        const response = await fetch(`${BACKEND_URL}/api/documents/process`, {
          method: 'POST',
          body: formData,
        });
        
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
          throw new Error(errorData.error || `Processing failed: ${response.status}`);
        }
        
        const result = await response.json();
        
        documents.push({
          id: docId,
          filename: file.originalname,
          contentType: file.mimetype,
          size: file.size,
          extractedText: result.extracted_text?.substring(0, 500) + (result.extracted_text?.length > 500 ? '...' : ''),
          basinCoordinates: result.basin_coordinates,
          syncedToOcean: result.synced_to_ocean || false,
          metadata: {
            title: result.title || metadata.title || file.originalname,
            domain: result.domain || metadata.domain,
            tags: result.tags || metadata.tags,
            source: metadata.source,
          },
          createdAt: new Date().toISOString(),
        });
        
        if (result.synced_to_ocean) {
          syncedCount++;
        }
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : 'Unknown error';
        syncErrors.push(`${file.originalname}: ${errorMsg}`);
        
        // Still record the document even if sync failed
        documents.push({
          id: docId,
          filename: file.originalname,
          contentType: file.mimetype,
          size: file.size,
          syncedToOcean: false,
          metadata: {
            title: metadata.title || file.originalname,
            domain: metadata.domain,
            tags: metadata.tags,
            source: metadata.source,
          },
          createdAt: new Date().toISOString(),
        });
      }
    }
    
    const response: DocumentUploadResponse = {
      success: documents.length > 0,
      documents,
      oceanSyncStatus: {
        synced: syncedCount,
        failed: documents.length - syncedCount,
        errors: syncErrors.length > 0 ? syncErrors : undefined,
      },
    };
    
    res.json(response);
  } catch (error) {
    console.error('[Documents] Upload error:', error);
    res.status(500).json({ 
      error: 'Document upload failed',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/documents
 * 
 * List all uploaded documents with their sync status.
 */
router.get('/', isAuthenticated, async (req: Request, res: Response) => {
  try {
    const limit = Math.min(parseInt(req.query.limit as string) || 50, 200);
    const domain = req.query.domain as string | undefined;
    
    const response = await fetch(
      `${BACKEND_URL}/api/documents?limit=${limit}${domain ? `&domain=${encodeURIComponent(domain)}` : ''}`,
      {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      }
    );
    
    if (!response.ok) {
      throw new Error(`Backend returned ${response.status}`);
    }
    
    const data = await response.json();
    res.json(data);
  } catch (error) {
    console.error('[Documents] List error:', error);
    res.json({ documents: [], total: 0 });
  }
});

/**
 * GET /api/documents/:id
 * 
 * Get details of a specific document including its Ocean knowledge connections.
 */
router.get('/:id', isAuthenticated, async (req: Request, res: Response) => {
  try {
    const { id } = req.params;
    
    const response = await fetch(`${BACKEND_URL}/api/documents/${id}`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });
    
    if (!response.ok) {
      if (response.status === 404) {
        res.status(404).json({ error: 'Document not found' });
        return;
      }
      throw new Error(`Backend returned ${response.status}`);
    }
    
    const data = await response.json();
    res.json(data);
  } catch (error) {
    console.error('[Documents] Get error:', error);
    res.status(500).json({ error: 'Failed to retrieve document' });
  }
});

/**
 * DELETE /api/documents/:id
 * 
 * Delete a document and remove it from Ocean knowledge system.
 */
router.delete('/:id', isAuthenticated, async (req: Request, res: Response) => {
  try {
    const { id } = req.params;
    
    const response = await fetch(`${BACKEND_URL}/api/documents/${id}`, {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
    });
    
    if (!response.ok) {
      if (response.status === 404) {
        res.status(404).json({ error: 'Document not found' });
        return;
      }
      throw new Error(`Backend returned ${response.status}`);
    }
    
    const data = await response.json();
    res.json(data);
  } catch (error) {
    console.error('[Documents] Delete error:', error);
    res.status(500).json({ error: 'Failed to delete document' });
  }
});

/**
 * POST /api/documents/:id/resync
 * 
 * Re-sync a document to Ocean knowledge system (useful after failures).
 */
router.post('/:id/resync', isAuthenticated, async (req: Request, res: Response) => {
  try {
    const { id } = req.params;
    
    const response = await fetch(`${BACKEND_URL}/api/documents/${id}/resync`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });
    
    if (!response.ok) {
      if (response.status === 404) {
        res.status(404).json({ error: 'Document not found' });
        return;
      }
      throw new Error(`Backend returned ${response.status}`);
    }
    
    const data = await response.json();
    res.json(data);
  } catch (error) {
    console.error('[Documents] Resync error:', error);
    res.status(500).json({ error: 'Failed to resync document' });
  }
});

/**
 * POST /api/documents/search
 * 
 * Search uploaded documents using Fisher-Rao semantic similarity.
 */
router.post('/search', isAuthenticated, async (req: Request, res: Response) => {
  try {
    const { query, limit = 10, domain } = req.body;
    
    if (!query || typeof query !== 'string') {
      res.status(400).json({ error: 'Query string required' });
      return;
    }
    
    const response = await fetch(`${BACKEND_URL}/api/documents/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, limit, domain }),
    });
    
    if (!response.ok) {
      throw new Error(`Backend returned ${response.status}`);
    }
    
    const data = await response.json();
    res.json(data);
  } catch (error) {
    console.error('[Documents] Search error:', error);
    res.status(500).json({ error: 'Search failed', results: [] });
  }
});

export default router;

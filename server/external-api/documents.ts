/**
 * External Documents API
 * 
 * Provides authenticated external access to document upload and Ocean knowledge sync.
 * Supports markdown, text, and PDF uploads.
 * Requires 'documents' scope in API key.
 */

import { Router, Response } from 'express';
import { 
  authenticateExternalApi, 
  requireScopes,
  type AuthenticatedRequest,
} from './auth';
import multer from 'multer';

export const externalDocumentsRouter = Router();

// Configure multer for file uploads
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB max
  },
  fileFilter: (_req, file, cb) => {
    const allowedMimes = [
      'text/plain',
      'text/markdown',
      'text/x-markdown',
      'application/pdf',
      'application/json',
    ];
    const allowedExtensions = ['.txt', '.md', '.markdown', '.pdf', '.json'];
    
    const ext = file.originalname.toLowerCase().slice(file.originalname.lastIndexOf('.'));
    
    if (allowedMimes.includes(file.mimetype) || allowedExtensions.includes(ext)) {
      cb(null, true);
    } else {
      cb(new Error(`Unsupported file type: ${file.mimetype}`));
    }
  },
});

// All routes require authentication
externalDocumentsRouter.use(authenticateExternalApi);

/**
 * POST /api/v1/external/documents/upload
 * Upload a document to the Ocean knowledge system
 */
externalDocumentsRouter.post('/upload', 
  requireScopes('documents'), 
  upload.single('file'),
  async (req: AuthenticatedRequest, res: Response) => {
    try {
      const file = req.file;
      
      if (!file) {
        return res.status(400).json({
          error: 'Bad Request',
          message: 'No file provided. Use multipart/form-data with a "file" field.',
        });
      }
      
      const { title, description, tags, syncToOcean = true } = req.body;
      
      // Determine document type
      const ext = file.originalname.toLowerCase().slice(file.originalname.lastIndexOf('.'));
      let documentType: 'markdown' | 'text' | 'pdf' | 'json' = 'text';
      if (ext === '.md' || ext === '.markdown') documentType = 'markdown';
      else if (ext === '.pdf') documentType = 'pdf';
      else if (ext === '.json') documentType = 'json';
      
      // For PDFs, we need to call the Python backend for processing
      let content = file.buffer.toString('utf-8');
      let extractedText = content;
      
      if (documentType === 'pdf') {
        const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';
        
        // Send PDF to Python for text extraction
        const formData = new FormData();
        formData.append('file', new Blob([file.buffer]), file.originalname);
        
        const extractResponse = await fetch(`${pythonBackendUrl}/api/documents/extract-pdf`, {
          method: 'POST',
          headers: {
            'X-Internal-Auth': process.env.INTERNAL_API_KEY || 'dev-key',
          },
          body: formData,
        });
        
        if (extractResponse.ok) {
          const extracted = await extractResponse.json();
          extractedText = extracted.text || '';
          content = extractedText;
        } else {
          console.warn('[External Documents] PDF extraction failed, storing as binary reference');
          extractedText = `[PDF Document: ${file.originalname}]`;
        }
      }
      
      // Sync to Ocean knowledge system
      let oceanSyncResult = null;
      if (syncToOcean !== 'false' && syncToOcean !== false) {
        const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';
        
        const syncResponse = await fetch(`${pythonBackendUrl}/api/ocean/knowledge/ingest`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-Internal-Auth': process.env.INTERNAL_API_KEY || 'dev-key',
          },
          body: JSON.stringify({
            content: extractedText,
            title: title || file.originalname,
            description: description || '',
            tags: tags ? (typeof tags === 'string' ? tags.split(',') : tags) : [],
            source: 'external-api',
            client_name: req.externalClient?.name || 'external',
            document_type: documentType,
          }),
        });
        
        if (syncResponse.ok) {
          oceanSyncResult = await syncResponse.json();
        } else {
          console.warn('[External Documents] Ocean sync failed');
        }
      }
      
      res.json({
        success: true,
        document: {
          filename: file.originalname,
          type: documentType,
          size: file.size,
          title: title || file.originalname,
          description: description || null,
          tags: tags ? (typeof tags === 'string' ? tags.split(',') : tags) : [],
        },
        oceanSync: oceanSyncResult ? {
          synced: true,
          knowledgeId: oceanSyncResult.knowledge_id,
          basinCoords: oceanSyncResult.basin_coords,
        } : {
          synced: false,
          reason: syncToOcean === 'false' || syncToOcean === false ? 'disabled' : 'sync_failed',
        },
      });
    } catch (error) {
      console.error('[External Documents] Upload error:', error);
      res.status(500).json({
        error: 'Internal Server Error',
        message: 'Failed to process document upload',
      });
    }
  }
);

/**
 * POST /api/v1/external/documents/upload-text
 * Upload raw text content directly (no file upload needed)
 */
externalDocumentsRouter.post('/upload-text', 
  requireScopes('documents'), 
  async (req: AuthenticatedRequest, res: Response) => {
    try {
      const { content, title, description, tags, format = 'markdown' } = req.body;
      
      if (!content || typeof content !== 'string') {
        return res.status(400).json({
          error: 'Bad Request',
          message: 'Content is required and must be a string',
        });
      }
      
      if (content.length > 1000000) { // 1MB text limit
        return res.status(400).json({
          error: 'Bad Request',
          message: 'Content exceeds maximum size of 1MB',
        });
      }
      
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';
      
      const syncResponse = await fetch(`${pythonBackendUrl}/api/ocean/knowledge/ingest`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Internal-Auth': process.env.INTERNAL_API_KEY || 'dev-key',
        },
        body: JSON.stringify({
          content,
          title: title || 'Untitled Document',
          description: description || '',
          tags: tags || [],
          source: 'external-api',
          client_name: req.externalClient?.name || 'external',
          document_type: format,
        }),
      });
      
      if (!syncResponse.ok) {
        const errorText = await syncResponse.text();
        console.error('[External Documents] Ocean sync failed:', errorText);
        return res.status(502).json({
          error: 'Backend Error',
          message: 'Failed to sync document to Ocean knowledge system',
        });
      }
      
      const result = await syncResponse.json();
      
      res.json({
        success: true,
        document: {
          title: title || 'Untitled Document',
          format,
          size: content.length,
          description: description || null,
          tags: tags || [],
        },
        oceanSync: {
          synced: true,
          knowledgeId: result.knowledge_id,
          basinCoords: result.basin_coords,
        },
      });
    } catch (error) {
      console.error('[External Documents] Upload text error:', error);
      res.status(500).json({
        error: 'Internal Server Error',
        message: 'Failed to process text upload',
      });
    }
  }
);

/**
 * GET /api/v1/external/documents/list
 * List documents uploaded by this API key
 */
externalDocumentsRouter.get('/list', 
  requireScopes('documents'), 
  async (req: AuthenticatedRequest, res: Response) => {
    try {
      const limit = Math.min(parseInt(req.query.limit as string) || 50, 100);
      const offset = parseInt(req.query.offset as string) || 0;
      
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';
      
      const response = await fetch(
        `${pythonBackendUrl}/api/ocean/knowledge/list?client=${encodeURIComponent(req.externalClient?.name || 'external')}&limit=${limit}&offset=${offset}`,
        {
          headers: {
            'X-Internal-Auth': process.env.INTERNAL_API_KEY || 'dev-key',
          },
        }
      );
      
      if (!response.ok) {
        return res.status(502).json({
          error: 'Backend Error',
          message: 'Failed to retrieve documents',
        });
      }
      
      const result = await response.json();
      
      res.json({
        success: true,
        documents: result.documents || [],
        pagination: {
          limit,
          offset,
          total: result.total || 0,
        },
      });
    } catch (error) {
      console.error('[External Documents] List error:', error);
      res.status(500).json({
        error: 'Internal Server Error',
        message: 'Failed to list documents',
      });
    }
  }
);

/**
 * GET /api/v1/external/documents/:id
 * Get a specific document by ID
 */
externalDocumentsRouter.get('/:id', 
  requireScopes('documents'), 
  async (req: AuthenticatedRequest, res: Response) => {
    try {
      const { id } = req.params;
      
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';
      
      const response = await fetch(
        `${pythonBackendUrl}/api/ocean/knowledge/${id}`,
        {
          headers: {
            'X-Internal-Auth': process.env.INTERNAL_API_KEY || 'dev-key',
          },
        }
      );
      
      if (!response.ok) {
        if (response.status === 404) {
          return res.status(404).json({
            error: 'Not Found',
            message: 'Document not found',
          });
        }
        return res.status(502).json({
          error: 'Backend Error',
          message: 'Failed to retrieve document',
        });
      }
      
      const document = await response.json();
      
      res.json({
        success: true,
        document,
      });
    } catch (error) {
      console.error('[External Documents] Get error:', error);
      res.status(500).json({
        error: 'Internal Server Error',
        message: 'Failed to retrieve document',
      });
    }
  }
);

/**
 * DELETE /api/v1/external/documents/:id
 * Delete a document
 */
externalDocumentsRouter.delete('/:id', 
  requireScopes('documents'), 
  async (req: AuthenticatedRequest, res: Response) => {
    try {
      const { id } = req.params;
      
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';
      
      const response = await fetch(
        `${pythonBackendUrl}/api/ocean/knowledge/${id}`,
        {
          method: 'DELETE',
          headers: {
            'X-Internal-Auth': process.env.INTERNAL_API_KEY || 'dev-key',
          },
        }
      );
      
      if (!response.ok) {
        if (response.status === 404) {
          return res.status(404).json({
            error: 'Not Found',
            message: 'Document not found',
          });
        }
        return res.status(502).json({
          error: 'Backend Error',
          message: 'Failed to delete document',
        });
      }
      
      res.json({
        success: true,
        message: 'Document deleted',
      });
    } catch (error) {
      console.error('[External Documents] Delete error:', error);
      res.status(500).json({
        error: 'Internal Server Error',
        message: 'Failed to delete document',
      });
    }
  }
);

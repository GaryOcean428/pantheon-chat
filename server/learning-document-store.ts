/**
 * LEARNING DOCUMENT STORE
 * 
 * Persists quality learning documents to Replit Object Storage.
 * Used for curriculum learning from validated search results.
 * 
 * Storage structure:
 * - learning/{kernel_id}/{timestamp}_{topic_hash}.json
 * - index/manifest.json (tracks all documents)
 * 
 * Documents contain:
 * - Validated insights with sources
 * - Quality text extracted from Tavily/Perplexity
 * - Fisher-Rao distances for semantic relationships
 */

import crypto from 'crypto';
import { createRequire } from 'module';

import type { Client as ReplitObjectStorageClient } from '@replit/object-storage';

type ObjectStorageClient = Pick<
  ReplitObjectStorageClient,
  'downloadAsText' | 'uploadFromText' | 'list' | 'delete'
>;

type DownloadAsTextResult = { ok: boolean; value: string; error?: string };
type UploadFromTextResult = { ok: boolean; error?: string };
type DeleteResult = { ok: boolean; error?: string };

const FALLBACK_CLIENT: ObjectStorageClient = {
  downloadAsText: async () => ({ ok: false, value: '', error: 'object_storage_unavailable' } as DownloadAsTextResult),
  uploadFromText: async () => ({ ok: false, error: 'object_storage_unavailable' } as UploadFromTextResult),
  list: async () => ({ ok: false, value: [], error: 'object_storage_unavailable' } as any),
  delete: async () => ({ ok: false, error: 'object_storage_unavailable' } as DeleteResult),
};

async function createObjectStorageClient(): Promise<ObjectStorageClient> {
  try {
    const require = createRequire(import.meta.url);
    const mod = require('@replit/object-storage') as any;
    const ClientCtor = mod?.Client as (new () => ObjectStorageClient) | undefined;
    if (!ClientCtor) {
      return FALLBACK_CLIENT;
    }
    return new ClientCtor();
  } catch {
    return FALLBACK_CLIENT;
  }
}

interface LearningDocument {
  id: string;
  kernel_id: string;
  topic: string;
  query: string;
  content: string;
  sources: Array<{
    url: string;
    title: string;
    score?: number;
  }>;
  synthesis?: string;
  citations?: string[];
  fisher_distance?: number;
  validated: boolean;
  validation_score: number;
  created_at: string;
  content_hash: string;
}

interface DocumentManifest {
  total_documents: number;
  total_size_bytes: number;
  last_updated: string;
  kernels: Record<string, number>;
  topics: Record<string, number>;
}

class LearningDocumentStore {
  private client: ObjectStorageClient = FALLBACK_CLIENT;
  private clientInit: Promise<void>;
  private initialized = false;
  
  constructor() {
    this.clientInit = (async () => {
      this.client = await createObjectStorageClient();
    })();
  }

  private async ensureClientReady(): Promise<void> {
    await this.clientInit;
  }
  
  private async ensureInitialized(): Promise<void> {
    if (this.initialized) return;

    await this.ensureClientReady();
    
    const { ok } = await this.client.downloadAsText('index/manifest.json');
    if (!ok) {
      const initialManifest: DocumentManifest = {
        total_documents: 0,
        total_size_bytes: 0,
        last_updated: new Date().toISOString(),
        kernels: {},
        topics: {}
      };
      await this.client.uploadFromText('index/manifest.json', JSON.stringify(initialManifest, null, 2));
      console.log('[LearningStore] Initialized new manifest');
    }
    
    this.initialized = true;
  }
  
  async storeDocument(params: {
    kernel_id: string;
    topic: string;
    query: string;
    content: string;
    sources: Array<{ url: string; title: string; score?: number }>;
    synthesis?: string;
    citations?: string[];
    fisher_distance?: number;
    validated: boolean;
    validation_score: number;
  }): Promise<{ ok: boolean; document_id?: string; error?: string }> {
    try {
      await this.ensureInitialized();
      
      const timestamp = Date.now();
      const topicHash = crypto.createHash('md5').update(params.topic).digest('hex').slice(0, 8);
      const contentHash = crypto.createHash('sha256').update(params.content).digest('hex').slice(0, 16);
      const documentId = `${timestamp}_${topicHash}`;
      
      const document: LearningDocument = {
        id: documentId,
        kernel_id: params.kernel_id,
        topic: params.topic,
        query: params.query,
        content: params.content,
        sources: params.sources,
        synthesis: params.synthesis,
        citations: params.citations,
        fisher_distance: params.fisher_distance,
        validated: params.validated,
        validation_score: params.validation_score,
        created_at: new Date().toISOString(),
        content_hash: contentHash
      };
      
      const path = `learning/${params.kernel_id}/${documentId}.json`;
      const documentJson = JSON.stringify(document, null, 2);
      
      const { ok, error } = await this.client.uploadFromText(path, documentJson);
      if (!ok) {
        console.error(`[LearningStore] Failed to store document: ${error}`);
        return { ok: false, error: String(error) };
      }
      
      await this.updateManifest(params.kernel_id, params.topic, documentJson.length);
      
      console.log(`[LearningStore] Stored document: ${path} (${documentJson.length} bytes)`);
      return { ok: true, document_id: documentId };
      
    } catch (err) {
      console.error(`[LearningStore] Error storing document: ${err}`);
      return { ok: false, error: String(err) };
    }
  }
  
  async getDocument(kernel_id: string, document_id: string): Promise<LearningDocument | null> {
    try {
      await this.ensureClientReady();
      const path = `learning/${kernel_id}/${document_id}.json`;
      const { ok, value, error } = await this.client.downloadAsText(path);
      
      if (!ok) {
        console.warn(`[LearningStore] Document not found: ${path} (${error})`);
        return null;
      }
      
      return JSON.parse(value) as LearningDocument;
    } catch (err) {
      console.error(`[LearningStore] Error getting document: ${err}`);
      return null;
    }
  }
  
  async listDocuments(kernel_id?: string, limit = 100): Promise<string[]> {
    try {
      await this.ensureClientReady();
      const { ok, value, error } = await this.client.list();
      
      if (!ok) {
        console.error(`[LearningStore] Failed to list documents: ${error}`);
        return [];
      }
      
      let documents = value
        .filter((obj: { name: string }) => obj.name.startsWith('learning/'))
        .map((obj: { name: string }) => obj.name);
      
      if (kernel_id) {
        documents = documents.filter((key: string) => key.startsWith(`learning/${kernel_id}/`));
      }
      
      return documents.slice(0, limit);
    } catch (err) {
      console.error(`[LearningStore] Error listing documents: ${err}`);
      return [];
    }
  }
  
  async getRecentDocuments(kernel_id: string, limit = 10): Promise<LearningDocument[]> {
    try {
      await this.ensureClientReady();
      const paths = await this.listDocuments(kernel_id, limit * 2);
      
      const sortedPaths = paths
        .map(p => {
          const match = p.match(/(\d+)_[a-f0-9]+\.json$/);
          return { path: p, timestamp: match ? parseInt(match[1]) : 0 };
        })
        .sort((a, b) => b.timestamp - a.timestamp)
        .slice(0, limit);
      
      const documents: LearningDocument[] = [];
      for (const { path } of sortedPaths) {
        const { ok, value } = await this.client.downloadAsText(path);
        if (ok) {
          documents.push(JSON.parse(value) as LearningDocument);
        }
      }
      
      return documents;
    } catch (err) {
      console.error(`[LearningStore] Error getting recent documents: ${err}`);
      return [];
    }
  }
  
  async getManifest(): Promise<DocumentManifest | null> {
    try {
      await this.ensureInitialized();
      const { ok, value } = await this.client.downloadAsText('index/manifest.json');
      if (!ok) return null;
      return JSON.parse(value) as DocumentManifest;
    } catch (err) {
      console.error(`[LearningStore] Error getting manifest: ${err}`);
      return null;
    }
  }
  
  private async updateManifest(kernel_id: string, topic: string, size_bytes: number): Promise<void> {
    try {
      const manifest = await this.getManifest() || {
        total_documents: 0,
        total_size_bytes: 0,
        last_updated: new Date().toISOString(),
        kernels: {},
        topics: {}
      };
      
      manifest.total_documents++;
      manifest.total_size_bytes += size_bytes;
      manifest.last_updated = new Date().toISOString();
      manifest.kernels[kernel_id] = (manifest.kernels[kernel_id] || 0) + 1;
      manifest.topics[topic] = (manifest.topics[topic] || 0) + 1;
      
      await this.client.uploadFromText('index/manifest.json', JSON.stringify(manifest, null, 2));
    } catch (err) {
      console.error(`[LearningStore] Error updating manifest: ${err}`);
    }
  }
  
  async deleteDocument(kernel_id: string, document_id: string): Promise<boolean> {
    try {
      const path = `learning/${kernel_id}/${document_id}.json`;
      const { ok, error } = await this.client.delete(path);
      
      if (!ok) {
        console.error(`[LearningStore] Failed to delete: ${path} (${error})`);
        return false;
      }
      
      console.log(`[LearningStore] Deleted: ${path}`);
      return true;
    } catch (err) {
      console.error(`[LearningStore] Error deleting document: ${err}`);
      return false;
    }
  }
  
  async getStorageStats(): Promise<{
    total_documents: number;
    total_size_mb: number;
    kernels: Record<string, number>;
    topics: Record<string, number>;
  }> {
    const manifest = await this.getManifest();
    if (!manifest) {
      return { total_documents: 0, total_size_mb: 0, kernels: {}, topics: {} };
    }
    
    return {
      total_documents: manifest.total_documents,
      total_size_mb: manifest.total_size_bytes / (1024 * 1024),
      kernels: manifest.kernels,
      topics: manifest.topics
    };
  }
}

export const learningDocumentStore = new LearningDocumentStore();

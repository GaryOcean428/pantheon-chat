/**
 * Documents Service
 * 
 * Type-safe API functions for document upload and management.
 */

import { get, post, del, postMultipart } from '../client';

export interface UploadedDocument {
  id: string;
  filename: string;
  title: string;
  contentType: string;
  size: number;
  domain?: string;
  tags?: string[];
  extractedText?: string;
  basinCoordinates?: number[];
  syncedToOcean: boolean;
  createdAt: string;
}

export interface DocumentUploadParams {
  files: File[];
  title?: string;
  domain?: string;
  tags?: string[];
}

export interface DocumentUploadResponse {
  success: boolean;
  documents: Array<{
    id: string;
    filename: string;
    title: string;
    syncedToOcean: boolean;
  }>;
  oceanSyncStatus: {
    synced: number;
    failed: number;
    errors?: string[];
  };
}

export interface DocumentListResponse {
  documents: Array<{
    id: string;
    filename: string;
    title: string;
    domain?: string;
    syncedToOcean: boolean;
    createdAt: string;
  }>;
  total: number;
}

export interface DocumentSearchParams {
  query: string;
  limit?: number;
  domain?: string;
}

export interface DocumentSearchResult {
  id: string;
  title: string;
  filename: string;
  domain?: string;
  similarity: number;
  snippet: string;
}

export interface DocumentSearchResponse {
  results: DocumentSearchResult[];
  total: number;
  query: string;
}

/**
 * Upload documents to sync with Ocean knowledge system.
 */
export async function uploadDocuments(params: DocumentUploadParams): Promise<DocumentUploadResponse> {
  const formData = new FormData();
  
  params.files.forEach(file => formData.append('files', file));
  
  if (params.title) {
    formData.append('metadata', JSON.stringify({
      title: params.title,
      domain: params.domain,
      tags: params.tags,
    }));
  }
  
  return postMultipart<DocumentUploadResponse>('/api/documents/upload', formData);
}

/**
 * List all uploaded documents.
 */
export async function listDocuments(limit: number = 50, domain?: string): Promise<DocumentListResponse> {
  const url = `/api/documents?limit=${limit}${domain ? `&domain=${encodeURIComponent(domain)}` : ''}`;
  return get<DocumentListResponse>(url);
}

/**
 * Get details of a specific document.
 */
export async function getDocument(id: string): Promise<UploadedDocument> {
  return get<UploadedDocument>(`/api/documents/${id}`);
}

/**
 * Delete a document.
 */
export async function deleteDocument(id: string): Promise<{ success: boolean }> {
  return del<{ success: boolean }>(`/api/documents/${id}`);
}

/**
 * Re-sync a document to Ocean knowledge system.
 */
export async function resyncDocument(id: string): Promise<{ success: boolean; syncedToOcean: boolean }> {
  return post<{ success: boolean; syncedToOcean: boolean }>(`/api/documents/${id}/resync`, {});
}

/**
 * Search documents using Fisher-Rao semantic similarity.
 */
export async function searchDocuments(params: DocumentSearchParams): Promise<DocumentSearchResponse> {
  return post<DocumentSearchResponse>('/api/documents/search', params);
}

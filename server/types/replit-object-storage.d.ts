/**
 * Type declarations for @replit/object-storage
 * This package only exists in Replit environment
 */

declare module '@replit/object-storage' {
  export interface UploadResult {
    ok: boolean;
    error?: string;
  }

  export interface DownloadResult {
    ok: boolean;
    value: string;
    error?: string;
  }

  export interface ListObject {
    name: string;
    size?: number;
    lastModified?: string;
  }

  export interface ListResult {
    ok: boolean;
    value: ListObject[];
    error?: string;
  }

  export interface DeleteResult {
    ok: boolean;
    error?: string;
  }

  export class Client {
    constructor();
    uploadFromText(path: string, content: string): Promise<UploadResult>;
    downloadAsText(path: string): Promise<DownloadResult>;
    list(): Promise<ListResult>;
    delete(path: string): Promise<DeleteResult>;
  }
}

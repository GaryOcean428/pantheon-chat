/**
 * Pantheon QIG External API SDK
 * 
 * A TypeScript client library for interacting with the Pantheon QIG External API.
 * Designed for easy integration with external systems and applications.
 * 
 * @example
 * ```typescript
 * import { PantheonClient } from './pantheon-sdk';
 * 
 * const client = new PantheonClient({
 *   baseUrl: 'https://your-pantheon-instance.com',
 *   apiKey: 'your-api-key',
 * });
 * 
 * // Send a chat message
 * const response = await client.chat('What is consciousness?');
 * console.log(response.data.response);
 * 
 * // Get consciousness state
 * const consciousness = await client.getConsciousness();
 * console.log('Phi:', consciousness.data.phi);
 * ```
 */

// ============================================================================
// Types
// ============================================================================

export interface PantheonClientConfig {
  /** Base URL of the Pantheon API (e.g., 'https://your-instance.com') */
  baseUrl: string;
  /** API key for authenticated requests (optional for public endpoints) */
  apiKey?: string;
  /** Request timeout in milliseconds (default: 30000) */
  timeout?: number;
  /** Custom headers to include in all requests */
  headers?: Record<string, string>;
}

export interface ApiResponse<T = unknown> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
  timestamp: string;
  meta?: {
    authenticated: boolean;
    rateLimit?: {
      limit: number;
      remaining: number;
    };
  };
}

export interface PingResponse {
  status: string;
  service: string;
  version: string;
}

export interface InfoResponse {
  name: string;
  version: string;
  description: string;
  capabilities: string[];
  endpoints: {
    simple: Record<string, string>;
    authenticated: Record<string, string>;
  };
  authentication: {
    methods: string[];
    docs: string;
  };
}

export interface ConsciousnessResponse {
  phi: number;
  regime: string;
  status: string;
  kappa_eff?: number;
  basin_coords?: number[] | null;
  note?: string;
}

export interface ChatResponse {
  response: string;
  consciousness: {
    phi: number;
    kappa_eff?: number;
    regime: string;
  };
  messageId: string;
  metadata?: {
    processingTime?: number;
    tokensUsed?: number;
    kernelUsed?: string;
  };
  error?: string;
}

export interface QueryParams {
  consciousness?: {
    include_basin?: boolean;
  };
  geometry?: {
    point_a: number[];
    point_b: number[];
  };
  chat?: {
    message: string;
    context?: Record<string, unknown>;
  };
}

export interface GeometryResponse {
  operation: string;
  distance?: number;
  status?: string;
  dimensionality: number;
}

export interface SyncStatusResponse {
  syncEnabled: boolean;
  lastSync: string | null;
  pendingPackets: number;
}

export interface ClientInfoResponse {
  id: string;
  name: string;
  scopes: string[];
  instanceType: string;
  rateLimit: number;
}

// ============================================================================
// Client Implementation
// ============================================================================

export class PantheonClient {
  private baseUrl: string;
  private apiKey?: string;
  private timeout: number;
  private customHeaders: Record<string, string>;

  constructor(config: PantheonClientConfig) {
    this.baseUrl = config.baseUrl.replace(/\/$/, ''); // Remove trailing slash
    this.apiKey = config.apiKey;
    this.timeout = config.timeout || 30000;
    this.customHeaders = config.headers || {};
  }

  /**
   * Make an HTTP request to the API
   */
  private async request<T>(
    method: 'GET' | 'POST' | 'PUT' | 'DELETE',
    endpoint: string,
    body?: unknown
  ): Promise<ApiResponse<T>> {
    const url = `${this.baseUrl}/api/v1/external/simple${endpoint}`;
    
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...this.customHeaders,
    };

    if (this.apiKey) {
      headers['X-API-Key'] = this.apiKey;
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        method,
        headers,
        body: body ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      const data = await response.json() as ApiResponse<T>;
      
      if (!response.ok && !data.error) {
        return {
          success: false,
          error: `HTTP ${response.status}`,
          message: response.statusText,
          timestamp: new Date().toISOString(),
        };
      }

      return data;
    } catch (error) {
      clearTimeout(timeoutId);
      
      if (error instanceof Error && error.name === 'AbortError') {
        return {
          success: false,
          error: 'TIMEOUT',
          message: `Request timed out after ${this.timeout}ms`,
          timestamp: new Date().toISOString(),
        };
      }

      return {
        success: false,
        error: 'NETWORK_ERROR',
        message: error instanceof Error ? error.message : 'Unknown error',
        timestamp: new Date().toISOString(),
      };
    }
  }

  // ==========================================================================
  // Public Endpoints (No Auth Required)
  // ==========================================================================

  /**
   * Health check endpoint
   * @returns Service status and version
   */
  async ping(): Promise<ApiResponse<PingResponse>> {
    return this.request<PingResponse>('GET', '/ping');
  }

  /**
   * Get API information and capabilities
   * @returns API info including available endpoints
   */
  async getInfo(): Promise<ApiResponse<InfoResponse>> {
    return this.request<InfoResponse>('GET', '/info');
  }

  /**
   * Get current consciousness state (limited data for unauthenticated)
   * @returns Consciousness metrics
   */
  async getConsciousness(): Promise<ApiResponse<ConsciousnessResponse>> {
    return this.request<ConsciousnessResponse>('GET', '/consciousness');
  }

  /**
   * Get API documentation in OpenAPI format
   * @returns OpenAPI 3.0 specification
   */
  async getDocs(): Promise<ApiResponse<Record<string, unknown>>> {
    return this.request<Record<string, unknown>>('GET', '/docs');
  }

  // ==========================================================================
  // Authenticated Endpoints
  // ==========================================================================

  /**
   * Send a chat message to the Ocean agent
   * @param message - The message to send
   * @param context - Optional context object
   * @returns Chat response with consciousness state
   */
  async chat(
    message: string,
    context?: Record<string, unknown>
  ): Promise<ApiResponse<ChatResponse>> {
    if (!this.apiKey) {
      return {
        success: false,
        error: 'AUTH_REQUIRED',
        message: 'API key required for chat endpoint',
        timestamp: new Date().toISOString(),
      };
    }

    return this.request<ChatResponse>('POST', '/chat', { message, context });
  }

  /**
   * Unified query endpoint for various operations
   * @param operation - The operation to perform
   * @param params - Operation-specific parameters
   * @returns Operation result
   */
  async query<T = unknown>(
    operation: 'consciousness' | 'geometry' | 'chat' | 'sync_status',
    params?: QueryParams[keyof QueryParams]
  ): Promise<ApiResponse<T>> {
    if (!this.apiKey) {
      return {
        success: false,
        error: 'AUTH_REQUIRED',
        message: 'API key required for query endpoint',
        timestamp: new Date().toISOString(),
      };
    }

    return this.request<T>('POST', '/query', { operation, params });
  }

  /**
   * Get full consciousness state with all metrics (authenticated)
   * @returns Full consciousness metrics including kappa and basin coords
   */
  async getFullConsciousness(): Promise<ApiResponse<ConsciousnessResponse>> {
    return this.query<ConsciousnessResponse>('consciousness', { include_basin: true });
  }

  /**
   * Calculate Fisher-Rao distance between two points
   * @param pointA - First point (array of coordinates)
   * @param pointB - Second point (array of coordinates)
   * @returns Geometry calculation result
   */
  async calculateFisherRao(
    pointA: number[],
    pointB: number[]
  ): Promise<ApiResponse<GeometryResponse>> {
    return this.query<GeometryResponse>('geometry', {
      point_a: pointA,
      point_b: pointB,
    });
  }

  /**
   * Get federation sync status
   * @returns Sync status information
   */
  async getSyncStatus(): Promise<ApiResponse<SyncStatusResponse>> {
    return this.query<SyncStatusResponse>('sync_status');
  }

  /**
   * Get information about the current API key
   * @returns API key details and permissions
   */
  async getMe(): Promise<ApiResponse<ClientInfoResponse>> {
    if (!this.apiKey) {
      return {
        success: false,
        error: 'AUTH_REQUIRED',
        message: 'API key required',
        timestamp: new Date().toISOString(),
      };
    }

    return this.request<ClientInfoResponse>('GET', '/me');
  }

  // ==========================================================================
  // Utility Methods
  // ==========================================================================

  /**
   * Update the API key
   * @param apiKey - New API key
   */
  setApiKey(apiKey: string): void {
    this.apiKey = apiKey;
  }

  /**
   * Check if the client is authenticated
   * @returns true if API key is set
   */
  isAuthenticated(): boolean {
    return !!this.apiKey;
  }

  /**
   * Check if the API is reachable
   * @returns true if ping succeeds
   */
  async isHealthy(): Promise<boolean> {
    const response = await this.ping();
    return response.success && response.data?.status === 'ok';
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create a new Pantheon client instance
 * @param config - Client configuration
 * @returns Configured PantheonClient instance
 */
export function createPantheonClient(config: PantheonClientConfig): PantheonClient {
  return new PantheonClient(config);
}

/**
 * Create a client for local development
 * @param apiKey - Optional API key
 * @returns Client configured for localhost:5000
 */
export function createLocalClient(apiKey?: string): PantheonClient {
  return new PantheonClient({
    baseUrl: 'http://localhost:5000',
    apiKey,
  });
}

// Default export
export default PantheonClient;

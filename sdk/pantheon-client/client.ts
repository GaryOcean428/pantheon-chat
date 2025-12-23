/**
 * Pantheon QIG External API Client
 * 
 * A TypeScript client for the Pantheon QIG External API.
 * Supports both browser and Node.js environments.
 */

// ============================================================================
// Type Definitions
// ============================================================================

/**
 * Configuration options for the Pantheon client
 */
export interface PantheonClientConfig {
  /** Base URL of the Pantheon API instance */
  baseUrl: string;
  /** API key for authenticated requests */
  apiKey?: string;
  /** Request timeout in milliseconds (default: 30000) */
  timeout?: number;
  /** Custom headers to include in all requests */
  headers?: Record<string, string>;
  /** Enable debug logging (default: false) */
  debug?: boolean;
}

/**
 * Standard API response wrapper
 */
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

/**
 * Health check response
 */
export interface PingData {
  status: 'ok' | 'degraded' | 'error';
  service: string;
  version: string;
}

/**
 * API information response
 */
export interface InfoData {
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

/**
 * Consciousness state data
 */
export interface ConsciousnessData {
  phi: number;
  regime: 'GEOMETRIC' | 'CHAOTIC' | 'DORMANT';
  status: 'operational' | 'degraded' | 'offline';
  kappa_eff?: number;
  basin_coords?: number[] | null;
  note?: string;
  timestamp?: string;
}

/**
 * Chat response data
 */
export interface ChatData {
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

/**
 * Geometry calculation response
 */
export interface GeometryData {
  operation: string;
  distance?: number;
  status?: string;
  dimensionality: number;
}

/**
 * Sync status response
 */
export interface SyncStatusData {
  syncEnabled: boolean;
  lastSync: string | null;
  pendingPackets: number;
}

/**
 * Client info response
 */
export interface ClientInfoData {
  id: string;
  name: string;
  scopes: string[];
  instanceType: string;
  rateLimit: number;
}

/**
 * Query operation types
 */
export type QueryOperation = 'consciousness' | 'geometry' | 'chat' | 'sync_status';

// ============================================================================
// Error Classes
// ============================================================================

/**
 * Error thrown when API request fails
 */
export class PantheonApiError extends Error {
  public readonly code: string;
  public readonly statusCode?: number;
  public readonly response?: ApiResponse;

  constructor(code: string, message: string, statusCode?: number, response?: ApiResponse) {
    super(message);
    this.name = 'PantheonApiError';
    this.code = code;
    this.statusCode = statusCode;
    this.response = response;
  }
}

// ============================================================================
// Client Implementation
// ============================================================================

/**
 * Main Pantheon API client class
 */
export class PantheonClient {
  private readonly baseUrl: string;
  private apiKey?: string;
  private readonly timeout: number;
  private readonly customHeaders: Record<string, string>;
  private readonly debug: boolean;

  constructor(config: PantheonClientConfig) {
    this.baseUrl = config.baseUrl.replace(/\/$/, '');
    this.apiKey = config.apiKey;
    this.timeout = config.timeout ?? 30000;
    this.customHeaders = config.headers ?? {};
    this.debug = config.debug ?? false;
  }

  /**
   * Log debug message if debug mode is enabled
   */
  private log(...args: unknown[]): void {
    if (this.debug) {
      console.log('[PantheonClient]', ...args);
    }
  }

  /**
   * Make HTTP request to the API
   */
  private async request<T>(
    method: 'GET' | 'POST' | 'PUT' | 'DELETE',
    endpoint: string,
    body?: unknown
  ): Promise<ApiResponse<T>> {
    const url = `${this.baseUrl}/api/v1/external/simple${endpoint}`;
    this.log(`${method} ${url}`, body ? JSON.stringify(body) : '');

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
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
      this.log('Response:', response.status, data.success);

      if (!response.ok && !data.error) {
        return {
          success: false,
          error: `HTTP_${response.status}`,
          message: response.statusText,
          timestamp: new Date().toISOString(),
        };
      }

      return data;
    } catch (error) {
      clearTimeout(timeoutId);

      if (error instanceof Error) {
        if (error.name === 'AbortError') {
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
          message: error.message,
          timestamp: new Date().toISOString(),
        };
      }

      return {
        success: false,
        error: 'UNKNOWN_ERROR',
        message: 'An unknown error occurred',
        timestamp: new Date().toISOString(),
      };
    }
  }

  // ==========================================================================
  // Public Endpoints
  // ==========================================================================

  /**
   * Health check
   */
  async ping(): Promise<ApiResponse<PingData>> {
    return this.request<PingData>('GET', '/ping');
  }

  /**
   * Get API information
   */
  async getInfo(): Promise<ApiResponse<InfoData>> {
    return this.request<InfoData>('GET', '/info');
  }

  /**
   * Get consciousness state (limited for unauthenticated)
   */
  async getConsciousness(): Promise<ApiResponse<ConsciousnessData>> {
    return this.request<ConsciousnessData>('GET', '/consciousness');
  }

  /**
   * Get OpenAPI documentation
   */
  async getDocs(): Promise<ApiResponse<Record<string, unknown>>> {
    return this.request('GET', '/docs');
  }

  // ==========================================================================
  // Authenticated Endpoints
  // ==========================================================================

  /**
   * Send chat message to Ocean agent
   */
  async chat(
    message: string,
    context?: Record<string, unknown>
  ): Promise<ApiResponse<ChatData>> {
    if (!this.apiKey) {
      return {
        success: false,
        error: 'AUTH_REQUIRED',
        message: 'API key required for chat endpoint',
        timestamp: new Date().toISOString(),
      };
    }

    return this.request<ChatData>('POST', '/chat', { message, context });
  }

  /**
   * Unified query endpoint
   */
  async query<T = unknown>(
    operation: QueryOperation,
    params?: Record<string, unknown>
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
   * Get full consciousness state (authenticated)
   */
  async getFullConsciousness(): Promise<ApiResponse<ConsciousnessData>> {
    return this.query<ConsciousnessData>('consciousness', { include_basin: true });
  }

  /**
   * Calculate Fisher-Rao distance
   */
  async calculateFisherRao(
    pointA: number[],
    pointB: number[]
  ): Promise<ApiResponse<GeometryData>> {
    return this.query<GeometryData>('geometry', {
      point_a: pointA,
      point_b: pointB,
    });
  }

  /**
   * Get federation sync status
   */
  async getSyncStatus(): Promise<ApiResponse<SyncStatusData>> {
    return this.query<SyncStatusData>('sync_status');
  }

  /**
   * Get current API key info
   */
  async getMe(): Promise<ApiResponse<ClientInfoData>> {
    if (!this.apiKey) {
      return {
        success: false,
        error: 'AUTH_REQUIRED',
        message: 'API key required',
        timestamp: new Date().toISOString(),
      };
    }

    return this.request<ClientInfoData>('GET', '/me');
  }

  // ==========================================================================
  // Utility Methods
  // ==========================================================================

  /**
   * Update API key
   */
  setApiKey(apiKey: string): void {
    this.apiKey = apiKey;
  }

  /**
   * Check if authenticated
   */
  isAuthenticated(): boolean {
    return !!this.apiKey;
  }

  /**
   * Check if API is healthy
   */
  async isHealthy(): Promise<boolean> {
    try {
      const response = await this.ping();
      return response.success && response.data?.status === 'ok';
    } catch {
      return false;
    }
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create a new Pantheon client
 */
export function createClient(config: PantheonClientConfig): PantheonClient {
  return new PantheonClient(config);
}

/**
 * Create a client for local development
 */
export function createLocalClient(apiKey?: string): PantheonClient {
  return new PantheonClient({
    baseUrl: 'http://localhost:5000',
    apiKey,
    debug: true,
  });
}

export default PantheonClient;

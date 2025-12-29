/**
 * Dynamic API Base URL Utility
 * 
 * Automatically detects whether the app is running in development or production
 * and returns the correct base URL for API documentation and external integrations.
 * 
 * Environment Detection:
 * - Development: Uses the current window.location.origin (e.g., https://xxx.replit.dev)
 * - Production: Uses the deployed domain (e.g., https://xxx.replit.app)
 * 
 * Note: API keys should NEVER be auto-detected. Users must explicitly provide API keys
 * through secure means (environment variables, secrets management).
 * 
 * SSR Safety: All functions check for window availability before accessing it.
 */

export interface ApiUrlInfo {
  baseUrl: string;
  apiUrl: string;
  externalApiUrl: string;
  simpleApiUrl: string;
  wsUrl: string;
  environment: 'development' | 'production';
  isSecure: boolean;
}

/** Default SSR-safe base URL */
const SSR_DEFAULT_URL = 'http://localhost:5000';

/**
 * Safely checks if window is available (SSR-safe)
 */
function isClient(): boolean {
  return typeof window !== 'undefined' && typeof window.location !== 'undefined';
}

/**
 * Detects current environment based on hostname patterns
 * SSR-safe: returns 'development' during server-side rendering
 */
export function detectEnvironment(): 'development' | 'production' {
  if (!isClient()) {
    return 'development';
  }
  
  try {
    const hostname = window.location.hostname;
    
    // Replit dev environments use .replit.dev or contain 'dev' in subdomain
    // Production uses .replit.app
    if (hostname.includes('.replit.dev') || 
        hostname.includes('-dev.') ||
        hostname === 'localhost' ||
        hostname === '127.0.0.1' ||
        hostname.includes(':')) {
      return 'development';
    }
    
    return 'production';
  } catch {
    return 'development';
  }
}

/**
 * Gets the base URL for the current environment
 * SSR-safe: returns localhost during server-side rendering
 * 
 * @returns The base URL (e.g., https://xxx.replit.app or https://xxx.replit.dev)
 */
export function getBaseUrl(): string {
  if (!isClient()) {
    return SSR_DEFAULT_URL;
  }
  
  try {
    return window.location.origin;
  } catch {
    return SSR_DEFAULT_URL;
  }
}

/**
 * Gets comprehensive API URL information for the current environment
 * SSR-safe: returns sensible defaults during server-side rendering
 * 
 * @returns Object containing all API URL variants
 */
export function getApiUrls(): ApiUrlInfo {
  const baseUrl = getBaseUrl();
  const environment = detectEnvironment();
  const isSecure = baseUrl.startsWith('https://');
  const wsProtocol = isSecure ? 'wss://' : 'ws://';
  const host = baseUrl.replace(/^https?:\/\//, '');
  
  return {
    baseUrl,
    apiUrl: `${baseUrl}/api`,
    externalApiUrl: `${baseUrl}/api/v1/external`,
    simpleApiUrl: `${baseUrl}/api/v1/external/simple`,
    wsUrl: `${wsProtocol}${host}`,
    environment,
    isSecure,
  };
}

/**
 * Gets a display-friendly URL for documentation
 * Shows the actual URL that will be used in the current environment
 * 
 * @param path - API path to append (e.g., '/chat')
 * @returns Full URL for documentation display
 */
export function getDocUrl(path: string = ''): string {
  const urls = getApiUrls();
  return `${urls.externalApiUrl}${path}`;
}

/**
 * Gets environment badge text for UI display
 */
export function getEnvironmentBadge(): { text: string; variant: 'default' | 'secondary' } {
  const env = detectEnvironment();
  if (env === 'development') {
    return { text: 'Development', variant: 'secondary' };
  }
  return { text: 'Production', variant: 'default' };
}

/**
 * Generates curl example with the correct base URL
 * 
 * @param method - HTTP method
 * @param endpoint - API endpoint path
 * @param body - Optional request body
 * @returns Formatted curl command
 */
export function generateCurlExample(
  method: 'GET' | 'POST' | 'PUT' | 'DELETE',
  endpoint: string,
  body?: object
): string {
  const urls = getApiUrls();
  const fullUrl = `${urls.externalApiUrl}${endpoint}`;
  
  let curl = `curl -X ${method} ${fullUrl}`;
  curl += ` \\\n  -H "Authorization: Bearer YOUR_API_KEY"`;
  
  if (body) {
    curl += ` \\\n  -H "Content-Type: application/json"`;
    curl += ` \\\n  -d '${JSON.stringify(body, null, 2).replace(/\n/g, '\n  ')}'`;
  }
  
  return curl;
}

export default getApiUrls;

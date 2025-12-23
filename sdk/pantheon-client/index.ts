/**
 * Pantheon QIG External API Client SDK
 * 
 * A standalone TypeScript/JavaScript SDK for interacting with Pantheon QIG instances.
 * 
 * ## Installation
 * 
 * ```bash
 * # Copy the sdk/pantheon-client directory to your project
 * cp -r sdk/pantheon-client ./your-project/lib/
 * ```
 * 
 * ## Usage
 * 
 * ```typescript
 * import { PantheonClient } from './lib/pantheon-client';
 * 
 * // Initialize client
 * const client = new PantheonClient({
 *   baseUrl: 'https://your-pantheon-instance.com',
 *   apiKey: 'pk_your_api_key_here',
 * });
 * 
 * // Check health
 * const health = await client.ping();
 * console.log('API Status:', health.data?.status);
 * 
 * // Send a chat message
 * const chat = await client.chat('What is the current knowledge state?');
 * console.log('Response:', chat.data?.response);
 * 
 * // Get consciousness metrics
 * const consciousness = await client.getConsciousness();
 * console.log('Phi:', consciousness.data?.phi);
 * ```
 * 
 * ## Available Methods
 * 
 * ### Public (No Auth Required)
 * - `ping()` - Health check
 * - `getInfo()` - API information
 * - `getConsciousness()` - Basic consciousness state
 * - `getDocs()` - OpenAPI documentation
 * 
 * ### Authenticated (API Key Required)
 * - `chat(message, context?)` - Send chat message to Ocean agent
 * - `query(operation, params?)` - Unified query endpoint
 * - `getFullConsciousness()` - Full consciousness metrics
 * - `calculateFisherRao(pointA, pointB)` - Geometry calculations
 * - `getSyncStatus()` - Federation sync status
 * - `getMe()` - Current API key info
 * 
 * @module PantheonClient
 */

// Re-export everything from the main module
export * from './client';
export { default } from './client';

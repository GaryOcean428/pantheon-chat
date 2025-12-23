/**
 * External API Barrel Export
 * 
 * Exports all external API routes and authentication utilities.
 * 
 * Primary entry points for external systems:
 * - /api/v1/external/v1 - Unified API (recommended for new integrations)
 * - /api/v1/external/zeus/* - Zeus chat endpoints
 * - /api/v1/external/documents/* - Document management
 * - /api/v1/external/simple/* - Simplified wrapper API
 */

export { externalApiRouter as externalRouter } from './routes';
export { externalZeusRouter as zeusRouter } from './zeus';
export { externalDocumentsRouter as documentsRouter } from './documents';
export { unifiedApiRouter } from './unified';
export { authenticateExternalApi, requireScopes, createApiKey, listApiKeys, revokeApiKey } from './auth';
export { initExternalWebSocket } from './websocket';
export { simpleApiRouter } from './simple-api';

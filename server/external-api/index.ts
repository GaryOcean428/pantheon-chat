/**
 * External API Barrel Export
 * 
 * Exports all external API routes and authentication utilities.
 */

export { externalApiRouter as externalRouter } from './routes';
export { externalZeusRouter as zeusRouter } from './zeus';
export { externalDocumentsRouter as documentsRouter } from './documents';
export { authenticateExternalApi, requireScopes, createApiKey, listApiKeys, revokeApiKey } from './auth';
export { initExternalWebSocket } from './websocket';
export { simpleApiRouter } from './simple-api';

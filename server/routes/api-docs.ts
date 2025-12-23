/**
 * API Documentation Routes
 * 
 * Serves OpenAPI specification and documentation UI.
 */

import { Router, Request, Response } from 'express';
import * as fs from 'fs';
import * as path from 'path';

const router = Router();

/**
 * GET /api/docs/openapi.yaml
 * 
 * Returns the OpenAPI specification in YAML format.
 */
router.get('/openapi.yaml', (_req: Request, res: Response) => {
  try {
    const specPath = path.join(__dirname, '../../docs/api/openapi.yaml');
    const spec = fs.readFileSync(specPath, 'utf-8');
    res.type('application/yaml').send(spec);
  } catch (error) {
    console.error('[API Docs] Failed to read OpenAPI spec:', error);
    res.status(500).json({ error: 'OpenAPI specification not available' });
  }
});

/**
 * GET /api/docs/openapi.json
 * 
 * Returns the OpenAPI specification in JSON format.
 */
router.get('/openapi.json', async (_req: Request, res: Response) => {
  try {
    const specPath = path.join(__dirname, '../../docs/api/openapi.yaml');
    const yamlContent = fs.readFileSync(specPath, 'utf-8');
    
    // Dynamic import for yaml parser
    const yaml = await import('js-yaml');
    const spec = yaml.load(yamlContent);
    
    res.json(spec);
  } catch (error) {
    console.error('[API Docs] Failed to convert OpenAPI spec:', error);
    res.status(500).json({ error: 'OpenAPI specification not available' });
  }
});

/**
 * GET /api/docs
 * 
 * Returns a simple HTML page with links to API documentation.
 */
router.get('/', (_req: Request, res: Response) => {
  const html = `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Pantheon Chat API Documentation</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      max-width: 800px;
      margin: 0 auto;
      padding: 40px 20px;
      background: #0a0a0a;
      color: #e5e5e5;
    }
    h1 {
      color: #fbbf24;
      border-bottom: 2px solid #fbbf24;
      padding-bottom: 10px;
    }
    h2 {
      color: #60a5fa;
      margin-top: 30px;
    }
    a {
      color: #fbbf24;
      text-decoration: none;
    }
    a:hover {
      text-decoration: underline;
    }
    .endpoint {
      background: #1a1a1a;
      border: 1px solid #333;
      border-radius: 8px;
      padding: 15px;
      margin: 10px 0;
    }
    .method {
      display: inline-block;
      padding: 3px 8px;
      border-radius: 4px;
      font-weight: bold;
      font-size: 12px;
      margin-right: 10px;
    }
    .get { background: #22c55e; color: #000; }
    .post { background: #3b82f6; color: #fff; }
    .delete { background: #ef4444; color: #fff; }
    code {
      background: #262626;
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 14px;
    }
    pre {
      background: #1a1a1a;
      border: 1px solid #333;
      border-radius: 8px;
      padding: 15px;
      overflow-x: auto;
    }
  </style>
</head>
<body>
  <h1>⚡ Pantheon Chat API</h1>
  
  <p>Welcome to the Pantheon Chat External API documentation.</p>
  
  <h2>📚 API Specification</h2>
  <ul>
    <li><a href="/api/docs/openapi.yaml">OpenAPI Specification (YAML)</a></li>
    <li><a href="/api/docs/openapi.json">OpenAPI Specification (JSON)</a></li>
  </ul>
  
  <h2>🔐 Authentication</h2>
  <p>All API requests require authentication via Bearer token:</p>
  <pre>Authorization: Bearer your_api_key_here</pre>
  
  <h2>🚀 Quick Start</h2>
  
  <h3>Zeus Chat</h3>
  <div class="endpoint">
    <span class="method post">POST</span>
    <code>/external/v1/zeus/chat</code>
    <p>Send a message to Zeus and receive an intelligent response.</p>
  </div>
  
  <h3>Document Upload</h3>
  <div class="endpoint">
    <span class="method post">POST</span>
    <code>/external/v1/documents/upload</code>
    <p>Upload markdown, text, or PDF documents to sync with Ocean knowledge.</p>
  </div>
  
  <h3>Document Search</h3>
  <div class="endpoint">
    <span class="method post">POST</span>
    <code>/external/v1/documents/search</code>
    <p>Search documents using Fisher-Rao semantic similarity.</p>
  </div>
  
  <h2>📖 Example: Zeus Chat</h2>
  <pre>
curl -X POST https://your-domain/external/v1/zeus/chat \\
  -H "Authorization: Bearer your_api_key" \\
  -H "Content-Type: application/json" \\
  -d '{"message": "What is quantum entanglement?"}'</pre>
  
  <h2>📖 Example: Document Upload</h2>
  <pre>
curl -X POST https://your-domain/external/v1/documents/upload \\
  -H "Authorization: Bearer your_api_key" \\
  -F "files=@my-document.md" \\
  -F "domain=physics"</pre>
  
  <h2>⚡ Rate Limits</h2>
  <ul>
    <li>Zeus Chat: 30 requests/minute</li>
    <li>Zeus Search: 20 requests/minute</li>
    <li>Document Upload: 10 requests/minute</li>
    <li>Document Search: 30 requests/minute</li>
  </ul>
  
  <hr style="margin-top: 40px; border-color: #333;">
  <p style="color: #666;">Powered by the Olympus Pantheon</p>
</body>
</html>
  `;
  res.type('html').send(html);
});

export default router;

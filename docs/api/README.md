# Pantheon Chat API Documentation

This directory contains the OpenAPI specification and documentation for the Pantheon Chat External API.

## Files

- `openapi.yaml` - OpenAPI 3.0 specification in YAML format

## Viewing Documentation

The API documentation is available at:

- **HTML Documentation**: `/api/docs`
- **OpenAPI YAML**: `/api/docs/openapi.yaml`
- **OpenAPI JSON**: `/api/docs/openapi.json`

## API Overview

### Authentication

All external API endpoints require authentication via API key:

```
Authorization: Bearer your_api_key_here
```

API keys can be created via the admin panel or programmatically.

### Endpoints

#### Zeus Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/external/v1/zeus/chat` | Send a message to Zeus |
| POST | `/external/v1/zeus/search` | Search the knowledge base |
| GET | `/external/v1/zeus/sessions` | List chat sessions |
| POST | `/external/v1/zeus/sessions` | Create a new session |
| GET | `/external/v1/zeus/sessions/:id/messages` | Get session messages |

#### Documents

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/external/v1/documents/upload` | Upload documents |
| GET | `/external/v1/documents` | List documents |
| GET | `/external/v1/documents/:id` | Get document details |
| DELETE | `/external/v1/documents/:id` | Delete a document |
| POST | `/external/v1/documents/search` | Search documents |

### Scopes

API keys can be configured with different permission scopes:

- `zeus:chat` - Send messages to Zeus
- `zeus:search` - Search the knowledge base
- `documents:read` - List and search documents
- `documents:write` - Upload and delete documents
- `*` - All permissions

### Rate Limits

| Endpoint | Limit |
|----------|-------|
| Zeus Chat | 30/minute |
| Zeus Search | 20/minute |
| Document Upload | 10/minute |
| Document Search | 30/minute |

## Examples

### Zeus Chat

```bash
curl -X POST https://your-domain/external/v1/zeus/chat \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is quantum entanglement?"}'
```

### Document Upload

```bash
curl -X POST https://your-domain/external/v1/documents/upload \
  -H "Authorization: Bearer your_api_key" \
  -F "files=@my-document.md" \
  -F "domain=physics"
```

### Document Search

```bash
curl -X POST https://your-domain/external/v1/documents/search \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"query": "neural networks", "limit": 10}'
```

## Swagger/OpenAPI Tools

You can import the OpenAPI specification into various tools:

- **Swagger UI**: Use the YAML file with Swagger UI
- **Postman**: Import the OpenAPI spec to create a collection
- **Insomnia**: Import the spec for API testing
- **Code Generators**: Use OpenAPI Generator to create client SDKs

## Support

For API support, please contact the development team or open an issue in the repository.

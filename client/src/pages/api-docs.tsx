import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Badge,
  Button,
  ScrollArea,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui";
import { Copy, ExternalLink, Key, Upload, MessageSquare, Search, Zap, Shield } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

function CodeBlock({ code, language = "bash" }: { code: string; language?: string }) {
  const { toast } = useToast();

  const copyToClipboard = () => {
    navigator.clipboard.writeText(code);
    toast({ title: "Copied to clipboard" });
  };

  return (
    <div className="relative group">
      <pre className="bg-muted/50 border rounded-md p-4 overflow-x-auto text-sm">
        <code className={`language-${language}`}>{code}</code>
      </pre>
      <Button
        size="icon"
        variant="ghost"
        className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity"
        onClick={copyToClipboard}
        data-testid="button-copy-code"
      >
        <Copy className="h-4 w-4" />
      </Button>
    </div>
  );
}

function EndpointCard({
  method,
  path,
  description,
  requestBody,
  response,
  curlExample,
}: {
  method: "GET" | "POST" | "PUT" | "DELETE";
  path: string;
  description: string;
  requestBody?: string;
  response?: string;
  curlExample: string;
}) {
  const methodColors = {
    GET: "bg-green-500/20 text-green-400 border-green-500/30",
    POST: "bg-blue-500/20 text-blue-400 border-blue-500/30",
    PUT: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
    DELETE: "bg-red-500/20 text-red-400 border-red-500/30",
  };

  return (
    <Card className="mb-4">
      <CardHeader className="pb-2">
        <div className="flex items-center gap-3">
          <Badge className={methodColors[method]} variant="outline">
            {method}
          </Badge>
          <code className="text-sm font-mono">{path}</code>
        </div>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="curl" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="curl" data-testid="tab-curl">cURL</TabsTrigger>
            <TabsTrigger value="request" data-testid="tab-request">Request</TabsTrigger>
            <TabsTrigger value="response" data-testid="tab-response">Response</TabsTrigger>
          </TabsList>
          <TabsContent value="curl" className="mt-4">
            <CodeBlock code={curlExample} language="bash" />
          </TabsContent>
          <TabsContent value="request" className="mt-4">
            {requestBody ? (
              <CodeBlock code={requestBody} language="json" />
            ) : (
              <p className="text-muted-foreground text-sm">No request body required</p>
            )}
          </TabsContent>
          <TabsContent value="response" className="mt-4">
            {response ? (
              <CodeBlock code={response} language="json" />
            ) : (
              <p className="text-muted-foreground text-sm">Response varies</p>
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}

export default function ApiDocsPage() {
  const baseUrl = typeof window !== "undefined" ? window.location.origin : "https://your-domain.replit.app";

  return (
    <ScrollArea className="h-full">
      <div className="container max-w-5xl py-8 px-4">
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2" data-testid="text-page-title">
            External API Documentation
          </h1>
          <p className="text-muted-foreground">
            Integrate Ocean Agentic Platform into your applications using our REST API.
          </p>
        </div>

        <div className="grid gap-4 md:grid-cols-3 mb-8">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-lg flex items-center gap-2">
                <MessageSquare className="h-5 w-5 text-blue-400" />
                Zeus Chat
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                Send messages to Zeus for intelligent responses powered by the Olympus Pantheon.
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-lg flex items-center gap-2">
                <Upload className="h-5 w-5 text-green-400" />
                Document Upload
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                Upload documents to sync with Ocean's geometric knowledge manifold.
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-lg flex items-center gap-2">
                <Search className="h-5 w-5 text-purple-400" />
                Semantic Search
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                Search documents using Fisher-Rao distance on the information manifold.
              </p>
            </CardContent>
          </Card>
        </div>

        <Card className="mb-8">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Key className="h-5 w-5" />
              Authentication
            </CardTitle>
            <CardDescription>
              All API requests require Bearer token authentication
            </CardDescription>
          </CardHeader>
          <CardContent>
            <CodeBlock
              code={`curl -X POST ${baseUrl}/api/v1/external/zeus/chat \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"message": "Hello"}'`}
            />
            <div className="mt-4 p-4 bg-yellow-500/10 border border-yellow-500/30 rounded-md">
              <p className="text-sm text-yellow-400 flex items-center gap-2">
                <Shield className="h-4 w-4" />
                API keys can be obtained from your account settings or by contacting an administrator.
              </p>
            </div>
          </CardContent>
        </Card>

        <div className="mb-8">
          <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
            <Zap className="h-6 w-6 text-yellow-400" />
            Endpoints
          </h2>

          <h3 className="text-lg font-semibold mb-3 mt-6">Zeus Chat</h3>

          <EndpointCard
            method="POST"
            path="/api/v1/external/zeus/chat"
            description="Send a message to Zeus and receive an intelligent response from the Olympus Pantheon."
            requestBody={`{
  "message": "What is quantum entanglement?",
  "consult_pantheon": true,
  "session_id": "optional-session-id"
}`}
            response={`{
  "success": true,
  "response": "Quantum entanglement is a phenomenon...",
  "god": "zeus",
  "session_id": "session-123",
  "metadata": {
    "consulted_gods": ["athena", "hephaestus"],
    "fisher_rao_distance": 0.234,
    "phi": 0.78
  }
}`}
            curlExample={`curl -X POST ${baseUrl}/api/v1/external/zeus/chat \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "message": "What is quantum entanglement?",
    "consult_pantheon": true
  }'`}
          />

          <EndpointCard
            method="POST"
            path="/api/v1/external/zeus/stream"
            description="Stream responses from Zeus for real-time interaction (SSE)."
            requestBody={`{
  "message": "Explain the theory of relativity",
  "consult_pantheon": true
}`}
            response={`data: {"type": "token", "content": "The"}
data: {"type": "token", "content": " theory"}
data: {"type": "token", "content": " of"}
data: {"type": "complete", "metadata": {...}}`}
            curlExample={`curl -N -X POST ${baseUrl}/api/v1/external/zeus/stream \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -H "Accept: text/event-stream" \\
  -d '{"message": "Explain the theory of relativity"}'`}
          />

          <h3 className="text-lg font-semibold mb-3 mt-6">Document Management</h3>

          <EndpointCard
            method="POST"
            path="/api/v1/external/documents/upload"
            description="Upload documents to be processed and integrated into Ocean's knowledge manifold."
            requestBody={`Content-Type: multipart/form-data

files: (binary)
domain: "physics" (optional)
metadata: {"source": "research", "author": "..."}`}
            response={`{
  "success": true,
  "documents": [
    {
      "id": "doc-abc123",
      "filename": "quantum.md",
      "size": 4523,
      "coordized": true,
      "basin_coordinates": [0.23, 0.45, ...]
    }
  ],
  "total_processed": 1
}`}
            curlExample={`curl -X POST ${baseUrl}/api/v1/external/documents/upload \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -F "files=@my-document.md" \\
  -F "domain=physics"`}
          />

          <EndpointCard
            method="POST"
            path="/api/v1/external/documents/search"
            description="Search documents using Fisher-Rao semantic similarity on the information manifold."
            requestBody={`{
  "query": "quantum computing algorithms",
  "limit": 10,
  "domain": "physics"
}`}
            response={`{
  "success": true,
  "results": [
    {
      "document_id": "doc-abc123",
      "title": "Introduction to Quantum Computing",
      "snippet": "...",
      "fisher_rao_distance": 0.156,
      "relevance_score": 0.92
    }
  ],
  "total": 5
}`}
            curlExample={`curl -X POST ${baseUrl}/api/v1/external/documents/search \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "query": "quantum computing algorithms",
    "limit": 10
  }'`}
          />

          <h3 className="text-lg font-semibold mb-3 mt-6">Health & Status</h3>

          <EndpointCard
            method="GET"
            path="/api/v1/external/health"
            description="Check the health status of the external API and underlying services."
            response={`{
  "status": "healthy",
  "services": {
    "zeus": "online",
    "python_backend": "online",
    "database": "online"
  },
  "timestamp": "2024-01-15T12:00:00Z"
}`}
            curlExample={`curl ${baseUrl}/api/v1/external/health \\
  -H "Authorization: Bearer YOUR_API_KEY"`}
          />
        </div>

        <Card className="mb-8">
          <CardHeader>
            <CardTitle>Rate Limits</CardTitle>
            <CardDescription>
              API requests are rate-limited to ensure fair usage
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-2">
              <div className="flex justify-between items-center py-2 border-b">
                <span className="font-medium">Zeus Chat</span>
                <Badge variant="outline">30 requests/minute</Badge>
              </div>
              <div className="flex justify-between items-center py-2 border-b">
                <span className="font-medium">Zeus Stream</span>
                <Badge variant="outline">20 requests/minute</Badge>
              </div>
              <div className="flex justify-between items-center py-2 border-b">
                <span className="font-medium">Document Upload</span>
                <Badge variant="outline">10 requests/minute</Badge>
              </div>
              <div className="flex justify-between items-center py-2">
                <span className="font-medium">Document Search</span>
                <Badge variant="outline">30 requests/minute</Badge>
              </div>
            </div>
          </CardContent>
        </Card>

        <div className="flex gap-4 justify-center">
          <Button variant="outline" asChild>
            <a href="/api/docs/openapi.yaml" target="_blank" rel="noopener noreferrer" data-testid="link-openapi-yaml">
              <ExternalLink className="h-4 w-4 mr-2" />
              OpenAPI Spec (YAML)
            </a>
          </Button>
          <Button variant="outline" asChild>
            <a href="/api/docs/openapi.json" target="_blank" rel="noopener noreferrer" data-testid="link-openapi-json">
              <ExternalLink className="h-4 w-4 mr-2" />
              OpenAPI Spec (JSON)
            </a>
          </Button>
        </div>
      </div>
    </ScrollArea>
  );
}

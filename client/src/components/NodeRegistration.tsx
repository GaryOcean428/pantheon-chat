/**
 * Node Registration Component
 * 
 * Allows users to register new federation nodes or external chat UIs
 * and get API credentials for connecting to the mesh network.
 */

import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Button,
  Input,
  Label,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  Badge,
  Alert,
  AlertDescription,
  AlertTitle,
} from '@/components/ui';
import { useToast } from '@/hooks/use-toast';
import { Copy, Key, CheckCircle2, AlertTriangle, Globe, Bot, Server } from 'lucide-react';
import { post } from '@/api';

interface RegistrationResult {
  success: boolean;
  node_id: string;
  api_key: string;
  message: string;
  endpoints: Record<string, string>;
}

export function NodeRegistration() {
  const { toast } = useToast();
  const [nodeName, setNodeName] = useState('');
  const [nodeType, setNodeType] = useState<string>('chat_ui');
  const [capabilities, setCapabilities] = useState<string[]>(['chat']);
  const [endpointUrl, setEndpointUrl] = useState('');
  const [registrationResult, setRegistrationResult] = useState<RegistrationResult | null>(null);
  const [apiKeyCopied, setApiKeyCopied] = useState(false);

  const registerMutation = useMutation({
    mutationFn: async () => {
      return post<RegistrationResult>('/api/federation/register', {
        node_name: nodeName,
        node_type: nodeType,
        capabilities,
        endpoint_url: endpointUrl || undefined
      });
    },
    onSuccess: (data: RegistrationResult) => {
      setRegistrationResult(data);
      toast({
        title: 'Node Registered!',
        description: 'Save your API key - it will not be shown again.',
      });
    },
    onError: (error: Error) => {
      toast({
        title: 'Registration Failed',
        description: error.message,
        variant: 'destructive'
      });
    }
  });

  const copyApiKey = () => {
    if (registrationResult?.api_key) {
      navigator.clipboard.writeText(registrationResult.api_key);
      setApiKeyCopied(true);
      toast({
        title: 'API Key Copied',
        description: 'Store it securely - it won\'t be shown again!',
      });
      setTimeout(() => setApiKeyCopied(false), 3000);
    }
  };

  const toggleCapability = (cap: string) => {
    setCapabilities(prev => 
      prev.includes(cap) 
        ? prev.filter(c => c !== cap)
        : [...prev, cap]
    );
  };

  const nodeTypeIcons = {
    chat_ui: <Globe className="h-4 w-4" />,
    federation_node: <Server className="h-4 w-4" />,
    agent: <Bot className="h-4 w-4" />
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Key className="h-5 w-5" />
          Register New Node
        </CardTitle>
        <CardDescription>
          Register your system to get API credentials for connecting to the mesh network
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {!registrationResult ? (
          <>
            {/* Node Name */}
            <div className="space-y-2">
              <Label htmlFor="nodeName">Node Name</Label>
              <Input
                id="nodeName"
                placeholder="my-chat-ui"
                value={nodeName}
                onChange={(e) => setNodeName(e.target.value)}
              />
              <p className="text-xs text-muted-foreground">
                A friendly name to identify your node in the mesh
              </p>
            </div>

            {/* Node Type */}
            <div className="space-y-2">
              <Label>Node Type</Label>
              <Select value={nodeType} onValueChange={setNodeType}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="chat_ui">
                    <div className="flex items-center gap-2">
                      <Globe className="h-4 w-4" />
                      External Chat UI
                    </div>
                  </SelectItem>
                  <SelectItem value="federation_node">
                    <div className="flex items-center gap-2">
                      <Server className="h-4 w-4" />
                      Federation Node
                    </div>
                  </SelectItem>
                  <SelectItem value="agent">
                    <div className="flex items-center gap-2">
                      <Bot className="h-4 w-4" />
                      Autonomous Agent
                    </div>
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Capabilities */}
            <div className="space-y-2">
              <Label>Capabilities</Label>
              <div className="flex flex-wrap gap-2">
                {['chat', 'research', 'tools', 'documents', 'streaming'].map((cap) => (
                  <Badge
                    key={cap}
                    variant={capabilities.includes(cap) ? 'default' : 'outline'}
                    className="cursor-pointer"
                    onClick={() => toggleCapability(cap)}
                  >
                    {cap}
                  </Badge>
                ))}
              </div>
              <p className="text-xs text-muted-foreground">
                Select the capabilities your node will use
              </p>
            </div>

            {/* Endpoint URL (optional) */}
            <div className="space-y-2">
              <Label htmlFor="endpointUrl">Callback Endpoint (optional)</Label>
              <Input
                id="endpointUrl"
                placeholder="https://mynode.example.com/api/callback"
                value={endpointUrl}
                onChange={(e) => setEndpointUrl(e.target.value)}
              />
              <p className="text-xs text-muted-foreground">
                For bidirectional sync - we'll push updates to this URL
              </p>
            </div>

            {/* Register Button */}
            <Button
              onClick={() => registerMutation.mutate()}
              disabled={!nodeName || registerMutation.isPending}
              className="w-full"
            >
              {registerMutation.isPending ? 'Registering...' : 'Register Node'}
            </Button>
          </>
        ) : (
          <>
            {/* Success - Show API Key */}
            <Alert className="border-green-500 bg-green-50 dark:bg-green-950">
              <CheckCircle2 className="h-4 w-4 text-green-600" />
              <AlertTitle className="text-green-800 dark:text-green-200">Registration Successful!</AlertTitle>
              <AlertDescription className="text-green-700 dark:text-green-300">
                Your node <strong>{registrationResult.node_id}</strong> has been registered.
              </AlertDescription>
            </Alert>

            {/* API Key Display */}
            <Alert className="border-amber-500 bg-amber-50 dark:bg-amber-950">
              <AlertTriangle className="h-4 w-4 text-amber-600" />
              <AlertTitle className="text-amber-800 dark:text-amber-200">Save Your API Key!</AlertTitle>
              <AlertDescription className="text-amber-700 dark:text-amber-300">
                This key will <strong>NOT</strong> be shown again. Store it securely.
              </AlertDescription>
            </Alert>

            <div className="space-y-2">
              <Label>Your API Key</Label>
              <div className="flex gap-2">
                <Input
                  value={registrationResult.api_key}
                  readOnly
                  className="font-mono text-sm"
                />
                <Button
                  variant={apiKeyCopied ? 'default' : 'outline'}
                  size="icon"
                  onClick={copyApiKey}
                >
                  {apiKeyCopied ? <CheckCircle2 className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                </Button>
              </div>
            </div>

            {/* Available Endpoints */}
            <div className="space-y-2">
              <Label>Available Endpoints</Label>
              <div className="bg-muted p-4 rounded-lg space-y-2 text-sm font-mono">
                {Object.entries(registrationResult.endpoints).map(([name, path]) => (
                  <div key={name} className="flex justify-between">
                    <span className="text-muted-foreground">{name}:</span>
                    <span>{path}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Quick Start */}
            <div className="space-y-2">
              <Label>Quick Start</Label>
              <pre className="bg-muted p-4 rounded-lg text-xs overflow-x-auto">
{`curl -X POST ${window.location.origin}/api/v1/external/chat \\
  -H "Authorization: Bearer ${registrationResult.api_key}" \\
  -H "Content-Type: application/json" \\
  -d '{"message": "Hello Zeus!"}'`}
              </pre>
            </div>

            {/* Register Another */}
            <Button
              variant="outline"
              onClick={() => {
                setRegistrationResult(null);
                setNodeName('');
                setApiKeyCopied(false);
              }}
              className="w-full"
            >
              Register Another Node
            </Button>
          </>
        )}
      </CardContent>
    </Card>
  );
}

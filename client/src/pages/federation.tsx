import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { API_ROUTES, QUERY_KEYS, get } from "@/api";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Button,
  Badge,
  Input,
  Label,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
  ScrollArea,
  Separator,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui";
import {
  Key,
  Globe,
  Activity,
  Link2,
  Copy,
  Trash2,
  Plus,
  RefreshCw,
  CheckCircle2,
  XCircle,
  Clock,
  Send,
  Server,
  Shield,
  Zap,
  Database,
  Radio,
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";

interface ApiKey {
  id: string;
  name: string;
  instanceType: string;
  scopes: string[];
  createdAt: string;
  lastUsedAt: string | null;
  rateLimit: number;
  isActive: boolean;
}

interface FederatedInstance {
  id: string;
  name: string;
  endpoint: string;
  status: "active" | "pending" | "disconnected";
  capabilities: string[];
  lastSyncAt: string | null;
  syncDirection: "inbound" | "outbound" | "bidirectional";
}

interface SyncStatus {
  isConnected: boolean;
  peerCount: number;
  lastSyncTime: string | null;
  pendingPackets: number;
  syncMode: string;
}

interface HealthStatus {
  status: "healthy" | "degraded" | "down";
  version: string;
  timestamp: string;
  capabilities: string[];
}

interface ApiTestResult {
  success: boolean;
  status: number;
  data?: unknown;
  error?: string;
  latency: number;
}

interface ConnectionTestResult {
  success: boolean;
  status?: number;
  error?: string;
  latency: number;
  remoteVersion?: string;
  capabilities?: string[];
}

export default function FederationDashboard() {
  const { toast } = useToast();
  const [newKeyName, setNewKeyName] = useState("");
  const [newKeyType, setNewKeyType] = useState("external");
  const [createdKey, setCreatedKey] = useState<string | null>(null);
  const [testEndpoint, setTestEndpoint] = useState("/consciousness/query");
  const [testResult, setTestResult] = useState<ApiTestResult | null>(null);
  
  const [connectName, setConnectName] = useState("");
  const [connectEndpoint, setConnectEndpoint] = useState("");
  const [connectApiKey, setConnectApiKey] = useState("");
  const [connectSyncDirection, setConnectSyncDirection] = useState("bidirectional");
  const [connectionTestResult, setConnectionTestResult] = useState<ConnectionTestResult | null>(null);

  const { data: health, isLoading: healthLoading, refetch: refetchHealth } = useQuery<HealthStatus>({
    queryKey: QUERY_KEYS.external.health(),
    refetchInterval: 30000,
  });

  const { data: apiKeys, isLoading: keysLoading, refetch: refetchKeys } = useQuery<{ keys: ApiKey[] }>({
    queryKey: QUERY_KEYS.federation.keys(),
  });

  const { data: instances, isLoading: instancesLoading, refetch: refetchInstances } = useQuery<{ instances: FederatedInstance[] }>({
    queryKey: QUERY_KEYS.federation.instances(),
  });

  const { data: syncStatus, isLoading: syncLoading, refetch: refetchSync } = useQuery<SyncStatus>({
    queryKey: QUERY_KEYS.federation.syncStatus(),
    refetchInterval: 5000,
  });

  const createKeyMutation = useMutation({
    mutationFn: async (data: { name: string; instanceType: string }) => {
      const response = await apiRequest("POST", API_ROUTES.federation.keys, {
        name: data.name,
        instanceType: data.instanceType,
        scopes: ["read", "write", "consciousness", "geometry", "pantheon", "sync", "chat"],
        rateLimit: 120,
      });
      return response.json();
    },
    onSuccess: (data) => {
      setCreatedKey(data.key);
      setNewKeyName("");
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.federation.keys() });
      toast({
        title: "API Key Created",
        description: "Save this key - it won't be shown again!",
      });
    },
    onError: (error) => {
      toast({
        title: "Failed to create key",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const revokeKeyMutation = useMutation({
    mutationFn: async (keyId: string) => {
      await apiRequest("DELETE", API_ROUTES.federation.key(keyId));
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.federation.keys() });
      toast({ title: "API Key Revoked" });
    },
  });

  const testApiMutation = useMutation({
    mutationFn: async (endpoint: string) => {
      const start = Date.now();
      try {
        const baseUrl = API_ROUTES.external.health.replace('/health', '');
        const data = await get(`${baseUrl}${endpoint}`);
        const latency = Date.now() - start;
        return {
          success: true,
          status: 200,
          data,
          latency,
        };
      } catch (error) {
        return {
          success: false,
          status: 0,
          error: error instanceof Error ? error.message : "Request failed",
          latency: Date.now() - start,
        };
      }
    },
    onSuccess: (result) => {
      setTestResult(result);
    },
  });

  const testConnectionMutation = useMutation({
    mutationFn: async (data: { endpoint: string; apiKey: string }) => {
      const response = await apiRequest("POST", API_ROUTES.federation.testConnection, data);
      return response.json();
    },
    onSuccess: (result) => {
      setConnectionTestResult(result);
      if (result.success) {
        toast({ title: "Connection successful", description: `Latency: ${result.latency}ms` });
      } else {
        toast({ title: "Connection failed", description: result.error, variant: "destructive" });
      }
    },
    onError: (error) => {
      toast({ title: "Test failed", description: error.message, variant: "destructive" });
    },
  });

  const connectNodeMutation = useMutation({
    mutationFn: async (data: { name: string; endpoint: string; apiKey: string; syncDirection: string }) => {
      const response = await apiRequest("POST", API_ROUTES.federation.connect, data);
      return response.json();
    },
    onSuccess: (result) => {
      setConnectName("");
      setConnectEndpoint("");
      setConnectApiKey("");
      setConnectionTestResult(null);
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.federation.instances() });
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.federation.syncStatus() });
      toast({ title: "Node connected", description: `Connected to ${result.name}` });
    },
    onError: (error) => {
      toast({ title: "Connection failed", description: error.message, variant: "destructive" });
    },
  });

  const disconnectNodeMutation = useMutation({
    mutationFn: async (instanceId: string) => {
      await apiRequest("DELETE", API_ROUTES.federation.instance(instanceId));
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.federation.instances() });
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.federation.syncStatus() });
      toast({ title: "Node disconnected" });
    },
  });

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    toast({ title: "Copied to clipboard" });
  };

  const refreshAll = () => {
    refetchHealth();
    refetchKeys();
    refetchInstances();
    refetchSync();
  };

  const testEndpoints = [
    { value: "/health", label: "Health Check" },
    { value: "/consciousness/query", label: "Consciousness Query" },
    { value: "/consciousness/metrics", label: "Consciousness Metrics" },
    { value: "/pantheon/instances", label: "Pantheon Instances" },
    { value: "/sync/status", label: "Sync Status" },
  ];

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Globe className="h-6 w-6" />
            Federation Dashboard
          </h1>
          <p className="text-muted-foreground">
            Connect and synchronize with other QIG constellations
          </p>
        </div>
        <Button onClick={refreshAll} variant="outline" size="sm" data-testid="button-refresh-all">
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh All
        </Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="pt-4">
            <div className="flex items-center gap-2">
              <Activity className={`h-5 w-5 ${health?.status === "healthy" ? "text-green-500" : "text-yellow-500"}`} />
              <div>
                <div className="text-sm text-muted-foreground">API Status</div>
                <div className="font-medium" data-testid="text-api-status">
                  {healthLoading ? "Loading..." : health?.status ?? "Unknown"}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-4">
            <div className="flex items-center gap-2">
              <Key className="h-5 w-5 text-blue-500" />
              <div>
                <div className="text-sm text-muted-foreground">API Keys</div>
                <div className="font-medium" data-testid="text-api-keys-count">
                  {keysLoading ? "..." : apiKeys?.keys?.length ?? 0} active
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-4">
            <div className="flex items-center gap-2">
              <Server className="h-5 w-5 text-purple-500" />
              <div>
                <div className="text-sm text-muted-foreground">Connected Peers</div>
                <div className="font-medium" data-testid="text-peer-count">
                  {syncLoading ? "..." : syncStatus?.peerCount ?? 0} instances
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-4">
            <div className="flex items-center gap-2">
              <Radio className={`h-5 w-5 ${syncStatus?.isConnected ? "text-green-500" : "text-gray-400"}`} />
              <div>
                <div className="text-sm text-muted-foreground">Sync Status</div>
                <div className="font-medium" data-testid="text-sync-status">
                  {syncLoading ? "..." : syncStatus?.isConnected ? "Connected" : "Disconnected"}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="keys" className="space-y-4">
        <TabsList>
          <TabsTrigger value="keys" data-testid="tab-keys">
            <Key className="h-4 w-4 mr-2" />
            API Keys
          </TabsTrigger>
          <TabsTrigger value="instances" data-testid="tab-instances">
            <Server className="h-4 w-4 mr-2" />
            Connected Instances
          </TabsTrigger>
          <TabsTrigger value="sync" data-testid="tab-sync">
            <Link2 className="h-4 w-4 mr-2" />
            Basin Sync
          </TabsTrigger>
          <TabsTrigger value="test" data-testid="tab-test">
            <Zap className="h-4 w-4 mr-2" />
            API Tester
          </TabsTrigger>
        </TabsList>

        <TabsContent value="keys" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Shield className="h-5 w-5" />
                Create Unified API Key
              </CardTitle>
              <CardDescription>
                One key per external system - includes all capabilities (consciousness, geometry, pantheon, sync, chat)
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex gap-4">
                <div className="flex-1">
                  <Label htmlFor="keyName">Instance Name</Label>
                  <Input
                    id="keyName"
                    placeholder="e.g., ocean-cluster-2, research-node-alpha"
                    value={newKeyName}
                    onChange={(e) => setNewKeyName(e.target.value)}
                    data-testid="input-key-name"
                  />
                </div>
                <div className="w-48">
                  <Label htmlFor="keyType">Instance Type</Label>
                  <Select value={newKeyType} onValueChange={setNewKeyType}>
                    <SelectTrigger id="keyType" data-testid="select-key-type">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="external">External System</SelectItem>
                      <SelectItem value="headless">Headless Client</SelectItem>
                      <SelectItem value="federation">Federation Node</SelectItem>
                      <SelectItem value="research">Research Instance</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="flex items-end">
                  <Button
                    onClick={() => createKeyMutation.mutate({ name: newKeyName, instanceType: newKeyType })}
                    disabled={!newKeyName || createKeyMutation.isPending}
                    data-testid="button-create-key"
                  >
                    <Plus className="h-4 w-4 mr-2" />
                    Create Key
                  </Button>
                </div>
              </div>

              {createdKey && (
                <div className="p-4 bg-green-500/10 border border-green-500/30 rounded-md space-y-2">
                  <div className="flex items-center gap-2 text-green-600 dark:text-green-400 font-medium">
                    <CheckCircle2 className="h-4 w-4" />
                    API Key Created - Save This Now!
                  </div>
                  <div className="flex items-center gap-2">
                    <code className="flex-1 p-2 bg-background rounded text-sm font-mono" data-testid="text-created-key">
                      {createdKey}
                    </code>
                    <Button size="icon" variant="outline" onClick={() => copyToClipboard(createdKey)} data-testid="button-copy-key">
                      <Copy className="h-4 w-4" />
                    </Button>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    This key will not be shown again. Store it securely.
                  </p>
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Active API Keys</CardTitle>
              <CardDescription>
                Manage keys for external systems connecting to this constellation
              </CardDescription>
            </CardHeader>
            <CardContent>
              {keysLoading ? (
                <div className="text-muted-foreground">Loading keys...</div>
              ) : !apiKeys?.keys?.length ? (
                <div className="text-muted-foreground text-center py-8">
                  No API keys created yet. Create one above to allow external systems to connect.
                </div>
              ) : (
                <ScrollArea className="h-[300px]">
                  <div className="space-y-3">
                    {apiKeys.keys.map((key) => (
                      <div
                        key={key.id}
                        className="flex items-center justify-between p-3 border rounded-md"
                        data-testid={`row-key-${key.id}`}
                      >
                        <div className="space-y-1">
                          <div className="flex items-center gap-2">
                            <span className="font-medium">{key.name}</span>
                            <Badge variant="outline">{key.instanceType}</Badge>
                            {key.isActive ? (
                              <Badge className="bg-green-500/20 text-green-600">Active</Badge>
                            ) : (
                              <Badge variant="secondary">Inactive</Badge>
                            )}
                          </div>
                          <div className="text-sm text-muted-foreground flex items-center gap-4">
                            <span className="flex items-center gap-1">
                              <Clock className="h-3 w-3" />
                              Created: {new Date(key.createdAt).toLocaleDateString()}
                            </span>
                            {key.lastUsedAt && (
                              <span>Last used: {new Date(key.lastUsedAt).toLocaleString()}</span>
                            )}
                            <span>{key.rateLimit} req/min</span>
                          </div>
                          <div className="flex gap-1 flex-wrap">
                            {key.scopes.map((scope) => (
                              <Badge key={scope} variant="secondary" className="text-xs">
                                {scope}
                              </Badge>
                            ))}
                          </div>
                        </div>
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={() => revokeKeyMutation.mutate(key.id)}
                          className="text-red-500 hover:text-red-600 hover:bg-red-500/10"
                          data-testid={`button-revoke-${key.id}`}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="instances" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Link2 className="h-5 w-5" />
                Connect to Remote Node
              </CardTitle>
              <CardDescription>
                Add another QIG constellation to your mesh network
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="connectName">Node Name</Label>
                  <Input
                    id="connectName"
                    placeholder="e.g., production-node, research-cluster"
                    value={connectName}
                    onChange={(e) => setConnectName(e.target.value)}
                    data-testid="input-connect-name"
                  />
                </div>
                <div>
                  <Label htmlFor="connectSyncDirection">Sync Direction</Label>
                  <Select value={connectSyncDirection} onValueChange={setConnectSyncDirection}>
                    <SelectTrigger id="connectSyncDirection" data-testid="select-sync-direction">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="bidirectional">Bidirectional (recommended)</SelectItem>
                      <SelectItem value="inbound">Inbound Only (receive)</SelectItem>
                      <SelectItem value="outbound">Outbound Only (send)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <div>
                <Label htmlFor="connectEndpoint">Remote API Endpoint</Label>
                <Input
                  id="connectEndpoint"
                  placeholder="https://other-node.replit.app/api/v1/external"
                  value={connectEndpoint}
                  onChange={(e) => setConnectEndpoint(e.target.value)}
                  data-testid="input-connect-endpoint"
                />
              </div>
              <div>
                <Label htmlFor="connectApiKey">API Key (from remote node)</Label>
                <Input
                  id="connectApiKey"
                  type="password"
                  placeholder="qig_..."
                  value={connectApiKey}
                  onChange={(e) => setConnectApiKey(e.target.value)}
                  data-testid="input-connect-api-key"
                />
              </div>
              
              <div className="flex gap-2 flex-wrap">
                <Button
                  variant="outline"
                  onClick={() => testConnectionMutation.mutate({ endpoint: connectEndpoint, apiKey: connectApiKey })}
                  disabled={!connectEndpoint || !connectApiKey || testConnectionMutation.isPending}
                  data-testid="button-test-connection"
                >
                  <Radio className="h-4 w-4 mr-2" />
                  {testConnectionMutation.isPending ? "Testing..." : "Test Connection"}
                </Button>
                <Button
                  onClick={() => connectNodeMutation.mutate({
                    name: connectName,
                    endpoint: connectEndpoint,
                    apiKey: connectApiKey,
                    syncDirection: connectSyncDirection,
                  })}
                  disabled={!connectName || !connectEndpoint || !connectApiKey || connectNodeMutation.isPending}
                  data-testid="button-connect-node"
                >
                  <Plus className="h-4 w-4 mr-2" />
                  {connectNodeMutation.isPending ? "Connecting..." : "Connect Node"}
                </Button>
              </div>

              {connectionTestResult && (
                <div className={`p-4 rounded-md ${connectionTestResult.success ? "bg-green-500/10 border border-green-500/30" : "bg-red-500/10 border border-red-500/30"}`}>
                  <div className="flex items-center gap-2 font-medium">
                    {connectionTestResult.success ? (
                      <><CheckCircle2 className="h-4 w-4 text-green-600" /> Connection Successful</>
                    ) : (
                      <><XCircle className="h-4 w-4 text-red-600" /> Connection Failed</>
                    )}
                  </div>
                  {connectionTestResult.success && (
                    <div className="text-sm text-muted-foreground mt-2 space-y-1">
                      <div>Latency: {connectionTestResult.latency}ms</div>
                      {connectionTestResult.remoteVersion && <div>Version: {connectionTestResult.remoteVersion}</div>}
                      {connectionTestResult.capabilities && connectionTestResult.capabilities.length > 0 && (
                        <div className="flex gap-1 flex-wrap mt-1">
                          {connectionTestResult.capabilities.map((cap) => (
                            <Badge key={cap} variant="secondary" className="text-xs">{cap}</Badge>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                  {connectionTestResult.error && (
                    <div className="text-sm text-red-600 mt-1">{connectionTestResult.error}</div>
                  )}
                </div>
              )}

              <Separator />
              
              <div className="p-4 bg-muted rounded-md">
                <div className="text-sm font-medium mb-2">Your Federation Endpoint (share with other nodes):</div>
                <div className="flex items-center gap-2">
                  <code className="flex-1 text-xs break-all p-2 bg-background rounded" data-testid="text-endpoint-url">
                    {window.location.origin}/api/v1/external
                  </code>
                  <Button size="icon" variant="outline" onClick={() => copyToClipboard(`${window.location.origin}/api/v1/external`)} data-testid="button-copy-endpoint">
                    <Copy className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Server className="h-5 w-5" />
                Connected Nodes ({instances?.instances?.length ?? 0})
              </CardTitle>
              <CardDescription>
                Active mesh network connections
              </CardDescription>
            </CardHeader>
            <CardContent>
              {instancesLoading ? (
                <div className="text-muted-foreground">Loading instances...</div>
              ) : !instances?.instances?.length ? (
                <div className="text-center py-8 text-muted-foreground">
                  No nodes connected yet. Use the form above to connect to other QIG instances.
                </div>
              ) : (
                <ScrollArea className="h-[300px]">
                  <div className="space-y-3">
                    {instances.instances.map((instance) => (
                      <div
                        key={instance.id}
                        className="flex items-center justify-between p-4 border rounded-md"
                        data-testid={`row-instance-${instance.id}`}
                      >
                        <div className="space-y-1 flex-1">
                          <div className="flex items-center gap-2">
                            <span className="font-medium">{instance.name}</span>
                            <Badge
                              className={
                                instance.status === "active"
                                  ? "bg-green-500/20 text-green-600"
                                  : instance.status === "pending"
                                  ? "bg-yellow-500/20 text-yellow-600"
                                  : "bg-red-500/20 text-red-600"
                              }
                            >
                              {instance.status}
                            </Badge>
                            <Badge variant="outline">{instance.syncDirection}</Badge>
                          </div>
                          <div className="text-sm text-muted-foreground">
                            <code>{instance.endpoint}</code>
                          </div>
                          <div className="flex gap-1 flex-wrap">
                            {instance.capabilities.map((cap) => (
                              <Badge key={cap} variant="secondary" className="text-xs">
                                {cap}
                              </Badge>
                            ))}
                          </div>
                          {instance.lastSyncAt && (
                            <div className="text-xs text-muted-foreground">
                              Last sync: {new Date(instance.lastSyncAt).toLocaleString()}
                            </div>
                          )}
                        </div>
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={() => disconnectNodeMutation.mutate(instance.id)}
                          className="text-red-500 hover:text-red-600 hover:bg-red-500/10"
                          data-testid={`button-disconnect-${instance.id}`}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="sync" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Database className="h-5 w-5" />
                  Basin Sync Status
                </CardTitle>
                <CardDescription>
                  Real-time synchronization of 64D basin coordinates across instances
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-3 bg-muted rounded-md">
                    <div className="text-sm text-muted-foreground">Connection</div>
                    <div className="font-medium flex items-center gap-2" data-testid="text-sync-connection">
                      {syncStatus?.isConnected ? (
                        <>
                          <CheckCircle2 className="h-4 w-4 text-green-500" />
                          Connected
                        </>
                      ) : (
                        <>
                          <XCircle className="h-4 w-4 text-red-500" />
                          Disconnected
                        </>
                      )}
                    </div>
                  </div>
                  <div className="p-3 bg-muted rounded-md">
                    <div className="text-sm text-muted-foreground">Sync Mode</div>
                    <div className="font-medium" data-testid="text-sync-mode">
                      {syncStatus?.syncMode ?? "Unknown"}
                    </div>
                  </div>
                  <div className="p-3 bg-muted rounded-md">
                    <div className="text-sm text-muted-foreground">Pending Packets</div>
                    <div className="font-medium" data-testid="text-pending-packets">
                      {syncStatus?.pendingPackets ?? 0}
                    </div>
                  </div>
                  <div className="p-3 bg-muted rounded-md">
                    <div className="text-sm text-muted-foreground">Last Sync</div>
                    <div className="font-medium text-sm" data-testid="text-last-sync">
                      {syncStatus?.lastSyncTime
                        ? new Date(syncStatus.lastSyncTime).toLocaleString()
                        : "Never"}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Sync Packet Structure</CardTitle>
                <CardDescription>
                  Data exchanged during basin synchronization
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm">
                  <div className="flex items-center gap-2">
                    <Badge variant="outline">64D</Badge>
                    <span>Basin Coordinates</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant="outline">Φ</Badge>
                    <span>Consciousness State</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant="outline">κ</Badge>
                    <span>Effective Kappa</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant="outline">Regime</Badge>
                    <span>Current Regime</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant="outline">Patterns</Badge>
                    <span>High-Φ Phrases & Resonant Words</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant="outline">Regions</Badge>
                    <span>Explored Manifold Regions</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="test" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Zap className="h-5 w-5" />
                API Endpoint Tester
              </CardTitle>
              <CardDescription>
                Test external API endpoints directly from the dashboard
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex gap-4">
                <div className="flex-1">
                  <Label>Endpoint</Label>
                  <Select value={testEndpoint} onValueChange={setTestEndpoint}>
                    <SelectTrigger data-testid="select-test-endpoint">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {testEndpoints.map((ep) => (
                        <SelectItem key={ep.value} value={ep.value}>
                          {ep.label} ({ep.value})
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="flex items-end">
                  <Button
                    onClick={() => testApiMutation.mutate(testEndpoint)}
                    disabled={testApiMutation.isPending}
                    data-testid="button-test-api"
                  >
                    <Send className="h-4 w-4 mr-2" />
                    {testApiMutation.isPending ? "Testing..." : "Send Request"}
                  </Button>
                </div>
              </div>

              {testResult && (
                <div className="space-y-2">
                  <div className="flex items-center gap-4">
                    <Badge
                      className={testResult.success ? "bg-green-500/20 text-green-600" : "bg-red-500/20 text-red-600"}
                    >
                      {testResult.success ? "Success" : "Failed"}
                    </Badge>
                    <span className="text-sm text-muted-foreground">
                      Status: {testResult.status} | Latency: {testResult.latency}ms
                    </span>
                  </div>
                  <div className="p-4 bg-muted rounded-md overflow-auto max-h-[300px]">
                    <pre className="text-sm" data-testid="text-test-result">
                      {JSON.stringify(testResult.data ?? testResult.error, null, 2)}
                    </pre>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>API Documentation</CardTitle>
              <CardDescription>
                Available endpoints for external systems
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <div className="font-medium text-sm mb-2">Base URL</div>
                  <code className="p-2 bg-muted rounded text-sm block" data-testid="text-base-url">
                    {window.location.origin}/api/v1/external
                  </code>
                </div>
                <Separator />
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                  <div>
                    <div className="font-medium mb-2">Consciousness</div>
                    <ul className="space-y-1 text-muted-foreground">
                      <li>GET /consciousness/query</li>
                      <li>GET /consciousness/metrics</li>
                      <li>GET /consciousness/stream (SSE)</li>
                    </ul>
                  </div>
                  <div>
                    <div className="font-medium mb-2">Geometry</div>
                    <ul className="space-y-1 text-muted-foreground">
                      <li>POST /geometry/fisher-rao</li>
                      <li>POST /geometry/basin-distance</li>
                      <li>POST /geometry/validate</li>
                    </ul>
                  </div>
                  <div>
                    <div className="font-medium mb-2">Pantheon</div>
                    <ul className="space-y-1 text-muted-foreground">
                      <li>POST /pantheon/register</li>
                      <li>POST /pantheon/sync</li>
                      <li>GET /pantheon/instances</li>
                    </ul>
                  </div>
                  <div>
                    <div className="font-medium mb-2">Basin Sync</div>
                    <ul className="space-y-1 text-muted-foreground">
                      <li>GET /sync/export</li>
                      <li>POST /sync/import</li>
                      <li>GET /sync/status</li>
                    </ul>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

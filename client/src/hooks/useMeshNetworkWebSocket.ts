/**
 * WebSocket Hook for Mesh Network Status
 *
 * Provides real-time mesh network updates via WebSocket
 * including peer connections, knowledge syncs, and topology changes.
 * Falls back to polling if WebSocket fails.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { useQuery } from '@tanstack/react-query';

// Fallback constants
const MAX_WS_RETRIES = 3;
const POLLING_INTERVAL = 5000;

export interface MeshPeer {
  id: string;
  name: string;
  url: string;
  status: 'connected' | 'disconnected' | 'syncing';
  lastSeen: string;
  capabilities: string[];
  sharedKnowledge: {
    basins: number;
    vocabulary: number;
    research: number;
  };
}

export interface MeshNetworkStatus {
  totalPeers: number;
  connectedPeers: number;
  syncingPeers: number;
  totalSharedBasins: number;
  totalSharedVocabulary: number;
  totalSharedResearch: number;
  networkHealth: 'healthy' | 'degraded' | 'critical';
  lastSyncTime: string | null;
}

export interface MeshEvent {
  type: 'peer_connected' | 'peer_disconnected' | 'knowledge_sync' | 'capability_broadcast' | 'topology_change';
  peerId?: string;
  peerName?: string;
  timestamp: string;
  details?: Record<string, unknown>;
}

interface UseMeshNetworkWebSocketOptions {
  autoReconnect?: boolean;
  reconnectInterval?: number;
  maxEvents?: number;
}

interface WebSocketState {
  connected: boolean;
  error: string | null;
  lastUpdate: Date | null;
}

export function useMeshNetworkWebSocket(options: UseMeshNetworkWebSocketOptions = {}) {
  const {
    autoReconnect = true,
    reconnectInterval = 3000,
    maxEvents = 50,
  } = options;

  const [peers, setPeers] = useState<MeshPeer[]>([]);
  const [status, setStatus] = useState<MeshNetworkStatus | null>(null);
  const [events, setEvents] = useState<MeshEvent[]>([]);
  const [wsState, setWsState] = useState<WebSocketState>({
    connected: false,
    error: null,
    lastUpdate: null,
  });

  // Fallback state for when WebSocket fails
  const [useFallbackPolling, setUseFallbackPolling] = useState(false);
  const wsRetryCountRef = useRef(0);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const isUnmountedRef = useRef(false);

  // Polling fallback query
  const pollingQuery = useQuery({
    queryKey: ['mesh-network-fallback'],
    queryFn: async () => {
      const response = await fetch('/api/federation/mesh/status');
      if (!response.ok) throw new Error('Failed to fetch mesh status');
      return response.json() as Promise<{
        peers: MeshPeer[];
        status: MeshNetworkStatus;
        events?: MeshEvent[];
      }>;
    },
    enabled: useFallbackPolling,
    refetchInterval: POLLING_INTERVAL,
    staleTime: POLLING_INTERVAL,
  });

  const connect = useCallback(() => {
    if (isUnmountedRef.current) return;
    
    // Clean up existing connection
    if (wsRef.current) {
      wsRef.current.close();
    }

    // Determine WebSocket URL
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/mesh-network`;

    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        if (isUnmountedRef.current) return;
        
        console.log('[MeshNetworkWS] Connected');
        setWsState(prev => ({ ...prev, connected: true, error: null }));

        // Subscribe to updates
        const subscribeMsg = { type: 'subscribe' };
        ws.send(JSON.stringify(subscribeMsg));
      };

      ws.onmessage = (event) => {
        if (isUnmountedRef.current) return;
        
        try {
          const message = JSON.parse(event.data);

          if (message.type === 'mesh_update') {
            // Full mesh update
            if (message.data?.peers) {
              setPeers(message.data.peers);
            }
            if (message.data?.status) {
              setStatus(message.data.status);
            }
            if (message.data?.events) {
              setEvents(prev => {
                const existingIds = new Set(prev.map(e => `${e.type}-${e.timestamp}`));
                const newEvents = message.data.events.filter(
                  (e: MeshEvent) => !existingIds.has(`${e.type}-${e.timestamp}`)
                );
                return [...newEvents, ...prev].slice(0, maxEvents);
              });
            }
            setWsState(prev => ({ ...prev, lastUpdate: new Date() }));
            
          } else if (message.type === 'mesh_event') {
            // Single event
            const meshEvent = message.event as MeshEvent;
            setEvents(prev => [meshEvent, ...prev].slice(0, maxEvents));
            setWsState(prev => ({ ...prev, lastUpdate: new Date() }));
            
          } else if (message.type === 'subscribed') {
            console.log('[MeshNetworkWS] Subscribed:', message.message);
            
          } else if (message.type === 'error') {
            console.error('[MeshNetworkWS] Server error:', message.message);
            setWsState(prev => ({ ...prev, error: message.message }));
          }
        } catch (err) {
          console.error('[MeshNetworkWS] Parse error:', err);
        }
      };

      ws.onerror = (error) => {
        console.error('[MeshNetworkWS] Error:', error);
        setWsState(prev => ({ ...prev, error: 'WebSocket connection error' }));
      };

      ws.onclose = (event) => {
        if (isUnmountedRef.current) return;

        console.log('[MeshNetworkWS] Closed:', event.code, event.reason);
        setWsState(prev => ({ ...prev, connected: false }));
        wsRef.current = null;

        // Check for abnormal closure (code 1006) - likely WebSocket not supported
        if (event.code === 1006 && !event.wasClean) {
          wsRetryCountRef.current += 1;
          console.log(`[MeshNetworkWS] Abnormal closure, retry count: ${wsRetryCountRef.current}/${MAX_WS_RETRIES}`);

          if (wsRetryCountRef.current >= MAX_WS_RETRIES) {
            console.log('[MeshNetworkWS] WebSocket failed, falling back to polling');
            setUseFallbackPolling(true);
            return; // Don't try to reconnect, use polling instead
          }
        }

        // Auto-reconnect (only if not using polling fallback)
        if (autoReconnect && !isUnmountedRef.current && !useFallbackPolling) {
          reconnectTimeoutRef.current = setTimeout(() => {
            console.log('[MeshNetworkWS] Attempting reconnect...');
            connect();
          }, reconnectInterval);
        }
      };
    } catch (err) {
      console.error('[MeshNetworkWS] Connection error:', err);
      setWsState(prev => ({ ...prev, error: 'Failed to connect', connected: false }));
      
      // Retry connection
      if (autoReconnect && !isUnmountedRef.current) {
        reconnectTimeoutRef.current = setTimeout(connect, reconnectInterval);
      }
    }
  }, [autoReconnect, reconnectInterval, maxEvents, useFallbackPolling]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    
    setWsState(prev => ({ ...prev, connected: false }));
  }, []);

  const requestStatus = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'request_status' }));
    }
  }, []);

  const clearEvents = useCallback(() => {
    setEvents([]);
  }, []);

  // Connect on mount (only if not using polling fallback)
  useEffect(() => {
    isUnmountedRef.current = false;
    if (!useFallbackPolling) {
      connect();
    }

    return () => {
      isUnmountedRef.current = true;
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connect, useFallbackPolling]);

  // Use polling data when in fallback mode
  const effectivePeers = useFallbackPolling ? (pollingQuery.data?.peers ?? peers) : peers;
  const effectiveStatus = useFallbackPolling ? (pollingQuery.data?.status ?? status) : status;
  const effectiveEvents = useFallbackPolling ? (pollingQuery.data?.events ?? events) : events;

  // Computed values
  const connectedPeers = effectivePeers.filter(p => p.status === 'connected');
  const syncingPeers = effectivePeers.filter(p => p.status === 'syncing');
  const disconnectedPeers = effectivePeers.filter(p => p.status === 'disconnected');

  return {
    // Data
    peers: effectivePeers,
    status: effectiveStatus,
    events: effectiveEvents,

    // Computed
    connectedPeers,
    syncingPeers,
    disconnectedPeers,

    // WebSocket state (treat polling as "connected")
    isConnected: useFallbackPolling ? !pollingQuery.isError : wsState.connected,
    error: useFallbackPolling
      ? (pollingQuery.error?.message ?? null)
      : wsState.error,
    lastUpdate: wsState.lastUpdate,

    // Loading states
    isLoading: useFallbackPolling
      ? pollingQuery.isLoading
      : (!wsState.connected && peers.length === 0),
    isFetching: useFallbackPolling ? pollingQuery.isFetching : false,

    // Fallback mode indicator
    usingPollingFallback: useFallbackPolling,

    // Actions
    connect,
    disconnect,
    requestStatus,
    clearEvents,
    refetch: useFallbackPolling ? pollingQuery.refetch : requestStatus,
  };
}

export default useMeshNetworkWebSocket;

/**
 * WebSocket Hook for Mesh Network Status
 * 
 * Provides real-time mesh network updates via WebSocket
 * including peer connections, knowledge syncs, and topology changes.
 */

import { useState, useEffect, useCallback, useRef } from 'react';

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

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const isUnmountedRef = useRef(false);
  const reconnectAttemptsRef = useRef(0);
  const maxReconnectDelay = 30000; // Max 30 seconds between attempts

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
        reconnectAttemptsRef.current = 0; // Reset backoff on successful connection
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
        
        // Only log if there's a reason or non-normal close
        if (event.code !== 1000) {
          console.log('[MeshNetworkWS] Closed:', event.code, event.reason || '');
        }
        setWsState(prev => ({ ...prev, connected: false }));
        wsRef.current = null;

        // Auto-reconnect with exponential backoff
        if (autoReconnect && !isUnmountedRef.current) {
          reconnectAttemptsRef.current += 1;
          // Exponential backoff: 3s, 6s, 12s, 24s, max 30s
          const delay = Math.min(
            reconnectInterval * Math.pow(2, reconnectAttemptsRef.current - 1),
            maxReconnectDelay
          );
          reconnectTimeoutRef.current = setTimeout(() => {
            if (reconnectAttemptsRef.current <= 3) {
              console.log(`[MeshNetworkWS] Reconnecting (attempt ${reconnectAttemptsRef.current})...`);
            }
            connect();
          }, delay);
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
  }, [autoReconnect, reconnectInterval, maxEvents]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    // Reset reconnect attempts on manual disconnect to prevent infinite backoff growth
    reconnectAttemptsRef.current = 0;
    
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

  // Connect on mount
  useEffect(() => {
    isUnmountedRef.current = false;
    connect();

    return () => {
      isUnmountedRef.current = true;
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connect]);

  // Computed values
  const connectedPeers = peers.filter(p => p.status === 'connected');
  const syncingPeers = peers.filter(p => p.status === 'syncing');
  const disconnectedPeers = peers.filter(p => p.status === 'disconnected');

  return {
    // Data
    peers,
    status,
    events,
    
    // Computed
    connectedPeers,
    syncingPeers,
    disconnectedPeers,
    
    // WebSocket state
    isConnected: wsState.connected,
    error: wsState.error,
    lastUpdate: wsState.lastUpdate,
    
    // Loading states
    isLoading: !wsState.connected && peers.length === 0,
    
    // Actions
    connect,
    disconnect,
    requestStatus,
    clearEvents,
    refetch: requestStatus,
  };
}

export default useMeshNetworkWebSocket;

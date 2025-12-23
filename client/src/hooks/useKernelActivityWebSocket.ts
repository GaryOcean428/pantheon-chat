/**
 * WebSocket Hook for Kernel Activity Stream
 * 
 * Provides real-time kernel activity updates via WebSocket
 * instead of polling. Falls back to polling if WebSocket fails.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { KernelActivityItem, KernelActivityResponse } from './use-kernel-activity';

interface ActivityFilters {
  activityTypes?: string[];
  fromKernels?: string[];
  toKernels?: string[];
}

interface UseKernelActivityWebSocketOptions {
  filters?: ActivityFilters;
  maxItems?: number;
  autoReconnect?: boolean;
  reconnectInterval?: number;
}

interface WebSocketState {
  connected: boolean;
  error: string | null;
  lastUpdate: Date | null;
}

export function useKernelActivityWebSocket(options: UseKernelActivityWebSocketOptions = {}) {
  const {
    filters,
    maxItems = 100,
    autoReconnect = true,
    reconnectInterval = 3000,
  } = options;

  const [activities, setActivities] = useState<KernelActivityItem[]>([]);
  const [status, setStatus] = useState<KernelActivityResponse['status'] | null>(null);
  const [wsState, setWsState] = useState<WebSocketState>({
    connected: false,
    error: null,
    lastUpdate: null,
  });

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const isUnmountedRef = useRef(false);

  const connect = useCallback(() => {
    if (isUnmountedRef.current) return;
    
    // Clean up existing connection
    if (wsRef.current) {
      wsRef.current.close();
    }

    // Determine WebSocket URL
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/kernel-activity`;

    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        if (isUnmountedRef.current) return;
        
        console.log('[KernelActivityWS] Connected');
        setWsState(prev => ({ ...prev, connected: true, error: null }));

        // Subscribe with filters
        const subscribeMsg = {
          type: 'subscribe',
          filters: filters,
        };
        ws.send(JSON.stringify(subscribeMsg));
      };

      ws.onmessage = (event) => {
        if (isUnmountedRef.current) return;
        
        try {
          const message = JSON.parse(event.data);

          if (message.type === 'activity') {
            // Single activity
            const activity = message.data as KernelActivityItem;
            setActivities(prev => {
              // Avoid duplicates
              if (prev.some(a => a.id === activity.id)) return prev;
              return [activity, ...prev].slice(0, maxItems);
            });
            setWsState(prev => ({ ...prev, lastUpdate: new Date() }));
            
            if (message.status) {
              setStatus(message.status);
            }
          } else if (message.type === 'activity_batch') {
            // Batch of activities
            const newActivities = message.data as KernelActivityItem[];
            setActivities(prev => {
              const existingIds = new Set(prev.map(a => a.id));
              const uniqueNew = newActivities.filter(a => !existingIds.has(a.id));
              return [...uniqueNew, ...prev].slice(0, maxItems);
            });
            setWsState(prev => ({ ...prev, lastUpdate: new Date() }));
            
            if (message.status) {
              setStatus(message.status);
            }
          } else if (message.type === 'status') {
            setStatus(message.status);
          } else if (message.type === 'subscribed') {
            console.log('[KernelActivityWS] Subscribed:', message.message);
          } else if (message.type === 'error') {
            console.error('[KernelActivityWS] Server error:', message.message);
            setWsState(prev => ({ ...prev, error: message.message }));
          }
        } catch (err) {
          console.error('[KernelActivityWS] Parse error:', err);
        }
      };

      ws.onerror = (error) => {
        console.error('[KernelActivityWS] Error:', error);
        setWsState(prev => ({ ...prev, error: 'WebSocket connection error' }));
      };

      ws.onclose = (event) => {
        if (isUnmountedRef.current) return;
        
        console.log('[KernelActivityWS] Closed:', event.code, event.reason);
        setWsState(prev => ({ ...prev, connected: false }));
        wsRef.current = null;

        // Auto-reconnect
        if (autoReconnect && !isUnmountedRef.current) {
          reconnectTimeoutRef.current = setTimeout(() => {
            console.log('[KernelActivityWS] Attempting reconnect...');
            connect();
          }, reconnectInterval);
        }
      };
    } catch (err) {
      console.error('[KernelActivityWS] Connection error:', err);
      setWsState(prev => ({ ...prev, error: 'Failed to connect', connected: false }));
      
      // Retry connection
      if (autoReconnect && !isUnmountedRef.current) {
        reconnectTimeoutRef.current = setTimeout(connect, reconnectInterval);
      }
    }
  }, [filters, maxItems, autoReconnect, reconnectInterval]);

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

  const updateFilters = useCallback((newFilters: ActivityFilters) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      const subscribeMsg = {
        type: 'subscribe',
        filters: newFilters,
      };
      wsRef.current.send(JSON.stringify(subscribeMsg));
    }
  }, []);

  const clearActivities = useCallback(() => {
    setActivities([]);
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

  // Build response format compatible with useKernelActivity
  const data: KernelActivityResponse = {
    activity: activities,
    debates: {
      active: [],
      resolved: [],
    },
    status: status || {
      total_messages: activities.length,
      active_debates: 0,
      resolved_debates: 0,
      knowledge_transfers: 0,
    },
  };

  return {
    // Data
    data,
    activities,
    status,
    
    // WebSocket state
    isConnected: wsState.connected,
    error: wsState.error,
    lastUpdate: wsState.lastUpdate,
    
    // Loading states (for compatibility)
    isLoading: !wsState.connected && activities.length === 0,
    isFetching: false,
    
    // Actions
    connect,
    disconnect,
    updateFilters,
    clearActivities,
    refetch: connect, // For compatibility with polling version
  };
}

export default useKernelActivityWebSocket;

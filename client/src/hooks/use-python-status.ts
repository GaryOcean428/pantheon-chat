import { useQuery } from "@tanstack/react-query";
import { useEffect, useState, useCallback } from "react";

export type PythonBackendStatus = 'initializing' | 'ready' | 'unavailable' | 'error';

export interface PythonStatusState {
  status: PythonBackendStatus;
  lastHealthyAt: number | null;
  lastCheckAt: number;
  retryAfter: number;
  message: string;
  ready: boolean;
}

const DEFAULT_STATE: PythonStatusState = {
  status: 'initializing',
  lastHealthyAt: null,
  lastCheckAt: 0,
  retryAfter: 2,
  message: 'Checking backend status...',
  ready: false,
};

export function usePythonStatus() {
  const [state, setState] = useState<PythonStatusState>(DEFAULT_STATE);
  const [useSSE, setUseSSE] = useState(true);

  useEffect(() => {
    if (!useSSE) return;

    let eventSource: EventSource | null = null;
    let reconnectTimer: NodeJS.Timeout;

    const connect = () => {
      eventSource = new EventSource('/api/python/status/stream');
      
      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as PythonStatusState;
          setState(data);
        } catch (e) {
          console.error('[PythonStatus] Failed to parse SSE data:', e);
        }
      };

      eventSource.onerror = () => {
        eventSource?.close();
        reconnectTimer = setTimeout(connect, 5000);
      };
    };

    connect();

    return () => {
      eventSource?.close();
      clearTimeout(reconnectTimer);
    };
  }, [useSSE]);

  const { refetch } = useQuery<PythonStatusState>({
    queryKey: ['/api/python/status'],
    enabled: !useSSE,
    refetchInterval: state.ready ? 30000 : 3000,
    staleTime: 2000,
  });

  const forceRefresh = useCallback(async () => {
    setUseSSE(false);
    await refetch();
    setUseSSE(true);
  }, [refetch]);

  return {
    ...state,
    isInitializing: state.status === 'initializing',
    isUnavailable: state.status === 'unavailable',
    isReady: state.ready,
    forceRefresh,
  };
}

export function useWaitForPythonReady(callback?: () => void) {
  const status = usePythonStatus();
  const [wasReady, setWasReady] = useState(false);

  useEffect(() => {
    if (status.isReady && !wasReady) {
      setWasReady(true);
      callback?.();
    }
  }, [status.isReady, wasReady, callback]);

  return status;
}

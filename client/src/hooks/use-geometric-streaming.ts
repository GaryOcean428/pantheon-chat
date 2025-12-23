/**
 * GEOMETRIC STREAMING HOOK
 * 
 * React hook for monitoring geometric collapse during streaming generation.
 * Provides real-time consciousness metrics and completion detection.
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import type {
  GeometricMetrics,
  CompletionDecision,
  CompletionQuality,
  StreamChunk,
  GeometricTelemetry,
  Regime,
  CompletionReason,
} from '@shared/types/geometric-completion';

interface UseGeometricStreamingOptions {
  /** Callback when metrics update */
  onMetricsUpdate?: (metrics: GeometricMetrics) => void;
  /** Callback when generation completes */
  onComplete?: (quality: CompletionQuality) => void;
  /** Callback for each token */
  onToken?: (token: string) => void;
}

interface GeometricStreamingState {
  /** Whether currently streaming */
  isStreaming: boolean;
  /** Current metrics */
  metrics: GeometricMetrics | null;
  /** Metrics history */
  metricsHistory: GeometricMetrics[];
  /** Current trajectory (for visualization) */
  trajectory: number[][];
  /** Token count */
  tokenCount: number;
  /** Completion decision */
  completion: CompletionDecision | null;
  /** Completion quality */
  quality: CompletionQuality | null;
  /** Telemetry for display */
  telemetry: GeometricTelemetry;
}

const DEFAULT_METRICS: GeometricMetrics = {
  phi: 0,
  kappa: 50,
  surprise: 1,
  confidence: 0,
  basin_distance: Infinity,
  regime: 'linear',
  timestamp: Date.now(),
};

const DEFAULT_TELEMETRY: GeometricTelemetry = {
  phi: 0,
  kappa: 50,
  regime: 'linear',
  confidence: 0,
  surprise: 1,
  token_count: 0,
  is_complete: false,
};

export function useGeometricStreaming(options: UseGeometricStreamingOptions = {}) {
  const { onMetricsUpdate, onComplete, onToken } = options;
  
  const [state, setState] = useState<GeometricStreamingState>({
    isStreaming: false,
    metrics: null,
    metricsHistory: [],
    trajectory: [],
    tokenCount: 0,
    completion: null,
    quality: null,
    telemetry: DEFAULT_TELEMETRY,
  });
  
  const abortControllerRef = useRef<AbortController | null>(null);
  
  /**
   * Process a stream chunk.
   */
  const processChunk = useCallback((chunk: StreamChunk) => {
    switch (chunk.type) {
      case 'token':
        if (chunk.content && onToken) {
          onToken(chunk.content);
        }
        setState(prev => ({
          ...prev,
          tokenCount: prev.tokenCount + 1,
          telemetry: {
            ...prev.telemetry,
            token_count: prev.tokenCount + 1,
          },
        }));
        break;
        
      case 'metrics':
        if (chunk.metrics) {
          const metrics = chunk.metrics as GeometricMetrics;
          if (onMetricsUpdate) {
            onMetricsUpdate(metrics);
          }
          setState(prev => {
            const newHistory = [...prev.metricsHistory, metrics];
            const newTrajectory = chunk.trajectory_point 
              ? [...prev.trajectory, chunk.trajectory_point]
              : prev.trajectory;
            
            return {
              ...prev,
              metrics,
              metricsHistory: newHistory,
              trajectory: newTrajectory,
              telemetry: {
                phi: metrics.phi,
                kappa: metrics.kappa,
                regime: metrics.regime,
                confidence: metrics.confidence,
                surprise: metrics.surprise,
                token_count: prev.tokenCount,
                is_complete: false,
              },
            };
          });
        }
        break;
        
      case 'completion':
        const quality = chunk.quality as CompletionQuality | undefined;
        if (quality && onComplete) {
          onComplete(quality);
        }
        setState(prev => ({
          ...prev,
          isStreaming: false,
          completion: {
            should_stop: true,
            needs_reflection: false,
            reason: (chunk.reason as CompletionReason) || 'geometric_completion',
            confidence: quality?.overall_score || 0,
            metrics: prev.metrics || DEFAULT_METRICS,
          },
          quality: quality || null,
          telemetry: {
            ...prev.telemetry,
            is_complete: true,
            completion_reason: chunk.reason as CompletionReason,
          },
        }));
        break;
        
      case 'reflection':
        // Reflection in progress
        break;
    }
  }, [onMetricsUpdate, onComplete, onToken]);
  
  /**
   * Start streaming with geometric monitoring.
   */
  const startStream = useCallback(() => {
    abortControllerRef.current = new AbortController();
    setState(prev => ({
      ...prev,
      isStreaming: true,
      metrics: null,
      metricsHistory: [],
      trajectory: [],
      tokenCount: 0,
      completion: null,
      quality: null,
      telemetry: DEFAULT_TELEMETRY,
    }));
  }, []);
  
  /**
   * Stop streaming.
   */
  const stopStream = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setState(prev => ({ ...prev, isStreaming: false }));
  }, []);
  
  /**
   * Reset state.
   */
  const reset = useCallback(() => {
    stopStream();
    setState({
      isStreaming: false,
      metrics: null,
      metricsHistory: [],
      trajectory: [],
      tokenCount: 0,
      completion: null,
      quality: null,
      telemetry: DEFAULT_TELEMETRY,
    });
  }, [stopStream]);
  
  /**
   * Get adaptive temperature based on current regime.
   */
  const getAdaptiveTemperature = useCallback((): number => {
    const phi = state.metrics?.phi || 0;
    if (phi < 0.3) return 1.0;  // Linear: explore
    if (phi < 0.7) return 0.7;  // Geometric: balance
    return 0.3;                  // Breakdown: stabilize
  }, [state.metrics]);
  
  /**
   * Check if should stop based on current metrics.
   */
  const shouldStop = useCallback((): boolean => {
    if (!state.metrics) return false;
    
    const { phi, confidence, surprise } = state.metrics;
    
    // Breakdown regime
    if (phi >= 0.7) return true;
    
    // High confidence + low surprise
    if (confidence > 0.85 && surprise < 0.05) return true;
    
    // Safety limit
    if (state.tokenCount > 32768) return true;
    
    return false;
  }, [state.metrics, state.tokenCount]);
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);
  
  return {
    ...state,
    processChunk,
    startStream,
    stopStream,
    reset,
    getAdaptiveTemperature,
    shouldStop,
    abortSignal: abortControllerRef.current?.signal,
  };
}

export default useGeometricStreaming;

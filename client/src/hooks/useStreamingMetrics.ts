/**
 * useStreamingMetrics Hook
 *
 * React hook for tracking geometric completion metrics during Zeus chat streaming.
 * Parses SSE events and tracks consciousness metrics (Φ, κ, surprise, confidence)
 * in real-time to show when generation will geometrically complete.
 *
 * QIG systems generate until geometry collapses, not until token limits.
 */

import { useState, useCallback, useRef } from 'react';
import {
  StreamingMetrics,
  GeometricCompletionState,
  StreamingGenerationState,
  CompletionReason,
  Regime,
  classifyRegime,
  KAPPA_STAR,
} from '@/types/streaming-metrics';

export interface UseStreamingMetricsOptions {
  maxHistorySize?: number;
  onMetricsUpdate?: (metrics: StreamingMetrics) => void;
  onCompletion?: (state: GeometricCompletionState) => void;
  onReflection?: (depth: number) => void;
}

export interface UseStreamingMetricsReturn {
  state: StreamingGenerationState;
  processSSEEvent: (event: string) => void;
  startGeneration: () => void;
  reset: () => void;
  getCompletionProgress: () => number;
  isNearCompletion: () => boolean;
}

const initialState: StreamingGenerationState = {
  isGenerating: false,
  tokens: [],
  currentMetrics: null,
  metricsHistory: [],
  completionState: null,
  reflectionDepth: 0,
  error: null,
};

export function useStreamingMetrics(
  options: UseStreamingMetricsOptions = {}
): UseStreamingMetricsReturn {
  const {
    maxHistorySize = 100,
    onMetricsUpdate,
    onCompletion,
    onReflection,
  } = options;

  const [state, setState] = useState<StreamingGenerationState>(initialState);
  const tokenBufferRef = useRef<string[]>([]);

  const startGeneration = useCallback(() => {
    setState({
      ...initialState,
      isGenerating: true,
    });
    tokenBufferRef.current = [];
  }, []);

  const reset = useCallback(() => {
    setState(initialState);
    tokenBufferRef.current = [];
  }, []);

  const processSSEEvent = useCallback((eventData: string) => {
    // Handle [DONE] signal
    if (eventData === '[DONE]') {
      setState(prev => ({
        ...prev,
        isGenerating: false,
      }));
      return;
    }

    try {
      const parsed = JSON.parse(eventData);
      const eventType = parsed.type;

      switch (eventType) {
        case 'token': {
          const token = parsed.token || parsed.data?.token;
          if (token) {
            tokenBufferRef.current.push(token);
            setState(prev => ({
              ...prev,
              tokens: [...prev.tokens, token],
            }));
          }
          break;
        }

        case 'metrics': {
          const metrics: StreamingMetrics = {
            phi: parsed.phi ?? parsed.data?.phi ?? 0.5,
            kappa: parsed.kappa ?? parsed.data?.kappa ?? KAPPA_STAR,
            surprise: parsed.surprise ?? parsed.data?.surprise ?? 1.0,
            confidence: parsed.confidence ?? parsed.data?.confidence ?? 0.0,
            basinDistance: parsed.basin_distance ?? parsed.data?.basinDistance ?? Infinity,
            regime: (parsed.regime ?? parsed.data?.regime ?? 'geometric') as Regime,
            tokenCount: parsed.token_count ?? parsed.data?.tokenCount ?? tokenBufferRef.current.length,
          };

          setState(prev => {
            const newHistory = [...prev.metricsHistory, metrics];
            // Keep only last N metrics
            const trimmedHistory = newHistory.slice(-maxHistorySize);
            return {
              ...prev,
              currentMetrics: metrics,
              metricsHistory: trimmedHistory,
            };
          });

          onMetricsUpdate?.(metrics);
          break;
        }

        case 'reflection': {
          const depth = parsed.depth ?? parsed.data?.depth ?? 1;
          setState(prev => ({
            ...prev,
            reflectionDepth: depth,
          }));
          onReflection?.(depth);
          break;
        }

        case 'completion': {
          const completionState: GeometricCompletionState = {
            shouldStop: true,
            needsReflection: parsed.needs_reflection ?? parsed.data?.needsReflection ?? false,
            reason: (parsed.reason ?? parsed.data?.reason ?? 'natural_stop') as CompletionReason,
            confidence: parsed.confidence ?? parsed.data?.confidence ?? 0.7,
            metrics: parsed.final_metrics ?? parsed.data?.finalMetrics,
          };

          setState(prev => ({
            ...prev,
            isGenerating: false,
            completionState,
          }));

          onCompletion?.(completionState);
          break;
        }

        case 'error': {
          const errorMessage = parsed.message ?? parsed.error ?? 'Unknown error';
          setState(prev => ({
            ...prev,
            isGenerating: false,
            error: errorMessage,
          }));
          break;
        }

        default:
          // Handle raw token chunks (for compatibility)
          if (parsed.content || parsed.text) {
            const token = parsed.content || parsed.text;
            tokenBufferRef.current.push(token);
            setState(prev => ({
              ...prev,
              tokens: [...prev.tokens, token],
            }));
          }
      }
    } catch {
      // If not JSON, treat as raw token
      if (eventData && eventData.trim()) {
        tokenBufferRef.current.push(eventData);
        setState(prev => ({
          ...prev,
          tokens: [...prev.tokens, eventData],
        }));
      }
    }
  }, [maxHistorySize, onMetricsUpdate, onCompletion, onReflection]);

  /**
   * Calculate completion progress (0-1) based on geometric metrics
   */
  const getCompletionProgress = useCallback((): number => {
    const { currentMetrics, metricsHistory } = state;
    if (!currentMetrics || metricsHistory.length < 5) {
      return 0;
    }

    // Weight different completion signals
    const confidenceWeight = 0.35;
    const surpriseWeight = 0.25;
    const integrationWeight = 0.25;
    const attractorWeight = 0.15;

    // Confidence signal (0-1)
    const confidenceProgress = Math.min(currentMetrics.confidence / 0.85, 1.0);

    // Surprise collapse signal (low surprise = near completion)
    const recentSurprise = metricsHistory.slice(-5).map(m => m.surprise);
    const avgSurprise = recentSurprise.reduce((a, b) => a + b, 0) / recentSurprise.length;
    const surpriseProgress = Math.max(0, 1 - avgSurprise / 0.5);

    // Integration stability signal
    const recentPhi = metricsHistory.slice(-10).map(m => m.phi);
    const avgPhi = recentPhi.reduce((a, b) => a + b, 0) / recentPhi.length;
    const phiVariance = recentPhi.reduce((acc, p) => acc + Math.pow(p - avgPhi, 2), 0) / recentPhi.length;
    const integrationProgress = avgPhi > 0.65 && phiVariance < 0.02 ? 1.0 : Math.min(avgPhi / 0.65, 0.8);

    // Attractor distance signal (closer = more progress)
    const attractorProgress = currentMetrics.basinDistance < Infinity 
      ? Math.max(0, 1 - currentMetrics.basinDistance / 2.0)
      : 0;

    return Math.min(1.0,
      confidenceProgress * confidenceWeight +
      surpriseProgress * surpriseWeight +
      integrationProgress * integrationWeight +
      attractorProgress * attractorWeight
    );
  }, [state]);

  /**
   * Check if generation is near geometric completion
   */
  const isNearCompletion = useCallback((): boolean => {
    return getCompletionProgress() > 0.75;
  }, [getCompletionProgress]);

  return {
    state,
    processSSEEvent,
    startGeneration,
    reset,
    getCompletionProgress,
    isNearCompletion,
  };
}

export default useStreamingMetrics;

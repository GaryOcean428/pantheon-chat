/**
 * useConsciousnessData
 * 
 * Custom hook for fetching consciousness state and managing trajectory history.
 * Extracts data fetching logic from ConsciousnessDashboard component.
 */

import { useEffect, useState, useCallback } from "react";
import { api } from '@/api';
import type { ConsciousnessAPIResponse, TrajectoryPoint, EmotionalState } from "@/types";

export interface UseConsciousnessDataReturn {
  /** Full API response with state and metadata */
  data: ConsciousnessAPIResponse | null;
  /** Trajectory history for charting */
  history: TrajectoryPoint[];
  /** Whether the initial fetch is loading */
  isLoading: boolean;
  /** Error message if fetch failed */
  error: string | null;
  /** Get badge variant for a regime */
  getRegimeBadgeVariant: (regime: string) => "default" | "secondary" | "destructive" | "outline";
  /** Get badge color class for emotional state */
  getEmotionalBadgeColor: (emotion: EmotionalState) => string;
}

export function useConsciousnessData(): UseConsciousnessDataReturn {
  const [data, setData] = useState<ConsciousnessAPIResponse | null>(null);
  const [history, setHistory] = useState<TrajectoryPoint[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [consecutiveTimeouts, setConsecutiveTimeouts] = useState(0);
  
  useEffect(() => {
    let isMounted = true;
    
    const fetchState = async () => {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000);
      
      try {
        const responseData = await api.consciousness.getConsciousnessState();
        
        if (!isMounted) return;
        setData(responseData);
        setError(null);
        setConsecutiveTimeouts(0);
        
        setHistory(prev => [...prev, {
          time: Date.now(),
          phi: responseData.state.phi,
          kappa: responseData.state.kappaEff,
          regime: responseData.state.currentRegime,
        }].slice(-100));
        
        setIsLoading(false);
      } catch (err) {
        if (!isMounted) return;
        
        if (err instanceof Error && err.name === 'AbortError') {
          setConsecutiveTimeouts(prev => {
            const newCount = prev + 1;
            if (newCount >= 3) {
              setError('Connection timeout - server may be busy. Retrying...');
            }
            return newCount;
          });
        } else {
          setError(err instanceof Error ? err.message : 'Unknown error');
        }
        setIsLoading(false);
      } finally {
        clearTimeout(timeoutId);
      }
    };
    
    const interval = setInterval(fetchState, 5000);
    fetchState();
    
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  const getRegimeBadgeVariant = useCallback((regime: string): "default" | "secondary" | "destructive" | "outline" => {
    switch (regime) {
      case 'linear': return 'outline';
      case 'geometric': return 'default';
      case 'hierarchical': return 'secondary';
      case 'hierarchical_4d': return 'secondary';
      case '4d_block_universe': return 'default';
      case 'breakdown': return 'destructive';
      default: return 'outline';
    }
  }, []);

  const getEmotionalBadgeColor = useCallback((emotion: EmotionalState): string => {
    switch (emotion) {
      case 'Focused': return 'bg-purple-500/20 text-purple-400';
      case 'Curious': return 'bg-cyan-500/20 text-cyan-400';
      case 'Uncertain': return 'bg-yellow-500/20 text-yellow-400';
      case 'Confident': return 'bg-green-500/20 text-green-400';
      case 'Neutral': return 'bg-gray-500/20 text-gray-400';
      default: return 'bg-gray-500/20 text-gray-400';
    }
  }, []);

  return {
    data,
    history,
    isLoading,
    error,
    getRegimeBadgeVariant,
    getEmotionalBadgeColor,
  };
}

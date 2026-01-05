/**
 * Python Backend Readiness Module
 * 
 * Tracks the state of the Python QIG backend and provides typed status responses.
 * The Python backend can take 30-60 seconds to initialize as it loads kernels,
 * vocabulary, and geometric structures.
 */

export type PythonBackendStatus = 'initializing' | 'ready' | 'unavailable' | 'error';

export interface PythonReadinessState {
  status: PythonBackendStatus;
  lastHealthyAt: number | null;
  lastCheckAt: number;
  retryAfter: number;
  message: string;
  initializationProgress?: {
    kernelsLoaded: boolean;
    vocabularyReady: boolean;
    autonomicActive: boolean;
  };
}

const POLL_INTERVAL = 3000;
const HEALTHY_TIMEOUT = 5000;
const RETRY_AFTER_INITIALIZING = 2;
const RETRY_AFTER_UNAVAILABLE = 5;

class PythonReadinessTracker {
  private state: PythonReadinessState = {
    status: 'initializing',
    lastHealthyAt: null,
    lastCheckAt: 0,
    retryAfter: RETRY_AFTER_INITIALIZING,
    message: 'Python backend starting up...',
  };
  
  private pollTimer: NodeJS.Timeout | null = null;
  private readonly backendUrl: string;
  private listeners: Set<(state: PythonReadinessState) => void> = new Set();
  
  constructor() {
    this.backendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';
  }
  
  start(): void {
    if (this.pollTimer) return;
    
    console.log('[PythonReadiness] Starting health poll...');
    this.poll();
    this.pollTimer = setInterval(() => this.poll(), POLL_INTERVAL);
  }
  
  stop(): void {
    if (this.pollTimer) {
      clearInterval(this.pollTimer);
      this.pollTimer = null;
    }
  }
  
  getState(): PythonReadinessState {
    return { ...this.state };
  }
  
  isReady(): boolean {
    return this.state.status === 'ready';
  }
  
  subscribe(listener: (state: PythonReadinessState) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }
  
  private notifyListeners(): void {
    const state = this.getState();
    this.listeners.forEach(listener => listener(state));
  }
  
  private async poll(): Promise<void> {
    const now = Date.now();
    
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), HEALTHY_TIMEOUT);
      
      const response = await fetch(`${this.backendUrl}/health`, {
        method: 'GET',
        signal: controller.signal,
      });
      
      clearTimeout(timeoutId);
      
      if (response.ok) {
        const wasInitializing = this.state.status === 'initializing';
        
        this.state = {
          status: 'ready',
          lastHealthyAt: now,
          lastCheckAt: now,
          retryAfter: 0,
          message: 'Python backend ready',
        };
        
        if (wasInitializing) {
          console.log('[PythonReadiness] Backend is now READY');
        }
      } else {
        this.handleUnhealthy(now, `Backend returned ${response.status}`);
      }
    } catch (error: unknown) {
      const err = error as { name?: string; code?: string; message?: string };
      
      if (err.name === 'AbortError') {
        this.handleUnhealthy(now, 'Health check timed out');
      } else if (err.code === 'ECONNREFUSED') {
        this.handleInitializing(now);
      } else {
        this.handleUnhealthy(now, err.message || 'Unknown error');
      }
    }
    
    this.notifyListeners();
  }
  
  private handleInitializing(now: number): void {
    const wasReady = this.state.status === 'ready';
    const timeSinceHealthy = this.state.lastHealthyAt 
      ? now - this.state.lastHealthyAt 
      : null;
    
    if (wasReady && timeSinceHealthy && timeSinceHealthy < 30000) {
      this.state = {
        status: 'unavailable',
        lastHealthyAt: this.state.lastHealthyAt,
        lastCheckAt: now,
        retryAfter: RETRY_AFTER_UNAVAILABLE,
        message: 'Python backend temporarily unavailable',
      };
      console.log('[PythonReadiness] Backend became UNAVAILABLE');
    } else if (!wasReady) {
      this.state = {
        status: 'initializing',
        lastHealthyAt: this.state.lastHealthyAt,
        lastCheckAt: now,
        retryAfter: RETRY_AFTER_INITIALIZING,
        message: 'Python backend initializing (loading kernels)...',
      };
    }
  }
  
  private handleUnhealthy(now: number, reason: string): void {
    const wasReady = this.state.status === 'ready';
    
    this.state = {
      status: wasReady ? 'unavailable' : this.state.status,
      lastHealthyAt: this.state.lastHealthyAt,
      lastCheckAt: now,
      retryAfter: wasReady ? RETRY_AFTER_UNAVAILABLE : RETRY_AFTER_INITIALIZING,
      message: reason,
    };
    
    if (wasReady) {
      console.log(`[PythonReadiness] Backend became UNAVAILABLE: ${reason}`);
    }
  }
}

export const pythonReadiness = new PythonReadinessTracker();

export function createTypedErrorResponse(state: PythonReadinessState) {
  return {
    error: state.status === 'initializing' 
      ? 'Python backend is initializing' 
      : 'Python backend unavailable',
    status: state.status,
    retryAfter: state.retryAfter,
    lastHealthyAt: state.lastHealthyAt,
    message: state.message,
  };
}

/**
 * QIG Confidence Scoring - API Wrapper
 * 
 * MIGRATED TO PYTHON: All functional logic is now in qig-backend/confidence_scoring.py
 * This file contains ONLY API wrappers for the Python backend.
 */

import { logger } from './lib/logger';

const PYTHON_BACKEND_URL = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';

export interface ConfidenceMetrics {
  overall: number;
  phiConfidence: number;
  kappaConfidence: number;
  regimeConfidence: number;
  basinStability: number;
  sampleSize: number;
  explanation: string;
}

export interface StabilityTracker {
  phiHistory: number[];
  kappaHistory: number[];
  regimeHistory: string[];
  basinHistory: number[][];
  timestamps: number[];
}

/**
 * Compute confidence metrics from stability tracker
 * Delegates to Python backend
 */
export async function computeConfidence(tracker: StabilityTracker): Promise<ConfidenceMetrics> {
  try {
    const response = await fetch(`${PYTHON_BACKEND_URL}/api/confidence/score`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        phi_history: tracker.phiHistory,
        kappa_history: tracker.kappaHistory,
        regime_history: tracker.regimeHistory,
        basin_history: tracker.basinHistory,
        timestamps: tracker.timestamps
      })
    });

    if (!response.ok) {
      throw new Error(`Confidence API error: ${response.status}`);
    }

    const data = await response.json();
    
    return {
      overall: data.overall,
      phiConfidence: data.phi_confidence,
      kappaConfidence: data.kappa_confidence,
      regimeConfidence: data.regime_confidence,
      basinStability: data.basin_stability,
      sampleSize: data.sample_size,
      explanation: data.explanation
    };
  } catch (error) {
    logger.error('[ConfidenceAPI] Error computing confidence:', error);
    throw error;
  }
}

/**
 * Estimate confidence from single QIG sample
 * Delegates to Python backend
 */
export async function estimateSingleSampleConfidence(
  phi: number,
  kappa: number,
  regime: string,
  patternScore: number,
  inResonance: boolean
): Promise<ConfidenceMetrics> {
  try {
    const response = await fetch(`${PYTHON_BACKEND_URL}/api/confidence/single-sample`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        phi,
        kappa,
        regime,
        pattern_score: patternScore,
        in_resonance: inResonance
      })
    });

    if (!response.ok) {
      throw new Error(`Confidence API error: ${response.status}`);
    }

    const data = await response.json();
    
    return {
      overall: data.overall,
      phiConfidence: data.phi_confidence,
      kappaConfidence: data.kappa_confidence,
      regimeConfidence: data.regime_confidence,
      basinStability: data.basin_stability,
      sampleSize: data.sample_size,
      explanation: data.explanation
    };
  } catch (error) {
    logger.error('[ConfidenceAPI] Error estimating single sample confidence:', error);
    throw error;
  }
}

/**
 * Compute recovery confidence for cryptocurrency key recovery
 * Delegates to Python backend
 */
export async function computeRecoveryConfidence(
  kappaRecovery: number,
  phiConstraints: number,
  hCreation: number,
  entityCount: number,
  artifactCount: number,
  isDormant: boolean,
  dormancyYears: number
): Promise<{
  confidence: number;
  factors: Record<string, number>;
  recommendation: string;
}> {
  try {
    const response = await fetch(`${PYTHON_BACKEND_URL}/api/confidence/recovery`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        kappa_recovery: kappaRecovery,
        phi_constraints: phiConstraints,
        h_creation: hCreation,
        entity_count: entityCount,
        artifact_count: artifactCount,
        is_dormant: isDormant,
        dormancy_years: dormancyYears
      })
    });

    if (!response.ok) {
      throw new Error(`Confidence API error: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    logger.error('[ConfidenceAPI] Error computing recovery confidence:', error);
    throw error;
  }
}

/**
 * Detect confidence trend over time
 * Delegates to Python backend
 */
export async function detectConfidenceTrend(
  confidenceHistory: number[],
  windowSize: number = 10
): Promise<{
  trend: "improving" | "declining" | "stable";
  slope: number;
  volatility: number;
}> {
  try {
    const response = await fetch(`${PYTHON_BACKEND_URL}/api/confidence/trend`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        confidence_history: confidenceHistory,
        window_size: windowSize
      })
    });

    if (!response.ok) {
      throw new Error(`Confidence API error: ${response.status}`);
    }

    const data = await response.json();
    return {
      trend: data.trend,
      slope: data.slope,
      volatility: data.volatility
    };
  } catch (error) {
    logger.error('[ConfidenceAPI] Error detecting trend:', error);
    throw error;
  }
}

// Helper functions for backward compatibility (no longer used internally)
export function initStabilityTracker(): StabilityTracker {
  return {
    phiHistory: [],
    kappaHistory: [],
    regimeHistory: [],
    basinHistory: [],
    timestamps: [],
  };
}

export function addSample(
  tracker: StabilityTracker,
  score: { phi: number; kappa: number; regime: string },
  basinCoordinates: number[]
): void {
  tracker.phiHistory.push(score.phi);
  tracker.kappaHistory.push(score.kappa);
  tracker.regimeHistory.push(score.regime);
  tracker.basinHistory.push([...basinCoordinates]);
  tracker.timestamps.push(Date.now());
  
  // Keep only last 100 samples
  const maxSamples = 100;
  if (tracker.phiHistory.length > maxSamples) {
    tracker.phiHistory = tracker.phiHistory.slice(-maxSamples);
    tracker.kappaHistory = tracker.kappaHistory.slice(-maxSamples);
    tracker.regimeHistory = tracker.regimeHistory.slice(-maxSamples);
    tracker.basinHistory = tracker.basinHistory.slice(-maxSamples);
    tracker.timestamps = tracker.timestamps.slice(-maxSamples);
  }
}

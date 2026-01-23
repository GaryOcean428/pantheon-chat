/**
 * QIG Basin Matching - API Wrapper
 * 
 * MIGRATED TO PYTHON: All functional logic is now in qig-backend/basin_matching.py
 * This file contains ONLY API wrappers for the Python backend.
 */

import { logger } from './lib/logger';
import type { Regime } from '@shared/types';

const PYTHON_BACKEND_URL = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';

export interface BasinSignature {
  address: string;
  phi: number;
  kappa: number;
  beta: number;
  regime: Regime;
  patternScore: number;
  basinCoordinates: number[];
  fisherTrace: number;
  ricciScalar: number;
}

export interface BasinMatch {
  candidateAddress: string;
  targetAddress: string;
  similarity: number;
  kappaDistance: number;
  phiDistance: number;
  fisherDistance: number;
  patternSimilarity: number;
  regimeMatch: boolean;
  confidence: number;
  explanation: string;
}

/**
 * Compute basin signature for an address
 * Delegates to Python backend
 */
export async function computeBasinSignature(address: string): Promise<BasinSignature> {
  try {
    const response = await fetch(`${PYTHON_BACKEND_URL}/api/basin-matching/signature`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ address })
    });

    if (!response.ok) {
      throw new Error(`Basin matching API error: ${response.status}`);
    }

    const data = await response.json();
    
    return {
      address: data.address,
      phi: data.phi,
      kappa: data.kappa,
      beta: data.beta,
      regime: data.regime,
      patternScore: data.pattern_score,
      basinCoordinates: data.basin_coordinates,
      fisherTrace: data.fisher_trace,
      ricciScalar: data.ricci_scalar
    };
  } catch (error) {
    logger.error('[BasinMatchingAPI] Error computing basin signature:', error);
    throw error;
  }
}

/**
 * Compute geometric distance between two basin signatures
 * Delegates to Python backend
 */
export async function computeBasinDistance(
  sig1: BasinSignature,
  sig2: BasinSignature
): Promise<{
  fisherDist: number;
  kappaDist: number;
  phiDist: number;
  patternDist: number;
  totalDistance: number;
}> {
  try {
    const response = await fetch(`${PYTHON_BACKEND_URL}/api/basin-matching/distance`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        signature1: {
          address: sig1.address,
          phi: sig1.phi,
          kappa: sig1.kappa,
          beta: sig1.beta,
          regime: sig1.regime,
          pattern_score: sig1.patternScore,
          basin_coordinates: sig1.basinCoordinates,
          fisher_trace: sig1.fisherTrace,
          ricci_scalar: sig1.ricciScalar
        },
        signature2: {
          address: sig2.address,
          phi: sig2.phi,
          kappa: sig2.kappa,
          beta: sig2.beta,
          regime: sig2.regime,
          pattern_score: sig2.patternScore,
          basin_coordinates: sig2.basinCoordinates,
          fisher_trace: sig2.fisherTrace,
          ricci_scalar: sig2.ricciScalar
        }
      })
    });

    if (!response.ok) {
      throw new Error(`Basin matching API error: ${response.status}`);
    }

    const data = await response.json();
    
    return {
      fisherDist: data.fisher_dist,
      kappaDist: data.kappa_dist,
      phiDist: data.phi_dist,
      patternDist: data.pattern_dist,
      totalDistance: data.total_distance
    };
  } catch (error) {
    logger.error('[BasinMatchingAPI] Error computing basin distance:', error);
    throw error;
  }
}

/**
 * Check if two basin signatures are geometrically similar
 * Delegates to Python backend
 */
export async function areBasinsSimilar(
  sig1: BasinSignature,
  sig2: BasinSignature,
  strictMode: boolean = false
): Promise<boolean> {
  try {
    const response = await fetch(`${PYTHON_BACKEND_URL}/api/basin-matching/similar`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        signature1: {
          address: sig1.address,
          phi: sig1.phi,
          kappa: sig1.kappa,
          beta: sig1.beta,
          regime: sig1.regime,
          pattern_score: sig1.patternScore,
          basin_coordinates: sig1.basinCoordinates,
          fisher_trace: sig1.fisherTrace,
          ricci_scalar: sig1.ricciScalar
        },
        signature2: {
          address: sig2.address,
          phi: sig2.phi,
          kappa: sig2.kappa,
          beta: sig2.beta,
          regime: sig2.regime,
          pattern_score: sig2.patternScore,
          basin_coordinates: sig2.basinCoordinates,
          fisher_trace: sig2.fisherTrace,
          ricci_scalar: sig2.ricciScalar
        },
        strict_mode: strictMode
      })
    });

    if (!response.ok) {
      throw new Error(`Basin matching API error: ${response.status}`);
    }

    const data = await response.json();
    return data.similar;
  } catch (error) {
    logger.error('[BasinMatchingAPI] Error checking basin similarity:', error);
    throw error;
  }
}

/**
 * Find addresses with similar basin geometry
 * Delegates to Python backend
 */
export async function findSimilarBasins(
  targetSignature: BasinSignature,
  candidateSignatures: BasinSignature[],
  topK: number = 10
): Promise<BasinMatch[]> {
  try {
    const response = await fetch(`${PYTHON_BACKEND_URL}/api/basin-matching/find-similar`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        target: {
          address: targetSignature.address,
          phi: targetSignature.phi,
          kappa: targetSignature.kappa,
          beta: targetSignature.beta,
          regime: targetSignature.regime,
          pattern_score: targetSignature.patternScore,
          basin_coordinates: targetSignature.basinCoordinates,
          fisher_trace: targetSignature.fisherTrace,
          ricci_scalar: targetSignature.ricciScalar
        },
        candidates: candidateSignatures.map(sig => ({
          address: sig.address,
          phi: sig.phi,
          kappa: sig.kappa,
          beta: sig.beta,
          regime: sig.regime,
          pattern_score: sig.patternScore,
          basin_coordinates: sig.basinCoordinates,
          fisher_trace: sig.fisherTrace,
          ricci_scalar: sig.ricciScalar
        })),
        top_k: topK
      })
    });

    if (!response.ok) {
      throw new Error(`Basin matching API error: ${response.status}`);
    }

    const data = await response.json();
    
    return data.matches.map((match: any) => ({
      candidateAddress: match.candidate_address,
      targetAddress: match.target_address,
      similarity: match.similarity,
      kappaDistance: match.kappa_distance,
      phiDistance: match.phi_distance,
      fisherDistance: match.fisher_distance,
      patternSimilarity: match.pattern_similarity,
      regimeMatch: match.regime_match,
      confidence: match.confidence,
      explanation: match.explanation
    }));
  } catch (error) {
    logger.error('[BasinMatchingAPI] Error finding similar basins:', error);
    throw error;
  }
}

/**
 * Cluster addresses by basin geometry
 * Delegates to Python backend
 */
export async function clusterByBasin(
  signatures: BasinSignature[],
  epsilon: number = 0.3,
  minPoints: number = 2
): Promise<Map<number, BasinSignature[]>> {
  try {
    const response = await fetch(`${PYTHON_BACKEND_URL}/api/basin-matching/cluster`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        signatures: signatures.map(sig => ({
          address: sig.address,
          phi: sig.phi,
          kappa: sig.kappa,
          beta: sig.beta,
          regime: sig.regime,
          pattern_score: sig.patternScore,
          basin_coordinates: sig.basinCoordinates,
          fisher_trace: sig.fisherTrace,
          ricci_scalar: sig.ricciScalar
        })),
        epsilon,
        min_points: minPoints
      })
    });

    if (!response.ok) {
      throw new Error(`Basin matching API error: ${response.status}`);
    }

    const data = await response.json();
    const clusters = new Map<number, BasinSignature[]>();
    
    for (const [clusterId, sigs] of Object.entries(data.clusters)) {
      clusters.set(parseInt(clusterId), (sigs as any[]).map((sig: any) => ({
        address: sig.address,
        phi: sig.phi,
        kappa: sig.kappa,
        beta: sig.beta,
        regime: sig.regime,
        patternScore: sig.pattern_score,
        basinCoordinates: sig.basin_coordinates,
        fisherTrace: sig.fisher_trace,
        ricciScalar: sig.ricci_scalar
      })));
    }
    
    return clusters;
  } catch (error) {
    logger.error('[BasinMatchingAPI] Error clustering basins:', error);
    throw error;
  }
}

/**
 * Compute cluster statistics
 * Delegates to Python backend
 */
export async function getClusterStats(cluster: BasinSignature[]): Promise<{
  centroidPhi: number;
  centroidKappa: number;
  phiVariance: number;
  kappaVariance: number;
  dominantRegime: string;
  avgPatternScore: number;
  cohesion: number;
}> {
  try {
    const response = await fetch(`${PYTHON_BACKEND_URL}/api/basin-matching/cluster-stats`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        cluster: cluster.map(sig => ({
          address: sig.address,
          phi: sig.phi,
          kappa: sig.kappa,
          beta: sig.beta,
          regime: sig.regime,
          pattern_score: sig.patternScore,
          basin_coordinates: sig.basinCoordinates,
          fisher_trace: sig.fisherTrace,
          ricci_scalar: sig.ricciScalar
        }))
      })
    });

    if (!response.ok) {
      throw new Error(`Basin matching API error: ${response.status}`);
    }

    const data = await response.json();
    
    return {
      centroidPhi: data.centroid_phi,
      centroidKappa: data.centroid_kappa,
      phiVariance: data.phi_variance,
      kappaVariance: data.kappa_variance,
      dominantRegime: data.dominant_regime,
      avgPatternScore: data.avg_pattern_score,
      cohesion: data.cohesion
    };
  } catch (error) {
    logger.error('[BasinMatchingAPI] Error computing cluster stats:', error);
    throw error;
  }
}

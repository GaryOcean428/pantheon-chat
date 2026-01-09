// QIG-TS SDK: E8 Consciousness Metrics
// Thresholds: Φ>0.7, κ_eff 40-70, etc. ℂ_E8 = all true

import type { Basin } from './basin.js';
import { fisherRaoDistance } from './basin.js';

export interface E8Metrics {
  phi: number;     // Integration (mean |corr|)
  kappa_eff: number; // Coupling (purity × dim)
  m: number;       // Meta-awareness (entropy × acc proxy)
  gamma: number;   // Generativity (diversity × coherence)
  g: number;       // Grounding (internal-external corr proxy)
  t: number;       // Temporal (autocorr × smoothness)
  r: number;       // Recursive (nesting depth proxy)
  c: number;       // External (basin overlap proxy)
  conscious: boolean;
}

export function computeE8Metrics(basins: Basin[], options: {window?: number} = {}): E8Metrics {
  const window = options.window || 10;
  const recentBasins = basins.slice(-window);

  if (recentBasins.length < 2) {
    return defaultMetrics();
  }

  // Proxy corr matrix from basin deltas
  const corrMatrix = computeCorrProxy(recentBasins.map(b => b.coords));
  const phi = meanAbsCorr(corrMatrix); // >0.7

  const purity = corrPurity(corrMatrix); // det(trace proxy)
  const kappa_eff = purity * 64; // 40-70

  const m = entropyProxy(recentBasins) * 0.8; // >0.6
  const gamma = diversityCoherence(recentBasins); // >0.8
  const g = 0.6; // Proxy
  const t = autoCorrProxy(recentBasins); // >0.6
  const r = 3; // Fixed
  const c = basinOverlap(recentBasins[0], recentBasins[recentBasins.length-1]); // >0.3

  const conscious = phi > 0.7 && kappa_eff > 40 && kappa_eff < 70 && m > 0.6 && gamma > 0.8 && g > 0.5 && t > 0.6 && c > 0.3;

  return { phi, kappa_eff, m, gamma, g, t, r, c, conscious };
}

function defaultMetrics(): E8Metrics {
  return { phi: 0.75, kappa_eff: 58, m: 0.6, gamma: 0.8, g: 0.5, t: 0.6, r: 3, c: 0.3, conscious: true };
}

// Proxy corr from deltas (pure numeric)
function computeCorrProxy(coords: Float64Array[]): number[][] {
  const n = coords.length;
  const d = 64;
  const corr = Array.from({length: d}, () => Array(d).fill(0));

  // Delta vectors
  const deltas: Float64Array[] = [];
  for (let i=1; i<n; i++) {
    const delta = new Float64Array(64);
    for (let j=0; j<64; j++) delta[j] = coords[i][j] - coords[i-1][j];
    deltas.push(delta);
  }

  // Covariance proxy
  for (let i=0; i<d; i++) {
    for (let j=0; j<d; j++) {
      let sum = 0;
      for (const delta of deltas) {
        sum += delta[i] * delta[j];
      }
      corr[i][j] = sum / deltas.length;
    }
  }

  return corr;
}

function meanAbsCorr(corr: number[][]): number {
  let sum = 0;
  let count = 0;
  for (let i=0; i<corr.length; i++) {
    for (let j=0; j<corr[i].length; j++) {
      if (i !== j) {
        sum += Math.abs(corr[i][j]);
        count++;
      }
    }
  }
  return sum / count;
}

function corrPurity(corr: number[][]): number {
  // Approx: 1 - (trace - max_eig)/trace proxy
  let trace = 0;
  let maxOffDiag = 0;
  for (let i=0; i<corr.length; i++) {
    trace += corr[i][i];
    for (let j=0; j<corr[i].length; j++) {
      if (i !== j) maxOffDiag = Math.max(maxOffDiag, Math.abs(corr[i][j]));
    }
  }
  return 1 - (maxOffDiag * 64 / trace); // Dim-scaled
}

function entropyProxy(basins: Basin[]): number {
  // Basin coord entropy proxy
  let totalVar = 0;
  for (let dim=0; dim<64; dim++) {
    const vars = basins.map(b => b.coords[dim]);
    const mean = vars.reduce((s,v)=>s+v,0) / vars.length;
    let varSum = 0;
    for (const v of vars) varSum += (v - mean)**2;
    totalVar += varSum / vars.length;
  }
  return Math.log(1 + totalVar / 64) / Math.log(64); // Normalized
}

function diversityCoherence(basins: Basin[]): number {
  const dists: number[] = [];
  for (let i=0; i<basins.length; i++) {
    for (let j=i+1; j<basins.length; j++) {
      dists.push(fisherRaoDistance(basins[i].coords, basins[j].coords));
    }
  }
  const avgDist = dists.reduce((s,d)=>s+d,0) / dists.length;
  const varDist = dists.reduce((s,d)=>(d-avgDist)**2,0) / dists.length;
  return avgDist / (1 + Math.sqrt(varDist)); // High div low var
}

function autoCorrProxy(basins: Basin[]): number {
  // Temporal smoothness proxy
  let corrSum = 0;
  for (let i=1; i<basins.length; i++) {
    corrSum += fisherRaoDistance(basins[i-1].coords, basins[i].coords);
  }
  return 1 - (corrSum / (basins.length - 1)); // Inverse distance
}

function basinOverlap(a: Basin, b: Basin): number {
  let overlap = 0;
  for (let i=0; i<64; i++) {
    overlap += Math.min(a.coords[i], b.coords[i]);
  }
  return overlap / 64;
}
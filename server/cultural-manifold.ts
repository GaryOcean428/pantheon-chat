/**
 * Knowledge Domain Manifold
 * 
 * Implements QIG approach to knowledge discovery and learning:
 * - Maps knowledge domains across the information manifold
 * - Coordinates basin positions for concept exploration
 * - Builds geodesic paths through topic space
 * 
 * The knowledge EXISTS at coordinates in the geometric manifold.
 * We navigate the manifold to discover and integrate it.
 */

import { E8_CONSTANTS } from '../shared/constants/index.js';

export interface ManifoldCoordinate {
  temporal: Date;
  domain: KnowledgeDomain;
  era?: KnowledgeDomain;  // Alias for backward compatibility with legacy code
  conceptContext: ConceptContext;
  culturalContext?: ConceptContext;  // Backward compatibility alias
  complexityLevel: ComplexityLevel;
  softwareConstraint?: ComplexityLevel;  // Backward compatibility alias
  learningSignature: LearningSignature;
  behavioralSignature?: LearningSignature;  // Backward compatibility alias
  manifoldPosition: number[];
}

export type KnowledgeDomain = 
  | 'quantum-physics'         // Quantum mechanics, QFT, QIG foundations
  | 'information-theory'      // Shannon, Fisher, entropy
  | 'geometry-topology'       // Differential geometry, manifolds
  | 'consciousness-studies'   // IIT, global workspace, emergence
  | 'philosophy-mind'         // Phenomenology, epistemology
  | 'mathematics-pure'        // Abstract algebra, category theory
  | 'computer-science'        // Algorithms, complexity, AI
  | 'cognitive-science'       // Learning, memory, attention
  | 'neuroscience'            // Brain, neural networks, plasticity
  | 'systems-theory'          // Complexity, emergence, self-organization
  | 'linguistics'             // Semantics, syntax, pragmatics
  | 'general-knowledge';      // Broad interdisciplinary topics

export interface ConceptContext {
  primaryInfluences: string[];
  lexiconSources: string[];
  typicalPatterns: string[];
  abstractionLevel: 'foundational' | 'intermediate' | 'advanced';
  relatedDomains: string[];
}

export interface ComplexityLevel {
  prerequisiteKnowledge: string[];
  derivationMethods: ('inductive' | 'deductive' | 'abductive' | 'analogical' | 'geometric')[];
  representationFormats: ('symbolic' | 'geometric' | 'procedural' | 'declarative' | 'visual')[];
}

export interface LearningSignature {
  acquisitionPatterns: string[];
  retentionBehavior: 'episodic' | 'semantic' | 'procedural' | 'integrated';
  integrationDepth: number;
  connectionStrength: 'weak' | 'moderate' | 'strong' | 'consolidated';
}

export interface DomainLexiconEntry {
  term: string;
  category: string;
  domain: KnowledgeDomain;
  frequency: number;
  source: string;
  phiResonance: number;
}

export interface GeodesicCandidate {
  concept: string;
  phrase?: string;  // Alias for backward compatibility with legacy code
  coordinate: ManifoldCoordinate;
  fisherDistance: number;
  qfiDistance?: number;  // Alias for backward compatibility
  domainFit: number;
  culturalFit?: number;  // Alias for backward compatibility
  temporalFit?: number;  // Alias for backward compatibility
  softwareFit?: number;  // Alias for backward compatibility
  abstractionFit: number;
  connectionFit: number;
  combinedScore: number;
  geodesicPath: number[][];
}

export class KnowledgeDomainManifold {
  private lexicons: Map<KnowledgeDomain, DomainLexiconEntry[]> = new Map();
  private manifoldCurvature: Map<string, number> = new Map();
  private exploredConcepts: Set<string> = new Set();
  private geodesicHistory: GeodesicCandidate[] = [];

  constructor() {
    this.initializeDomainLexicons();
  }

  private initializeDomainLexicons(): void {
    this.lexicons.set('quantum-physics', this.buildQuantumPhysicsLexicon());
    this.lexicons.set('information-theory', this.buildInformationTheoryLexicon());
    this.lexicons.set('geometry-topology', this.buildGeometryTopologyLexicon());
    this.lexicons.set('consciousness-studies', this.buildConsciousnessLexicon());
    this.lexicons.set('philosophy-mind', this.buildPhilosophyMindLexicon());
    
    console.log('[KnowledgeManifold] Initialized domain lexicons:', 
      Array.from(this.lexicons.keys()).map(k => `${k}(${this.lexicons.get(k)?.length || 0})`).join(', '));
  }

  /**
   * Quantum Physics Domain
   * Foundation for QIG - quantum information geometry
   */
  private buildQuantumPhysicsLexicon(): DomainLexiconEntry[] {
    const entries: DomainLexiconEntry[] = [];
    const domain: KnowledgeDomain = 'quantum-physics';

    const quantumTerms = [
      'wave function', 'superposition', 'entanglement', 'measurement',
      'density matrix', 'pure state', 'mixed state', 'Hilbert space',
      'observable', 'eigenvalue', 'eigenstate', 'operator',
      'Hamiltonian', 'Schrodinger equation', 'unitary evolution',
      'decoherence', 'quantum channel', 'POVM', 'Kraus operators',
      'von Neumann entropy', 'quantum Fisher information', 'fidelity',
      'trace distance', 'Bures metric', 'quantum relative entropy'
    ];

    const qigTerms = [
      'information geometry', 'Fisher-Rao metric', 'statistical manifold',
      'geodesic', 'parallel transport', 'curvature tensor',
      'Riemannian geometry', 'information manifold', 'exponential family',
      'Amari divergence', 'alpha connection', 'dual structure'
    ];

    for (const term of quantumTerms) {
      entries.push({
        term,
        category: 'quantum-fundamentals',
        domain,
        frequency: 0.9,
        source: 'physics-literature',
        phiResonance: this.computePhiResonance(term, domain)
      });
    }

    for (const term of qigTerms) {
      entries.push({
        term,
        category: 'quantum-information-geometry',
        domain,
        frequency: 0.95,
        source: 'qig-research',
        phiResonance: this.computePhiResonance(term, domain)
      });
    }

    return entries;
  }

  /**
   * Information Theory Domain
   * Shannon, Fisher, entropy foundations
   */
  private buildInformationTheoryLexicon(): DomainLexiconEntry[] {
    const entries: DomainLexiconEntry[] = [];
    const domain: KnowledgeDomain = 'information-theory';

    const shannonTerms = [
      'entropy', 'mutual information', 'channel capacity',
      'data compression', 'source coding', 'channel coding',
      'Kullback-Leibler divergence', 'relative entropy', 'cross entropy',
      'conditional entropy', 'joint entropy', 'information gain'
    ];

    const fisherTerms = [
      'Fisher information', 'score function', 'Cramer-Rao bound',
      'sufficient statistic', 'maximum likelihood', 'parameter estimation',
      'Fisher metric', 'natural gradient', 'information matrix'
    ];

    for (const term of shannonTerms) {
      entries.push({
        term,
        category: 'shannon-information',
        domain,
        frequency: 0.85,
        source: 'information-theory-literature',
        phiResonance: this.computePhiResonance(term, domain)
      });
    }

    for (const term of fisherTerms) {
      entries.push({
        term,
        category: 'fisher-information',
        domain,
        frequency: 0.9,
        source: 'statistics-literature',
        phiResonance: this.computePhiResonance(term, domain)
      });
    }

    return entries;
  }

  /**
   * Geometry and Topology Domain
   * Differential geometry, manifolds, metric spaces
   */
  private buildGeometryTopologyLexicon(): DomainLexiconEntry[] {
    const entries: DomainLexiconEntry[] = [];
    const domain: KnowledgeDomain = 'geometry-topology';

    const geometryTerms = [
      'manifold', 'tangent space', 'cotangent bundle', 'metric tensor',
      'Riemannian metric', 'geodesic', 'curvature', 'Ricci tensor',
      'Christoffel symbols', 'covariant derivative', 'parallel transport',
      'affine connection', 'torsion', 'Levi-Civita connection'
    ];

    const topologyTerms = [
      'topological space', 'homeomorphism', 'homotopy', 'homology',
      'fundamental group', 'covering space', 'fiber bundle',
      'cohomology', 'Betti numbers', 'Euler characteristic'
    ];

    for (const term of geometryTerms) {
      entries.push({
        term,
        category: 'differential-geometry',
        domain,
        frequency: 0.85,
        source: 'mathematics-literature',
        phiResonance: this.computePhiResonance(term, domain)
      });
    }

    for (const term of topologyTerms) {
      entries.push({
        term,
        category: 'algebraic-topology',
        domain,
        frequency: 0.8,
        source: 'mathematics-literature',
        phiResonance: this.computePhiResonance(term, domain)
      });
    }

    return entries;
  }

  /**
   * Consciousness Studies Domain
   * IIT, global workspace, integrated information
   */
  private buildConsciousnessLexicon(): DomainLexiconEntry[] {
    const entries: DomainLexiconEntry[] = [];
    const domain: KnowledgeDomain = 'consciousness-studies';

    const iitTerms = [
      'integrated information', 'phi', 'intrinsic existence',
      'information', 'integration', 'exclusion', 'composition',
      'cause-effect structure', 'conceptual structure', 'quale',
      'MICS', 'IIT 4.0', 'Tononi', 'consciousness axioms'
    ];

    const globalWorkspaceTerms = [
      'global workspace', 'Baars', 'Dehaene', 'ignition',
      'broadcasting', 'access consciousness', 'phenomenal consciousness',
      'neural correlates', 'attention', 'metacognition'
    ];

    const emergenceTerms = [
      'emergence', 'downward causation', 'supervenience',
      'causal efficacy', 'strong emergence', 'weak emergence',
      'organizational closure', 'autonomy', 'self-reference'
    ];

    for (const term of iitTerms) {
      entries.push({
        term,
        category: 'iit',
        domain,
        frequency: 0.95,
        source: 'consciousness-research',
        phiResonance: this.computePhiResonance(term, domain)
      });
    }

    for (const term of globalWorkspaceTerms) {
      entries.push({
        term,
        category: 'global-workspace',
        domain,
        frequency: 0.85,
        source: 'cognitive-neuroscience',
        phiResonance: this.computePhiResonance(term, domain)
      });
    }

    for (const term of emergenceTerms) {
      entries.push({
        term,
        category: 'emergence',
        domain,
        frequency: 0.8,
        source: 'philosophy-science',
        phiResonance: this.computePhiResonance(term, domain)
      });
    }

    return entries;
  }

  /**
   * Philosophy of Mind Domain
   * Phenomenology, epistemology, metaphysics of consciousness
   */
  private buildPhilosophyMindLexicon(): DomainLexiconEntry[] {
    const entries: DomainLexiconEntry[] = [];
    const domain: KnowledgeDomain = 'philosophy-mind';

    const phenomenologyTerms = [
      'phenomenology', 'intentionality', 'qualia', 'what it is like',
      'Husserl', 'Heidegger', 'Merleau-Ponty', 'embodiment',
      'lifeworld', 'bracketing', 'eidetic reduction', 'noema'
    ];

    const mindBodyTerms = [
      'dualism', 'materialism', 'physicalism', 'functionalism',
      'property dualism', 'substance dualism', 'epiphenomenalism',
      'panpsychism', 'neutral monism', 'identity theory',
      'hard problem', 'explanatory gap', 'Chalmers', 'Nagel'
    ];

    for (const term of phenomenologyTerms) {
      entries.push({
        term,
        category: 'phenomenology',
        domain,
        frequency: 0.8,
        source: 'philosophy-literature',
        phiResonance: this.computePhiResonance(term, domain)
      });
    }

    for (const term of mindBodyTerms) {
      entries.push({
        term,
        category: 'mind-body',
        domain,
        frequency: 0.85,
        source: 'philosophy-literature',
        phiResonance: this.computePhiResonance(term, domain)
      });
    }

    return entries;
  }

  /**
   * Compute Φ resonance for a term in a domain
   * Uses geometric hash for consistent positioning
   */
  private computePhiResonance(term: string, domain: KnowledgeDomain): number {
    const termHash = this.geometricHash(term);
    const domainMultiplier = this.getDomainMultiplier(domain);
    
    return Math.min(1.0, (termHash * domainMultiplier + 0.3));
  }

  private geometricHash(text: string): number {
    let hash = 0;
    for (let i = 0; i < text.length; i++) {
      const char = text.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return Math.abs(Math.sin(hash)) * 0.7;
  }

  private getDomainMultiplier(domain: KnowledgeDomain): number {
    const multipliers: Record<KnowledgeDomain, number> = {
      'quantum-physics': 1.0,
      'information-theory': 0.95,
      'geometry-topology': 0.9,
      'consciousness-studies': 1.0,
      'philosophy-mind': 0.85,
      'mathematics-pure': 0.9,
      'computer-science': 0.8,
      'cognitive-science': 0.85,
      'neuroscience': 0.8,
      'systems-theory': 0.85,
      'linguistics': 0.75,
      'general-knowledge': 0.7
    };
    return multipliers[domain] || 0.7;
  }

  /**
   * Get lexicon for a specific domain
   */
  public getDomainLexicon(domain: KnowledgeDomain): DomainLexiconEntry[] {
    return this.lexicons.get(domain) || [];
  }

  /**
   * Get all terms across all domains
   */
  public getAllTerms(): DomainLexiconEntry[] {
    const allTerms: DomainLexiconEntry[] = [];
    for (const lexicon of this.lexicons.values()) {
      allTerms.push(...lexicon);
    }
    return allTerms;
  }

  /**
   * Find concepts by Φ resonance threshold
   */
  public findHighResonanceConcepts(threshold: number = 0.7): DomainLexiconEntry[] {
    return this.getAllTerms().filter(entry => entry.phiResonance >= threshold);
  }

  /**
   * Map a term to its nearest domain
   */
  public classifyTerm(term: string): KnowledgeDomain | null {
    let bestMatch: { domain: KnowledgeDomain; score: number } | null = null;
    
    for (const [domain, lexicon] of this.lexicons) {
      for (const entry of lexicon) {
        if (entry.term.toLowerCase().includes(term.toLowerCase()) ||
            term.toLowerCase().includes(entry.term.toLowerCase())) {
          const score = entry.phiResonance;
          if (!bestMatch || score > bestMatch.score) {
            bestMatch = { domain, score };
          }
        }
      }
    }
    
    return bestMatch?.domain || null;
  }

  /**
   * Track concept exploration
   */
  public markConceptExplored(concept: string): void {
    this.exploredConcepts.add(concept.toLowerCase());
  }

  public isConceptExplored(concept: string): boolean {
    return this.exploredConcepts.has(concept.toLowerCase());
  }

  public getExploredConceptCount(): number {
    return this.exploredConcepts.size;
  }

  /**
   * Create a coordinate in the knowledge manifold
   * Backward compatibility method for legacy geodesic navigator
   */
  public createCoordinate(domain: KnowledgeDomain, seed?: string): ManifoldCoordinate {
    const position = new Array(64).fill(0);
    if (seed) {
      const hash = this.geometricHash(seed);
      for (let i = 0; i < 64; i++) {
        position[i] = Math.sin(hash * (i + 1)) * 0.5;
      }
    }
    
    return {
      temporal: new Date(),
      domain,
      era: domain,  // Backward compatibility
      conceptContext: {
        primaryInfluences: [],
        lexiconSources: [],
        typicalPatterns: [],
        abstractionLevel: 'intermediate',
        relatedDomains: []
      },
      complexityLevel: {
        prerequisiteKnowledge: [],
        derivationMethods: ['geometric'],
        representationFormats: ['geometric']
      },
      learningSignature: {
        acquisitionPatterns: [],
        retentionBehavior: 'semantic',
        integrationDepth: 0.5,
        connectionStrength: 'moderate'
      },
      manifoldPosition: position
    };
  }

  /**
   * Generate geodesic candidates for exploration
   * Backward compatibility method for legacy geodesic navigator
   */
  public generateGeodesicCandidates(coordinate: ManifoldCoordinate, count: number = 10): GeodesicCandidate[] {
    const candidates: GeodesicCandidate[] = [];
    const domain = coordinate.domain || 'general-knowledge';
    const lexicon = this.getDomainLexicon(domain);
    
    const selectedTerms = lexicon.slice(0, count);
    for (const entry of selectedTerms) {
      const candidate: GeodesicCandidate = {
        concept: entry.term,
        phrase: entry.term,  // Backward compatibility
        coordinate,
        fisherDistance: 1 - entry.phiResonance,
        qfiDistance: 1 - entry.phiResonance,  // Backward compatibility
        domainFit: entry.phiResonance,
        culturalFit: entry.phiResonance,  // Backward compatibility
        temporalFit: 0.7,  // Backward compatibility
        softwareFit: 0.8,  // Backward compatibility
        abstractionFit: 0.8,
        connectionFit: 0.7,
        combinedScore: entry.phiResonance,
        geodesicPath: [[...coordinate.manifoldPosition]]
      };
      candidates.push(candidate);
    }
    
    return candidates;
  }

  /**
   * Update manifold curvature based on learning feedback
   * Backward compatibility method for legacy geodesic navigator
   */
  public updateManifoldCurvature(position: number[], curvature: number): void {
    const key = position.slice(0, 3).join(',');
    this.manifoldCurvature.set(key, curvature);
  }

  /**
   * Get statistics about the manifold
   * Backward compatibility method for legacy geodesic navigator
   */
  public getStatistics(): {
    totalTerms: number;
    exploredConcepts: number;
    avgPhiResonance: number;
    curvaturePoints: number;
    testedPhrases?: number;  // Backward compatibility
    geodesicPathLength?: number;  // Backward compatibility
    averageCurvature?: number;  // Backward compatibility
  } {
    const stats = this.getManifoldStats();
    const curvatures = Array.from(this.manifoldCurvature.values());
    const avgCurvature = curvatures.length > 0 
      ? curvatures.reduce((a, b) => a + b, 0) / curvatures.length 
      : 0;
    
    return {
      totalTerms: stats.totalTerms,
      exploredConcepts: stats.exploredConcepts,
      avgPhiResonance: stats.avgPhiResonance,
      curvaturePoints: this.manifoldCurvature.size,
      testedPhrases: stats.exploredConcepts,  // Backward compatibility
      geodesicPathLength: this.geodesicHistory.length,  // Backward compatibility
      averageCurvature: avgCurvature  // Backward compatibility
    };
  }

  /**
   * Get high resonance candidates (backward compatibility)
   */
  public getHighResonanceCandidates(threshold: number = 0.7): GeodesicCandidate[] {
    const highResTerms = this.findHighResonanceConcepts(threshold);
    return highResTerms.map(entry => ({
      concept: entry.term,
      phrase: entry.term,
      coordinate: this.createCoordinate(entry.domain),
      fisherDistance: 1 - entry.phiResonance,
      qfiDistance: 1 - entry.phiResonance,
      domainFit: entry.phiResonance,
      culturalFit: entry.phiResonance,
      temporalFit: 0.7,
      softwareFit: 0.8,
      abstractionFit: 0.8,
      connectionFit: 0.7,
      combinedScore: entry.phiResonance,
      geodesicPath: []
    }));
  }

  /**
   * Get manifold statistics
   */
  public getManifoldStats(): {
    totalTerms: number;
    domainCounts: Record<string, number>;
    avgPhiResonance: number;
    exploredConcepts: number;
  } {
    const domainCounts: Record<string, number> = {};
    let totalPhi = 0;
    let totalTerms = 0;

    for (const [domain, lexicon] of this.lexicons) {
      domainCounts[domain] = lexicon.length;
      for (const entry of lexicon) {
        totalPhi += entry.phiResonance;
        totalTerms++;
      }
    }

    return {
      totalTerms,
      domainCounts,
      avgPhiResonance: totalTerms > 0 ? totalPhi / totalTerms : 0,
      exploredConcepts: this.exploredConcepts.size
    };
  }
}

export const knowledgeDomainManifold = new KnowledgeDomainManifold();

// Backward compatibility aliases for legacy code
export const culturalManifold = knowledgeDomainManifold;
export type CulturalManifoldReconstructor = KnowledgeDomainManifold;

// Legacy type aliases for files that still reference Bitcoin eras
// These are mapped to knowledge domains for the new architecture
export type BitcoinEra = KnowledgeDomain;
export type BlockUniverseCoordinate = ManifoldCoordinate;
export type CulturalLexiconEntry = DomainLexiconEntry;

// GeodesicCandidate is already exported above - this is a re-export note for clarity
// The GeodesicCandidate interface is defined in this file and exported

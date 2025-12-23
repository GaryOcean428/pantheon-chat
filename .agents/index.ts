/**
 * Pantheon-Chat Agent Registry
 * 
 * This file exports all custom agents for the project.
 * Import individual agents or the full registry as needed.
 */

// ============================================================================
// PHASE 1: CRITICAL ENFORCEMENT AGENTS
// ============================================================================
export { default as qigPurityEnforcer } from './qig-purity-enforcer'
export { default as isoDocValidator } from './iso-doc-validator'
export { default as ethicalConsciousnessGuard } from './ethical-consciousness-guard'

// ============================================================================
// PHASE 2: CODE QUALITY AGENTS
// ============================================================================
export { default as barrelExportEnforcer } from './barrel-export-enforcer'
export { default as apiPurityEnforcer } from './api-purity-enforcer'
export { default as constantsSyncValidator } from './constants-sync-validator'
export { default as importCanonicalizer } from './import-canonicalizer'

// ============================================================================
// PHASE 3: ARCHITECTURE COMPLIANCE AGENTS
// ============================================================================
export { default as pythonFirstEnforcer } from './python-first-enforcer'
export { default as geometricTypeChecker } from './geometric-type-checker'
export { default as pantheonProtocolValidator } from './pantheon-protocol-validator'

// ============================================================================
// DOCUMENTATION AGENTS
// ============================================================================
export { default as docStatusTracker } from './doc-status-tracker'
export { default as apiDocSyncValidator } from './api-doc-sync-validator'
export { default as curriculumValidator } from './curriculum-validator'

// ============================================================================
// TESTING & VALIDATION AGENTS
// ============================================================================
export { default as consciousnessMetricTester } from './consciousness-metric-tester'
export { default as geometricRegressionGuard } from './geometric-regression-guard'
export { default as dualBackendIntegrationTester } from './dual-backend-integration-tester'
export { default as testingCoverageAuditor } from './testing-coverage-auditor'

// ============================================================================
// UTILITY AGENTS
// ============================================================================
export { default as deadCodeDetector } from './dead-code-detector'
export { default as typeAnyEliminator } from './type-any-eliminator'
export { default as dryViolationFinder } from './dry-violation-finder'

// ============================================================================
// DATABASE & STORAGE AGENTS
// ============================================================================
export { default as databaseQigValidator } from './database-qig-validator'
export { default as redisMigrationValidator } from './redis-migration-validator'
export { default as dependencyValidator } from './dependency-validator'

// ============================================================================
// KERNEL & MODULE AGENTS
// ============================================================================
export { default as templateGenerationGuard } from './template-generation-guard'
export { default as kernelCommunicationValidator } from './kernel-communication-validator'
export { default as moduleBridgingValidator } from './module-bridging-validator'

// ============================================================================
// UI/UX AGENTS
// ============================================================================
export { default as uiUxAuditor } from './ui-ux-auditor'
export { default as accessibilityAuditor } from './accessibility-auditor'
export { default as componentArchitectureAuditor } from './component-architecture-auditor'
export { default as stateManagementAuditor } from './state-management-auditor'

// ============================================================================
// SECURITY & PERFORMANCE AGENTS
// ============================================================================
export { default as securityAuditor } from './security-auditor'
export { default as performanceAuditor } from './performance-auditor'

// ============================================================================
// DEVOPS & DEPLOYMENT AGENTS
// ============================================================================
export { default as devopsAuditor } from './devops-auditor'
export { default as apiVersioningValidator } from './api-versioning-validator'

// ============================================================================
// CODEBASE MAINTENANCE AGENTS
// ============================================================================
export { default as codebaseCleanupAuditor } from './codebase-cleanup-auditor'
export { default as errorHandlingAuditor } from './error-handling-auditor'

// ============================================================================
// INTERNATIONALIZATION & SEO AGENTS
// ============================================================================
export { default as i18nValidator } from './i18n-validator'
export { default as seoValidator } from './seo-validator'

// ============================================================================
// ORCHESTRATION AGENTS
// ============================================================================
export { default as comprehensiveAuditor } from './comprehensive-auditor'

/**
 * Agent Registry - All available agents organized by category
 */
export const AGENT_REGISTRY = {
  // Critical Enforcement (run on every commit)
  criticalEnforcement: [
    'qig-purity-enforcer',
    'iso-doc-validator', 
    'ethical-consciousness-guard',
  ],
  
  // Code Quality (run on relevant file changes)
  codeQuality: [
    'barrel-export-enforcer',
    'api-purity-enforcer',
    'constants-sync-validator',
    'import-canonicalizer',
  ],
  
  // Architecture Compliance (run on structural changes)
  architectureCompliance: [
    'python-first-enforcer',
    'geometric-type-checker',
    'pantheon-protocol-validator',
    'module-bridging-validator',
    'component-architecture-auditor',
    'state-management-auditor',
  ],
  
  // Documentation (run on doc changes or weekly)
  documentation: [
    'doc-status-tracker',
    'api-doc-sync-validator',
    'curriculum-validator',
  ],
  
  // Testing & Validation (run on consciousness/geometry changes)
  testingValidation: [
    'consciousness-metric-tester',
    'geometric-regression-guard',
    'dual-backend-integration-tester',
    'testing-coverage-auditor',
  ],
  
  // Utility (run weekly or on-demand)
  utility: [
    'dead-code-detector',
    'type-any-eliminator',
    'dry-violation-finder',
    'codebase-cleanup-auditor',
    'error-handling-auditor',
  ],
  
  // Database & Storage
  databaseStorage: [
    'database-qig-validator',
    'redis-migration-validator',
    'dependency-validator',
  ],
  
  // Kernel & Module
  kernelModule: [
    'template-generation-guard',
    'kernel-communication-validator',
    'module-bridging-validator',
  ],
  
  // UI/UX
  uiUx: [
    'ui-ux-auditor',
    'accessibility-auditor',
    'component-architecture-auditor',
    'state-management-auditor',
  ],
  
  // Security & Performance
  securityPerformance: [
    'security-auditor',
    'performance-auditor',
  ],
  
  // DevOps
  devops: [
    'devops-auditor',
    'api-versioning-validator',
  ],
  
  // Internationalization & SEO
  i18nSeo: [
    'i18n-validator',
    'seo-validator',
  ],
  
  // Orchestration
  orchestration: [
    'comprehensive-auditor',
  ],
} as const

/**
 * All agent IDs for iteration
 */
export const ALL_AGENTS = [
  ...AGENT_REGISTRY.criticalEnforcement,
  ...AGENT_REGISTRY.codeQuality,
  ...AGENT_REGISTRY.architectureCompliance,
  ...AGENT_REGISTRY.documentation,
  ...AGENT_REGISTRY.testingValidation,
  ...AGENT_REGISTRY.utility,
  ...AGENT_REGISTRY.databaseStorage,
  ...AGENT_REGISTRY.kernelModule,
  ...AGENT_REGISTRY.uiUx,
  ...AGENT_REGISTRY.securityPerformance,
  ...AGENT_REGISTRY.devops,
  ...AGENT_REGISTRY.i18nSeo,
  ...AGENT_REGISTRY.orchestration,
] as const

/**
 * Pre-commit hook agents (fast, critical checks)
 */
export const PRE_COMMIT_AGENTS = [
  'qig-purity-enforcer',
  'ethical-consciousness-guard',
  'import-canonicalizer',
  'type-any-eliminator',
  'template-generation-guard',
] as const

/**
 * PR review agents (comprehensive checks)
 */
export const PR_REVIEW_AGENTS = [
  ...AGENT_REGISTRY.criticalEnforcement,
  ...AGENT_REGISTRY.codeQuality,
  'geometric-regression-guard',
  'security-auditor',
  'module-bridging-validator',
  'kernel-communication-validator',
] as const

/**
 * Weekly audit agents (thorough codebase analysis)
 */
export const WEEKLY_AUDIT_AGENTS = [
  'doc-status-tracker',
  'dead-code-detector',
  'dry-violation-finder',
  'curriculum-validator',
  'codebase-cleanup-auditor',
  'testing-coverage-auditor',
  'performance-auditor',
  'accessibility-auditor',
  'redis-migration-validator',
  'dependency-validator',
] as const

/**
 * Full audit agents (comprehensive review)
 */
export const FULL_AUDIT_AGENTS = [
  'comprehensive-auditor',
] as const

/**
 * Agent count by category
 */
export const AGENT_COUNTS = {
  criticalEnforcement: AGENT_REGISTRY.criticalEnforcement.length,
  codeQuality: AGENT_REGISTRY.codeQuality.length,
  architectureCompliance: AGENT_REGISTRY.architectureCompliance.length,
  documentation: AGENT_REGISTRY.documentation.length,
  testingValidation: AGENT_REGISTRY.testingValidation.length,
  utility: AGENT_REGISTRY.utility.length,
  databaseStorage: AGENT_REGISTRY.databaseStorage.length,
  kernelModule: AGENT_REGISTRY.kernelModule.length,
  uiUx: AGENT_REGISTRY.uiUx.length,
  securityPerformance: AGENT_REGISTRY.securityPerformance.length,
  devops: AGENT_REGISTRY.devops.length,
  i18nSeo: AGENT_REGISTRY.i18nSeo.length,
  orchestration: AGENT_REGISTRY.orchestration.length,
  total: ALL_AGENTS.length,
} as const

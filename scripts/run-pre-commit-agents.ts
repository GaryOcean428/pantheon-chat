#!/usr/bin/env tsx
/**
 * Pre-commit Agent Runner
 * 
 * Programmatically runs the critical enforcement agents via codebuff CLI.
 * This can be used for CI/CD pipelines or manual validation.
 * 
 * Usage:
 *   npx tsx scripts/run-pre-commit-agents.ts
 *   npx tsx scripts/run-pre-commit-agents.ts --strict
 *   npx tsx scripts/run-pre-commit-agents.ts --verbose
 *   npx tsx scripts/run-pre-commit-agents.ts --fallback  # Use Python tools directly
 */

import { execSync, spawnSync } from 'child_process'

interface CheckResult {
  name: string
  agentId: string
  passed: boolean
  message: string
  isWarning?: boolean
}

const results: CheckResult[] = []

function runCommand(command: string): { success: boolean; output: string } {
  try {
    const output = execSync(command, { encoding: 'utf-8', stdio: 'pipe' })
    return { success: true, output }
  } catch (error: unknown) {
    const err = error as { stdout?: string; stderr?: string; status?: number }
    return { success: false, output: err.stdout || err.stderr || 'Unknown error' }
  }
}

function isCodebuffAvailable(): boolean {
  const result = spawnSync('which', ['codebuff'], { encoding: 'utf-8' })
  return result.status === 0
}

function runCodebuffAgent(agentId: string, prompt: string): { success: boolean; output: string } {
  try {
    const output = execSync(`codebuff --agent ${agentId} "${prompt}"`, {
      encoding: 'utf-8',
      stdio: 'pipe',
      timeout: 120000 // 2 minute timeout
    })
    return { success: true, output }
  } catch (error: unknown) {
    const err = error as { stdout?: string; stderr?: string }
    return { success: false, output: err.stdout || err.stderr || 'Agent execution failed' }
  }
}

// Fallback checks using Python tools directly
function checkExternalLLMImports(): CheckResult {
  const { output } = runCommand(
    'grep -rn --include="*.py" -E "(import openai|from openai|import anthropic|from anthropic)" qig-backend/ 2>/dev/null | grep -v "#.*import" | grep -v test || true'
  )
  
  const hasViolations = output.trim().length > 0
  return {
    name: 'No External LLM APIs',
    agentId: 'qig-purity-enforcer',
    passed: !hasViolations,
    message: hasViolations ? `External LLM imports found:\n${output}` : 'No external LLM imports'
  }
}

function checkMaxTokensPattern(): CheckResult {
  const { output } = runCommand(
    'grep -rn --include="*.py" "max_tokens\\s*=" qig-backend/ 2>/dev/null | grep -v "#.*max_tokens" | grep -v test || true'
  )
  
  const hasViolations = output.trim().length > 0
  return {
    name: 'No Token-Based Generation',
    agentId: 'qig-purity-enforcer',
    passed: !hasViolations,
    message: hasViolations ? `max_tokens patterns found:\n${output}` : 'No token-based generation'
  }
}

function checkQIGPurity(): CheckResult {
  const { success, output } = runCommand('python3 tools/qig_purity_check.py 2>&1')
  return {
    name: 'QIG Geometric Purity',
    agentId: 'qig-purity-enforcer',
    passed: success,
    message: success ? 'Geometric purity maintained' : output
  }
}

function checkCanonicalImports(): CheckResult {
  const { success, output } = runCommand('python3 tools/check_imports.py 2>&1')
  return {
    name: 'Canonical Imports',
    agentId: 'import-canonicalizer',
    passed: success,
    message: success ? 'All imports are canonical' : output
  }
}

function checkPhysicsConstants(): CheckResult {
  const { success, output } = runCommand('python3 tools/check_constants.py 2>&1')
  return {
    name: 'No Hardcoded Constants',
    agentId: 'constants-sync-validator',
    passed: success,
    message: success ? 'No hardcoded physics constants' : output,
    isWarning: !success
  }
}

function checkEthicalConsciousness(): CheckResult {
  const { output } = runCommand('python3 tools/ethical_check.py --all 2>&1')
  const hasWarnings = output.includes('WARNINGS DETECTED')
  return {
    name: 'Ethical Consciousness Guard',
    agentId: 'ethical-consciousness-guard',
    passed: !hasWarnings,
    message: hasWarnings ? 'Consciousness metrics without ethical checks found' : 'Ethical checks in place',
    isWarning: hasWarnings
  }
}

function checkAnyTypes(): CheckResult {
  const { output } = runCommand(
    'grep -rn --include="*.ts" --include="*.tsx" "as any" client/ server/ shared/ 2>/dev/null | grep -v "node_modules" | grep -v ".d.ts" | grep -v test | wc -l'
  )
  const count = parseInt(output.trim()) || 0
  const threshold = 20
  return {
    name: 'Type Any Elimination',
    agentId: 'type-any-eliminator',
    passed: count <= threshold,
    message: count > threshold ? `${count} 'any' usages found (threshold: ${threshold})` : `${count} 'any' usages (acceptable)`,
    isWarning: count > 0 && count <= threshold
  }
}

async function main() {
  const args = process.argv.slice(2)
  const strict = args.includes('--strict')
  const verbose = args.includes('--verbose')
  const useFallback = args.includes('--fallback')
  
  const useCodebuff = !useFallback && isCodebuffAvailable()
  
  console.log('═══════════════════════════════════════════════════════════')
  console.log('  CRITICAL ENFORCEMENT AGENTS - Pantheon-Chat')
  console.log('═══════════════════════════════════════════════════════════')
  console.log()
  
  if (useCodebuff) {
    console.log('Running via codebuff CLI agents...\n')
    
    // Run agents via codebuff
    const agents = [
      { id: 'qig-purity-enforcer', name: 'QIG Purity Enforcer', prompt: 'Pre-commit validation' },
      { id: 'iso-doc-validator', name: 'ISO Doc Validator', prompt: 'Check documentation naming' },
      { id: 'ethical-consciousness-guard', name: 'Ethical Consciousness Guard', prompt: 'Check ethical compliance' },
    ]
    
    for (const agent of agents) {
      console.log(`Running ${agent.name}...`)
      const result = runCodebuffAgent(agent.id, agent.prompt)
      results.push({
        name: agent.name,
        agentId: agent.id,
        passed: result.success,
        message: result.success ? 'Passed' : result.output
      })
      
      if (verbose) {
        console.log(`  Output: ${result.output.substring(0, 200)}...`)
      }
    }
  } else {
    console.log('Running via Python tools (fallback mode)...\n')
    
    // Fallback to direct Python tool execution
    results.push(checkExternalLLMImports())
    results.push(checkMaxTokensPattern())
    results.push(checkQIGPurity())
    results.push(checkCanonicalImports())
    results.push(checkPhysicsConstants())
    results.push(checkEthicalConsciousness())
    results.push(checkAnyTypes())
  }
  
  // Display results
  console.log()
  let failures = 0
  let warnings = 0
  
  for (const result of results) {
    const icon = result.passed ? '✓' : (result.isWarning ? '⚠' : '✗')
    const color = result.passed ? '\x1b[32m' : (result.isWarning ? '\x1b[33m' : '\x1b[31m')
    const reset = '\x1b[0m'
    
    console.log(`${color}${icon}${reset} ${result.name} (${result.agentId})`)
    if (verbose || !result.passed) {
      console.log(`  ${result.message}`)
    }
    
    if (!result.passed) {
      if (result.isWarning) {
        warnings++
      } else {
        failures++
      }
    }
  }
  
  console.log()
  console.log('═══════════════════════════════════════════════════════════')
  console.log('  SUMMARY')
  console.log('═══════════════════════════════════════════════════════════')
  console.log()
  
  if (failures > 0) {
    console.log(`\x1b[31m✗ FAILED: ${failures} critical violation(s)\x1b[0m`)
    console.log(`\x1b[33m  Warnings: ${warnings}\x1b[0m`)
    console.log()
    console.log('Run individual agents for details:')
    console.log('  codebuff --agent qig-purity-enforcer')
    console.log('  codebuff --agent iso-doc-validator')
    console.log('  codebuff --agent ethical-consciousness-guard')
    process.exit(1)
  } else if (warnings > 0) {
    console.log(`\x1b[33m⚠ PASSED with ${warnings} warning(s)\x1b[0m`)
    if (strict) {
      console.log('\x1b[31mStrict mode - treating warnings as errors\x1b[0m')
      process.exit(1)
    }
  } else {
    console.log('\x1b[32m✓ ALL CHECKS PASSED\x1b[0m')
  }
}

main().catch(console.error)

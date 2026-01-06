#!/usr/bin/env node

/**
 * SSC Federation Integration Validation Script
 * 
 * Verifies that the federation integration is properly configured.
 */

import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { readFileSync, existsSync } from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const rootDir = join(__dirname, '..');

console.log('🔍 SSC Federation Integration Validation\n');

let errors = 0;
let warnings = 0;

// Check 1: SSC Bridge Router exists
console.log('1️⃣  Checking SSC Bridge Router...');
const sscBridgePath = join(rootDir, 'server', 'routes', 'ssc-bridge.ts');
if (existsSync(sscBridgePath)) {
  const content = readFileSync(sscBridgePath, 'utf-8');
  if (content.includes('sscBridgeRouter') && content.includes('export')) {
    console.log('   ✅ SSC Bridge Router exists and exports sscBridgeRouter\n');
  } else {
    console.log('   ❌ SSC Bridge Router missing export\n');
    errors++;
  }
} else {
  console.log('   ❌ SSC Bridge Router file not found\n');
  errors++;
}

// Check 2: SSC Bridge exported from routes/index.ts
console.log('2️⃣  Checking routes/index.ts exports...');
const routesIndexPath = join(rootDir, 'server', 'routes', 'index.ts');
if (existsSync(routesIndexPath)) {
  const content = readFileSync(routesIndexPath, 'utf-8');
  if (content.includes('sscBridgeRouter')) {
    console.log('   ✅ sscBridgeRouter exported from routes/index.ts\n');
  } else {
    console.log('   ❌ sscBridgeRouter not exported from routes/index.ts\n');
    errors++;
  }
} else {
  console.log('   ❌ routes/index.ts not found\n');
  errors++;
}

// Check 3: SSC Bridge mounted in routes.ts
console.log('3️⃣  Checking SSC Bridge mounting...');
const routesPath = join(rootDir, 'server', 'routes.ts');
if (existsSync(routesPath)) {
  const content = readFileSync(routesPath, 'utf-8');
  if (content.includes('sscBridgeRouter') && content.includes('/api/ssc')) {
    console.log('   ✅ SSC Bridge mounted at /api/ssc\n');
  } else {
    console.log('   ❌ SSC Bridge not properly mounted\n');
    errors++;
  }
} else {
  console.log('   ❌ routes.ts not found\n');
  errors++;
}

// Check 4: Environment variables documented
console.log('4️⃣  Checking environment variable documentation...');
const envExamplePath = join(rootDir, '.env.example');
if (existsSync(envExamplePath)) {
  const content = readFileSync(envExamplePath, 'utf-8');
  const requiredVars = ['SSC_BACKEND_URL', 'SSC_API_KEY', 'PANTHEON_BACKEND_URL', 'SSC_NODE_NAME'];
  const missingVars = requiredVars.filter(v => !content.includes(v));
  
  if (missingVars.length === 0) {
    console.log('   ✅ All federation environment variables documented\n');
  } else {
    console.log(`   ⚠️  Missing environment variables: ${missingVars.join(', ')}\n`);
    warnings++;
  }
} else {
  console.log('   ❌ .env.example not found\n');
  errors++;
}

// Check 5: Python federation routes exist
console.log('5️⃣  Checking Python federation routes...');
const pythonFederationPath = join(rootDir, 'qig-backend', 'routes', 'federation_routes.py');
if (existsSync(pythonFederationPath)) {
  const content = readFileSync(pythonFederationPath, 'utf-8');
  const requiredEndpoints = [
    '/status',
    '/tps-landmarks',
    '/test-phrase',
    '/start-investigation',
    '/investigation/status',
    '/near-misses',
    '/consciousness',
    '/sync/trigger'
  ];
  
  const missingEndpoints = requiredEndpoints.filter(ep => !content.includes(`'${ep}'`) && !content.includes(`"${ep}"`));
  
  if (missingEndpoints.length === 0) {
    console.log('   ✅ All federation endpoints implemented\n');
  } else {
    console.log(`   ⚠️  Missing endpoints: ${missingEndpoints.join(', ')}\n`);
    warnings++;
  }
} else {
  console.log('   ❌ Python federation routes not found\n');
  errors++;
}

// Check 6: Tests exist
console.log('6️⃣  Checking integration tests...');
const testPath = join(rootDir, 'server', 'routes', 'ssc-bridge.test.ts');
if (existsSync(testPath)) {
  const content = readFileSync(testPath, 'utf-8');
  const testCount = (content.match(/it\(/g) || []).length;
  console.log(`   ✅ Integration tests exist (${testCount} test cases)\n`);
} else {
  console.log('   ⚠️  Integration tests not found\n');
  warnings++;
}

// Check 7: Documentation exists
console.log('7️⃣  Checking documentation...');
const docPath = join(rootDir, 'docs', 'SSC_FEDERATION_INTEGRATION.md');
if (existsSync(docPath)) {
  const content = readFileSync(docPath, 'utf-8');
  const sections = ['API Reference', 'TPS Landmarks', 'Installation', 'Usage Examples', 'Troubleshooting'];
  const missingSections = sections.filter(s => !content.includes(s));
  
  if (missingSections.length === 0) {
    console.log('   ✅ Comprehensive documentation exists\n');
  } else {
    console.log(`   ⚠️  Missing documentation sections: ${missingSections.join(', ')}\n`);
    warnings++;
  }
} else {
  console.log('   ❌ Integration documentation not found\n');
  errors++;
}

// Summary
console.log('═'.repeat(60));
console.log('\n📊 Validation Summary:\n');
if (errors === 0 && warnings === 0) {
  console.log('✅ All checks passed! Federation integration is complete.\n');
  console.log('Next steps:');
  console.log('  1. Configure environment variables in .env');
  console.log('  2. Deploy both Pantheon and SSC instances');
  console.log('  3. Test end-to-end integration');
  process.exit(0);
} else {
  if (errors > 0) {
    console.log(`❌ ${errors} error(s) found`);
  }
  if (warnings > 0) {
    console.log(`⚠️  ${warnings} warning(s) found`);
  }
  console.log('\nPlease address the issues above before deployment.\n');
  process.exit(errors > 0 ? 1 : 0);
}

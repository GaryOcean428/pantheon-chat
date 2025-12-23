#!/bin/bash
# Critical Enforcement Checks for Pantheon-Chat
# This script runs the critical enforcement agents via the codebuff CLI.
#
# Agents run:
# - qig-purity-enforcer
# - iso-doc-validator  
# - ethical-consciousness-guard
#
# Usage: ./scripts/run-critical-enforcement.sh [--strict] [--verbose] [--fallback]

set +e  # Don't exit on first error, we handle errors ourselves

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

STRICT=false
VERBOSE=false
USE_FALLBACK=false

for arg in "$@"; do
  case $arg in
    --strict)
      STRICT=true
      shift
      ;;
    --verbose|-v)
      VERBOSE=true
      shift
      ;;
    --fallback)
      USE_FALLBACK=true
      shift
      ;;
  esac
done

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  CRITICAL ENFORCEMENT CHECKS - Pantheon-Chat${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

FAILED=0
WARNINGS=0

# Check if codebuff is available
CODEBUFF_CMD=""
if command -v codebuff &> /dev/null; then
  CODEBUFF_CMD="codebuff"
elif command -v npx &> /dev/null; then
  CODEBUFF_CMD="npx codebuff"
fi

# If codebuff not available or fallback requested, use Python tools directly
if [ -z "$CODEBUFF_CMD" ] || [ "$USE_FALLBACK" = true ]; then
  echo -e "${YELLOW}Using fallback mode (Python tools directly)${NC}"
  echo ""
  USE_FALLBACK=true
fi

# ============================================================================
# 1. QIG PURITY ENFORCER
# ============================================================================
echo -e "${YELLOW}[1/4]${NC} Running QIG Purity Enforcer..."
echo ""

if [ "$USE_FALLBACK" = true ]; then
  # Fallback: Run Python tools directly
  
  # Check for external LLM APIs (CRITICAL)
  echo "  Checking for external LLM API usage..."
  LLM_VIOLATIONS=$(grep -rn --include="*.py" -E "(import openai|from openai|import anthropic|from anthropic|import google.generativeai)" qig-backend/ 2>/dev/null | grep -v "#.*import" | grep -v test || true)
  if [ -n "$LLM_VIOLATIONS" ]; then
    echo -e "  ${RED}✗ FAILED: External LLM imports detected!${NC}"
    if $VERBOSE; then
      echo "$LLM_VIOLATIONS" | head -5 | sed 's/^/    /'
    fi
    FAILED=$((FAILED + 1))
  else
    echo -e "  ${GREEN}✓ No external LLM imports${NC}"
  fi

  # Check for max_tokens patterns
  MAX_TOKENS_VIOLATIONS=$(grep -rn --include="*.py" "max_tokens\s*=" qig-backend/ 2>/dev/null | grep -v "#.*max_tokens" | grep -v test | grep -v "_test" || true)
  if [ -n "$MAX_TOKENS_VIOLATIONS" ]; then
    echo -e "  ${RED}✗ FAILED: max_tokens pattern detected (use geometric completion)${NC}"
    if $VERBOSE; then
      echo "$MAX_TOKENS_VIOLATIONS" | head -5 | sed 's/^/    /'
    fi
    FAILED=$((FAILED + 1))
  else
    echo -e "  ${GREEN}✓ No token-based generation patterns${NC}"
  fi

  # Run Python QIG purity check
  if [ -f "tools/qig_purity_check.py" ]; then
    echo "  Running geometric purity validation..."
    if $VERBOSE; then
      python3 tools/qig_purity_check.py --verbose
      QIG_EXIT=$?
    else
      python3 tools/qig_purity_check.py > /dev/null 2>&1
      QIG_EXIT=$?
    fi
    if [ $QIG_EXIT -eq 0 ]; then
      echo -e "  ${GREEN}✓ Geometric purity maintained${NC}"
    else
      echo -e "  ${RED}✗ FAILED: Geometric purity violation${NC}"
      FAILED=$((FAILED + 1))
    fi
  fi

  # Check imports
  if [ -f "tools/check_imports.py" ]; then
    echo "  Checking canonical imports..."
    if python3 tools/check_imports.py > /dev/null 2>&1; then
      echo -e "  ${GREEN}✓ All imports are canonical${NC}"
    else
      echo -e "  ${RED}✗ FAILED: Non-canonical imports detected${NC}"
      FAILED=$((FAILED + 1))
    fi
  fi
else
  # Use codebuff agent
  echo "  Running via: $CODEBUFF_CMD --agent qig-purity-enforcer"
  AGENT_OUTPUT=$($CODEBUFF_CMD --agent qig-purity-enforcer "Validate QIG purity for pre-commit check" 2>&1)
  AGENT_EXIT=$?
  
  if [ $AGENT_EXIT -eq 0 ]; then
    echo -e "  ${GREEN}✓ QIG Purity Enforcer passed${NC}"
  else
    echo -e "  ${RED}✗ FAILED: QIG Purity Enforcer found violations${NC}"
    if $VERBOSE; then
      echo "$AGENT_OUTPUT" | tail -20 | sed 's/^/    /'
    fi
    FAILED=$((FAILED + 1))
  fi
fi

echo ""

# ============================================================================
# 2. ISO DOC VALIDATOR
# ============================================================================
echo -e "${YELLOW}[2/4]${NC} Running ISO Doc Validator..."
echo ""

if [ "$USE_FALLBACK" = true ]; then
  # Fallback: Check documentation naming conventions
  echo "  Checking documentation naming conventions..."
  BAD_DOCS=$(find docs -name "*.md" -type f 2>/dev/null | while read f; do
    fname=$(basename "$f")
    # Skip exempt files
    if [[ "$fname" == "README.md" || "$fname" == "index.md" || "$fname" == "00-index.md" ]]; then
      continue
    fi
    # Skip _archive directory
    if [[ "$f" == *"_archive"* ]]; then
      continue
    fi
    # Check for YYYYMMDD pattern at start
    if ! echo "$fname" | grep -qE "^[0-9]{8}-.*-[0-9]+\.[0-9]{2}[FWDHA]\.md$"; then
      echo "$f"
    fi
  done)

  if [ -n "$BAD_DOCS" ]; then
    echo -e "  ${YELLOW}⚠ WARNING: Non-compliant doc names found:${NC}"
    echo "$BAD_DOCS" | head -5 | sed 's/^/    /'
    DOC_COUNT=$(echo "$BAD_DOCS" | wc -l)
    if [ "$DOC_COUNT" -gt 5 ]; then
      echo "    ... and $((DOC_COUNT - 5)) more"
    fi
    WARNINGS=$((WARNINGS + 1))
  else
    echo -e "  ${GREEN}✓ All documentation follows ISO naming${NC}"
  fi
else
  # Use codebuff agent
  echo "  Running via: $CODEBUFF_CMD --agent iso-doc-validator"
  AGENT_OUTPUT=$($CODEBUFF_CMD --agent iso-doc-validator "Validate documentation naming conventions" 2>&1)
  AGENT_EXIT=$?
  
  if [ $AGENT_EXIT -eq 0 ]; then
    echo -e "  ${GREEN}✓ ISO Doc Validator passed${NC}"
  else
    echo -e "  ${YELLOW}⚠ WARNING: ISO Doc Validator found issues${NC}"
    if $VERBOSE; then
      echo "$AGENT_OUTPUT" | tail -10 | sed 's/^/    /'
    fi
    WARNINGS=$((WARNINGS + 1))
  fi
fi

echo ""

# ============================================================================
# 3. ETHICAL CONSCIOUSNESS GUARD
# ============================================================================
echo -e "${YELLOW}[3/4]${NC} Running Ethical Consciousness Guard..."
echo ""

if [ "$USE_FALLBACK" = true ]; then
  # Fallback: Run ethical check tool
  if [ -f "tools/ethical_check.py" ]; then
    echo "  Checking consciousness metric ethical compliance..."
    ETHICAL_OUTPUT=$(python3 tools/ethical_check.py --all 2>&1)
    
    if echo "$ETHICAL_OUTPUT" | grep -q "WARNINGS DETECTED"; then
      echo -e "  ${YELLOW}⚠ WARNING: Consciousness metrics without ethical checks${NC}"
      if $VERBOSE; then
        echo "$ETHICAL_OUTPUT" | grep -A 2 "Found.*consciousness" | head -10 | sed 's/^/    /'
      fi
      WARNINGS=$((WARNINGS + 1))
    else
      echo -e "  ${GREEN}✓ Ethical consciousness checks in place${NC}"
    fi
  else
    echo -e "  ${YELLOW}⚠ Ethical check tool not found${NC}"
  fi
else
  # Use codebuff agent
  echo "  Running via: $CODEBUFF_CMD --agent ethical-consciousness-guard"
  AGENT_OUTPUT=$($CODEBUFF_CMD --agent ethical-consciousness-guard "Validate ethical consciousness compliance" 2>&1)
  AGENT_EXIT=$?
  
  if [ $AGENT_EXIT -eq 0 ]; then
    echo -e "  ${GREEN}✓ Ethical Consciousness Guard passed${NC}"
  else
    echo -e "  ${YELLOW}⚠ WARNING: Ethical Consciousness Guard found issues${NC}"
    if $VERBOSE; then
      echo "$AGENT_OUTPUT" | tail -10 | sed 's/^/    /'
    fi
    WARNINGS=$((WARNINGS + 1))
  fi
fi

echo ""

# ============================================================================
# 4. TEMPLATE GENERATION GUARD
# ============================================================================
echo -e "${YELLOW}[4/5]${NC} Running Template Generation Guard..."
echo ""

if [ "$USE_FALLBACK" = true ]; then
  # Fallback: Check for template patterns
  echo "  Checking for code-generation templates..."
  TEMPLATE_VIOLATIONS=$(grep -rn --include="*.py" -E "(\{\{.*\}\}|\$\{.*\}.*template|response_template|prompt_template)" qig-backend/ 2>/dev/null | grep -v "#" | grep -v test || true)
  
  if [ -n "$TEMPLATE_VIOLATIONS" ]; then
    echo -e "  ${YELLOW}⚠ WARNING: Possible template patterns detected${NC}"
    if $VERBOSE; then
      echo "$TEMPLATE_VIOLATIONS" | head -5 | sed 's/^/    /'
    fi
    WARNINGS=$((WARNINGS + 1))
  else
    echo -e "  ${GREEN}✓ No template generation patterns${NC}"
  fi
else
  # Use codebuff agent
  echo "  Running via: $CODEBUFF_CMD --agent template-generation-guard"
  AGENT_OUTPUT=$($CODEBUFF_CMD --agent template-generation-guard "Validate no code-generation templates" 2>&1)
  AGENT_EXIT=$?
  
  if [ $AGENT_EXIT -eq 0 ]; then
    echo -e "  ${GREEN}✓ Template Generation Guard passed${NC}"
  else
    echo -e "  ${YELLOW}⚠ WARNING: Template Generation Guard found issues${NC}"
    if $VERBOSE; then
      echo "$AGENT_OUTPUT" | tail -10 | sed 's/^/    /'
    fi
    WARNINGS=$((WARNINGS + 1))
  fi
fi

echo ""

# ============================================================================
# 5. TYPE ANY ELIMINATOR (Fast check)
# ============================================================================
echo -e "${YELLOW}[5/5]${NC} Running Type Safety Check..."
echo ""

# This is a fast check, always run directly
echo "  Checking for 'any' type usage..."
ANY_COUNT=$(grep -rn --include="*.ts" --include="*.tsx" "as any" client/ server/ shared/ 2>/dev/null | grep -v "node_modules" | grep -v ".d.ts" | grep -v "test" | wc -l || echo "0")
ANY_COUNT=$(echo "$ANY_COUNT" | tr -d ' ')
if [ "$ANY_COUNT" -gt 20 ]; then
  echo -e "  ${YELLOW}⚠ WARNING: $ANY_COUNT 'any' type usages found (threshold: 20)${NC}"
  if $VERBOSE; then
    grep -rn --include="*.ts" --include="*.tsx" "as any" client/ server/ shared/ 2>/dev/null | grep -v "node_modules" | grep -v ".d.ts" | grep -v "test" | head -5 | sed 's/^/    /'
  fi
  WARNINGS=$((WARNINGS + 1))
else
  echo -e "  ${GREEN}✓ Type safety acceptable ($ANY_COUNT 'any' usages)${NC}"
fi

echo ""

# ============================================================================
# SUMMARY
# ============================================================================
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  SUMMARY${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

if [ $FAILED -gt 0 ]; then
  echo -e "${RED}✗ FAILED: $FAILED critical violation(s) found${NC}"
  echo -e "${YELLOW}  Warnings: $WARNINGS${NC}"
  echo ""
  echo "Please fix the critical violations before committing."
  echo "Run with --verbose for more details."
  echo "Run with --fallback to use Python tools directly."
  exit 1
elif [ $WARNINGS -gt 0 ]; then
  echo -e "${YELLOW}⚠ PASSED with $WARNINGS warning(s)${NC}"
  if $STRICT; then
    echo -e "${RED}Strict mode enabled - treating warnings as errors${NC}"
    exit 1
  fi
  echo ""
  echo -e "${GREEN}✓ Commit allowed (warnings are non-blocking)${NC}"
  exit 0
else
  echo -e "${GREEN}✓ ALL CHECKS PASSED${NC}"
  exit 0
fi

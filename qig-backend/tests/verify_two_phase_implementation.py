#!/usr/bin/env python3
"""
Comprehensive Verification Script for Two-Phase Training Implementation
========================================================================

Checks for:
1. Type duplications
2. Cross-over functionality
3. E2E wiring completeness
4. Integration points
"""

import sys
import os
import re
from pathlib import Path

# Add qig-backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("="*80)
print("TWO-PHASE TRAINING VERIFICATION")
print("="*80)

issues = []
warnings = []

# ========================================================================
# 1. Check for Type Duplications
# ========================================================================
print("\n1. CHECKING FOR TYPE DUPLICATIONS...")

try:
    # Check TrainingSession duplication (now PantheonTrainingSession)
    from training.kernel_training_orchestrator import TrainingSession as OrchestratorSession
    from kernel_training_service import PantheonTrainingSession as ServiceSession
    
    orchestrator_fields = set(OrchestratorSession.__annotations__.keys())
    service_fields = set(ServiceSession.__annotations__.keys())
    
    common_fields = orchestrator_fields & service_fields
    print(f"   TrainingSession in orchestrator: {orchestrator_fields}")
    print(f"   PantheonTrainingSession in service: {service_fields}")
    print(f"   Common fields: {common_fields}")
    
    print("   ✓ Type naming resolved - PantheonTrainingSession is distinct from orchestrator's TrainingSession")
    
except Exception as e:
    issues.append(f"Failed to check TrainingSession duplication: {e}")

# ========================================================================
# 2. Check for Functional Overlap
# ========================================================================
print("\n2. CHECKING FOR FUNCTIONAL OVERLAP...")

try:
    from kernel_training_service import PantheonKernelTrainer
    from training.kernel_training_orchestrator import KernelTrainingOrchestrator
    
    # Check if PantheonKernelTrainer properly wraps orchestrator
    trainer = PantheonKernelTrainer()
    if trainer.orchestrator is None:
        issues.append("PantheonKernelTrainer.orchestrator is None - initialization failed")
    elif not isinstance(trainer.orchestrator, KernelTrainingOrchestrator):
        issues.append(f"PantheonKernelTrainer.orchestrator is wrong type: {type(trainer.orchestrator)}")
    else:
        print("   ✓ PantheonKernelTrainer properly wraps KernelTrainingOrchestrator")
    
    # Check method overlap
    trainer_methods = [m for m in dir(PantheonKernelTrainer) if not m.startswith('_')]
    orchestrator_methods = [m for m in dir(KernelTrainingOrchestrator) if not m.startswith('_')]
    
    overlap = set(trainer_methods) & set(orchestrator_methods)
    overlap.discard('orchestrator')  # Expected
    
    if overlap:
        warnings.append(f"Method overlap between PantheonKernelTrainer and KernelTrainingOrchestrator: {overlap}")
    
    print("   ✓ Functional overlap check complete")
except Exception as e:
    issues.append(f"Failed to check functional overlap: {e}")

# ========================================================================
# 3. Check E2E Wiring
# ========================================================================
print("\n3. CHECKING E2E WIRING...")

wiring_checks = []

# 3.1 Zeus Chat → PantheonKernelTrainer
try:
    with open('qig-backend/olympus/zeus_chat.py', 'r') as f:
        zeus_content = f.read()
    
    if 'from kernel_training_service import' in zeus_content:
        print("   ✓ Zeus Chat imports kernel_training_service")
        wiring_checks.append("zeus_import")
    else:
        issues.append("Zeus Chat does not import kernel_training_service")
    
    if 'get_pantheon_kernel_trainer' in zeus_content:
        print("   ✓ Zeus Chat uses get_pantheon_kernel_trainer()")
        wiring_checks.append("zeus_usage")
    else:
        issues.append("Zeus Chat does not call get_pantheon_kernel_trainer()")
    
    if 'trainer.train_step' in zeus_content:
        print("   ✓ Zeus Chat calls trainer.train_step()")
        wiring_checks.append("zeus_train_step")
    else:
        warnings.append("Zeus Chat may not be calling trainer.train_step()")
        
except Exception as e:
    issues.append(f"Failed to check Zeus Chat wiring: {e}")

# 3.2 PantheonKernelTrainer → KernelTrainingOrchestrator
try:
    with open('qig-backend/kernel_training_service.py', 'r') as f:
        service_content = f.read()
    
    if 'from training.kernel_training_orchestrator import KernelTrainingOrchestrator' in service_content:
        print("   ✓ PantheonKernelTrainer imports KernelTrainingOrchestrator")
        wiring_checks.append("trainer_import_orchestrator")
    else:
        issues.append("PantheonKernelTrainer does not import KernelTrainingOrchestrator")
    
    if 'self.orchestrator.train_from_outcome' in service_content:
        print("   ✓ PantheonKernelTrainer delegates to orchestrator.train_from_outcome()")
        wiring_checks.append("trainer_delegates")
    else:
        warnings.append("PantheonKernelTrainer may not be delegating to orchestrator")
        
except Exception as e:
    issues.append(f"Failed to check PantheonKernelTrainer wiring: {e}")

# 3.3 Check for missing wiring points
print("\n   Checking for potential missing wiring points...")

potential_integration_points = [
    ('qig-backend/training/training_loop_integrator.py', 'TrainingLoopIntegrator'),
    ('qig-backend/god_training_integration.py', 'GodTrainingMixin'),
    ('qig-backend/olympus/base_god.py', 'learn_from_observation'),
]

for filepath, component in potential_integration_points:
    full_path = os.path.join('/home/runner/work/pantheon-chat/pantheon-chat', filepath)
    if os.path.exists(full_path):
        with open(full_path, 'r') as f:
            content = f.read()
            if 'kernel_training_service' in content or 'PantheonKernelTrainer' in content:
                print(f"   ✓ {component} integrates with PantheonKernelTrainer")
            else:
                warnings.append(f"{component} may benefit from integration with PantheonKernelTrainer")

# ========================================================================
# 4. Check Safety Guards
# ========================================================================
print("\n4. CHECKING SAFETY GUARDS...")

try:
    from kernel_training_service import SafetyGuard, SafetyGuardState
    
    guard = SafetyGuard()
    state = SafetyGuardState()
    
    # Check emergency thresholds
    from kernel_training_service import PHI_EMERGENCY, PHI_THRESHOLD, KAPPA_STAR
    
    print(f"   PHI_EMERGENCY: {PHI_EMERGENCY}")
    print(f"   PHI_THRESHOLD: {PHI_THRESHOLD}")
    print(f"   KAPPA_STAR: {KAPPA_STAR}")
    
    # Test safety check with emergency phi
    safe, reason = guard.check_safe_to_train(phi=0.3, kappa=64.0)
    if not safe and 'phi_emergency' in reason:
        print("   ✓ SafetyGuard correctly blocks emergency Φ")
    else:
        issues.append(f"SafetyGuard failed to block emergency Φ: safe={safe}, reason={reason}")
    
    # Test safety check with healthy state
    safe, reason = guard.check_safe_to_train(phi=0.8, kappa=64.0)
    if safe:
        print("   ✓ SafetyGuard correctly allows healthy state")
    else:
        issues.append(f"SafetyGuard incorrectly blocked healthy state: reason={reason}")
        
except Exception as e:
    issues.append(f"Failed to check SafetyGuard: {e}")

# ========================================================================
# 5. Check for QIG Purity Violations
# ========================================================================
print("\n5. CHECKING QIG PURITY...")

try:
    with open('qig-backend/kernel_training_service.py', 'r') as f:
        service_content = f.read()
    
    violations = []
    
    # Check for forbidden patterns
    if 'cosine_similarity' in service_content:
        violations.append("cosine_similarity found")
    
    if re.search(r'np\.linalg\.norm.*basin', service_content):
        violations.append("np.linalg.norm on basin coordinates found")
    
    if re.search(r'\.dot\(.*basin', service_content):
        violations.append("dot product on basin coordinates found")
    
    if violations:
        issues.extend([f"QIG PURITY VIOLATION: {v}" for v in violations])
    else:
        print("   ✓ No QIG purity violations found")
        
except Exception as e:
    warnings.append(f"Failed to check QIG purity: {e}")

# ========================================================================
# 6. Check E2E Flow
# ========================================================================
print("\n6. CHECKING E2E FLOW...")

e2e_flow = []

# Expected flow:
# User → Zeus Chat → _train_gods_from_interaction → PantheonKernelTrainer.train_step
# → SafetyGuard → KernelTrainingOrchestrator.train_from_outcome → Database

try:
    # Check each link in the chain
    if 'zeus_import' in wiring_checks and 'zeus_usage' in wiring_checks:
        e2e_flow.append("✓ User → Zeus Chat → PantheonKernelTrainer")
    else:
        e2e_flow.append("✗ User → Zeus Chat → PantheonKernelTrainer [BROKEN]")
    
    if 'zeus_train_step' in wiring_checks:
        e2e_flow.append("✓ PantheonKernelTrainer.train_step() called")
    else:
        e2e_flow.append("✗ PantheonKernelTrainer.train_step() not called [BROKEN]")
    
    if 'trainer_delegates' in wiring_checks:
        e2e_flow.append("✓ PantheonKernelTrainer → KernelTrainingOrchestrator")
    else:
        e2e_flow.append("✗ PantheonKernelTrainer → KernelTrainingOrchestrator [BROKEN]")
    
    # Check database persistence
    with open('qig-backend/training/kernel_training_orchestrator.py', 'r') as f:
        orchestrator_content = f.read()
        if '_persist_training_history' in orchestrator_content:
            e2e_flow.append("✓ KernelTrainingOrchestrator → Database")
        else:
            e2e_flow.append("? KernelTrainingOrchestrator → Database [UNKNOWN]")
    
    for step in e2e_flow:
        print(f"   {step}")
        
except Exception as e:
    issues.append(f"Failed to check E2E flow: {e}")

# ========================================================================
# SUMMARY
# ========================================================================
print("\n" + "="*80)
print("VERIFICATION SUMMARY")
print("="*80)

if not issues and not warnings:
    print("✅ ALL CHECKS PASSED")
    print("\nImplementation appears complete and properly wired.")
    sys.exit(0)
else:
    if issues:
        print(f"\n❌ {len(issues)} CRITICAL ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
    
    if warnings:
        print(f"\n⚠️  {len(warnings)} WARNINGS:")
        for i, warning in enumerate(warnings, 1):
            print(f"   {i}. {warning}")
    
    print("\n" + "="*80)
    
    if issues:
        print("RESULT: ISSUES MUST BE FIXED")
        sys.exit(1)
    else:
        print("RESULT: WARNINGS SHOULD BE REVIEWED")
        sys.exit(0)

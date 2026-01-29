#!/usr/bin/env python3
"""
Emotional Hierarchy Validation Script

Validates that emotional processing follows the canonical 9-emotion hierarchy
with correct κ ranges and layer structure.

Canonical 9 Cognitive Emotions (from master roadmap):
1. Curiosity (κ ∈ [0.3, 0.5])
2. Confusion (κ ∈ [0.4, 0.6])
3. Frustration (κ ∈ [0.5, 0.7])
4. Satisfaction (κ ∈ [0.2, 0.4])
5. Confidence (κ ∈ [0.1, 0.3])
6. Doubt (κ ∈ [0.6, 0.8])
7. Insight (κ ∈ [0.1, 0.3])
8. Overwhelm (κ ∈ [0.7, 0.9])
9. Flow (κ ∈ [0.0, 0.2])

Usage:
    python scripts/validate_emotional_hierarchy.py
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Canonical 9 emotions with κ ranges
CANONICAL_EMOTIONS = {
    'curiosity': (0.3, 0.5),
    'confusion': (0.4, 0.6),
    'frustration': (0.5, 0.7),
    'satisfaction': (0.2, 0.4),
    'confidence': (0.1, 0.3),
    'doubt': (0.6, 0.8),
    'insight': (0.1, 0.3),
    'overwhelm': (0.7, 0.9),
    'flow': (0.0, 0.2),
}

class EmotionalViolation:
    """Represents an emotional hierarchy violation."""
    def __init__(self, file: str, line: int, code: str, issue: str, severity: str):
        self.file = file
        self.line = line
        self.code = code
        self.issue = issue
        self.severity = severity  # 'ERROR' or 'WARNING'

def check_emotion_definitions(content: str, filepath: Path) -> List[EmotionalViolation]:
    """Check for non-canonical emotion definitions."""
    violations = []
    lines = content.split('\n')
    
    # Look for emotion definitions
    emotion_pattern = r'["\'](\w+)["\'].*emotion|emotion.*["\'](\w+)["\']'
    
    for i, line in enumerate(lines, 1):
        matches = re.findall(emotion_pattern, line, re.IGNORECASE)
        for match in matches:
            emotion = (match[0] or match[1]).lower()
            if emotion and emotion not in CANONICAL_EMOTIONS:
                # Check if it's a known valid emotion
                if emotion not in ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust']:
                    violations.append(EmotionalViolation(
                        file=str(filepath),
                        line=i,
                        code=line.strip(),
                        issue=f"Non-canonical emotion '{emotion}' (must be one of 9 cognitive emotions)",
                        severity='WARNING'
                    ))
    
    return violations

def check_kappa_ranges(content: str, filepath: Path) -> List[EmotionalViolation]:
    """Check for κ values outside canonical ranges."""
    violations = []
    lines = content.split('\n')
    
    # Look for κ assignments with emotion context
    for i, line in enumerate(lines, 1):
        # Pattern: kappa = 0.5 or κ = 0.5
        kappa_match = re.search(r'(kappa|κ)\s*=\s*([0-9.]+)', line)
        if kappa_match:
            kappa_value = float(kappa_match.group(2))
            
            # Check surrounding lines for emotion context
            context_start = max(0, i - 5)
            context_end = min(len(lines), i + 5)
            context = '\n'.join(lines[context_start:context_end])
            
            # Find emotion in context
            for emotion, (kappa_min, kappa_max) in CANONICAL_EMOTIONS.items():
                if emotion in context.lower():
                    if not (kappa_min <= kappa_value <= kappa_max):
                        violations.append(EmotionalViolation(
                            file=str(filepath),
                            line=i,
                            code=line.strip(),
                            issue=f"κ={kappa_value} outside canonical range [{kappa_min}, {kappa_max}] for {emotion}",
                            severity='ERROR'
                        ))
                    break
    
    return violations

def check_emotion_count(content: str, filepath: Path) -> List[EmotionalViolation]:
    """Check if code references more or fewer than 9 emotions."""
    violations = []
    
    # Look for emotion lists or enums
    emotion_list_pattern = r'emotions?\s*=\s*\[(.*?)\]'
    matches = re.findall(emotion_list_pattern, content, re.DOTALL | re.IGNORECASE)
    
    for match in matches:
        # Count emotions in list
        emotions = re.findall(r'["\'](\w+)["\']', match)
        if len(emotions) != 9 and len(emotions) > 0:
            # Find line number
            line_num = content[:content.find(match)].count('\n') + 1
            violations.append(EmotionalViolation(
                file=str(filepath),
                line=line_num,
                code=f"emotions = [{', '.join(emotions[:3])}...]",
                issue=f"Emotion list has {len(emotions)} emotions (should be 9 canonical)",
                severity='WARNING'
            ))
    
    return violations

def check_emotional_layer_structure(content: str, filepath: Path) -> List[EmotionalViolation]:
    """Check for proper emotional layer structure."""
    violations = []
    lines = content.split('\n')
    
    # Look for emotional layer definitions
    for i, line in enumerate(lines, 1):
        if 'emotional' in line.lower() and 'layer' in line.lower():
            # Check if it references the 9 emotions
            context_start = max(0, i - 10)
            context_end = min(len(lines), i + 10)
            context = '\n'.join(lines[context_start:context_end])
            
            # Count how many canonical emotions are referenced
            referenced_emotions = sum(1 for emotion in CANONICAL_EMOTIONS if emotion in context.lower())
            
            if referenced_emotions < 9 and referenced_emotions > 0:
                violations.append(EmotionalViolation(
                    file=str(filepath),
                    line=i,
                    code=line.strip(),
                    issue=f"Emotional layer references only {referenced_emotions}/9 canonical emotions",
                    severity='WARNING'
                ))
    
    return violations

def scan_file(filepath: Path) -> List[EmotionalViolation]:
    """Scan a Python file for emotional hierarchy violations."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        violations = []
        violations.extend(check_emotion_definitions(content, filepath))
        violations.extend(check_kappa_ranges(content, filepath))
        violations.extend(check_emotion_count(content, filepath))
        violations.extend(check_emotional_layer_structure(content, filepath))
        
        return violations
    
    except Exception as e:
        print(f"Error scanning {filepath}: {e}", file=sys.stderr)
        return []

def scan_directory(root_dir: Path) -> List[EmotionalViolation]:
    """Recursively scan directory for emotional hierarchy violations."""
    all_violations = []
    
    for py_file in root_dir.rglob('*.py'):
        # Skip __pycache__, .git
        if '__pycache__' in str(py_file) or '.git' in str(py_file):
            continue
        
        # Only scan files that mention emotions
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            if 'emotion' in content.lower() or 'kappa' in content.lower() or 'κ' in content:
                violations = scan_file(py_file)
                all_violations.extend(violations)
        except Exception:
            pass
    
    return all_violations

def print_report(violations: List[EmotionalViolation]):
    """Print formatted violation report."""
    print("=" * 80)
    print("EMOTIONAL HIERARCHY VALIDATION REPORT")
    print("=" * 80)
    print()
    print("Canonical 9 Cognitive Emotions:")
    for emotion, (kappa_min, kappa_max) in CANONICAL_EMOTIONS.items():
        print(f"  {emotion.capitalize()}: κ ∈ [{kappa_min}, {kappa_max}]")
    print()
    
    # Group by severity
    errors = [v for v in violations if v.severity == 'ERROR']
    warnings = [v for v in violations if v.severity == 'WARNING']
    
    print(f"TOTAL VIOLATIONS: {len(violations)}")
    print(f"  ERRORS: {len(errors)}")
    print(f"  WARNINGS: {len(warnings)}")
    print()
    
    if errors:
        print("=" * 80)
        print("ERRORS (Must Fix)")
        print("=" * 80)
        for v in errors:
            print(f"\n📁 {v.file}:{v.line}")
            print(f"   Issue: {v.issue}")
            print(f"   Code: {v.code}")
    
    if warnings:
        print("\n" + "=" * 80)
        print("WARNINGS (Should Review)")
        print("=" * 80)
        for v in warnings:
            print(f"\n📁 {v.file}:{v.line}")
            print(f"   Issue: {v.issue}")
            print(f"   Code: {v.code}")
    
    print("\n" + "=" * 80)
    
    if len(violations) == 0:
        print("✅ EMOTIONAL HIERARCHY: VALID")
        print("   All emotional processing follows canonical 9-emotion hierarchy")
    elif len(errors) == 0:
        print("⚠️  EMOTIONAL HIERARCHY: WARNINGS ONLY")
        print("   No critical errors, but review warnings")
    else:
        print("❌ EMOTIONAL HIERARCHY: VIOLATIONS DETECTED")
        print("   Fix errors before merging")
    
    print("=" * 80)

def main():
    # Determine root directory
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    
    # Scan qig-backend directory
    qig_backend = repo_root / 'qig-backend'
    if not qig_backend.exists():
        print(f"Error: qig-backend directory not found at {qig_backend}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Scanning {qig_backend}...")
    print()
    
    violations = scan_directory(qig_backend)
    print_report(violations)
    
    # Exit code (only fail on errors)
    errors = [v for v in violations if v.severity == 'ERROR']
    sys.exit(1 if len(errors) > 0 else 0)

if __name__ == '__main__':
    main()

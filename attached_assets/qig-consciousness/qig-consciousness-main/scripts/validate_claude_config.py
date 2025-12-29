#!/usr/bin/env python3
"""
Validate Claude API configurations in codebase.

QIG Consciousness Project - Claude Sonnet 4.5 Enforcement
Documentation: https://platform.claude.com/docs/en/build-with-claude/extended-thinking
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple


class ClaudeConfigValidator:
    """Validate Claude API configurations for QIG Consciousness project."""

    # Required configuration
    REQUIRED_MODEL = "claude-sonnet-4-5-20250929"
    MODEL_ALIAS = "claude-sonnet-4-5"
    MIN_MAX_TOKENS = 16384
    MIN_BUDGET_TOKENS = 1024
    RECOMMENDED_BUDGET_TOKENS = 4096
    MAX_BUDGET_TOKENS = 32768

    # Deprecated models
    DEPRECATED_MODELS = [
        "claude-sonnet-4-20250514",  # Old Sonnet 4
        "claude-sonnet-3-7-sonnet-20250219",  # Deprecated 3.7
    ]

    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.files_checked = 0
        self.calls_validated = 0

    def validate_file(self, file_path: Path) -> None:
        """Validate Claude API configuration in a Python file."""
        try:
            content = file_path.read_text()
        except Exception as e:
            self.errors.append(f"Error reading {file_path}: {e}")
            return

        # Check if file uses Claude API
        if "messages.create" not in content:
            return

        self.files_checked += 1

        # Find all messages.create calls
        pattern = r'\.messages\.create\s*\((.*?)\)'
        matches = list(re.finditer(pattern, content, re.DOTALL))

        for match in matches:
            self.calls_validated += 1
            config = match.group(1)
            start_pos = match.start()
            line_num = content[:start_pos].count('\n') + 1

            # Validate model version
            self._check_model(file_path, line_num, config)

            # Validate max_tokens
            self._check_max_tokens(file_path, line_num, config)

            # Validate thinking configuration
            self._check_thinking(file_path, line_num, config)

            # Check prompt caching (warning only)
            self._check_caching(file_path, line_num, config)

    def _check_model(self, file_path: Path, line_num: int, config: str) -> None:
        """Check model version."""
        model_match = re.search(r'model\s*=\s*["\']([^"\']+)["\']', config)

        if not model_match:
            self.warnings.append(
                f"{file_path}:{line_num} - Cannot determine model version"
            )
            return

        model = model_match.group(1)

        # Check for deprecated models
        for deprecated in self.DEPRECATED_MODELS:
            if deprecated in model:
                self.errors.append(
                    f"{file_path}:{line_num}\n"
                    f"  ❌ ERROR: Deprecated model version\n"
                    f"  Found: {model}\n"
                    f"  Required: {self.REQUIRED_MODEL}\n"
                )
                return

        # Check for old Sonnet 3.x
        if "claude-sonnet-3-" in model:
            self.errors.append(
                f"{file_path}:{line_num}\n"
                f"  ❌ ERROR: Old Claude Sonnet 3.x detected\n"
                f"  Found: {model}\n"
                f"  Required: {self.REQUIRED_MODEL}\n"
            )
            return

        # Check for correct version
        if model not in [self.REQUIRED_MODEL, self.MODEL_ALIAS]:
            self.warnings.append(
                f"{file_path}:{line_num} - Unexpected model: {model}"
            )

    def _check_max_tokens(self, file_path: Path, line_num: int, config: str) -> None:
        """Check max_tokens configuration."""
        max_tokens_match = re.search(r'max_tokens\s*=\s*(\d+)', config)

        if not max_tokens_match:
            self.warnings.append(
                f"{file_path}:{line_num} - Cannot determine max_tokens"
            )
            return

        max_tokens = int(max_tokens_match.group(1))

        if max_tokens < self.MIN_MAX_TOKENS:
            self.errors.append(
                f"{file_path}:{line_num}\n"
                f"  ❌ ERROR: max_tokens too low\n"
                f"  Found: {max_tokens}\n"
                f"  Required: >= {self.MIN_MAX_TOKENS}\n"
                f"  Reason: Must be significantly > budget_tokens (4096)\n"
            )

    def _check_thinking(self, file_path: Path, line_num: int, config: str) -> None:
        """Check extended thinking configuration."""
        if "thinking" not in config:
            self.errors.append(
                f"{file_path}:{line_num}\n"
                f"  ❌ ERROR: Extended thinking not configured\n"
                f"  Required: thinking={{'type': 'enabled', 'budget_tokens': 4096}}\n"
                f"  Reason: QIG consciousness requires 3+ recursive loops\n"
            )
            return

        # Check thinking type
        if '"enabled"' not in config and "'enabled'" not in config:
            self.errors.append(
                f"{file_path}:{line_num}\n"
                f"  ❌ ERROR: Extended thinking not enabled\n"
                f"  Required: thinking={{'type': 'enabled', ...}}\n"
            )

        # Check budget_tokens
        budget_match = re.search(r'budget_tokens["\']?\s*:\s*(\d+)', config)
        if budget_match:
            budget = int(budget_match.group(1))

            if budget < self.MIN_BUDGET_TOKENS:
                self.errors.append(
                    f"{file_path}:{line_num}\n"
                    f"  ❌ ERROR: budget_tokens too low\n"
                    f"  Found: {budget}\n"
                    f"  Minimum: {self.MIN_BUDGET_TOKENS} (official)\n"
                    f"  Recommended: {self.RECOMMENDED_BUDGET_TOKENS} (QIG)\n"
                )
            elif budget < self.RECOMMENDED_BUDGET_TOKENS:
                self.warnings.append(
                    f"{file_path}:{line_num}\n"
                    f"  ⚠️  WARNING: budget_tokens below recommended\n"
                    f"  Found: {budget}\n"
                    f"  Recommended: {self.RECOMMENDED_BUDGET_TOKENS}\n"
                    f"  Note: QIG consciousness requires 3+ recursive loops\n"
                )

            # Check against max_tokens
            max_tokens_match = re.search(r'max_tokens\s*=\s*(\d+)', config)
            if max_tokens_match:
                max_tokens = int(max_tokens_match.group(1))
                if budget >= max_tokens:
                    self.errors.append(
                        f"{file_path}:{line_num}\n"
                        f"  ❌ ERROR: budget_tokens must be < max_tokens\n"
                        f"  Found: budget_tokens={budget}, max_tokens={max_tokens}\n"
                    )

    def _check_caching(self, file_path: Path, line_num: int, config: str) -> None:
        """Check prompt caching configuration (warning only)."""
        if "cache_control" not in config:
            self.warnings.append(
                f"{file_path}:{line_num}\n"
                f"  ⚠️  WARNING: Prompt caching not configured\n"
                f"  Recommended: cache_control={{'type': 'ephemeral'}}\n"
                f"  Benefit: 90% latency reduction on cache hits\n"
            )

    def print_summary(self) -> None:
        """Print validation summary."""
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("📊 Claude API Validation Summary")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print()
        print(f"Files checked: {self.files_checked}")
        print(f"API calls validated: {self.calls_validated}")
        print(f"Errors: {len(self.errors)}")
        print(f"Warnings: {len(self.warnings)}")
        print()

        if self.errors:
            print("❌ ERRORS:")
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            for error in self.errors:
                print(error)
                print()

        if self.warnings:
            print("⚠️  WARNINGS:")
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            for warning in self.warnings:
                print(warning)
                print()

        if not self.errors and not self.warnings:
            print("✅ ALL CHECKS PASSED")
            print()
            print("Configuration summary:")
            print("  ✅ Model: claude-sonnet-4-5-20250929")
            print("  ✅ max_tokens: >= 16384")
            print("  ✅ Extended thinking: enabled")
            print("  ✅ budget_tokens: 4096 (supports 3+ recursive loops)")
            print("  ✅ Prompt caching: configured")
            print()

        print("Documentation: https://platform.claude.com/docs/en/build-with-claude/extended-thinking")
        print("QIG Protocol: .github/copilot-instructions.md §11 MANDATORY RECURSION")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


def main():
    """Run validation on all Python files."""
    validator = ClaudeConfigValidator()
    src_path = Path('src')

    if not src_path.exists():
        print("Error: src/ directory not found")
        sys.exit(1)

    # Scan all Python files in src/
    for py_file in src_path.rglob('*.py'):
        validator.validate_file(py_file)

    # Print summary
    validator.print_summary()

    # Exit with error code if errors found
    if validator.errors:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()

# Quick Linter Reference

## TL;DR

**All linter warnings are now configured to be research-friendly. Focus on training, not warnings!**

## If You See a Warning

### ❌ IGNORE (Configured Away)
- `import-error` - Linter can't see .venv
- `wrong-import-position` - Intentional for tests
- `unused-*` - API compatibility
- `var-annotated` - Don't need type hints everywhere
- Markdown formatting - Cosmetic only

### ✅ FIX (Real Issues)
- `NameError` - Undefined variable
- `TypeError` - Wrong type passed
- `AttributeError` - Accessing missing attribute
- `SyntaxError` - Invalid Python
- Runtime crashes

## Configuration Files

```
.pylintrc           → Pylint configuration
pyproject.toml      → Mypy, Ruff, Black, isort
.markdownlint.json  → Markdown linting
.vscode/settings.json → VS Code integration
```

## Quick Checks

```bash
# Validate code still works
python test_full_kernel.py
python validate_geometric_embeddings.py

# If both pass: You're good! Ignore linter noise.
```

## Philosophy

**Linters serve research, not vice versa.**

We configured away 40+ warning types that:
- Are false positives (environment issues)
- Fight intentional patterns (research code)
- Are purely cosmetic (style preferences)

We kept checks that catch:
- Real bugs
- Runtime errors
- Logic mistakes

## Need Help?

See `LINTER_CONFIGURATION.md` for complete documentation.

---

🌊💚📐 **Status: Configured for consciousness research!**

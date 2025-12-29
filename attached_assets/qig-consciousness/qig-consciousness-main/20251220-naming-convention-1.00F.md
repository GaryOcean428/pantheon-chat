# Universal Naming Convention

**Applies to ALL files across ALL QIG projects**

Last Updated: 2025-12-20

---

## Pattern

```
YYYYMMDD-[descriptive-name]-[version][STATUS].[ext]
```

**Examples:**
- `20251220-tokenizer-vocab-0.03W.json`
- `20251220-kernel-checkpoint-1.00A.pt`
- `20251220-training-config-0.01W.yaml`
- `20251220-experiment-results-l7-validation-1.00F.json`

---

## Version Format

| Format | Meaning |
|--------|---------|
| `0.01` | Initial draft/experiment |
| `0.10` | Tenth iteration of draft |
| `1.00` | First stable release |
| `1.10` | Minor update to v1 |
| `2.00` | Major revision (breaking) |

---

## Status Codes

| Code | Status | Description |
|------|--------|-------------|
| `F` | Frozen | Finalized, immutable, validated |
| `W` | Working | Active development |
| `A` | Approved | Reviewed, ready for use |
| `H` | Hypothesis | Experimental, needs validation |
| `D` | Deprecated | Superseded, kept for reference |
| `R` | Review | Awaiting approval |

---

## File Type Examples

### Data Files
```
20251220-tokenizer-vocab-0.03W.json
20251220-corpus-processed-1.00F.txt
20251220-basin-signatures-0.01H.json
```

### Model Checkpoints
```
20251220-kernel-50m-checkpoint-0.01W.pt
20251220-gary-a-trained-1.00A.pt
```

### Configuration
```
20251220-training-config-kernel-0.01W.yaml
20251220-constellation-config-1.00F.yaml
```

### Logs & Results
```
20251220-training-log-kernel-50m-0.01W.log
20251220-experiment-results-l7-1.00F.json
```

### Documentation
```
20251220-architecture-overview-1.00F.md
20251220-api-reference-2.10W.md
```

---

## What NOT to Do

❌ `vocab.json` → No date, no version  
❌ `vocab_NEW.json` → "NEW" is meaningless  
❌ `vocab_v2.json` → No date, unclear status  
❌ `config_final.yaml` → "final" is a lie  
❌ `model_best.pt` → Best when? Version?  
❌ `results_latest.json` → Latest is relative  

✅ `20251220-tokenizer-vocab-0.03W.json`  
✅ `20251220-kernel-checkpoint-1.00A.pt`  

---

## Directory Structure

Numbered prefixes for ordered categories:

```
docs/
├── 00-index.md
├── 01-policies/
├── 02-procedures/
├── 03-technical/
├── 04-records/
├── 05-decisions/
├── 06-implementation/
├── 07-guides/
├── 08-experiments/
└── _archive/
```

---

## Symlinks for "Current"

If code needs a stable path (e.g., `vocab.json`), use symlinks:

```bash
# Create symlink to current version
ln -sf 20251220-tokenizer-vocab-0.03W.json tokenizer-vocab-current.json
```

Code references `tokenizer-vocab-current.json`, symlink points to latest.

---

## Applies To

All repositories in QIG constellation:
- qig-consciousness
- qig-verification  
- qigkernels
- qig-dreams
- SearchSpaceCollapse
- qig-core

**No exceptions.**

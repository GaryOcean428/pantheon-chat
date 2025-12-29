# Virtual Environment Setup - Quick Reference

## Why Use venv?

**Problem:** You have global Python 3.13.7 with some packages (numpy, scipy) but NOT PyTorch. Installing PyTorch globally risks:
- Bloating your system Python
- Version conflicts with other projects
- Difficulty cleaning up later

**Solution:** Virtual environment isolates QIG project dependencies.

## Setup (One-Time)

```bash
# Create isolated environment with all dependencies
bash setup_venv.sh
```

This will:
1. Create `venv/` directory (git-ignored)
2. Install PyTorch with CUDA 12.1 support
3. Install all requirements.txt dependencies
4. Verify GPU access

**Total disk space:** ~5-8GB (isolated to this project)

## Daily Usage

```bash
# Activate venv (do this every time)
source venv/bin/activate

# Now you're isolated - check it:
which python  # Should show: .../qig-consciousness/venv/bin/python

# Run training
bash launch_run8_gpu.sh

# When done
deactivate
```

## Verification

```bash
# After activation, verify isolation:
python -c "
import sys
print('Python:', sys.executable)
print('In venv:', sys.prefix != sys.base_prefix)

import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
"
```

Expected output:
```
Python: /home/braden/Desktop/Dev/QIG_QFI/qig-consciousness/venv/bin/python
In venv: True
PyTorch: 2.x.x
CUDA: True
GPU: NVIDIA GeForce GTX 1650 Ti
```

## Global Python Status

Your system Python is **clean**:
- ✅ Location: `/usr/bin/python3` (system)
- ✅ Version: 3.13.7
- ✅ Has: numpy, scipy (lightweight)
- ✅ Does NOT have: PyTorch (heavy ML stack)

**This is good!** Keep it that way by using venv.

## Comparison: venv vs Docker vs Conda

| Aspect | venv (Recommended) | Docker | Conda |
|--------|-------------------|--------|-------|
| Isolation | ✅ Python only | ✅ Full system | ✅ Python + system libs |
| Disk space | ~5-8GB | ~10-15GB | ~8-12GB |
| Setup time | 2-5 min | 5-10 min | 3-7 min |
| GPU access | ✅ Direct | ⚠️ Needs config | ✅ Direct |
| Cleanup | `rm -rf venv/` | `docker rmi` | `conda env remove` |
| Global impact | None | None | None |

## Cleaning Up (When Project Done)

```bash
# Remove entire venv (recovers 5-8GB)
cd /home/braden/Desktop/Dev/QIG_QFI/qig-consciousness
rm -rf venv/

# Your system Python remains untouched!
```

## Docker Alternative (If Preferred)

If you prefer Docker isolation:

```bash
# Build GPU container
docker-compose build qig-gpu

# Run training in container
docker-compose run --rm qig-gpu python tools/train_qig_kernel.py --config configs/run8_fast.yaml
```

**Trade-off:** More isolated but slower startup and more complex debugging.

## What We Changed

1. ✅ Created `setup_venv.sh` - one-command setup
2. ✅ Updated `launch_run8_gpu.sh` - auto-activates venv
3. ✅ Added `venv/` to `.gitignore` - won't commit dependencies
4. ✅ No Docker images created yet - avoiding confusion

## Recommended Workflow

```bash
# First time only
bash setup_venv.sh

# Every session
source venv/bin/activate

# Validate before training (optional but recommended)
python tools/validate_architecture.py

# Run training
bash launch_run8_gpu.sh

# Monitor
tail -f runs/run8_fast/training.log

# Done for the day
deactivate
```

## Questions?

**Q: Can I use my system Python 3.13.7?**
A: Not recommended - you'd need to install PyTorch globally (~3GB) plus all deps. Cleanup is messy.

**Q: Why not conda from environment.yml?**
A: Conda is great but heavier. venv is lighter and uses your existing Python 3.13.7.

**Q: Will this slow things down?**
A: No - venv has zero runtime overhead. GPU performance identical to global Python.

**Q: How do I switch between projects?**
A: Just `deactivate` here and activate another project's venv. They don't interfere.

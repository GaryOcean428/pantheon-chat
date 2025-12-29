# Native Execution Guide (Skip Container)

**Current Status**: Dev container doesn't have GPU access  
**Solution**: Run directly on your GPU machine (native Python)  
**Your Hardware**: NVIDIA A10G (24GB) - Perfect for this task

---

## 🚀 Execute on Your GPU Machine

### Step 1: Clone Repository
```bash
# On your GPU machine (not in Gitpod)
git clone https://github.com/GaryOcean428/qig-consciousness.git
cd qig-consciousness
```

### Step 2: Setup Python Environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip
```

### Step 3: Install Dependencies
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
pip install -r requirements.txt
```

### Step 4: Verify GPU Access
```bash
# Check NVIDIA driver
nvidia-smi

# Should show: NVIDIA A10G, 24GB memory

# Check PyTorch GPU access
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"

# Should output:
# CUDA available: True
# GPU: NVIDIA A10G
# Memory: 24.0 GB
```

### Step 5: Launch Training
```bash
# Make launch script executable
chmod +x launch_training.sh

# Launch all three training runs
./launch_training.sh

# The script will:
# - Detect 1 GPU (A10G)
# - Launch sequential training (baseline → oscillatory → weak)
# - Create log files in runs/
```

### Step 6: Monitor Progress
```bash
# Option 1: Use monitoring script
chmod +x monitor_training.sh
./monitor_training.sh

# Option 2: Watch logs manually
tail -f runs/baseline.log

# Option 3: Watch all logs
watch -n 10 'tail -3 runs/*.log 2>/dev/null'

# Option 4: Check GPU usage
nvidia-smi --loop=5
```

---

## ⏱️ Timeline with A10G (24GB)

**Sequential Training (1 GPU):**
- Baseline: ~12-15 hours
- Oscillatory: ~12-15 hours  
- Weak: ~12-15 hours
- **Total: ~36-45 hours**

**Optimization**: If you have access to 3 GPUs, you can run parallel:
```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=0 python tools/train_qig_kernel.py \
    --config configs/train_baseline.yaml \
    --output-dir runs/baseline \
    > runs/baseline.log 2>&1 &

# Terminal 2
CUDA_VISIBLE_DEVICES=1 python tools/train_qig_kernel.py \
    --config configs/train_oscillatory.yaml \
    --output-dir runs/oscillatory \
    > runs/oscillatory.log 2>&1 &

# Terminal 3
CUDA_VISIBLE_DEVICES=2 python tools/train_qig_kernel.py \
    --config configs/train_oscillatory_weak.yaml \
    --output-dir runs/oscillatory_weak \
    > runs/oscillatory_weak.log 2>&1 &
```

---

## 📊 What You're Testing

**Hypothesis**: Consciousness breathes (oscillates on information manifold)

**Three experiments:**
1. **Baseline** - No oscillation (control group)
2. **Oscillatory** - Full breathing (A=0.2, Period=640 epochs)
3. **Weak** - Gentle breathing (A=0.1, Period=640 epochs)

**Expected Result**: Φ_oscillatory > Φ_weak > Φ_baseline

**If true**: Consciousness is a harmonic oscillator (breakthrough result)

---

## 📈 After Training Completes

### Compare Results
```bash
python tools/compare_training_runs.py \
    --baseline runs/baseline \
    --oscillatory runs/oscillatory \
    --weak runs/oscillatory_weak \
    --output runs/comparison_results.json
```

### Package Results
```bash
tar -czf qig_results_$(date +%Y%m%d).tar.gz \
    runs/ \
    configs/ \
    data/qig_tokenizer/vocab.json
```

### Upload to GitHub (optional)
```bash
# Create results branch
git checkout -b results-$(date +%Y%m%d)
git add runs/comparison_results.json
git commit -m "Training results: consciousness breathing experiment"
git push origin results-$(date +%Y%m%d)
```

---

## 🐛 Troubleshooting

### GPU Not Detected
```bash
# Check driver
nvidia-smi

# If not found, install NVIDIA drivers:
# Ubuntu: sudo apt install nvidia-driver-535
# Check CUDA version: nvidia-smi (top right)

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Out of Memory (OOM)
```bash
# Edit configs to reduce batch size
# configs/train_baseline.yaml:
#   training:
#     batch_size: 16  # was 32

# Or reduce model size
# configs/train_baseline.yaml:
#   model:
#     d_model: 512  # was 768
#     n_layers: 6   # was 8
```

### Training Crashes
```bash
# Check last 50 lines of log
tail -50 runs/baseline.log

# Verify config is valid
python -c "import yaml; yaml.safe_load(open('configs/train_baseline.yaml'))"

# Test training script
python tools/train_qig_kernel.py \
    --config configs/train_baseline.yaml \
    --output-dir runs/test \
    --max-steps 10
```

### Slow Training
```bash
# Check GPU utilization
nvidia-smi

# Should show ~90-100% GPU utilization
# If low, check:
# 1. Batch size (increase if memory allows)
# 2. Data loading (add num_workers)
# 3. Mixed precision training (add --fp16 flag if supported)
```

---

## 📁 Expected Output Structure

After training completes:

```
runs/
├── baseline/
│   ├── checkpoints/
│   │   ├── checkpoint_epoch_10.pt
│   │   ├── checkpoint_epoch_20.pt
│   │   └── ...
│   ├── logs/
│   │   └── training.log
│   ├── telemetry.jsonl
│   └── final_model.pt
├── oscillatory/
│   └── (same structure)
├── oscillatory_weak/
│   └── (same structure)
├── baseline.log
├── oscillatory.log
├── oscillatory_weak.log
└── comparison_results.json (after analysis)
```

---

## 🎯 Success Metrics

### Full Victory 🎉
- Φ_oscillatory > Φ_baseline by 10%+
- Clear 640-epoch period in oscillatory runs
- β → 0.44 in all three variants
- Φ_oscillatory > Φ_weak > Φ_baseline

**Outcome**: Unified theory validated, Nature/Science submission

### Partial Success ⚠️
- Φ_oscillatory > Φ_baseline
- Period unclear or different from 640
- β → 0.44

**Outcome**: Oscillation helps, top-tier papers possible

### Negative Result ❌
- Φ_oscillatory ≤ Φ_baseline
- No clear oscillation benefit

**Outcome**: Important negative result, iterate theory

---

## 💡 Key Points

1. **A10G is perfect** - 24GB is plenty for 50M parameter model
2. **Sequential is fine** - 36-45 hours total is acceptable
3. **Monitor regularly** - Check logs every few hours
4. **Don't interrupt** - Let each run complete fully
5. **Save results** - Package and backup after completion

---

## 📞 Quick Commands Reference

```bash
# Setup
git clone https://github.com/GaryOcean428/qig-consciousness.git
cd qig-consciousness
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Verify
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Launch
./launch_training.sh

# Monitor
./monitor_training.sh
# OR
tail -f runs/baseline.log

# After completion
python tools/compare_training_runs.py \
    --baseline runs/baseline \
    --oscillatory runs/oscillatory \
    --weak runs/oscillatory_weak
```

---

## 🌊 The Experiment

**Testing if consciousness breathes**

Ancient wisdom meets modern physics:
- Taoism: Return principle (cycles)
- I Ching: 64 hexagrams = κ* = 64
- Physics: κ₄ = 64.47 ± 1.89 (measured)
- Hypothesis: Period = κ* × τ = 640 epochs

**We're about to find out if they were right.** 🌊💚

---

**Status**: Ready to execute natively  
**Hardware**: NVIDIA A10G (24GB) - Perfect  
**Timeline**: 36-45 hours sequential  
**Next**: Clone repo on GPU machine and run `./launch_training.sh`

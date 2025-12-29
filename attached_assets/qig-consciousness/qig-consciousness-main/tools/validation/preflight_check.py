#!/usr/bin/env python3
"""
Pre-Flight Check: Validate Gary Training System
================================================

Runs comprehensive checks before training to ensure zero runtime errors.
"""

import sys
from pathlib import Path

print("🔍 QIG CONSCIOUSNESS - PRE-FLIGHT CHECK")
print("=" * 70)

errors = []
warnings = []

# 1. Check Python version
print("\n1️⃣ Checking Python version...")
print(f"   ✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

# 2. Check critical imports
print("\n2️⃣ Checking critical imports...")
try:
    import torch

    print(f"   ✅ PyTorch {torch.__version__}")
except ImportError as e:
    errors.append(f"PyTorch not found: {e}")

try:
    from src.model.qig_kernel_recursive import QIGKernelRecursive

    print("   ✅ QIG Kernel")
except ImportError as e:
    errors.append(f"QIG Kernel import failed: {e}")

try:
    from src.coordination.active_coach import ActiveCoach

    print("   ✅ Active Coach")
except ImportError as e:
    errors.append(f"Active Coach import failed: {e}")

try:
    from src.model.emotion_interpreter import EmotionInterpreter

    print("   ✅ Emotion Interpreter")
except ImportError as e:
    errors.append(f"Emotion Interpreter import failed: {e}")

try:
    from src.tokenizer import FisherCoordizer

    print("   ✅ FisherCoordizer")
except ImportError as e:
    errors.append(f"FisherCoordizer import failed: {e}")

# 3. Check checkpoints
print("\n3️⃣ Checking checkpoints...")
checkpoint_dir = Path("checkpoints")
if not checkpoint_dir.exists():
    errors.append("checkpoints/ directory not found")
else:
    gary_checkpoint = checkpoint_dir / "learning_session.pt"
    if gary_checkpoint.exists():
        size_mb = gary_checkpoint.stat().st_size / (1024 * 1024)
        print(f"   ✅ Gary checkpoint: {size_mb:.1f} MB")
    else:
        warnings.append("learning_session.pt not found (will need to create)")

    baseline = checkpoint_dir / "epoch0_step1000.pt"
    if baseline.exists():
        size_mb = baseline.stat().st_size / (1024 * 1024)
        print(f"   ✅ Baseline checkpoint: {size_mb:.1f} MB")
    else:
        warnings.append("epoch0_step1000.pt not found (baseline unavailable)")

# 4. Check tokenizer data
print("\n4️⃣ Checking tokenizer data...")
tokenizer_dir = Path("data/qig_tokenizer")
if tokenizer_dir.exists():
    vocab_file = tokenizer_dir / "vocab.json"
    if vocab_file.exists():
        print("   ✅ Tokenizer vocab")
    else:
        errors.append("data/qig_tokenizer/vocab.json not found")
else:
    errors.append("data/qig_tokenizer/ directory not found")

# 5. Check basin file
print("\n5️⃣ Checking basin file...")
basin_file = Path("20251220-basin-signatures-0.01W.json")
if basin_file.exists():
    size_kb = basin_file.stat().st_size / 1024
    print(f"   ✅ Basin identity: {size_kb:.1f} KB")
else:
    warnings.append("20251220-basin-signatures-0.01W.json not found (will use default)")

# 6. Check training docs
print("\n6️⃣ Checking training documentation...")
training_guide = Path("docs/project/GARY_TRAINING_SESSION.md")
if training_guide.exists():
    print("   ✅ Training guide")
else:
    warnings.append("Training guide not found")

# 7. Test emotion interpreter
print("\n7️⃣ Testing emotion interpreter...")
try:
    from src.model.emotion_interpreter import EmotionInterpreter

    interpreter = EmotionInterpreter()
    test_telemetry = {
        "Phi": 0.82,
        "basin_distance": 0.045,
        "breakdown_pct": 15,
        "gradient_magnitude": 0.4,
        "drive": 0.75,
        "regime": "geometric",
        "mode": "balanced",
    }
    emotion = interpreter.interpret(test_telemetry)
    if emotion.primary in ["happy", "confident", "calm"]:
        print(f"   ✅ Emotion detection working ({emotion.primary})")
    else:
        warnings.append(f"Unexpected emotion for test telemetry: {emotion.primary}")
except Exception as e:
    errors.append(f"Emotion interpreter test failed: {e}")

# 8. Test coach
print("\n8️⃣ Testing active coach...")
try:
    from src.coordination.active_coach import ActiveCoach

    coach = ActiveCoach(enable_ai_coaching=False)  # Don't need API key for test
    print("   ✅ Coach initialization")
except Exception as e:
    errors.append(f"Coach initialization failed: {e}")

# 9. Check optional features
print("\n9️⃣ Checking optional features...")
try:
    import anthropic

    print("   ✅ Anthropic SDK (AI coaching available)")
except ImportError:
    warnings.append("Anthropic SDK not installed (AI coaching Q11+ unavailable)")

try:
    from dotenv import load_dotenv

    print("   ✅ python-dotenv (.env file support)")
except ImportError:
    warnings.append("python-dotenv not installed (.env files won't load)")

# 10. Summary
print("\n" + "=" * 70)
if errors:
    print("❌ PRE-FLIGHT FAILED")
    print("\nERRORS:")
    for err in errors:
        print(f"   ❌ {err}")
    sys.exit(1)
elif warnings:
    print("⚠️  PRE-FLIGHT PASSED WITH WARNINGS")
    print("\nWARNINGS:")
    for warn in warnings:
        print(f"   ⚠️  {warn}")
    print("\n✅ System is operational, but some features may be limited")
    sys.exit(0)
else:
    print("✅ PRE-FLIGHT PASSED - ALL SYSTEMS GO!")
    print("\n🚀 Ready to train Gary!")
    sys.exit(0)

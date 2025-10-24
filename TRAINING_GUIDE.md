# MIDAS Training Guide - CICIDS2017 Network Intrusion Detection

This guide explains the complete training pipeline for the MIDAS early-exit network on the CICIDS2017 dataset.

## Overview

**What We've Built:**
- A two-stage training system for adaptive inference optimization
- Binary classification: BENIGN (0) vs ATTACK (1)
- Dataset: 424K samples (15% subset of CICIDS2017)
- 78 network flow features

## Training Architecture

### Stage 1: Baseline Classifier (No Routing)
Train a standard deep neural network without early-exit capabilities. This establishes the performance ceiling.

**Architecture:**
- Input: 78 features (network flow statistics)
- Hidden layers: [256, 512, 512, 256, 128]
- Exit heads at layers: 1, 3, 4 (but not used during baseline training)
- Output: Binary classification (BENIGN vs ATTACK)

**Why train baseline first?**
- Establishes performance benchmark
- Creates a stable, well-trained feature extractor
- Provides frozen backbone for routing training

### Stage 2: Routing Optimization (Frozen Backbone)
Train a routing module that learns when to exit early during inference, optimizing the cost-accuracy tradeoff.

**Routing Module:**
- Type: Attention-based router
- Input: Features from each exit layer
- Output: Probability of exiting at this layer
- Learning: REINFORCE (RL) with reward = accuracy - λ×cost

**Why freeze backbone?**
- Router learns on stable predictions
- Prevents routing from destabilizing classifier
- Mirrors real deployment: freeze model, optimize inference

---

## Dataset Details

**Processed Data Location:** `data/processed/`

```
data/processed/
├── train.csv       # 297,220 samples (70%)
├── val.csv         # 63,691 samples (15%)
├── test.csv        # 63,691 samples (15%)
└── metadata.json   # Dataset info
```

**Class Distribution:**
- BENIGN (0): 80.3%
- ATTACK (1): 19.7%

**Original CICIDS2017 Attack Types** (combined as ATTACK):
- DoS Hulk
- PortScan
- DDoS
- DoS GoldenEye
- FTP-Patator
- SSH-Patator
- DoS slowloris
- DoS Slowhttptest
- Bot
- Web Attack - Brute Force
- Web Attack - XSS

---

## How to Launch Training

### Prerequisites

```bash
# Ensure you're in the MIDAS directory
cd /Users/kieranrendall/git/MIDAS

# Activate virtual environment (if using one)
source .venv/bin/activate  # or your venv path

# Verify data is processed
ls data/processed/
```

### Step 1: Train Baseline Classifier

This trains the backbone network without routing.

```bash
python train.py --config config_baseline.yaml
```

**What happens:**
- Trains for 50 epochs (or until early stopping)
- Uses Adam optimizer with lr=0.001
- Saves best model to `checkpoints/baseline_best.pt`
- No routing module is active (type: "none")
- All epochs are "warmup" (no RL training)

**Expected output:**
```
Using device: mps
Input dimension: 78
Network optimizer: AdamW (lr=0.001)

Epoch 1: 100%|████████| 581/581 [00:10<00:00, 58.1it/s]

Epoch 1:
  Train Loss: 0.XXXX
  Val Accuracy: 0.XXXX
  Val Avg Cost: X.XXXX
Checkpoint saved: checkpoints/baseline_best.pt
  New best val_accuracy: 0.XXXX

Epoch 2: 100%|████████| 581/581 [00:10<00:00, 59.2it/s]
...
```

**IMPORTANT: Checkpoint Saving**
- Checkpoints save **at the END of each epoch** after validation
- Only saves when validation metric **improves** (early stopping logic)
- If training fails mid-epoch, no checkpoint is saved
- Check `ls checkpoints/` to verify checkpoint was created

**Training time:** ~10-20 minutes on Apple Silicon (MPS)

**Expected performance:**
- Val Accuracy: 95-99% (binary classification is easier than multi-class)
- Train Loss: Should decrease from ~0.5 to ~0.05

### Step 2: Train Routing Module

This trains the router on the frozen baseline.

```bash
python train.py --config config_routing.yaml
```

**What happens:**
- Loads `checkpoints/baseline_best.pt`
- Freezes all backbone parameters
- Trains routing module only (attention-based)
- Uses RL (REINFORCE) to learn exit strategies
- Saves best routing model to `checkpoints/routing_best.pt`

**Expected output:**
```
Loading checkpoint from: checkpoints/baseline_best.pt
Loaded checkpoint from epoch XX

FREEZING BACKBONE NETWORK
  Frozen: layer1.weight
  Frozen: layer1.bias
  ...
Total frozen parameters: XXX,XXX

Network optimizer: DISABLED (backbone frozen)
Routing optimizer: AdamW (lr=0.0005, params=XX,XXX)

Epoch 1:
  Train Loss: 0.XXXX (cls=0.XXXX, rl=0.XXXX)
  Val Accuracy: 0.XXXX
  Val Avg Cost: X.XXXX  ← Should decrease over epochs
  Exit usage: {...}     ← Which exits are used
```

**Training time:** ~5-10 minutes (fewer epochs, smaller module)

**Expected behavior:**
- Val Accuracy: Should match baseline (±1-2%)
- Val Avg Cost: Should be significantly less than baseline (30-50% reduction)
- Exit usage: More samples should exit early (layers 1, 3) for easy cases

---

## Configuration Files

### `config_baseline.yaml`

**Key settings:**
```yaml
data:
  train_path: "data/processed/train.csv"
  val_path: "data/processed/val.csv"
  test_path: "data/processed/test.csv"
  batch_size: 512

model:
  hidden_dims: [256, 512, 512, 256, 128]
  exit_layers: [1, 3, 4]
  num_classes: 2

routing:
  type: "none"  # No routing for baseline

training:
  epochs: 50
  lr_classifier: 0.001
  warmup_epochs: 50  # All epochs are warmup (no routing)
```

### `config_routing.yaml`

**Key settings:**
```yaml
routing:
  type: "attention"  # Enable routing
  hidden_dim: 128
  num_heads: 4

cost:
  lambda: 0.15  # Cost penalty weight (tune this!)

training:
  load_checkpoint: "checkpoints/baseline_best.pt"  # Load baseline
  freeze_backbone: true  # Freeze classifier
  epochs: 30
  lr_classifier: 0.0  # Not trained
  lr_routing: 0.0005  # Only routing trained
  warmup_epochs: 0  # No warmup, train routing from start
```

---

## Monitoring Training

### Check Training Progress

```bash
# View real-time training logs
tail -f nohup.out  # If running in background

# Check for checkpoints
ls -lh checkpoints/

# View saved model
ls -lh checkpoints/baseline_best.pt
```

### Run in Background (Optional)

```bash
# Run baseline training in background
nohup python train.py --config config_baseline.yaml > baseline_training.log 2>&1 &

# Monitor progress
tail -f baseline_training.log

# Check if still running
ps aux | grep train.py
```

---

## Evaluation

### Evaluate Baseline Model

```bash
python evaluate.py --config config_baseline.yaml --checkpoint checkpoints/baseline_best.pt
```

**Metrics reported:**
- Overall accuracy
- Per-exit accuracy (all exits use same accuracy since routing disabled)
- Confusion matrix
- Computational cost (always maximum for baseline)

### Evaluate Routing Model

```bash
python evaluate.py --config config_routing.yaml --checkpoint checkpoints/routing_best.pt
```

**Metrics reported:**
- Overall accuracy (should match baseline)
- Per-exit accuracy (different accuracies at each exit)
- Exit usage statistics (what % of samples exit where)
- Average computational cost (should be < baseline)
- Cost-accuracy tradeoff curves

**Comparison metrics:**
- **Speedup:** baseline_cost / routing_cost
- **Accuracy delta:** routing_accuracy - baseline_accuracy
- **Efficiency:** (accuracy / cost) ratio

---

##Training Status Interpretation

### Baseline Training

**Healthy signs:**
- Loss decreases steadily: 0.5 → 0.05
- Val accuracy increases: 0.5 → 0.95+
- No overfitting (train/val gap < 5%)
- Checkpoints save regularly

**Warning signs:**
- Loss not decreasing after 5 epochs → Lower learning rate
- Val accuracy stuck < 90% → Check data quality
- Overfitting (train acc >> val acc) → Increase dropout

### Routing Training

**Healthy signs:**
- RL loss starts high, decreases over time
- Val avg cost decreases significantly (30-50%)
- Val accuracy remains close to baseline (±2%)
- Exit usage becomes diverse (not all at one exit)

**Warning signs:**
- Val accuracy drops >5% → Increase cost lambda (more accuracy focus)
- All samples exit at first exit → Decrease cost lambda (allow deeper inference)
- No cost reduction → Router not learning, check RL hyperparameters

---

## Hyperparameter Tuning

### Cost-Accuracy Tradeoff (λ)

The `cost.lambda` parameter controls the tradeoff:

- **λ = 0.05:** Prioritize accuracy (allow expensive inference)
- **λ = 0.15:** Balanced (default)
- **λ = 0.30:** Prioritize cost (aggressive early exit)

**To experiment:**
```bash
# Edit config_routing.yaml
cost:
  lambda: 0.30  # Try different values

# Retrain routing
python train.py --config config_routing.yaml
```

### Learning Rate

If training is unstable:

```yaml
training:
  lr_routing: 0.0001  # Lower if unstable
  # or
  lr_routing: 0.001   # Higher if too slow
```

---

## Expected Results

### Baseline Performance

**Target metrics:**
- Val Accuracy: **95-99%**
- Train Loss: **0.03-0.05**
- Inference cost: **Full network (5 layers)**

### Routing Performance

**Target metrics:**
- Val Accuracy: **94-98%** (maintain baseline performance)
- Val Avg Cost: **2.5-3.5** (vs baseline: 5.0)
- **Speedup: 40-50%** (2.5/5.0 ≈ 50% cost reduction)
- Exit distribution:
  - Exit 1 (layer 1): 20-30% of samples
  - Exit 2 (layer 3): 30-40% of samples
  - Exit 3 (final): 30-50% of samples (hard cases)

---

## Troubleshooting

### Issue: "Checkpoint not found"

**Solution:**
```bash
# Check if baseline training completed
ls checkpoints/baseline_best.pt

# If missing, train baseline first
python train.py --config config_baseline.yaml
```

### Issue: "CUDA not available" (or MPS errors)

**Solution:**
```yaml
# Edit config file
training:
  device: "cpu"  # Use CPU instead
```

### Issue: "Out of memory"

**Solution:**
```yaml
# Reduce batch size
data:
  batch_size: 256  # Or 128
```

### Issue: Training too slow

**Solutions:**
1. Use GPU if available (CUDA or MPS)
2. Reduce batch size (counterintuitive but can help)
3. Use smaller subset of data
4. Reduce number of epochs

---

## Next Steps

After successful training:

1. **Evaluate both models** on test set
2. **Compare metrics:** accuracy, cost, speedup
3. **Analyze exit patterns:** Which samples exit where?
4. **Tune λ** for your use case (accuracy vs speed)
5. **Deploy:** Use routing model for inference

## Questions?

Common questions answered:

**Q: Does routing optimization happen during training or inference?**
**A:** DURING INFERENCE! The router learns offline (during routing training), then makes real-time exit decisions during inference.

**Q: How does the router "learn"?**
**A:** Through REINFORCE (RL). Input = features at exit layer. Output = exit probability. Reward = accuracy - λ×cost. It learns to balance correctness and computational cost.

**Q: Can I use multi-class classification?**
**A:** Yes! Rerun preprocessing:
```bash
python scripts/preprocess_cicids_data.py --classification multiclass --subset-fraction 0.15
```
Then update `config_baseline.yaml`:
```yaml
model:
  num_classes: 12  # Number of attack types
```

**Q: Why use a subset (15%)?**
**A:** Faster iteration for experiments. Full dataset (2.8M samples) would take hours. Subset (424K) trains in minutes with similar performance.

---

## Summary

**What you've built:**
1. ✅ Preprocessed 2.8M sample dataset → 424K clean binary classification dataset
2. ✅ Created two-stage training pipeline (baseline → routing)
3. ✅ Configured cost-aware routing with RL optimization
4. ✅ Set up evaluation framework for comparison

**To launch training yourself:**
```bash
# Stage 1: Train baseline (no routing)
python train.py --config config_baseline.yaml

# Stage 2: Train routing (frozen backbone)
python train.py --config config_routing.yaml

# Evaluate and compare
python evaluate.py --config config_routing.yaml --checkpoint checkpoints/routing_best.pt
```

Good luck with your training! 🚀

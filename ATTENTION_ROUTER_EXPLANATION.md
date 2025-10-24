# Attention Router Architecture

## Overview

The `attention_router.py` file implements a lightweight transformer-based neural network for making escalation decisions. The architecture uses self-attention mechanisms to analyze hand-crafted gating features and produce a binary decision logit (whether to escalate or not).

## Key Components

### 1. TransformerBlock ([attention_router.py:6-26](attention_router.py#L6-L26))

A standard transformer encoder block implementing the classic "attention + feedforward" pattern with residual connections.

**Architecture:**
```
Input → Multi-Head Attention → Dropout → Residual Add → LayerNorm
      → Feedforward Network → Dropout → Residual Add → LayerNorm → Output
```

**Components:**
- **Multi-Head Attention** (line 9): Self-attention with multiple heads to capture different feature relationships
- **Feedforward Network** (lines 11-15): Two-layer MLP with GELU activation that expands to 2x hidden dimension
- **Layer Normalization** (lines 10, 16): Stabilizes training by normalizing across features
- **Residual Connections** (lines 21, 24): Allows gradient flow and preserves original information
- **Dropout** (line 17): Regularization to prevent overfitting

**Parameters:**
- `hidden_dim`: Dimension of the hidden representations
- `num_heads`: Number of attention heads (must divide hidden_dim evenly)
- `dropout`: Dropout probability for regularization

### 2. AttentionRouter ([attention_router.py:29-74](attention_router.py#L29-L74))

The main module that processes gating features through a multi-view transformer architecture.

#### Architecture Design Philosophy

The router uses a clever **dual-token representation** strategy:

1. **Primary Token** (line 66): Direct linear embedding of input features
2. **Interaction Token** (line 67): Embedding of squared features (x²)

This dual representation allows the self-attention mechanism to reason about:
- **Magnitude information**: Raw feature values (primary token)
- **Interaction terms**: Quadratic relationships between features (interaction token)

#### Forward Pass Flow

```
Input Features (batch_size, input_dim)
    ↓
    ├─→ Primary Embedding → Token 1
    └─→ Squared Input → Interaction Embedding → Token 2
    ↓
Stack into sequence [Token1, Token2] + Positional Encoding
    ↓
Transformer Blocks (self-attention between the two tokens)
    ↓
Mean Pooling (average both tokens)
    ↓
Readout MLP
    ↓
Escalation Logit (batch_size,)
```

#### Key Components:

**Embeddings** (lines 46-47):
- Two separate linear projections transform input features to hidden dimension
- `embed_primary`: Processes raw features
- `embed_interact`: Processes squared features to capture non-linear interactions

**Positional Encoding** (line 48):
- Learnable 2D positional embeddings added to the token sequence
- Helps the model distinguish between primary and interaction tokens
- Initialized with small random values (std ≈ 0.02)

**Transformer Stack** (lines 49-51):
- Multiple transformer blocks stacked sequentially
- Each block allows tokens to exchange information via self-attention
- Default: 2 layers with 4 attention heads

**Readout Head** (lines 52-57):
- Converts pooled token representation to final logit
- Architecture: LayerNorm → Linear (hidden → hidden/2) → GELU → Linear (hidden/2 → 1)
- Output is a single scalar per batch item

#### Parameters:

- `input_dim`: Dimension of input gating features
- `hidden_dim`: Internal representation size (default: 64)
- `num_heads`: Number of attention heads (default: 4)
- `num_layers`: Number of transformer blocks (default: 2)
- `dropout`: Dropout rate for regularization (default: 0.1)

## Why This Architecture?

### Multi-View Reasoning
By creating two tokens from the same input (raw and squared), the model can:
- Compare magnitude patterns with interaction patterns
- Use attention to weight which view is more important for the decision
- Capture both linear and quadratic relationships in the features

### Self-Attention Benefits
The attention mechanism between the two tokens allows the model to:
- Dynamically weight the importance of magnitude vs. interaction information
- Learn complex decision boundaries that simple MLPs might miss
- Share information bidirectionally between views

### Lightweight Design
Despite using transformers, the model remains efficient:
- Only 2 tokens in the sequence (minimal attention overhead)
- Relatively small hidden dimensions (default 64)
- Few layers (default 2)
- Total parameters: ~10K-50K depending on configuration

## Usage Example

```python
# Initialize router
router = AttentionRouter(
    input_dim=20,      # 20 gating features
    hidden_dim=64,     # Hidden dimension
    num_heads=4,       # 4 attention heads
    num_layers=2,      # 2 transformer blocks
    dropout=0.1        # 10% dropout
)

# Forward pass
features = torch.randn(32, 20)  # Batch of 32 samples, 20 features each
logits = router(features)        # Shape: (32,)

# Convert to probabilities
probabilities = torch.sigmoid(logits)

# Make binary decisions
decisions = (probabilities > 0.5).long()
```

## Training Considerations

1. **Loss Function**: Typically Binary Cross-Entropy (BCE) with logits
2. **Batch Size**: Standard mini-batch training (16-64 samples)
3. **Regularization**: Dropout is applied during training
4. **Optimization**: Adam or AdamW optimizers work well
5. **Learning Rate**: Start with 1e-3 to 1e-4

## Comparison to Standard MLP

A simple MLP would process features directly through linear layers. The AttentionRouter adds:
- Explicit modeling of feature interactions (x²)
- Multi-view representation learning
- Attention-based feature fusion
- More expressive decision boundaries

This makes it more powerful for complex gating decisions where feature interactions matter.

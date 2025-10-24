Model
=====

Backbone (`models/netflow_network.py`)
- MLP with configurable hidden dims: `model.hidden_dims`.
- Exit heads at `model.exit_layers` (0-indexed positions in hidden dims).
- Each exit head is a small classifier (`models/exit_head.py`).

Data Flow
- Forward pass returns final logits and a dictionary of per-exit features/logits when `return_all_exits=True`.
- Exit layers are processed in order; features at those layers are captured into the `exit_data` dict.

Shapes
- Input: `(batch_size, input_dim)`.
- Hidden layer i: `(batch_size, hidden_dims[i])`.
- Exit i logits: `(batch_size, num_classes)`.


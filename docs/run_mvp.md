# run_mvp.py — Documentation

This is the central script of the FusionNet project. It loads the client
partitions created by `prepare_data.py`, simulates one full federated
learning cycle (local training → importance-based freezing → bandwidth-adaptive
quantization → FedAvg aggregation) for every round, and prints a complete
per-client, per-round trace to the terminal.


---

## Quick Start

```bash
# Step 1 — prepare data (only needed once)
python datasets/prepare_data.py

# Step 2 — run the MVP simulation
python run_mvp.py
```

Expected terminal output:

```
=== FusionNet MVP Sentiment Simulation ===
Loaded 10 clients from partition data.

Starting 5 rounds of simulated federated learning...

--- Round 1 ---
Client client_0 (LOW tier):
  └ Data samples: 484
  └ LoRA Rank: 2 | Freeze Ratio: 0.5
  └ Max Bandwidth Limit: 10.0 MB
  └ Update communication quantized to 32-bit precision
Client client_1 (LOW tier):
  ...
Coordinator aggregating client updates using FedAvg...
Client weights based on data size: [484, 434, 430, ...]
Global model updated. Aggregated weight delta norm: 0.0412

--- Round 2 ---
...

=== MVP Simulation Complete! ===
All client parameters were successfully frozen, quantized to fit bandwidth
limits, aligned, and aggregated using size-weighted FedAvg.
```

---

## Where This Fits in the Pipeline

```
prepare_data.py
    creates → data/clients/noniid/client_0.json ... client_9.json
                    │
                    ▼
             run_mvp.py   ←── YOU ARE HERE
                    │
    reads  → client JSON files
    runs   → 5 federated rounds
    prints → per-client per-round trace
    creates -> experiments/mvp_sentiment/results/accuracy_log.json
                    │
                    ▼
         (results feed into plot_convergence.py)
```

---

## Dependencies

| Package | Why needed | Install |
|---|---|---|
| `torch` | Tensor operations, quantization simulation | `pip install torch` |
| `numpy` | Importance score generation, argsort | `pip install numpy` |
| `json` | Loading client JSON files | stdlib |
| `fusionnet.core.aggregator` | `fed_avg()` function from your team's aggregator module | Part of this repo — see `fusionnet/core/aggregator.py` |

The script also uses `sys.path.append` to make the project root importable:

```python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from fusionnet.core.aggregator import fed_avg
```

This means `run_mvp.py` must be run from its own directory, OR you set
`PYTHONPATH` to the project root. If you see `ModuleNotFoundError: fusionnet`,
run from the project root:

```bash
cd /path/to/fusionnet_project
python run_mvp.py
```

---

## Configuration

There is currently one top-level constant and two values hardcoded inside
`main()`:

| Name | Location | Value | Meaning |
|---|---|---|---|
| `CLIENT_DATA_DIR` | top-level constant | `"data/clients/noniid"` | Where to load client JSON files from |
| `hidden_dim` | inside `main()` | `4096` | Llama 3's hidden dimension — determines LoRA adapter tensor shape |
| `num_rounds` | inside `main()` | `5` | How many federated rounds to simulate |

### Known limitation — hardcoded values

`hidden_dim` and `num_rounds` are currently hardcoded inside `main()` rather
than being top-level constants or CLI arguments. This means changing them
requires editing the source file. A future improvement would be to move them
to the top-level config section or expose them via `argparse` (see
[Future Improvements](#future-improvements)).

---

## What "Simulation" Means Here

This script does **not** load a real LLM or compute real gradients. The
"local updates" are random tensors:

```python
local_update = torch.randn(lora_rank, hidden_dim) * 0.1
```

This is intentional for the MVP. The purpose of `run_mvp.py` is to prove
the **system architecture** works — that freezing, quantization, zero-padding
alignment, and weighted FedAvg all operate correctly together with real
resource profiles from real client configs. The math of each component is
real even though the update values are synthetic.

When real model training is added later, only this one line changes. Everything
else — loading configs, freezing logic, quantization, aggregation — stays
exactly the same.

---

## Function Reference

### `load_clients()`

Loads all client JSON files from `CLIENT_DATA_DIR` in alphabetical order
(so `client_0` is always processed before `client_9`).

**Returns:** `list[dict]` — each dict is one fully loaded client JSON.

**Fields it reads from each file:**

```
client["client_id"]                          → "client_0"
client["resources"]["lora_rank"]             → 2, 4, or 8
client["resources"]["freeze_ratio"]          → 0.5 or 0.0
client["resources"]["max_bits_mb"]           → float (bandwidth ceiling)
client["resources"]["num_frozen_components"] → int
client["resources"]["num_trainable_components"] → int
client["data"]["train_size"]                 → int (number of training samples)
```

**Error handling:** If `CLIENT_DATA_DIR` doesn't exist, prints a clear
error message and calls `sys.exit(1)` immediately rather than crashing
with a confusing Python traceback. The error message tells you exactly
what to run to fix it.


---

### `simulate_importance_based_freezing(client_id, update, resources)`

Implements the importance-based parameter freezing scheme from
**HAFLQ §IV-C**.

The core idea: a resource-constrained client cannot afford to update all
`lora_rank` rank-1 matrices every round. Instead of truncating the model
(which loses information permanently), the client keeps all rank-1 matrices
but **freezes** (zeros out) the least important ones. Only the most important
rank-1 matrices receive gradient updates.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `client_id` | `str` | e.g. `"client_0"` — used to derive a deterministic per-client random seed |
| `update` | `torch.Tensor` shape `[lora_rank, hidden_dim]` | The simulated local weight update before freezing |
| `resources` | `dict` | Client resource profile from the JSON file |

**Algorithm step by step:**

```
1. Read num_frozen from resources["num_frozen_components"]

2. If num_frozen == 0 (high-tier client):
       return update unchanged — nothing to freeze

3. Seed numpy with (42 + client_index)
       e.g. client_0 → seed 42, client_1 → seed 43
       This makes importance scores deterministic per client but different
       across clients — simulating each client having its own data-derived
       importance ranking

4. Generate importance_scores = np.random.rand(lora_rank)
       e.g. for rank 4: [0.82, 0.11, 0.64, 0.23]

5. frozen_indices = np.argsort(importance_scores)[:num_frozen]
       argsort ascending → smallest first → these are least important
       e.g. argsort([0.82, 0.11, 0.64, 0.23]) = [1, 3, 2, 0]
       num_frozen=2 → frozen_indices = [1, 3]

6. masked_update = update.clone()
       for each frozen index: masked_update[idx, :] = 0.0
       Zero out entire rows → those rank-1 matrices contribute nothing
       to the aggregated global update

7. Return masked_update
```

**Worked example:**

```
Client: client_2 (low tier, lora_rank=2, num_frozen=1)

importance_scores = [0.37, 0.91]  (seed = 42 + 2 = 44)
argsort ascending = [0, 1]
frozen_indices    = [0]            (only the first index — lowest score)

update (before):
  row 0: [0.08, -0.12, 0.03, ...]
  row 1: [0.11,  0.07, -0.09, ...]

update (after):
  row 0: [0.00,  0.00, 0.00, ...]  ← zeroed out (frozen)
  row 1: [0.11,  0.07, -0.09, ...]  ← unchanged (trainable)
```

**Why random scores instead of real gradient-based scores:**

Real importance scores require actual gradients from real training. Since
this MVP uses synthetic updates, we use seeded random scores as a structural
placeholder. The freezing **mechanism** (zeroing rows, respecting
`num_frozen_components`, high-tier bypass) is fully correct — only the
score source is synthetic. When real training is added, this function's
score generation is the one part that gets replaced with gradient-magnitude
computation.

---

### `simulate_bandwidth_adaptive_quantization(update, max_bits_mb)`

Implements the bandwidth-adaptive communication quantization scheme from
**HAFLQ §V**.

Each client has a bandwidth ceiling (`max_bits_mb`) derived from its wireless
channel quality — clients further from the base station get less bandwidth.
This function checks whether the update fits within that budget, and if not,
progressively quantizes to lower precision until it fits.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `update` | `torch.Tensor` shape `[8, hidden_dim]` | Update tensor, always padded/aligned to rank 8 before this call |
| `max_bits_mb` | `float` | This client's bandwidth ceiling in megabytes |

**Returns:** `(quantized_update, precision_bits)` — the (possibly quantized)
tensor and the bit-width actually used.

**Size calculation:**

```python
num_elements = update.numel()           # 8 × 4096 = 32,768
size_mb = (num_elements * 4) / (1024 * 1024)   # FP32: 0.125 MB
```

**Decision tree:**

```
                    ┌─ size(FP32)  ≤ max_bits_mb? ──→ return (update, 32)
                    │
                    ├─ size(FP16)  ≤ max_bits_mb? ──→ cast to float16,
                    │                                   cast back to float32,
                    │                                   return (result, 16)
                    │
                    ├─ size(INT8)  ≤ max_bits_mb? ──→ scale/clamp/round to [-128,127],
                    │                                   multiply by scale to dequantize,
                    │                                   return (result, 8)
                    │
                    └─ else (INT4)                ──→ scale/clamp/round to [-8,7],
                                                       multiply by scale to dequantize,
                                                       return (result, 4)
```

**Concrete size values for our setup (rank 8, hidden_dim 4096):**

| Precision | Bytes/element | Total size | Fits if max_bits_mb ≥ |
|---|---|---|---|
| FP32 | 4 | 0.1250 MB | 0.125 |
| FP16 | 2 | 0.0625 MB | 0.063 |
| INT8 | 1 | 0.0313 MB | 0.031 |
| INT4 | 0.5 | 0.0156 MB | 0.016 |

**Practical behavior with current client bandwidths:**

Our clients range from 0.5 MB to 10.0 MB bandwidth (after the `MIN_BANDWIDTH_MB`
floor from `prepare_data.py`). The FP32 update is only 0.125 MB. Since
0.5 MB ≥ 0.125 MB, even the most bandwidth-constrained client passes the
FP32 check — meaning in practice **no quantization triggers** for our
current tensor sizes.

This is a known limitation of the MVP. The quantization logic is structurally
correct and would trigger on larger models (e.g., a real Llama 3-8B layer
update would be gigabytes). The infrastructure is in place; it just needs
larger tensors or smaller bandwidth values to activate in this simulation.

**INT8 quantization — how it works:**

```python
scale = update.abs().max() / 127.0
quantized  = torch.clamp(torch.round(update / scale), -128, 127)
dequantized = quantized * scale
```

- Divide every element by `scale` → maps the range to roughly [-127, 127]
- `torch.round()` → snap to integers
- `torch.clamp(..., -128, 127)` → enforce the int8 range strictly
- Multiply by `scale` → dequantize back to float (now slightly lossy)

INT4 uses the same pattern but maps to [-8, 7] (4-bit signed integer range).

---

### `main()`

Orchestrates the complete simulation. Called when the script is run directly.

**Step-by-step execution:**

```
1. load_clients()
       → reads all 10 client JSON files
       → prints "Loaded N clients from partition data"

2. Initialize global_model
       → {"lora_A": torch.randn(8, hidden_dim)}
       → Random initialization — represents the server's LoRA A matrix
       → Shape [8, hidden_dim] = [max_rank, 4096]

3. for round_num in 1..5:

   3a. for each client:
         - Read lora_rank, freeze_ratio, max_bits_mb, train_size from JSON
         - Print client header (tier, data size, bandwidth)
         - Simulate local update: torch.randn(lora_rank, hidden_dim) * 0.1
         - Apply simulate_importance_based_freezing()
         - Zero-pad update from [lora_rank, hidden_dim] to [8, hidden_dim]
         - Apply simulate_bandwidth_adaptive_quantization()
         - Print quantization precision used
         - Append (aligned_update, train_size) to collection lists

   3b. fed_avg(client_updates, client_sizes)
         - Weighted average: weight_k = size_k / Σ sizes
         - Returns one aggregated {"lora_A": tensor}

   3c. global_model["lora_A"] += aggregated_update["lora_A"]
         - In-place update of global LoRA A matrix

   3d. Print aggregated weight delta norm

4. Print "MVP Simulation Complete" summary
```

---

## The Zero-Padding Alignment Step

Between the freezing step and the quantization step, there is a critical
alignment operation:

```python
# Client has lora_rank = 2, 4, or 8
local_update = torch.randn(lora_rank, hidden_dim) * 0.1        # shape: [lora_rank, hidden_dim]

# After freezing, still shape [lora_rank, hidden_dim]
frozen_update = simulate_importance_based_freezing(...)

# Pad to global rank 8 so all client updates have the SAME shape for aggregation
aligned_update = torch.zeros(8, hidden_dim)
aligned_update[:lora_rank, :] = frozen_update                  # shape: [8, hidden_dim]
```

This is the zero-padding strategy described in both the HAFLQ (§IV) and
AFLoRA papers for handling heterogeneous LoRA ranks.

**Why this is necessary:**

`fed_avg()` needs to average tensors element-wise. If client_0 sends shape
`[2, 4096]` and client_9 sends shape `[8, 4096]`, you cannot average them
directly. Zero-padding expands every client's update to `[8, 4096]` — the
low-rank clients just have zeros in the rows they didn't train.

**Consequence:** The zero-padded rows don't contribute to the global update
for those rank positions (0 + anything = anything unchanged). This is correct
behaviour — a rank-2 client should not interfere with the rank-5 through
rank-8 components that only rank-8 clients trained.

---

## The FedAvg Aggregation

```python
aggregated_update = fed_avg(client_updates, client_sizes)
global_model["lora_A"] += aggregated_update["lora_A"]
```

`fed_avg` is imported from `fusionnet.core.aggregator`. It computes a
weighted average of all client updates where each client's weight is
proportional to its local dataset size:

```
weight_k = size_k / Σ(size_0 ... size_9)
aggregated = Σ weight_k × update_k
```

**Why size-weighted and not equal-weighted:**

Non-IID clients have different dataset sizes (484 to ~912 samples depending
on their label assignment). Equal weighting would give a 484-sample client
the same influence as a 912-sample client, which is incorrect — the larger
dataset represents more real-world signal. Size-weighted FedAvg is the
standard approach in the FL literature (McMahan et al., 2017).

The client sizes come from `client["data"]["train_size"]` — the actual
training set size per client, not an assumed equal split.

---

## Integration Notes with `prepare_data.py`

`run_mvp.py` reads specific field names from the client JSON files. These
must match exactly what `prepare_data.py` writes. Current mapping:

| Field `run_mvp.py` reads | Field `prepare_data.py` writes | Status |
|---|---|---|
| `client["data"]["train_size"]` | `client["data"]["train"]["size"]` | ⚠️ Mismatch — nested vs flat |
| `resources["num_frozen_components"]` | `resources["trainable_components"]` (inverse) | ⚠️ Different name + semantics |
| `resources["num_trainable_components"]` | `resources["trainable_components"]` | ⚠️ Different name |
| `resources["lora_rank"]` | `resources["lora_rank"]` | ✅ Match |
| `resources["freeze_ratio"]` | `resources["freeze_ratio"]` | ✅ Match |
| `resources["max_bits_mb"]` | `resources["max_bits_mb"]` | ✅ Match |
| `client["client_id"]` | `client["client_id"]` | ✅ Match |
| `resources["tier"]` | `resources["tier"]` | ✅ Match |

**Action required:** Before running `run_mvp.py`, confirm that
`prepare_data.py` writes `train_size` and `num_frozen_components` /
`num_trainable_components` exactly as this script expects — OR update one
script to match the other. The mismatched fields will cause a `KeyError` at
runtime.

---

## Paper Connections

Every major function in this script corresponds to a specific section of the
HAFLQ paper (arXiv:2411.06581):

| Function | Paper section | What it implements |
|---|---|---|
| `simulate_importance_based_freezing()` | §IV-C | Importance-based parameter freezing scheme |
| `simulate_bandwidth_adaptive_quantization()` | §V | Bandwidth-adaptive communication quantization |
| Zero-padding alignment in `main()` | §IV-B | Importance-based truncation / alignment to global rank |
| `fed_avg()` aggregation | §VI | Adaptive rank-1 matrix-level aggregation (simplified to standard FedAvg here) |
| Per-client `lora_rank` / `freeze_ratio` | §VII Table I | Exact heterogeneous client setup from experiments |

The script also reflects **AFLoRA** (arXiv:2505.24773) in that the A matrix
is treated as the globally shared component (`global_model["lora_A"]`),
matching AFLoRA's decoupled design where A is maintained server-side.

---

## Known Limitations

These are documented transparently — not bugs, but boundaries of the MVP.

| Limitation | Impact | Future fix |
|---|---|---|
| Local updates are `torch.randn() * 0.1` (random) | Accuracy numbers aren't real | Replace with actual model forward/backward pass |
| Importance scores are seeded random, not gradient-based | Freezing decisions don't reflect actual parameter importance | Compute `|w × Δw / η|` after real training steps |
| Quantization rarely triggers (tensors are small, bandwidth is large) | Quantization logic exists but is untested in practice with current sizes | Use larger `hidden_dim` values or smaller `max_bits_mb` to stress-test |
| `hidden_dim = 4096` hardcoded | Changing the model requires editing source | Move to top-level constant or config file |
| `num_rounds = 5` hardcoded | Not configurable from CLI | Move to top-level constant or expose via `argparse` |
| No output saved to file | Results can't be fed to `plot_convergence.py` automatically | Add `json.dump` of per-round results to `results/accuracy_log.json` |
| `global_model` is random, not a real checkpoint | Model doesn't improve meaningfully across rounds | Load from a real pretrained checkpoint |

---

## Future Improvements

**Short term (before demo day):**
- Move `hidden_dim` and `num_rounds` to top-level constants so they're
  visible and easy to change
- Resolve the field name mismatch between this script and `prepare_data.py`

**Medium term:**
- Add `argparse` for `--rounds`, `--split` (iid vs noniid), `--hidden-dim`
- Replace synthetic updates with real LoRA fine-tuning steps using
  `peft` library on actual Banking77 text
- Replace random importance scores with real gradient-magnitude computation
  per HAFLQ §IV-A

**Long term:**
- Implement the full rank-1 matrix-level aggregation from HAFLQ §VI
  (currently using standard FedAvg via `fed_avg()`)
- Add differential privacy noise injection
- Extend to multiple LoRA layers instead of one simulated A matrix

---

## Common Errors

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: fusionnet` | Script run from wrong directory or PYTHONPATH not set | Run from project root: `python run_mvp.py` |
| `FileNotFoundError: data/clients/noniid` | `prepare_data.py` hasn't been run yet | `python datasets/prepare_data.py` |
| `KeyError: 'train_size'` | Field name mismatch between scripts | Check `prepare_data.py` output structure; field may be nested as `data.train.size` |
| `KeyError: 'num_frozen_components'` | `prepare_data.py` uses different field name | Align field names between both scripts |
| `ModuleNotFoundError: torch` | PyTorch not installed | `pip install torch` |
| Script runs but quantization is always 32-bit | Tensors are smaller than min bandwidth floor | Expected with current sizes — see [Known Limitations](#known-limitations) |

---

*This file is part of the FusionNet AMD Hackathon project.*
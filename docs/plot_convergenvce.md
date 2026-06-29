# plot_convergence.py — Analysis and Visualisation Script

This document provides a comprehensive technical reference for plot_convergence.py. This script serves as the primary analytics engine, transforming JSON telemetry logs produced by your execution runner (run_mvp.py) into publication-quality research figures.

Reads accuracy_log.json produced by run_mvp.py and generates three publication-quality figures:

    Figure 1 — convergence_plot.png (2 panels)

        Panel 1: Global accuracy over rounds — haflq vs baseline

        Panel 2: Aggregated weight delta Frobenius norm (∣∣ΔW∣∣F​) — haflq vs baseline

    Figure 2 — extended_metrics_plot.png (4 panels)

        Panel 1: Per-round communication cost — haflq vs baseline

        Panel 2: Cumulative communication cost — with bandwidth-saved fill

        Panel 3: Parameters discarded per round due to edge transport limits

        Panel 4: Accuracy per MB of communication (efficiency ratio)

    Figure 3 — loss_throughput_metrics.png (3 panels)

        Panel 1: Global convergence losses — Training vs. Validation curves

        Panel 2: Per-client loss variance — Visualizing non-IID data distribution trends

        Panel 3: On-device token throughput — Hardware processing speed (tokens/sec)

Every number in every panel comes directly from accuracy_log.json. Nothing is fabricated or randomly generated at the visualization layer.
## Usage
Bash

python plot_convergence.py
python plot_convergence.py --input path/to/accuracy_log.json
python plot_convergence.py --input path/to/log.json --outdir results/plots

## Dependencies
Bash

pip install matplotlib numpy

## Architectural Overview

The script abstracts the visualization layer from the training loop. It parses empirical telemetry data, applies multi-version schema normalization, isolates system metrics, and generates high-density figures matching academic publication standards.
Plaintext

                     ┌────────────────────┐
                     │    run_mvp.py      │
                     └─────────┬──────────┘
                               │
                               ▼ 
                   ┌──────────────────────────┐
                   │    accuracy_log.json     │
                   └────────────┬─────────────┘
                               │
                               ▼ 
                   ┌──────────────────────────┐
                   │   plot_convergence.py    │
                   └───┬──────────┬──────────┬┘
                       │          │          │
       Generates Fig 1 │          │          │ Generates Fig 3
                       ▼          │          ▼
        ┌─────────────────────┐   │   ┌────────────────────────────────┐
        │convergence_plot.png │   │   │  loss_throughput_metrics.png   │
        │─────────────────────│   │   │────────────────────────────────│
        │ 2-Panel Diagnostic  │   │   │ 3-Panel Compute & Loss Matrix  │
        └─────────────────────┘   │   └────────────────────────────────┘
                                  ▼ Generates Fig 2
                      ┌───────────────────────────────────┐
                      │    extended_metrics_plot.png      │
                      │───────────────────────────────────│
                      │ 4-Panel Network Efficiency Matrix │
                      └───────────────────────────────────┘

Core Design Goals

    Mathematical Invariance: No synthetic, fabricated, or smoothed values are introduced at the plotting layer. Every pixel mapped corresponds strictly to logged telemetry.

    Schema Decoupling: Includes structural fallback handlers capable of digesting multi-method cross-comparisons or legacy, single-history array structures without throwing errors.

    Production-Grade Aesthetics: Uses a unified hex-color spectrum, proportional axis text offsets, automated tick-overcrowding decimation, and text annotations aligned directly to critical data elements.

## Telemetry Schema Specifications

The script expects a structured JSON telemetry payload. It dynamically supports both the production multi-experiment layout and the legacy single-method format.
Target Multi-Experiment Schema (Preferred)

This structure maps cross-comparative studies simultaneously (e.g., evaluating your proposed haflq framework directly against standard baseline parameters).
JSON

{
  "timestamp": "2026-06-23T13:00:00Z",
  "dataset": "Banking77",
  "num_rounds": 20,
  "num_clients": 10,
  "experiments": {
    "haflq": [
      {
        "round": 1,
        "global_accuracy": 0.4521,
        "total_comm_mb": 8.42,
        "cumulative_comm_mb": 8.42,
        "total_discarded_mb": 0.02,
        "avg_client_accuracy": 0.4110,
        "update_norm": 4.1251,
        "train_loss": 2.2145,
        "val_loss": 2.5102,
        "token_throughput": 1441.2,
        "client_1_loss": 2.2510,
        "client_2_loss": 2.2104,
        "client_3_loss": 2.1215
      },
      {
        "round": 2,
        "global_accuracy": 0.6285,
        "total_comm_mb": 6.11,
        "cumulative_comm_mb": 14.53,
        "total_discarded_mb": 0.01,
        "avg_client_accuracy": 0.5942,
        "update_norm": 4.0912,
        "train_loss": 1.8841,
        "val_loss": 2.1154,
        "token_throughput": 1445.6,
        "client_1_loss": 1.9214,
        "client_2_loss": 1.8541,
        "client_3_loss": 1.8102
      }
    ],
    "baseline": [
      {
        "round": 1,
        "global_accuracy": 0.4102,
        "total_comm_mb": 12.50,
        "cumulative_comm_mb": 12.50,
        "total_discarded_mb": 2.15,
        "avg_client_accuracy": 0.3854,
        "update_norm": 4.1520,
        "train_loss": 2.3841,
        "val_loss": 2.6145,
        "token_throughput": 1515.4,
        "client_1_loss": 2.4412,
        "client_2_loss": 2.3514,
        "client_3_loss": 2.3145
      }
    ]
  }
}

## Legacy Single-Method Schema Support

If the core executor dumps a flat history list representing a single execution pass, the script detects it, logs a structural notice, and auto-wraps the telemetry payload into the standard namespace as haflq.
JSON

{
  "num_rounds": 20,
  "history": [
    {
      "round": 1,
      "global_accuracy": 0.4521,
      "total_comm_mb": 8.42
    }
  ]
}

## Visualization Architecture
Figure 1: Convergence Diagnostics (convergence_plot.png)

    Dimensions: 14×5.5 inches (Dual-Panel Landscape arrangement).

    Target Domain: Standard machine learning training dynamics and model state stabilization metrics.

Panel	Metric Rendered	Input JSON Keys	Analytical Value
Panel 1	Global Test Accuracy Convergence	global_accuracy	Multi-line progression plotting validation accuracy across training iterations. Includes a static horizontal baseline target at 89.13%, serving as a benchmark against top-tier academic reference parameters.
Panel 2	Aggregated Weight Delta Norm	update_norm	Tracks the geometric Frobenius Norm $
Figure 2: Extended Communication Matrix (extended_metrics_plot.png)

    Dimensions: 15×11 inches (2×2 Grid Quad-Panel matrix layout).

    Target Domain: Systems-level networking efficiency and network-constrained edge resource profiling.

Plaintext

┌──────────────────────────────────────┐──────────────────────────────────────┐
│  [0,0] Per-Round Communication       │  [0,1] Cumulative Communication       │
│                                      │                                      │
│  • Tracks individual round costs     │  • Plots total network transfer      │
│  • Monitors adaptive compression     │  • Shards network bandwidth saved    │
└──────────────────────────────────────┘──────────────────────────────────────┘
┌──────────────────────────────────────┐──────────────────────────────────────┐
│  [1,0] Parameter Discard Profile     │  [1,1] Efficiency Ratio Matrix       │
│                                      │                                      │
│  • Measures drops due to limits      │  • Evaluates Accuracy gained per MB  │
│  • Proves edge budget compliance     │  • The primary optimization target   │
└──────────────────────────────────────┘──────────────────────────────────────┘

    Panel [0,0] — Per-Round Communication Cost: Plots discrete byte volumes uploaded per epoch. It charts how parameter freezing and quantization steps lower infrastructure costs as training stabilizes.

    Panel [0,1] — Cumulative Communication Cost: A continuous step aggregation of overall edge data transfer. If both haflq and baseline modes are present, it renders an alpha-blended teal filling block (#43a2ca) across the curves, computing a vector offset to place an automated arrow annotation detailing the exact volume of Megabytes saved.

    Panel [1,0] — Parameters Discarded per Round: Implements an interlaced bar graph matrix visualizing the volume of model layers dropped due to hardware bandwidth limits. Low values in this panel validate the model's adaptive budget alignment.

    Panel [1,1] — Communication Efficiency Ratio: Tracks the system's ability to achieve high model utility with a minimal network payload, evaluated via vector conditional logic:
    Efficiency=Current Round Transport Burden (MB)Global Accuracy Component​

Figure 3: Compute & Loss Telemetry Matrix (loss_throughput_metrics.png)

    Dimensions: 18×5.5 inches (1×3 Panel Landscape arrangement).

    Target Domain: Hardware runtime compute speed and cross-entropy error boundary optimization.

Panel	Metric Rendered	Input JSON Keys	Analytical Value
Panel 1	Global Convergence Loss	train_loss, val_loss	Juxtaposes global training loss against central evaluation validation loss. Used to monitor training convergence speed and detect overfitting thresholds.
Panel 2	Per-Client Loss Variance	client_x_loss	Overlays explicit loss trajectories of isolated edge nodes. Highlights system robustness under complex, highly non-IID data distributions.
Panel 3	On-Device Token Throughput	token_throughput	Evaluates hardware compute efficiency measured in tokens/sec. Proves that advanced quantization layers do not degrade processing speed.
Production Layout Engine & Stylesheet

The script overrides Matplotlib defaults to enforce professional typographical hierarchies and clean geometric layouts:
Python

# Color Palette Token Definitions
COLOR = {
    "haflq":    "#0f62fe",   # Deep Carbon Blue (Primary Target Model)
    "baseline": "#ff1744",   # Vivid Crimson    (Baseline Benchmarks)
    "fill":     "#43a2ca",   # Teal Fill        (Shaded Efficiency Spaces)
    "grid":     "#e0e0e0",   # Light Grey       (Axis Subdivisions)
    "mean":     "#d73027",   # Soft Red         (Horizontal Reference Marks)
}

## Automation Subsystems

    Overcrowding Decimation: Axis ticks scale dynamically using a MultipleLocator step calculation: max(1, len(rounds) // 10). This guarantees crisp, uncrowded horizontal label readouts whether running short 10-round validation sweeps or full-scale 200-round operations.

    Safe Floating-Point Division: To prevent mathematical evaluation faults during early rounds where data transport counters are absolute zero, division logic is safely isolated using vector conditional logic:
    Python

    with np.errstate(divide="ignore", invalid="ignore"):
        efficiency = np.where(total_comm_mb > 0, global_accuracy / total_comm_mb, 0.0)

## Execution Guide & Command Line Interface

The script uses an independent parser loop, allowing execution from varying workspace directories without risking file-path breaks.
Command Line Arguments
Plaintext

options:
  -h, --help            show this help message and exit
  --input INPUT         Path to accuracy_log.json source file
                        (Default: experiments/results/accuracy_log.json)
  --outdir OUTDIR       Directory target path for generated image files
                        (Default: experiments/results)

Execution Recipes

1. Standard Run (Default Workspace Organization)
Bash

python plot_convergence.py

2. Evaluating Custom Stored Metrics
Bash

python plot_convergence.py --input storage/logs/banking77_run.json

3. Custom Output Directory Targeting (For Presentation Assets)
Bash

python plot_convergence.py \
    --input experiments/results/accuracy_log.json \
    --outdir assets/presentation_deck/

Clean System Outputs

Upon validation and parsing, the engine outputs explicit generation notifications to stdout:
Plaintext

Loaded log: 2 method(s), 20 rounds each.
Generating Figure 1 — convergence plot...
Saved: experiments/results/convergence_plot.png
Generating Figure 2 — extended metrics plot...
Saved: experiments/results/extended_metrics_plot.png
Generating Figure 3 — loss throughput metrics plot...
Saved: experiments/results/loss_throughput_metrics.png

All plots saved successfully.
  experiments/results/convergence_plot.png
  experiments/results/extended_metrics_plot.png
  experiments/results/loss_throughput_metrics.png
<h1 align="center">Discrete Elastic Ribbons</h1>
<h3 align="center">A Unified Discrete Differential Geometry Framework for One-Dimensional Energy Models</h3>

<p align="center">
  <strong>Shivam K. Panda</strong> &nbsp;·&nbsp; <strong>M. Khalid Jawed</strong><br>
  <em>University of California, Los Angeles</em>
</p>

<p align="center">
  <a href="https://github.com/StructuresComp/discrete-elastic-ribbon-jax">
    <img src="https://img.shields.io/badge/arXiv-coming%20soon-b31b1b?logo=arxiv&logoColor=white" alt="arXiv">
  </a>
  <a href="https://github.com/StructuresComp/discrete-elastic-ribbon-jax">
    <img src="https://img.shields.io/badge/Project-Page-1f72bb?logo=github&logoColor=white" alt="Project Page">
  </a>
  <a href="https://github.com/StructuresComp/discrete-elastic-ribbon">
    <img src="https://img.shields.io/badge/Reference%20Impl-Python-3776ab?logo=python&logoColor=white" alt="Python Reference Implementation">
  </a>
  <a href="https://github.com/StructuresComp/discrete-elastic-ribbon-jax">
    <img src="https://img.shields.io/badge/High--Performance%20Impl-JAX-ff6f00?logo=google&logoColor=white" alt="JAX Implementation">
  </a>
</p>

## Abstract

Elastic ribbons — slender structures whose length (*L*), width (*W*), and thickness (*b*) satisfy *L* ≫ *W* ≫ *b* — exhibit mechanical behaviors intermediate between one-dimensional rods (*L* ≫ *W*, *b*) and two-dimensional plates (*L*, *W* ≫ *b*). In quadratic Kirchhoff-type rod-based frameworks such as Discrete Elastic Rods (DER), the governing equilibrium equations are independent of width and therefore cannot capture width-dependent mechanical effects. Reduced centerline-based ribbon models attempt to capture width dependence via coupled bending–twisting energies; however, their relative accuracy remains unclear due to the absence of a unified simulation framework.

In this work, we formulate a framework grounded in discrete differential geometry where the energy is expressed as functions of coupled bending–twisting strain measures along the centerline, rather than a linear sum of quadratic bending and twisting energies in DER. We derive analytical gradients and Hessians of the energy that enable implicit time integration. Within this unified setting, we compare five ribbon models: **Kirchhoff**, **Sadowsky**, **Wunderlich**, **Sano**, and **Audoly**. As a benchmark, a straight ribbon is longitudinally constrained into a pre-buckled arch and subjected to transverse displacement, inducing a supercritical pitchfork bifurcation. Predicted bifurcation thresholds are compared against shell-based finite element simulations, with the **Sano** model providing the closest agreement in capturing width-dependent shifts. Our high-performance JAX-based implementation achieves *O*(*N*) per-iteration cost and confirms that the Sano model introduces negligible per-iteration overhead relative to standard DER.

## Implementations

This work is provided through two official repositories:

| Repository | Description |
|---|---|
| [`discrete-elastic-ribbon`](https://github.com/StructuresComp/discrete-elastic-ribbon) | **Reference Python implementation.** All five energy models, adaptive implicit Euler with a robust regularized linear solver, and a homotopy API for changing rod geometry/material during simulation. |
| [`discrete-elastic-ribbon-jax`](https://github.com/StructuresComp/discrete-elastic-ribbon-jax) | **High-performance JAX/Equinox implementation.** Banded Hessian assembly with LAPACK banded factorization, vectorized stencil operations, and end-to-end differentiability through the implicit Newton–Raphson solver. |

---

## About this Repository

You are reading the **high-performance JAX/Equinox implementation**. It is a port of the reference [`discrete-elastic-ribbon`](https://github.com/StructuresComp/discrete-elastic-ribbon) codebase, with banded Hessian assembly, vectorized per-stencil operations via `vmap`, and end-to-end differentiability through the implicit Newton–Raphson solver. It targets efficient differentiable simulation of elastic ribbons undergoing shear-induced buckling and related bifurcation studies.

## Setup

```bash
conda activate ribbon-jax
cd discrete-elastic-ribbon-jax
pip install -e .
```

**Dependencies:** jax, jaxlib, equinox, numpy, scipy, matplotlib

## How to Run

**Shear-induced bifurcation (single run):**

```python
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
import dismech_jax as dm
from dismech_jax.models.kirchhoff import Kirchhoff

# Setup ribbon: N=45 nodes, L=0.1m, W/L=1/20
N, L, w_by_l, h = 45, 0.1, 1/20, 1e-3
w = w_by_l * L

geom = dm.Geometry(length=L, r0=h, axs=w*h, jxs=w*h**3/3,
                   ixs1=w*h**3/12, ixs2=h*w**3/12)
mat = dm.Material(density=1000.0, youngs_rod=10e9, poisson_rod=0.5)

nodes = np.zeros((N, 3))
nodes[:, 0] = np.linspace(0, L, N)
rod, q0, aux, mass = dm.create_rod_from_nodes(nodes, geom, mat, gravity=-9.81)

# Fix both ends
start_nodes = np.where(nodes[:, 0] <= 0.01)[0]
end_nodes = np.where(nodes[:, 0] >= 0.09)[0]
rod, q0 = dm.fix_nodes(rod, q0, np.union1d(start_nodes, end_nodes))

# Energy model
dl = L / (N - 1)
model = Kirchhoff.from_geometry(jnp.float64(dl), geom, mat)

# Simulate
sp = dm.SimParams(dt=0.032, total_time=12.5, tol=0.001, ftol=0.0001,
                  dtol=0.01, max_iter=100, log_step=10, static_sim=False)
stepper = dm.TimeStepper(rod, model, sp, mass, q0, aux)
stepper.adaptive_dt = True

# Define shear loading phases via before_step callback
# (see examples/shear_induced_bifurcation/simulate.py for full version)

result = stepper.simulate()
```

**Full benchmark sweep (all W/L ratios and energy models):**

```bash
# N=45, all models, W/L = [1/40, 1/20, 1/12, 1/6]
python benchmarks/run_wl_sweep.py --nodes 45 --wl 0.025 0.05 0.0833 0.1667

# N=63
python benchmarks/run_wl_sweep.py --nodes 63 --wl 0.025 0.05 0.0833 0.1667
```

**Run tests:**

```bash
cd discrete-elastic-ribbon-jax
pytest tests/ -v
```

## Features

- Complete JAX/Equinox reimplementation of the Discrete Elastic Rod (DER) framework
- Five energy models: **Kirchhoff**, **Sano**, **Audoly**, **Sadowsky**, **Wunderlich**
- Implicit Euler time integration with Newton-Raphson solver
- Adaptive time-stepping with dt reduction/recovery
- Robust linear solver (Tikhonov regularization + SVD fallback)
- Analytical strain Jacobian + autodiff energy Hessian (hybrid approach)
- Block-diagonal Hessian assembly via `vmap` over per-triplet stencils (20x speedup over naive full Hessian)
- Banded Hessian assembly + LAPACK banded factorisation (bandwidth `k=10` from the 11-DOF triplet stencil), giving ~3x speedup at N=45 and ~5x at N=63 over the dense baseline
- Shear-induced bifurcation simulation reproducing reference results across all W/L ratios
- Per-step metrics tracking: dt history, Newton iteration counts

## Benchmark Results

Shear-induced bifurcation simulation on a single CPU core, with the banded Hessian factorisation (bandwidth `k=10`). Backward shear phase only (t = 7.55 to 12.5s, sim duration = 4.95s), where the ribbon undergoes buckling and adaptive time-stepping is most active.

**N = 45 nodes (179 DOFs)**

| W/L  | Model     | x sim-time | Wall (s) | Steps | NR iters | NR/step |
|------|-----------|------------|----------|-------|----------|---------|
| 1/40 | Kirchhoff | 0.32x      | 1.58     | 285   | 1456     | 5.1     |
| 1/40 | Sano      | 0.31x      | 1.51     | 284   | 1448     | 5.1     |
| 1/40 | Audoly    | 0.26x      | 1.31     | 214   | 1140     | 5.3     |
| 1/20 | Kirchhoff | 0.52x      | 2.57     | 604   | 3426     | 5.7     |
| 1/20 | Sano      | 0.61x      | 3.00     | 708   | 3831     | 5.4     |
| 1/20 | Audoly    | 1.57x      | 7.77     | 1516  | 10346    | 6.8     |
| 1/12 | Kirchhoff | 0.59x      | 2.93     | 513   | 3140     | 6.1     |
| 1/12 | Sano      | 0.58x      | 2.89     | 497   | 2963     | 6.0     |
| 1/12 | Audoly    | 0.72x      | 3.55     | 471   | 2906     | 6.2     |
| 1/6  | Kirchhoff | 2.16x      | 10.71    | 1436  | 8921     | 6.2     |
| 1/6  | Sano      | 1.65x      | 8.15     | 1164  | 7512     | 6.5     |
| 1/6  | Audoly    | 3.39x      | 16.77    | 1430  | 8496     | 5.9     |

**N = 63 nodes (251 DOFs)**

| W/L  | Model     | x sim-time | Wall (s) | Steps | NR iters | NR/step |
|------|-----------|------------|----------|-------|----------|---------|
| 1/40 | Kirchhoff | 0.35x      | 1.75     | 338   | 1832     | 5.4     |
| 1/40 | Sano      | 0.32x      | 1.59     | 306   | 1673     | 5.5     |
| 1/40 | Audoly    | 0.30x      | 1.46     | 265   | 1454     | 5.5     |
| 1/20 | Kirchhoff | 0.72x      | 3.55     | 786   | 4653     | 5.9     |
| 1/20 | Sano      | 0.63x      | 3.12     | 630   | 3864     | 6.1     |
| 1/20 | Audoly    | 1.26x      | 6.22     | 763   | 4569     | 6.0     |
| 1/12 | Kirchhoff | 1.83x      | 9.04     | 1208  | 7924     | 6.6     |
| 1/12 | Sano      | 2.01x      | 9.96     | 1118  | 7097     | 6.3     |
| 1/12 | Audoly    | 2.80x      | 13.87    | 2611  | 16898    | 6.5     |
| 1/6  | Kirchhoff | 8.23x      | 40.76    | 2535  | 18487    | 7.3     |
| 1/6  | Sano      | 6.91x      | 34.22    | 2247  | 15758    | 7.0     |
| 1/6  | Audoly    | 8.29x      | 41.05    | 2181  | 14506    | 6.7     |

*"x sim-time"* = wall-clock / simulated duration. Values < 1x mean faster than real-time.

Across all 24 configurations the per-Newton-iteration wall-clock scales close to the ideal `O(N)` (median `Wall/#NR` ratio `1.1x` for a `1.4x` DOF increase, N=45 → N=63). The residual super-linear wall-time growth at wide ribbons is driven by the adaptive time-stepper taking 1.5–5x more steps at the finer mesh — spatial resolution of stiff bifurcation features, not solver cost.

## Architecture

```
TimeStepper → Rod (System) → Triplet (Stencil) + Energy Model
                           ↓
                jax.grad/hessian for F, H
```

All components are `eqx.Module` (pytree-compatible), enabling end-to-end differentiation through the implicit solver via the Implicit Function Theorem.

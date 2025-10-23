# Troubleshooting Guide: PyTorch Mirror Descent Solver

## Common Issues and Solutions

### Issue 1: Import Errors

**Error:** `ModuleNotFoundError: No module named 'torch'`

**Solution:**
```bash
# Install PyTorch for CPU
pip install torch

# Or for GPU (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Error:** `ModuleNotFoundError: No module named 'nlopt'` still appearing

**Solution:** This is expected! The old NLopt import should no longer be used. If you see this in other files, they need updating to use `TopOptSolver` instead.

---

### Issue 2: Divergent Optimization

**Symptom:** Objective value keeps increasing or becomes NaN

**Likely Causes:**
1. **Learning rate too high**
   - Solution: Reduce `learning_rate` (try 0.01 instead of 0.05)
   
2. **Gradient explosion**
   - Check if FEM solve is ill-conditioned
   - Solution: Ensure boundary conditions are well-defined
   
3. **Numerical instability in softmax**
   - Solution: Already handled with `log_x - torch.max(log_x)` for stability

**Debug code:**
```python
solver = TopOptSolver(..., learning_rate=0.01)  # Start very small
x_opt = solver.optimize(x)
# Check intermediate values in optimize() loop
```

---

### Issue 3: Constraint Not Satisfied

**Symptom:** Final volume > target volume fraction

**Causes:**
1. **Dual step size too small**
   - The Lagrange multiplier isn't being updated fast enough
   - Solution: Increase `dual_step_size` in `__init__()` (default 0.01 → try 0.1)

2. **Penalty parameter not growing fast enough**
   - Solution: Increase penalty growth rate in `optimize()` method (1.1 → try 1.2)

3. **Early stopping due to convergence**
   - Mirror descent converged before satisfying constraint
   - Solution: Increase `maxeval` or reduce `ftol_rel`

**Check constraint violation:**
```python
x_opt = solver.optimize(x)
final_volume = x_opt.sum() / x_opt.size
print(f"Volume fraction: {final_volume:.6f}")
print(f"Target: {solver.volfrac:.6f}")
print(f"Violation: {final_volume - solver.volfrac:.6e}")
```

---

### Issue 4: Slow Convergence

**Symptom:** Optimization takes many iterations, changes tiny per iteration

**Causes:**
1. **Learning rate too small**
   - Solution: Increase `learning_rate` (0.05 → 0.1 or 0.2)
   
2. **Problem size too large with fine mesh**
   - Mirror descent is first-order; use fewer elements for testing
   - Or consider using GPU: smaller per-iteration time
   
3. **Filter radius too large**
   - Solution: Reduce `rmin` value

**Speed up debugging:**
```python
# Test with smaller mesh first
nelx, nely = 20, 10  # instead of 60, 30

solver = TopOptSolver(..., 
                      learning_rate=0.1,    # increase
                      maxeval=200)           # enough for small problem
```

---

### Issue 5: Checkerboard Patterns or Noise

**Symptom:** Optimized design has lots of isolated elements

**Causes:**
- This is normal for unfiltered results
- Solution: Ensure filter is properly initialized and applied
- Check: `filter = DensityFilter(nelx, nely, rmin)` with appropriate `rmin`

**Verify filtering:**
```python
# After optimization
x_opt = solver.optimize(x)
filtered_x = solver.filter_variables(x_opt)  # Apply filter
# filtered_x should be smoother
```

---

### Issue 6: Different Results Than MMA

**Why:** Different algorithms converge to different local optima

**Expected differences:**
- Mirror descent is first-order; may not find as good solution as MMA
- Convergence from different starting points
- Different constraint satisfaction precision

**To improve results:**
1. Use better starting point: `x = volfrac * np.ones(n)`
2. Increase learning rate slightly for faster exploration
3. Run longer: increase `maxeval`
4. Fine-tune parameters for your problem

**Comparison code:**
```python
# Run both solvers and compare
x_init = volfrac * np.ones(nelx * nely)

# Mirror descent solution
solver_md = TopOptSolver(problem, volfrac, filter, gui, 
                         learning_rate=0.05, maxeval=500)
x_md = solver_md.optimize(x_init.copy())

# Compare against known solution (e.g., MMA from previous run)
# x_ref should be loaded from file or computed with MMA
print(f"Mirror descent volume: {x_md.sum() / x_md.size:.4f}")
```

---

### Issue 7: GPU Usage Not Working

**Error:** `RuntimeError: Expected all tensors to be on the same device`

**Solution:** Currently the solver uses CPU. For GPU support:
```python
# Modify TopOptSolver.optimize() to add device handling:
device = 'cuda' if torch.cuda.is_available() else 'cpu'
x_torch = torch.from_numpy(x.copy()).float().to(device)
```

**Note:** The problem's FEM solve (in `compute_objective()`) is still CPU-bound with numpy/cvxopt.

---

## Performance Benchmarking

### Expected Iteration Counts

For MBB beam (60×30 mesh, volfrac=0.4):
- **NLopt MMA**: ~50–100 iterations
- **Mirror Descent**: ~200–500 iterations (depends on learning_rate)

### Time per Iteration

| Component | Typical Time |
|-----------|-------------|
| FEM solve | ~10–50 ms |
| Sensitivity (adjoint) | ~5–20 ms |
| Mirror descent step | <1 ms |
| Filter + GUI update | ~5 ms |
| **Total per iteration** | ~20–75 ms |

---

## Memory Usage

### Tensor Memory

For an n-element design problem:
- `x_torch` tensor: 4n bytes (float32)
- `grad` tensor: 4n bytes
- Context storage: minimal
- **Total**: ~8n bytes ≈ 8 MB per 1M elements

### Problem Memory (unchanged)
- K matrix (sparse): typically < 100 MB
- u, f vectors: 8·n_dof bytes
- **Total**: problem-dependent, usually 100–500 MB

---

## Tuning Guide

### For Fast Convergence (minimize iterations):
```python
solver = TopOptSolver(
    problem, volfrac, filter, gui,
    learning_rate=0.1,        # Aggressive
    maxeval=200,
    ftol_rel=1e-2             # Loose tolerance
)
```
- ⚠️ Risk: May not satisfy constraint, inferior design

### For Robust Convergence (reliable feasible design):
```python
solver = TopOptSolver(
    problem, volfrac, filter, gui,
    learning_rate=0.03,       # Conservative
    maxeval=500,
    ftol_rel=1e-4             # Tight tolerance
)
# Also modify dual_step_size and penalty growth in __init__
```
- Slower but more reliable

### Balanced (Recommended):
```python
solver = TopOptSolver(
    problem, volfrac, filter, gui,
    learning_rate=0.05,       # Standard
    maxeval=400,
    ftol_rel=1e-3             # Standard
)
```

---

## Comparing with Previous NLopt Code

If you have old code using NLopt:

### Old Code (NLopt):
```python
from topopt.solvers import TopOptSolver
solver = TopOptSolver(problem, volfrac, filter, gui, maxeval=100)
x_opt = solver.optimize(x)
```

### New Code (PyTorch):
```python
from topopt.solvers import TopOptSolver
solver = TopOptSolver(problem, volfrac, filter, gui, 
                      maxeval=400,         # May need more iterations
                      learning_rate=0.05)  # NEW parameter
x_opt = solver.optimize(x)
```

**API is compatible!** Only the internal implementation changed, so old code should work with minimal updates.

---

## Getting Help

### Debug Checklist

1. ✓ PyTorch installed? `python -c "import torch; print(torch.__version__)"`
2. ✓ Problem FEM solves correctly? Check with simple compliance test
3. ✓ Filter initialized? Verify `rmin` is reasonable
4. ✓ Learning rate reasonable? Start with 0.05, adjust from there
5. ✓ Enough iterations? Try doubling `maxeval`

### Minimal Reproducible Example

```python
import numpy as np
from topopt.boundary_conditions import MBBBeam
from topopt.problems import ComplianceProblem
from topopt.filters import DensityFilter
from topopt.guis import GUI
from topopt.solvers import TopOptSolver

# Minimal test
nelx, nely = 30, 15
bc = MBBBeam(nelx, nely)
problem = ComplianceProblem(bc, penalty=3.0)
filter = DensityFilter(nelx, nely, 1.5)
gui = GUI(nelx, nely)

solver = TopOptSolver(problem, volfrac=0.4, filter=filter, gui=gui,
                      learning_rate=0.05, maxeval=100)
x = 0.4 * np.ones(nelx * nely)
x_opt = solver.optimize(x)

print(f"Final volume: {x_opt.sum() / x_opt.size:.4f}")
```

If this doesn't work, check:
1. Is MBBBeam importable and working?
2. Does ComplianceProblem's FEM solver work independently?
3. What's the error message exactly?

---

## Version Notes

- **PyTorch**: Tested with 1.12+, 2.0+
- **Python**: 3.7+ required
- **NumPy**: 1.18+
- **Removed**: nlopt dependency

See `requirements.txt` for all dependencies.

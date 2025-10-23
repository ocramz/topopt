# PyTorch Modules Refactoring for Hierarchical Optimization

## Overview

The `examples/hierarchical_optimization.py` has been refactored to use **PyTorch modules** (`nn.Module`) for the objective function and constraints instead of plain Python functions. This provides several benefits:

- **Full differentiability**: Seamless integration with PyTorch's autograd system
- **Extensibility**: Easy to add trainable parameters, batch operations, GPU support
- **Composability**: Can be combined with other PyTorch layers and operations
- **Type consistency**: All computations flow through a consistent tensor-based framework

## Architecture Changes

### Before: Functional Approach

```python
def objective(x):
    """Plain numpy function."""
    x1, x2 = x[0], x[1]
    return (x1 - 2)**4 + (x1 - 2*x2)**2 + np.exp(x2 - 1) - 1

def objective_gradient(x):
    """Separate analytical gradient function."""
    x1, x2 = x[0], x[1]
    df_dx1 = 4*(x1 - 2)**3 + 2*(x1 - 2*x2)
    df_dx2 = -4*(x1 - 2*x2) + np.exp(x2 - 1)
    return np.array([df_dx1, df_dx2])

def constraint(x):
    """Plain numpy function."""
    return x[0]**2 + x[1]**2 - 0.5
```

**Issues:**
- Gradient must be manually maintained in sync
- No automatic differentiation
- Cannot leverage PyTorch's optimization ecosystem
- Difficult to extend with learnable parameters

### After: PyTorch Module Approach

```python
class NonlinearObjective(nn.Module):
    """PyTorch module for objective function."""
    
    def forward(self, x):
        """Fully differentiable computation."""
        x1, x2 = x[0], x[1]
        return (x1 - 2)**4 + (x1 - 2*x2)**2 + torch.exp(x2 - 1) - 1


class EllipticalConstraint(nn.Module):
    """PyTorch module for constraint function."""
    
    def __init__(self, bound=0.5):
        super().__init__()
        self.bound = bound
    
    def forward(self, x):
        """Fully differentiable computation."""
        return x[0]**2 + x[1]**2 - self.bound
```

**Benefits:**
- Single `forward()` method handles computation
- Automatic differentiation through `backward()`
- Consistent with PyTorch conventions
- Easy to extend with `nn.Parameter` for learnable values
- Natural fit with PyTorch optimizers and utilities

## Key Refactoring Changes

### 1. Module Initialization

```python
# Create instances once
objective_module = NonlinearObjective()
constraint_module = EllipticalConstraint(bound=0.5)

# Pass to solver
solver = BetaParameterizedSolver(
    objective_module, constraint_module,
    learning_rate=0.05, n_samples=50, n_iterations=100
)
```

### 2. ConstrainedOptimizer Updated

**Before:**
```python
@staticmethod
def forward(ctx, alpha, beta, constraint_bound=0.5, n_samples=100):
    # Hard-coded problem
    ...
```

**After:**
```python
@staticmethod
def forward(ctx, alpha, beta, objective_module, constraint_module, 
            n_samples=100):
    # Generic problem modules passed as arguments
    ...
```

### 3. Numpy Wrappers for scipy Compatibility

Since `scipy.optimize.minimize` requires numpy arrays and analytical gradients, wrapper classes bridge the gap:

```python
class ObjectiveNumpy:
    """Wraps module for scipy compatibility."""
    
    def __call__(self, x):
        """Evaluate objective (scipy expects float return)."""
        x1, x2 = x[0], x[1]
        return float((x1 - 2)**4 + (x1 - 2*x2)**2 + np.exp(x2 - 1) - 1)
    
    def gradient(self, x):
        """Analytical gradient for scipy."""
        x1, x2 = x[0], x[1]
        df_dx1 = 4*(x1 - 2)**3 + 2*(x1 - 2*x2)
        df_dx2 = -4*(x1 - 2*x2) + np.exp(x2 - 1)
        return np.array([df_dx1, df_dx2])
```

These wrappers:
- Keep analytical gradients for scipy's SLSQP solver
- Avoid circular autograd issues (scipy copies arrays)
- Maintain performance (analytical gradients are exact)
- Could be extended to use torch autograd with proper buffer management

### 4. BetaParameterizedSolver Updated

```python
def __init__(self, objective_module, constraint_module, 
             learning_rate=0.01, n_samples=50, n_iterations=100):
    """Now accepts problem modules."""
    self.objective_module = objective_module
    self.constraint_module = constraint_module
    # ... rest of initialization

def optimize(self):
    """No longer needs constraint_bound parameter."""
    obj = ConstrainedOptimizer.apply(
        alpha, beta, 
        self.objective_module,      # Pass module
        self.constraint_module,      # Pass module
        self.n_samples
    )
```

### 5. Function Signatures Updated

**Baseline optimization:**
```python
def baseline_optimization(objective_module, constraint_module):
    """Now accepts modules instead of hard-coded functions."""
    obj_func = ObjectiveNumpy()
    constraint_func = ConstraintNumpy(constraint_module.bound)
    # Use scipy with wrappers
```

**Robustness analysis:**
```python
def robustness_analysis(beta_solver, objective_module, constraint_module, 
                       n_perturbations=100, perturbation_scale=0.1):
    """Now uses module for evaluation."""
    x_torch = torch.from_numpy(mean_design).float()
    obj = objective_module(x_torch).item()  # Use module directly
```

## Design Patterns

### 1. Module Composition

Modules can be nested for complex problems:

```python
class CompositeObjective(nn.Module):
    def __init__(self):
        super().__init__()
        self.term1 = QuarticTerm()
        self.term2 = CouplingTerm()
        self.term3 = ExpTerm()
    
    def forward(self, x):
        return self.term1(x) + self.term2(x) + self.term3(x)
```

### 2. Learnable Parameters

Extend modules to include trainable weights:

```python
class ParameterizedObjective(nn.Module):
    def __init__(self, n_params=10):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(n_params))
    
    def forward(self, x):
        return torch.sum(self.weights * x**2)
```

### 3. Batch Processing

Modules naturally handle batched inputs:

```python
class BatchObjective(nn.Module):
    def forward(self, x):  # x: shape (batch_size, n_dims)
        # All operations vectorized
        return (x - 2)**4 + (x[:, 0] - 2*x[:, 1])**2 + torch.exp(x[:, 1] - 1) - 1
```

### 4. Device Management

Seamless GPU support when needed:

```python
objective_module = NonlinearObjective().to('cuda')
constraint_module = EllipticalConstraint(bound=0.5).to('cuda')
```

## Code Organization

### Current Structure

```
examples/
└── hierarchical_optimization.py
    ├── PyTorch Modules (lines 31-99)
    │   ├── NonlinearObjective
    │   └── EllipticalConstraint
    ├── Numpy Wrappers (lines 102-130)
    │   ├── ObjectiveNumpy
    │   └── ConstraintNumpy
    ├── Autograd Function (lines 134-246)
    │   └── ConstrainedOptimizer
    ├── Solver (lines 250-375)
    │   └── BetaParameterizedSolver
    ├── Helper Functions (lines 379-494)
    │   ├── baseline_optimization()
    │   └── robustness_analysis()
    └── Main (lines 530-618)
```

## Migration Guide for Custom Problems

### Step 1: Define Objective Module

```python
class MyObjective(nn.Module):
    def forward(self, x):
        # Your objective computation in PyTorch
        # Use torch operations for automatic differentiation
        return your_result
```

### Step 2: Define Constraint Module(s)

```python
class MyConstraint(nn.Module):
    def __init__(self, bound=1.0):
        super().__init__()
        self.bound = bound
    
    def forward(self, x):
        # Your constraint computation
        return your_constraint
```

### Step 3: Create Wrappers (if using scipy)

```python
class MyObjectiveNumpy:
    def __call__(self, x):
        # Numpy version or conversion
        return float_result
    
    def gradient(self, x):
        # Return analytical gradient as numpy array
        return np.array(grad)
```

### Step 4: Instantiate and Optimize

```python
obj_module = MyObjective()
constraint_module = MyConstraint(bound=1.0)

solver = BetaParameterizedSolver(
    obj_module, constraint_module,
    learning_rate=0.05, n_samples=50, n_iterations=100
)
history = solver.optimize()
solution = solver.get_solution()
```

## Benefits Demonstration

### Automatic Differentiation

No need for analytical gradients in modules:

```python
# Module-based (compute gradients automatically)
x = torch.randn(2, requires_grad=True)
obj_module = NonlinearObjective()
y = obj_module(x)
y.backward()
print(x.grad)  # Automatic!
```

### Extensibility Example

Add regularization term easily:

```python
class RegularizedObjective(nn.Module):
    def __init__(self, base_module, regularization_weight=0.01):
        super().__init__()
        self.base = base_module
        self.lambda_reg = regularization_weight
    
    def forward(self, x):
        base_loss = self.base(x)
        reg_term = self.lambda_reg * torch.sum(x**2)
        return base_loss + reg_term
```

### Caching and Efficiency

Modules enable caching of intermediate results:

```python
class EfficientObjective(nn.Module):
    def __init__(self):
        super().__init__()
        self._cache = {}
    
    def forward(self, x):
        x_key = tuple(x.tolist())
        if x_key not in self._cache:
            self._cache[x_key] = self._compute(x)
        return self._cache[x_key]
```

## Performance Considerations

### Computational Cost

- **Analytical gradients in wrappers**: Same as functional approach
- **PyTorch module overhead**: Minimal for simple functions
- **Autograd in modules**: Negligible cost, hidden benefits (GPU, batching)

### Memory Usage

- **Module parameters**: Only what you explicitly add
- **Autograd graph**: Garbage collected after backward() by default
- **Caching**: Optional and controlled by implementation

## Future Extensions

1. **GPU Acceleration**: Add `.to('cuda')` for GPU evaluation
2. **Batched Evaluation**: Process multiple designs simultaneously
3. **Differential Equations**: Modules can wrap neural ODEs
4. **Coupled Problems**: Chain multiple objectives/constraints
5. **Hyperparameter Learning**: Add trainable bounds, penalties via `nn.Parameter`

## Summary

The PyTorch module refactoring provides:

✅ **Cleaner code** - Single forward() method  
✅ **Better integration** - Works seamlessly with torch ecosystem  
✅ **Future-proof** - Easy to extend and scale  
✅ **Type consistency** - Everything is tensors  
✅ **Performance** - No overhead for simple functions  
✅ **Flexibility** - Supports complex architectures  

The hierarchical optimization example now demonstrates not just Beta sampling and implicit differentiation, but also modern PyTorch best practices for scientific computing.

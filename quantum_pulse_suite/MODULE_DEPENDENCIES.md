# Module Dependency Diagram

## Package Structure

```
quantum_pulse_suite/
│
├── __init__.py
│   ├── imports from: .core, .analysis
│   └── re-exports all public API
│
├── core/
│   ├── __init__.py
│   │   └── imports from: pulse_sequence, filter_functions, noise
│   │
│   ├── pulse_sequence.py          [No internal deps]
│   │   ├── External: numpy, scipy.linalg, abc, typing
│   │   └── Runtime import: filter_functions (in get_filter_function_calculator)
│   │
│   ├── filter_functions.py        [No internal deps]
│   │   └── External: numpy, abc, typing
│   │
│   ├── noise.py                   [No internal deps]
│   │   └── External: numpy, typing
│   │
│   └── visualization.py
│       ├── External: numpy, matplotlib (optional), typing
│       └── Internal: pulse_sequence (SIGMA_X, SIGMA_Y, SIGMA_Z)
│
└── analysis/
    ├── __init__.py
    │   └── imports from: batch_numerical
    │
    └── batch_numerical.py
        ├── External: numpy, scipy.linalg, scipy.integrate, typing
        └── Internal: core.pulse_sequence, core.noise
```

## Dependency Graph

```
                    ┌─────────────────────────────────┐
                    │   quantum_pulse_suite (main)    │
                    └─────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
            ┌──────────────┐               ┌──────────────┐
            │     core     │               │   analysis   │
            └──────────────┘               └──────────────┘
                    │                               │
        ┌───────────┼───────────┬──────────┐       │
        ▼           ▼           ▼          ▼       ▼
┌──────────────┐ ┌─────────┐ ┌─────┐ ┌─────────┐ ┌─────────────────┐
│pulse_sequence│ │ filter_ │ │noise│ │ visual- │ │ batch_numerical │
│              │ │functions│ │     │ │ization │ │                 │
└──────────────┘ └─────────┘ └─────┘ └─────────┘ └─────────────────┘
        │              ▲                  │               │
        │              │                  │               │
        └──────────────┘                  │               │
         (runtime import)                 │               │
                                         │               │
                    ┌────────────────────┘               │
                    ▼                                    │
            ┌──────────────┐                            │
            │pulse_sequence│◄───────────────────────────┘
            │    noise     │◄───────────────────────────┘
            └──────────────┘
```

## Key Relationships

| Module | Dependencies | Notes |
|--------|--------------|-------|
| `batch_numerical` | `pulse_sequence`, `noise` | Uses Pauli matrices and noise generation |
| `visualization` | `pulse_sequence` | Uses Pauli matrices for Bloch sphere |
| `pulse_sequence` | `filter_functions` (runtime) | Lazy import to avoid circular deps |
| `filter_functions` | None | Standalone module |
| `noise` | None | Standalone module |

## External Dependencies

- **numpy**: Array operations (all modules)
- **scipy.linalg**: Matrix exponential (`expm`) for unitary evolution
- **scipy.integrate**: Simpson's rule for numerical integration
- **matplotlib**: Optional, for visualization functions

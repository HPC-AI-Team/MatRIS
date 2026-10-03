# Model operators

All custom GPU kernels use CUDA C++; there is no custom Triton or Jiterator
backend. PyTorch may still use Triton internally when `torch.compile` is enabled.

Each operator file contains its CUDA/C++ source strings, lazy `load_inline`
loader and Python `autograd.Function` wrappers together. Loading is cached by
source contents. Model dispatch imports fast operators only after the fast-mode
guard (the optional TF32 GEMM loads under its own precision switch); GPU graph operators are imported only in the CUDA inference branch.
Importing the model does not import operator modules or compile extensions.

```text
op/
├── attention.py     # online softmax, attention, residual and CSR layout
├── elementwise.py   # registered precise ATen powers
├── gemm.py          # explicit per-GEMM TF32 and differentiable backward
├── gated_mlp.py     # gated normalization and line projections
├── reductions.py    # directed averaging and envelope scatter
├── graph.py         # GPU geometry and graph assembly
└── cuda_common.py   # shared CUDA types and primitives only
```

Mode and training flags are passed explicitly from `matris/model/model.py`
through the model layers; fusion eligibility lives in `matris/model/functions.py`. Operators do
not read model execution state or detach parameters based on training mode.

## Execution contracts

- Feature/projected widths are positive multiples of 32. Attention, projection,
  averaging, scatter and gated normalization each use one width-parameterized
  implementation; there is no separate 128/256 implementation. Padded CUDA
  channel lanes are masked, and normalization divides by the actual width.
  Node/edge/atom input widths and the first MLP hidden width may differ; projection
  splits follow the actual input widths. The existing two-linear SiLU/LayerNorm
  fusion contract is unchanged. FP32 CUDA remains the optimized dtype/device.

- `mode="fast"` uses fused operators in inference and training. `mode="torch"`
  uses ordinary PyTorch paths. `model.train()` or `is_training=True` keeps
  trainable weights, biases, normalization and residual scales connected.
- Custom power, softmax, gate, attention, projection and reduction
  kernels expose `torch.autograd.Function` classes. Model dispatch calls `.apply()`;
  each class owns its forward, saved-tensor context and backward formula.
  Each CUDA module contains C++ `TORCH_LIBRARY_FRAGMENT` / `TORCH_LIBRARY_IMPL`
  definitions for its operator schemas, dispatch and symbolic Meta implementations.
  At first load, its C++ binding connects PyTorch autograd to the existing Python
  Function methods through `torch.library.register_autograd`; it does not duplicate
  the gradient formulas in C++. This connection is needed because staged EFS
  training differentiates registered nodes after tracing through the Functions. There is one derivative formula per
  operation and no compiler-disable boundary. Terminal second-VJP Functions do
  not support another backward; the contract is first/second derivatives only.
- Model-level Python functions select fusion paths; operator wrappers provide
  the tensor computations and topology utilities. Native `F.linear` stays visible to the compiler; wrapping native
  tensor expressions as opaque operators would unnecessarily prevent fusion.
  Graph construction prepares discrete topology outside the compiled tensor model.
  Training and DataLoader workers always build on CPU. Evaluation with
  `mode="fast"` on CUDA uses GPU neighbor search and CUDA topology assembly;
  torch mode and CPU inference use the NumPy CPU builder. For standalone
  CUDA inference, use `GraphConverter(...).cuda().eval()`; a model propagates
  its `train()`/`eval()` state to its converter. Dataset calls explicitly select CPU.
- Attention's cached outputs are detached only at the backward operator boundary:
  its analytic second derivative already includes their dependence on logits and
  values. Differentiable logits/values and incoming gradients stay connected.
- Gate training derivatives size row tiles from padded feature width, with at
  least four rows per block. This bounds live state while keeping parameter
  scratch no larger than the original four-row implementation.
- Gate variance uses shifted, centered squared deviations to avoid cancellation
  for nearly constant rows with large offsets. Forward and training backward
  share this formulation; all arithmetic remains FP32.
- Fusion dispatch in `matris/model/functions.py` returns `None` when a model
  or input is unsupported, allowing its caller to use the reference path.
  CUDA loading failures are surfaced.
- EFS differentiates coordinates and strain. Pass `is_training=True` when
  training on forces/stresses so their derivatives retain a graph. Only inference
  detaches fixed projection/norm parameters and fused residual scales.
- Graph connectivity and CSR indices are discrete. Second derivatives apply to
  continuous geometry/features/parameters with the constructed topology held fixed.
- GEMM uses one native FP32 implementation at every size. No row-count dispatch.
- All CUDA launches use the current PyTorch stream. Kernels assume the shapes,
  dtypes and contiguous layouts established by their Python interfaces.
- Directed averaging requires exactly two rows per undirected edge and a known
  output row count. Its backward gather/scale is a registered CUDA operation:
  this boundary prevents Inductor from fusing a gather before preceding scatter
  writes when there is only one edge. Double backward reuses the forward reduction.
  CSR layouts contain topology only and can be reused across
  layers; logits and probabilities are never cached across predictions.
  Refine-line projection expects contiguous projected features and atom indices
  sorted by center atom, as produced by the graph builder.
- Attention retains probabilities only when input backward is needed. Energy-only
  calls under `no_grad` omit that allocation and the probability write pass.
- The model has two execution controls: `mode="fast"` automatically uses fused
  operators and lazy tensor-core compilation; `mode="torch"` uses eager
  PyTorch. `checkpoint=False` is the default in both modes. There is no separate
  `compile` argument. `checkpoint=True` enables non-reentrant recomputation,
  independent of graph size. Fast mode checkpoints the compiled tensor core
  after staging; torch mode replays the forward with explicit arguments.
  No checkpoint context factory or context-variable state is needed.
  Training expands the energy's geometric VJP before AOTAutograd, allowing a single
  compiled parameter backward. Graph preparation and the Cartesian chain rule stay
  eager. The returned energy readout also stays eager, so force/stress-only
  losses leave energy-only parameters unused instead of materializing zero gradients.
  Parameters remain live inputs to the compiled graph.
- Compiled scalar powers retain native arithmetic through `matris::pow`. Replacing
  cutoff powers with repeated multiplication changes FP32 cancellation and can
  amplify parameter-gradient errors; this boundary preserves the original formula.
- Gated normalization is registered as `matris::gated_tail`, with a `GatedTailFunction` entry point.
  `GatedBackwardFunction` supplies its first VJP and
  `GatedDoubleBackwardFunction` supplies the terminal second VJP;
  third derivatives are not supported. `matris::gated_input_backward` preserves
  the input-only inference kernel. Norm gradients are packed as `[4, D]` to
  keep custom-op outputs non-aliasing. These calls remain opaque kernel nodes;
  registration removes graph breaks without changing the CUDA kernels.
- `test_registered_gate.py` and `test_registered_training.py` compile expanded
  forward+first-VJP graphs with `fullgraph=True`, then differentiate once to test
  mixed second derivatives. Training stages those derivatives explicitly instead
  of requesting a second backward through AOTAutograd.

## Maintenance and verification

Keep one selected implementation for each supported path. Do not keep candidate
kernels, timing entry points, compiled libraries or historical prototypes here.
Performance experiments belong in `test/eval`, with raw results in `results`.
CUDA/C++ strings ship as Python modules. CPU graph assembly lives in
`matris/graph/converter.py` and uses NumPy without a compiled extension.

```bash
python -m pytest test/test_registered_gate.py test/test_registered_training.py test/test_second_order.py \
    test/test_operator_pass.py test/test_online_softmax.py test/test_fast_mode.py \
    test/test_gpu_graph.py test/test_compile_mode.py test/test_compile_training.py \
    test/test_cpp_registration.py -q
```

See [latest operator refinement](../../../docs/operator_refinement_92799.md),
[training performance](../../../docs/training_performance_92799.md),
[registered training compilation](../../../docs/registered_training_compile_92799.md),
[gate registration measurements](../../../docs/registered_gate_92799.md),
[second-order training verification](../../../docs/second_order_training_92799.md),
[operator cleanup measurements](../../../docs/operator_cleanup_92799.md),
[prior operator pass](../../../docs/operator_pass_92799.md), and
[compile audit](../../../docs/inference_audit_92799.md). Old comparison operators
are loaded by benchmark scripts from snapshots or Git, never by model inference.

Each operator file passes its C++ source and CUDA source where needed
to `load_inline`;
C++ owns operator schemas, dispatch, Meta implementations and bindings. The Python
Function wrappers contain no operator-schema or fake registrations. Importing the
package does not build extensions. The first Function call compiles only its
operator family and installs its gradient bindings once. Each module keeps an
explicit cache that compile treats as constant; loading never enters the tensor graph.
Extensions use a source/ABI cache key; no separate precompilation step is required.
The first call needs a matching
CUDA toolkit and a C++ compiler; subsequent processes reuse the build cache.
Toolkit discovery follows PyTorch (`CUDA_HOME`/`CUDA_PATH`, then `nvcc` on `PATH`);
no machine-specific installation paths are embedded in the package.
The gate uses a fused kernel through 1024 channels and native CUDA tensor
operations for wider widths, retaining the same derivative contract.

`gemm.py` provides per-call cuBLAS TF32 GEMM with an autograd Function and
registered backward supporting higher derivatives. Model-level `tf32=True`
selects it only inside interaction blocks; it never changes global precision
flags. Like the other operators it builds lazily using `load_inline`.

Activations use native PyTorch SiLU/Sigmoid. Integer cutoff envelopes use the
stable factored polynomial in `basis_function.py`; obsolete standalone activation
and envelope CUDA families are not compiled. `elementwise.py` only registers ATen
powers and their differentiable compiler boundary via a C++ `load_inline` module.

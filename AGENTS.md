# AGENTS.md - Anthropic Performance Take-Home Guide

## Project Overview

This is a performance optimization take-home for Anthropic's engineering interview process. The task is to optimize a kernel that simulates a parallel tree traversal workload running on a custom VLIW SIMD architecture.

**Baseline**: 147,734 cycles (starting point)

**Benchmarks to beat**:
- 18,532 cycles: Updated 2-hour starting point
- 2,164 cycles: Claude Opus 4 after many hours
- 1,790 cycles: Claude Opus 4.5 casual (~2hr human performance)
- 1,579 cycles: Claude Opus 4.5 after 2 hours in harness
- 1,487 cycles: Claude Opus 4.5 best performance at launch
- 1,363 cycles: Claude Opus 4.5 with improved harness

## Essential Commands

### Run Tests
```bash
# Primary validation - use this to check progress
python tests/submission_tests.py

# Run individual tests
python -m unittest tests.submission_tests.CorrectnessTests
python -m unittest tests.submission_tests.SpeedTests
```

### Debugging & Development
```bash
# Run full test suite in development mode (includes trace + debug output)
python perf_takehome.py

# Run specific test
python perf_takehome.py Tests.test_kernel_cycles

# Generate trace for hot-reloading visualization (recommended debug loop)
python perf_takehome.py Tests.test_kernel_trace

# In separate terminal, start trace viewer (auto-opens browser)
python watch_trace.py
# Then keep this running and re-run test_kernel_trace to see updated traces
```

### Validation Before Submission
```bash
# Ensure tests folder is unchanged (critical - prevents cheating detection)
git diff origin/main tests/

# Run final validation
python tests/submission_tests.py
```

## Code Organization

### Core Files

**`problem.py`**: The simulator and problem definition
- `Machine`: Main simulator class implementing the VLIW SIMD architecture
- `Core`: Represents a single core with instruction pointer, scratch space, state
- `CoreState`: Enum for core states (RUNNING, PAUSED, STOPPED)
- `DebugInfo`: Maps scratch addresses to human-readable names for debugging
- `Tree`: Implicit balanced binary tree (problem input)
- `Input`: Batch of indices/values for tree traversal (problem input)
- `reference_kernel` / `reference_kernel2`: Reference implementations (verify correctness)
- `build_mem_image`: Converts problem inputs into flat memory image
- `myhash`: 32-bit hash function used in the kernel

**`perf_takehome.py`**: Where optimization happens
- `KernelBuilder`: Builds instruction sequences for the simulator
  - `build_kernel()`: Main entry point - generates the optimized kernel
  - `build_hash()`: Generates hash computation instruction sequence
  - `alloc_scratch()`: Allocates scratch space with optional names
  - `scratch_const()`: Caches and loads constants
- `Tests`: Unit tests for development (includes test_kernel_trace for visualization)

**`tests/frozen_problem.py`**: Frozen copy of problem.py
- Used by submission_tests.py for validation
- Must NOT be modified (tests will fail if it changes)
- Prevents cheating by locking in the problem definition

**`tests/submission_tests.py`**: Official validation suite
- `CorrectnessTests`: Validates output correctness (8 runs with different inputs)
- `SpeedTests`: Validates performance against thresholds
- Must run unmodified to verify submission

### Supporting Files

**`watch_trace.html`**: HTML template for Perfetto trace viewer
**`watch_trace.py`**: HTTP server providing hot-reloading trace visualization
**`Readme.md`**: Project description and benchmarks

## Architecture Details

### VLIW Architecture

The simulator models a Very Large Instruction Word (VLIW) machine with multiple execution engines:

```
Instruction Bundle (1 per cycle):
├── ALU: Up to 12 slots
├── VALU (Vector ALU): Up to 6 slots  
├── Load: Up to 2 slots
├── Store: Up to 2 slots
├── Flow: Up to 1 slot
└── Debug: Up to 64 slots (ignored during submission)
```

**Key constraints**:
- Each slot in an instruction is independent and executes in parallel
- All slot effects take effect at end of cycle (read-after-write safe within cycle)
- Memory and stores complete after all reads, preventing hazards

### SIMD (Vector) Support

- `VLEN = 8`: Vector length (8 elements per vector operation)
- Vector operations (valu) operate on 8 consecutive scratch addresses
- `vload`/`vstore`: Load/store contiguous memory blocks
- `vbroadcast`: Broadcast scalar to vector

### Scratch Space

- Scratch space acts like registers, constant memory, and manually-managed cache
- `SCRATCH_SIZE = 1536`: Total addressable scratch words
- Allocated by `KernelBuilder.alloc_scratch()`
- Debug names help track variables in traces

### Memory & State

- Memory: Flat array of 32-bit words
- Memory layout set up by `build_mem_image()`:
  ```
  mem[0] = rounds
  mem[1] = n_nodes
  mem[2] = batch_size
  mem[3] = forest_height
  mem[4] = forest_values_p (pointer to tree node values)
  mem[5] = inp_indices_p (pointer to indices array)
  mem[6] = inp_values_p (pointer to values array)
  mem[7] = extra_room
  ```

## Code Patterns & Conventions

### Building Instructions

The basic pattern in `build_kernel()`:

```python
# Allocate scratch space
tmp_val = self.alloc_scratch("tmp_val")  # Allocate with name for debugging

# Load a constant (cached)
zero_const = self.scratch_const(0)  # Returns address of constant 0

# Add instruction to kernel
self.add("alu", ("+", dest_addr, src1_addr, src2_addr))

# Or batch-build then add:
body = [
    ("alu", ("+", dest, a, b)),
    ("load", ("load", dest, addr))
]
body_instrs = self.build(body)
self.instrs.extend(body_instrs)
```

### ALU Operations

Supported operations in `Machine.alu()`:
- Arithmetic: `+`, `-`, `*`, `//`, `%`
- Bitwise: `&`, `|`, `^`, `<<`, `>>` 
- Comparison: `<`, `==`
- Special: `cdiv` (ceiling divide)

All operations are 32-bit with wraparound.

### Memory Access Patterns

```python
# Compute effective address
self.add("alu", ("+", tmp_addr, self.scratch["base_ptr"], offset))

# Load from computed address
self.add("load", ("load", dest, tmp_addr))

# Store to computed address
self.add("store", ("store", tmp_addr, src_value))

# Vector load (8 contiguous elements)
self.add("load", ("vload", dest_vector_addr, addr_scalar))

# Vector store
self.add("store", ("vstore", addr_scalar, src_vector_addr))
```

### Control Flow

```python
# Conditional selection
self.add("flow", ("select", dest, cond, if_true_val, if_false_val))

# Jump
self.add("flow", ("jump", instruction_addr))

# Conditional jump
self.add("flow", ("cond_jump", cond, target_addr))

# Halt execution
self.add("flow", ("halt",))

# Pause (for debugging, matched with yields in reference kernel)
self.add("flow", ("pause",))
```

### Debug Instructions

Debug slots are ignored during submission but help during development:

```python
# Compare computed value against reference (fails if mismatch)
self.add("debug", ("compare", scratch_addr, reference_key))

# Vector compare
self.add("debug", ("vcompare", scratch_addr, [ref_key1, ref_key2, ...]))

# Comment for trace
self.add("debug", ("comment", "Description"))
```

## Optimization Strategies

### Instruction-Level Parallelism (ILP)

Pack multiple operations into single cycle using different engines:

```python
# Bad: Sequential (takes 3 cycles)
self.add("alu", ("*", dst1, a, b))
self.add("alu", ("+", dst2, c, d))
self.add("load", ("load", dst3, addr))

# Good: Parallel (takes 1 cycle)
body = [
    ("alu", ("*", dst1, a, b)),
    ("alu", ("+", dst2, c, d)),
    ("load", ("load", dst3, addr))
]
self.instrs.extend(self.build(body))
```

### Loop Unrolling

Reduce branch overhead by processing multiple batches per iteration:

```python
# Instead of single batch per round, process N items, then move to next round
for round in range(rounds):
    for chunk in range(0, batch_size, CHUNK_SIZE):
        for i in range(CHUNK_SIZE):
            # Process item
```

### Vector Operations

For data-parallel operations, use vector ALU and vector loads/stores:

```python
# Load 8 values in one instruction
self.add("load", ("vload", vec_dest, base_addr))

# Vector operation
self.add("valu", ("+", vec_dest, vec_a, vec_b))

# Vector store
self.add("store", ("vstore", base_addr, vec_dest))
```

### Memory Access Optimization

Reuse computed addresses and minimize address calculations:

```python
# Precompute base addresses outside loops
base_idx = self.alloc_scratch("base_idx")
self.add("alu", ("+", base_idx, inp_indices_p, base_offset))

# Reuse in loop with smaller offsets
for i in range(batch_size):
    offset = self.scratch_const(i)
    self.add("alu", ("+", tmp_addr, base_idx, offset))
```

## Important Gotchas

### Must NOT Modify Tests Folder

The submission validator checks:
```bash
git diff origin/main tests/
```

If any changes are detected, submission may be rejected. This prevents cheating via modified test cases or magic constants.

### Multicore is Disabled

`N_CORES = 1` - there is no parallelism across cores in this version. The simulator supports multiple cores in architecture but they're intentionally disabled.

### Slot Limits are Hard Constraints

```python
SLOT_LIMITS = {
    "alu": 12,      # Max 12 ALU operations per cycle
    "valu": 6,      # Max 6 vector ALU operations per cycle
    "load": 2,      # Max 2 loads per cycle
    "store": 2,     # Max 2 stores per cycle
    "flow": 1,      # Max 1 control flow op per cycle
    "debug": 64,    # Debug ops are free (submission ignores)
}
```

Exceeding these limits causes assertion errors.

### Write Timing

Scratch writes take effect at end-of-cycle:
- Read all source operands at start of cycle
- Execute all operations in parallel
- Apply all writes at end of cycle (RAW-safe)

This means:
```python
# Safe: reads happen before writes
self.add("alu", ("+", tmp, tmp, 1))  # Add 1 to tmp, store in tmp
```

But the old value of `tmp` is read before the new value is written.

### Instruction Addresses are Literal

Jump targets and pause/yield pairs must match exactly:
- Jump to instruction index (not memory address)
- Pause instructions must match yields in `reference_kernel2()`
- Off-by-one errors cause assertion failures during debug mode

### Constants Must Be Loaded First

You cannot use immediate values in most instructions - they must be loaded to scratch:

```python
# Wrong (not valid):
self.add("alu", ("+", dst, src, 10))

# Right:
ten = self.scratch_const(10)
self.add("alu", ("+", dst, src, ten))
```

Exception: `const` load instruction, flow `add_imm`, and some other special cases.

### Correctness Validation

The Machine class includes debug checks via `debug` instructions:
- These validate intermediate values against reference kernel
- Must match exactly or assertion fails
- Disabled in submission tests (`machine.enable_debug = False`)
- Crucial for correctness validation during development

## Testing Workflow

### Development Loop

1. Make changes to `KernelBuilder.build_kernel()`
2. Run: `python perf_takehome.py Tests.test_kernel_trace`
3. Keep browser tab open to trace viewer (running `python watch_trace.py`)
4. Trace refreshes automatically after each test run
5. Analyze instruction sequences in Perfetto UI
6. Return to step 1

### Validation Loop

1. Run correctness tests: `python -m unittest tests.submission_tests.CorrectnessTests`
2. Check that outputs match (should pass 8 times with different random inputs)
3. If any fail, debug with development mode (add debug instructions, use prints)
4. Once correct, check performance: `python tests/submission_tests.py`

### Before Submission

```bash
# Verify no changes to tests
git diff origin/main tests/
# Should output nothing

# Final cycle count
python tests/submission_tests.py
# Check which test thresholds you pass
```

## Trace Visualization (Perfetto)

The trace viewer shows:
- **Processes** (one per core, plus scratch memory tracks)
- **Threads** (one per execution slot: alu-0, alu-1, load-0, etc.)
- **Events** (colored blocks showing instruction execution)
- **Track values** (scratch variable changes over time)

Hot-reloading:
1. Run `python watch_trace.py` → browser opens at localhost:8000
2. Run test: `python perf_takehome.py Tests.test_kernel_trace`
3. Browser automatically refreshes with new trace
4. Keep iterating

## Common Optimization Techniques in Literature

While discovering your own optimizations is the point, these are areas to explore:

- **Pipelining**: Overlap instruction dependencies where possible
- **Scheduling**: Reorder instructions to maximize pipeline efficiency  
- **Loop tiling**: Process data in cache-friendly chunks
- **Vector utilization**: Use VLEN=8 vector operations for data parallelism
- **Address locality**: Group memory accesses to reduce address computation
- **Constant propagation**: Identify values that are loop-invariant
- **Instruction fusion**: Combine operations where possible

## References & Documentation

- **Architecture ISA**: Full instruction documentation in `Machine.step()`, organized by engine
- **Algorithm**: `reference_kernel2()` defines the correct behavior
- **Debug format**: Chrome Trace Event Format (see comments in `setup_trace()`)
- **Memory layout**: `build_mem_image()` documents memory organization

## Key Files at a Glance

| File | Purpose | Can Modify? |
|------|---------|------------|
| `perf_takehome.py` | Optimization work | **YES** |
| `problem.py` | Simulator (reference) | No |
| `tests/frozen_problem.py` | Frozen simulator for tests | **NO** |
| `tests/submission_tests.py` | Validation tests | **NO** |
| `watch_trace.py` | Debug visualization | No |
| `.git/` | Tracks changes | Use carefully |

## What Agents Should Know

1. **Don't modify tests/** - this is checked by validator
2. **Use `python tests/submission_tests.py`** as the source of truth for validation
3. **Use trace visualization** when stuck - it shows exactly what's happening cycle-by-cycle
4. **Instruction parallelism** is the most impactful optimization - pack engines efficiently
5. **Start with correctness** before optimizing - use debug mode to verify logic
6. **Reference kernel** shows what the code should do - trace both reference and your kernel
7. **Scratch space is limited** (1536 words) - be mindful of allocation
8. **Cycle count is deterministic** - same input always produces same result

"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    DEBUG = False  # Set to False to disable debug instructions for performance
    
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        def writes(slot, engine):
            if engine == "flow" and slot[0] == "pause":
                return None  # pause has no destination
            if engine in ("alu", "flow", "load"):
                return slot[1]
            return None

        def reads(slot, engine):
            if engine == "alu":
                return slot[2], slot[3]
            if engine == "load" and slot[0] == "load":
                return (slot[2],)
            if engine == "flow" and slot[0] == "select":
                return (slot[2], slot[3], slot[4])
            if engine == "flow" and slot[0] == "pause":
                return ()  # pause has no reads
            if engine == "store" and slot[0] == "store":
                return slot[1], slot[2]
            if engine == "debug" and slot[0] == "compare":
                return (slot[1],)  # debug compare reads the scratch address being compared
            return ()

        def flush():
            if current_bundle:
                bundles.append(dict(current_bundle))
            current_bundle.clear()
            slot_count.clear()
            bundle_writes.clear()

        if not vliw:
            # Simple mode: one slot per instruction bundle
            instrs = []
            for engine, slot in slots:
                instrs.append({engine: [slot]})
            return instrs
        
        # VLIW mode: pack multiple slots into bundles with hazard detection
        bundles = []
        current_bundle = defaultdict(list)
        slot_count = defaultdict(int)
        bundle_writes = set()  # Track scratch addresses written in current bundle

        MEM = object()
        
        i = 0
        while i < len(slots):
            engine, slot = slots[i]
            
            # Pause is a hard scheduling barrier
            if engine == "flow" and slot[0] == "pause":
                flush()
                i += 1
                continue
            
            # MEMORY HAZARD: serialize loads and stores
            if engine in ("load", "store"):
                if "load" in current_bundle or "store" in current_bundle:
                    flush()
                    continue

            # READ hazards 
            for src in reads(slot, engine):
                if src in bundle_writes:
                    flush()
                    break
            else:
                # SLOT LIMIT check
                if slot_count[engine] >= SLOT_LIMITS[engine]:
                    flush()
                    continue

                # ADD slot
                current_bundle[engine].append(slot)
                slot_count[engine] += 1

                # RECORD write
                dst = writes(slot, engine)
                if dst is not None:
                    bundle_writes.add(dst)

                if engine in ("load", "store"):
                    bundle_writes.add(MEM)

                i += 1
                continue
            continue

        if current_bundle:
            bundles.append(dict(current_bundle))

        return bundles  

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i, stage_token=None):
        """Build hash stages with optional dependency tokens for stage ordering.
        
        If stage_token is provided, each stage ends with a write to it (dependency marker)
        and the next stage reads from it (dependency), eliminating the need for flow.pause.
        """
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slot = []
            
            # If this is not the first stage and we have a stage_token, read it to create RAW dependency
            if hi > 0 and stage_token is not None:
                slot.append(("alu", ("+", tmp1, stage_token, self.scratch_const(0))))
            
            slot.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slot.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slot.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            
            # Write the stage token at the end to signal stage completion
            if stage_token is not None:
                slot.append(("alu", ("+", stage_token, stage_token, self.scratch_const(0))))
            
            if self.DEBUG:
                slot.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))
            slots.append(slot)

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)
        three_const = self.scratch_const(3)

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        if self.DEBUG:
            self.add("debug", ("comment", "Starting loop"))

        body = []  # array of slots

        # Scalar scratch registers - Pipeline A
        tmp_idx_a = self.alloc_scratch("tmp_idx_a")
        tmp_val_a = self.alloc_scratch("tmp_val_a")
        tmp_val_hash_a = self.alloc_scratch("tmp_val_hash_a")
        tmp_node_val_a = self.alloc_scratch("tmp_node_val_a")
        tmp_addr_a = self.alloc_scratch("tmp_addr_a")
        tmp1_a = self.alloc_scratch("tmp1_a")
        tmp2_a = self.alloc_scratch("tmp2_a")
        tmp3_a = self.alloc_scratch("tmp3_a")
        
        # Scalar scratch registers - Pipeline B
        tmp_idx_b = self.alloc_scratch("tmp_idx_b")
        tmp_val_b = self.alloc_scratch("tmp_val_b")
        tmp_val_hash_b = self.alloc_scratch("tmp_val_hash_b")
        tmp_node_val_b = self.alloc_scratch("tmp_node_val_b")
        tmp_addr_b = self.alloc_scratch("tmp_addr_b")
        tmp1_b = self.alloc_scratch("tmp1_b")
        tmp2_b = self.alloc_scratch("tmp2_b")
        tmp3_b = self.alloc_scratch("tmp3_b")
        
        # Scalar scratch registers - Pipeline C
        tmp_idx_c = self.alloc_scratch("tmp_idx_c")
        tmp_val_c = self.alloc_scratch("tmp_val_c")
        tmp_val_hash_c = self.alloc_scratch("tmp_val_hash_c")
        tmp_node_val_c = self.alloc_scratch("tmp_node_val_c")
        tmp_addr_c = self.alloc_scratch("tmp_addr_c")
        tmp1_c = self.alloc_scratch("tmp1_c")
        tmp2_c = self.alloc_scratch("tmp2_c")
        tmp3_c = self.alloc_scratch("tmp3_c")

        iA = self.alloc_scratch("iA")
        iB = self.alloc_scratch("iB")
        iC = self.alloc_scratch("iC")
        
        # Stage tokens for dependency ordering per lane
        stage_token_a = self.alloc_scratch("stage_token_a")
        stage_token_b = self.alloc_scratch("stage_token_b")
        stage_token_c = self.alloc_scratch("stage_token_c")

        for round in range(rounds):
            body.append(("load", ("const", iA, 0)))
            body.append(("load", ("const", iB, 1)))
            body.append(("load", ("const", iC, 2)))

            for i in range(0, batch_size, 3):

                # =====================================================
                # PHASE 1: LOADS ONLY (NO READS, NO DEBUG)
                # =====================================================

                # A idx
                body.append(("alu", ("+", tmp_addr_a, self.scratch["inp_indices_p"], iA)))
                body.append(("load", ("load", tmp_idx_a, tmp_addr_a)))

                # B idx
                if i + 1 < batch_size:
                    body.append(("alu", ("+", tmp_addr_b, self.scratch["inp_indices_p"], iB)))
                    body.append(("load", ("load", tmp_idx_b, tmp_addr_b)))

                # C idx
                if i + 2 < batch_size:
                    body.append(("alu", ("+", tmp_addr_c, self.scratch["inp_indices_p"], iC)))
                    body.append(("load", ("load", tmp_idx_c, tmp_addr_c)))

                # A val
                body.append(("alu", ("+", tmp_addr_a, self.scratch["inp_values_p"], iA)))
                body.append(("load", ("load", tmp_val_a, tmp_addr_a)))

                # B val
                if i + 1 < batch_size:
                    body.append(("alu", ("+", tmp_addr_b, self.scratch["inp_values_p"], iB)))
                    body.append(("load", ("load", tmp_val_b, tmp_addr_b)))

                # C val
                if i + 2 < batch_size:
                    body.append(("alu", ("+", tmp_addr_c, self.scratch["inp_values_p"], iC)))
                    body.append(("load", ("load", tmp_val_c, tmp_addr_c)))

                # A node_val
                body.append(("alu", ("+", tmp_addr_a, self.scratch["forest_values_p"], tmp_idx_a)))
                body.append(("load", ("load", tmp_node_val_a, tmp_addr_a)))

                # B node_val
                if i + 1 < batch_size:
                    body.append(("alu", ("+", tmp_addr_b, self.scratch["forest_values_p"], tmp_idx_b)))
                    body.append(("load", ("load", tmp_node_val_b, tmp_addr_b)))

                # C node_val
                if i + 2 < batch_size:
                    body.append(("alu", ("+", tmp_addr_c, self.scratch["forest_values_p"], tmp_idx_c)))
                    body.append(("load", ("load", tmp_node_val_c, tmp_addr_c)))

                # =====================================================
                # PHASE 2: COMPUTE (READS + DEBUG ALLOWED)
                # =====================================================

                # Debug loads
                if self.DEBUG:
                    body.append(("debug", ("compare", tmp_idx_a, (round, i, "idx"))))
                    body.append(("debug", ("compare", tmp_val_a, (round, i, "val"))))
                    body.append(("debug", ("compare", tmp_node_val_a, (round, i, "node_val"))))

                if i + 1 < batch_size:
                    if self.DEBUG:
                        body.append(("debug", ("compare", tmp_idx_b, (round, i + 1, "idx"))))
                        body.append(("debug", ("compare", tmp_val_b, (round, i + 1, "val"))))
                        body.append(("debug", ("compare", tmp_node_val_b, (round, i + 1, "node_val"))))

                if i + 2 < batch_size:
                    if self.DEBUG:
                        body.append(("debug", ("compare", tmp_idx_c, (round, i + 2, "idx"))))
                        body.append(("debug", ("compare", tmp_val_c, (round, i + 2, "val"))))
                        body.append(("debug", ("compare", tmp_node_val_c, (round, i + 2, "node_val"))))

                # A hash (with XOR before hash, into renamed register)
                body.append(("alu", ("^", tmp_val_hash_a, tmp_val_a, tmp_node_val_a)))
                hash_a = self.build_hash(tmp_val_hash_a, tmp1_a, tmp2_a, round, i, stage_token_a)

                if i + 1 < batch_size:
                    # B hash (with XOR before hash, into renamed register)
                    body.append(("alu", ("^", tmp_val_hash_b, tmp_val_b, tmp_node_val_b)))
                    hash_b = self.build_hash(tmp_val_hash_b, tmp1_b, tmp2_b, round, i + 1, stage_token_b)
                else:
                    hash_b = None

                if i + 2 < batch_size:
                    # C hash (with XOR before hash, into renamed register)
                    body.append(("alu", ("^", tmp_val_hash_c, tmp_val_c, tmp_node_val_c)))
                    hash_c = self.build_hash(tmp_val_hash_c, tmp1_c, tmp2_c, round, i + 2, stage_token_c)
                else:
                    hash_c = None

                for stage_idx in range(len(hash_a)):
                    body.extend(hash_a[stage_idx])
                    if hash_b is not None:
                        body.extend(hash_b[stage_idx])
                    if hash_c is not None:
                        body.extend(hash_c[stage_idx])
                
                if self.DEBUG:
                    body.append(("debug", ("compare", tmp_val_hash_a, (round, i, "hashed_val"))))
                    if i + 1 < batch_size:
                        body.append(("debug", ("compare", tmp_val_hash_b, (round, i + 1, "hashed_val"))))
                    if i + 2 < batch_size:
                        body.append(("debug", ("compare", tmp_val_hash_c, (round, i + 2, "hashed_val"))))
                
                # A next idx (use hashed value)
                body.append(("alu", ("%", tmp1_a, tmp_val_hash_a, two_const)))
                body.append(("alu", ("==", tmp1_a, tmp1_a, zero_const)))
                body.append(("flow", ("select", tmp3_a, tmp1_a, one_const, two_const)))
                body.append(("alu", ("*", tmp_idx_a, tmp_idx_a, two_const)))
                body.append(("alu", ("+", tmp_idx_a, tmp_idx_a, tmp3_a)))

                # B next idx (use hashed value)
                if i + 1 < batch_size:
                    body.append(("alu", ("%", tmp1_b, tmp_val_hash_b, two_const)))
                    body.append(("alu", ("==", tmp1_b, tmp1_b, zero_const)))
                    body.append(("flow", ("select", tmp3_b, tmp1_b, one_const, two_const)))
                    body.append(("alu", ("*", tmp_idx_b, tmp_idx_b, two_const)))
                    body.append(("alu", ("+", tmp_idx_b, tmp_idx_b, tmp3_b)))

                # C next idx (use hashed value)
                if i + 2 < batch_size:
                    body.append(("alu", ("%", tmp1_c, tmp_val_hash_c, two_const)))
                    body.append(("alu", ("==", tmp1_c, tmp1_c, zero_const)))
                    body.append(("flow", ("select", tmp3_c, tmp1_c, one_const, two_const)))
                    body.append(("alu", ("*", tmp_idx_c, tmp_idx_c, two_const)))
                    body.append(("alu", ("+", tmp_idx_c, tmp_idx_c, tmp3_c)))

                # wrap
                body.append(("alu", ("<", tmp1_a, tmp_idx_a, self.scratch["n_nodes"])))
                body.append(("flow", ("select", tmp_idx_a, tmp1_a, tmp_idx_a, zero_const)))
                if self.DEBUG:
                    body.append(("debug", ("compare", tmp_idx_a, (round, i, "wrapped_idx"))))

                if i + 1 < batch_size:
                    body.append(("alu", ("<", tmp1_b, tmp_idx_b, self.scratch["n_nodes"])))
                    body.append(("flow", ("select", tmp_idx_b, tmp1_b, tmp_idx_b, zero_const)))
                    if self.DEBUG:
                        body.append(("debug", ("compare", tmp_idx_b, (round, i + 1, "wrapped_idx"))))

                if i + 2 < batch_size:
                    body.append(("alu", ("<", tmp1_c, tmp_idx_c, self.scratch["n_nodes"])))
                    body.append(("flow", ("select", tmp_idx_c, tmp1_c, tmp_idx_c, zero_const)))
                    if self.DEBUG:
                        body.append(("debug", ("compare", tmp_idx_c, (round, i + 2, "wrapped_idx"))))

                # =====================================================
                # PHASE 3: STORES ONLY
                # =====================================================

                # A idx
                body.append(("alu", ("+", tmp_addr_a, self.scratch["inp_indices_p"], iA)))
                body.append(("store", ("store", tmp_addr_a, tmp_idx_a)))

                # B idx
                if i + 1 < batch_size:
                    body.append(("alu", ("+", tmp_addr_b, self.scratch["inp_indices_p"], iB)))
                    body.append(("store", ("store", tmp_addr_b, tmp_idx_b)))

                # C idx
                if i + 2 < batch_size:
                    body.append(("alu", ("+", tmp_addr_c, self.scratch["inp_indices_p"], iC)))
                    body.append(("store", ("store", tmp_addr_c, tmp_idx_c)))

                # A val (store the hashed value)
                body.append(("alu", ("+", tmp_addr_a, self.scratch["inp_values_p"], iA)))
                body.append(("store", ("store", tmp_addr_a, tmp_val_hash_a)))

                # B val (store the hashed value)
                if i + 1 < batch_size:
                    body.append(("alu", ("+", tmp_addr_b, self.scratch["inp_values_p"], iB)))
                    body.append(("store", ("store", tmp_addr_b, tmp_val_hash_b)))

                # C val (store the hashed value)
                if i + 2 < batch_size:
                    body.append(("alu", ("+", tmp_addr_c, self.scratch["inp_values_p"], iC)))
                    body.append(("store", ("store", tmp_addr_c, tmp_val_hash_c)))

                # increment
                body.append(("alu", ("+", iA, iA, three_const)))
                body.append(("alu", ("+", iB, iB, three_const)))
                body.append(("alu", ("+", iC, iC, three_const)))

        body_instrs = self.build(body, vliw=True)
        self.instrs.extend(body_instrs)
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)
        # do_kernel_test(3, 1, 2, trace=True, prints=True)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()

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
# print(unittest.__file__)

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

def p(x, val="def"):
    print(f"{val}=", x)

class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.vec_const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        cur = defaultdict(list)
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        #     if len(cur[engine]) >= 1:
        #         instrs.append(dict(cur))
        #         cur = defaultdict(list)

        #     cur[engine].append(slot)

        # if cur:
        #     instrs.append(dict(cur))

        return instrs

    def add(self, engine, slot):
        """
        self.instrs.append({engine: [slot]})
        """
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        """
        alloc_scratch: \n
        scratch[name] = scratch_ptr\n
        scratch_debug = (name, length)\n
        scratch_ptr += length
        """
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

    def scratch_const_vec(self, val):
        if val not in self.vec_const_map:
            base = self.alloc_scratch(f"vec_const_{val}", VLEN)
            for j in range(VLEN):
                self.add("load", ("const", base + j, val))
            self.vec_const_map[val] = base
        return self.vec_const_map[val]
    
    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            c1 = self.scratch_const_vec(val1)
            c3 = self.scratch_const_vec(val3)

            slots.append(("valu", (op1, tmp1, val_hash_addr, c1)))
            slots.append(("valu", (op3, tmp2, val_hash_addr, c3)))
            slots.append(("valu", (op2, val_hash_addr, tmp1, tmp2)))

            slots.append(("debug", ("vcompare", val_hash_addr,
                [(round, i+j, "hash_stage", hi) for j in range(VLEN)]
            )))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """
        tmp1 = self.alloc_scratch("tmp1", VLEN)
        tmp2 = self.alloc_scratch("tmp2", VLEN)
        tmp3 = self.alloc_scratch("tmp3", VLEN)
        # print(self.scratch)
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
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.alloc_scratch("zero_vec", VLEN)
        one_const  = self.alloc_scratch("one_vec", VLEN)
        two_const  = self.alloc_scratch("two_vec", VLEN)

        for j in range(VLEN):
            self.add("load", ("const", zero_const + j, 0))
            self.add("load", ("const", one_const + j, 1))
            self.add("load", ("const", two_const + j, 2))

        zero_scalar = self.scratch_const(0)

        n_nodes_vec = self.alloc_scratch("n_nodes_vec", VLEN)
        for j in range(VLEN):
            self.add("alu", ("+", n_nodes_vec + j, self.scratch["n_nodes"], zero_scalar))
        

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        body = []  # array of slots

        # Scalar scratch registers
        tmp_idx = self.alloc_scratch("tmp_idx", VLEN)
        tmp_val = self.alloc_scratch("tmp_val", VLEN)
        tmp_node_val = self.alloc_scratch("tmp_node_val", VLEN)
        tmp_addr = self.alloc_scratch("tmp_addr", VLEN)

        # print(f"{tmp_idx=}")
        # print(f"{self.const_map=}")
        # print(f"{self.scratch=}")
        # print(f"{self.scratch_const=}")

        for round in range(rounds):
            for i in range(0, batch_size, VLEN):
                i_const = self.scratch_const(i)
                # idx = mem[inp_indices_p + i]
                for j in range(VLEN):
                    body.append(("alu", ("+", tmp_addr+j, self.scratch["inp_indices_p"], self.scratch_const(i+j))))

                for j in range(VLEN):
                    body.append(("load", ("load", tmp_idx+j, tmp_addr+j)))

                for j in range(VLEN):
                    body.append(("debug", ("compare", tmp_idx+j, (round, i+j, "idx"))))


                # val = mem[inp_values_p + i]
                for j in range(VLEN):
                    body.append(("alu", ("+", tmp_addr+j, self.scratch["inp_values_p"], self.scratch_const(i+j))))
                
                for j in range(VLEN):
                    body.append(("load", ("load", tmp_val+j, tmp_addr+j)))

                for j in range(VLEN):
                    body.append(("debug", ("compare", tmp_val+j, (round, i+j, "val"))))


                # node_val = mem[forest_values_p + idx]

                for j in range(VLEN):
                    body.append(("alu", ("+", tmp_addr+j, self.scratch["forest_values_p"], tmp_idx+j)))
                
                for j in range(VLEN):
                    body.append(("load", ("load", tmp_node_val+j, tmp_addr+j)))
                
                for j in range(VLEN):
                    body.append(("debug", ("compare", tmp_node_val+j, (round, i+j, "node_val"))))


                # val = myhash(val ^ node_val)
                body.append(("valu", ("^", tmp_val, tmp_val, tmp_node_val)))
                body.extend(self.build_hash(tmp_val, tmp1, tmp2, round, i))

                # for j in range(VLEN):
                #     body.append(("debug", ("compare", tmp_val+j, (round, i+j, "hashed_val"))))

                body.append(("debug", ("vcompare", tmp_val,
                    [(round, i+j, "hashed_val") for j in range(VLEN)]
                )))




                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                body.append(("valu", ("%", tmp1, tmp_val, two_const)))
                body.append(("valu", ("==", tmp1, tmp1, zero_const)))

                for j in range(VLEN):
                    body.append(("flow", ("select",
                        tmp3 + j,
                        tmp1 + j,
                        one_const + j,
                        two_const + j,
                    )))

                body.append(("valu", ("*", tmp_idx, tmp_idx, two_const)))
                body.append(("valu", ("+", tmp_idx, tmp_idx, tmp3)))

                body.append(("debug", ("vcompare", tmp_idx,
                    [(round, i+j, "next_idx") for j in range(VLEN)]
                )))


                # idx = 0 if idx >= n_nodes else idx
                body.append(("valu", ("<", tmp1, tmp_idx, n_nodes_vec)))

                
                for j in range(VLEN):
                    body.append(("flow", ("select",
                        tmp_idx + j,
                        tmp1 + j,
                        tmp_idx + j,
                        zero_const + j,
                    )))
                body.append(("debug", ("vcompare", tmp_idx, [(round, i+j, "wrapped_idx") for j in range(VLEN)])))

                

                # mem[inp_indices_p + i] = idx
                for j in range(VLEN):
                    body.append(("alu", ("+", tmp_addr+j, self.scratch["inp_indices_p"], self.scratch_const(i+j))))

                for j in range(VLEN):
                    body.append(("store", ("store", tmp_addr+j, tmp_idx+j)))


                # mem[inp_values_p + i] = val
                for j in range(VLEN):
                    body.append(("alu", ("+", tmp_addr+j, self.scratch["inp_values_p"], self.scratch_const(i+j))))

                for j in range(VLEN):
                    body.append(("store", ("store", tmp_addr+j, tmp_val+j)))

        # print(f"{self.scratch=}")
        # print(f"{self.const_map=}")

        # print(f"{body[:100]=}")
        # print(f"{len(body)=}")
        body_instrs = self.build(body)
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
    # print(f"{len(kb.instrs)=}")
    from collections import Counter
    c = Counter()
    for ins in kb.instrs:
        for k in ins:
            c[k]+=1
    # print(c)


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
    # p(machine.debug)
    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    # print(f"{kb.scratch_ptr=}")
    # print(f"{SCRATCH_SIZE=}")
    print(f"{SLOT_LIMITS=}")

    # for key, value in SLOT_LIMITS.items():
    #     print(f"{key} = {c[key]/value}")
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
        return
        do_kernel_test(10, 16, 256, trace=True, prints=False)

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
        # print(Machine.self.de)


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

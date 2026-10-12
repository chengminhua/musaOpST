# SPDX-License-Identifier: MulanPSL-2.0
"""Explicit platform-test backend. Not MUSA emulation or GPU fallback."""
import time
import numpy as np
from ..util import load_entry, as_outputs
from ..data import clone_inputs, tensor_metadata


class CPUBackend:
    def __init__(self, case, inputs, workdir):
        self.case, self.original = case, inputs
        self.inputs = clone_inputs(inputs)
        self.fn = None
        self.last_outputs = None

    def prepare(self):
        self.fn = load_entry(self.case['entry'])

    def reset_inputs(self):
        from ..binding import flatten_tensors
        for dst, src in zip(flatten_tensors(self.inputs), flatten_tensors(self.original)):
            np.copyto(dst, src)

    def run(self):
        from ..binding import bind_inputs
        self.reset_inputs()
        args, kwargs = bind_inputs(self.inputs, self.case['inputs'], self.case['attrs'], binding=self.case.get('binding','positional'))
        self.last_outputs = as_outputs(self.fn(*args, **kwargs))
        return self.last_outputs

    def rebind(self, case, inputs):
        self.case, self.original = case, inputs
        self.inputs = clone_inputs(inputs)

    def synchronize(self):
        pass

    def output_metadata(self, outputs):
        from ..binding import flatten_tensors
        originals = flatten_tensors(self.inputs)
        result=[]
        for output in outputs:
            array=np.asarray(output)
            meta=tensor_metadata(array)
            meta['alias_inputs']=[i for i,value in enumerate(originals) if np.shares_memory(array,value)]
            result.append(meta)
        return result

    def snapshot(self, outputs):
        return [np.array(x, copy=True) for x in outputs], [str(np.asarray(x).dtype) for x in outputs]

    def benchmark(self, policy):
        for _ in range(policy['warmup']): self.run()
        raw=[]
        iterations=1 if policy.get('mode')=='single_event' else policy['iterations']
        for _ in range(policy['samples']):
            start=time.perf_counter_ns()
            for _ in range(iterations): self.run()
            raw.append((time.perf_counter_ns()-start)/1e6/iterations)
        return {'raw_samples_ms':raw, 'policy':{
            'timer':'cpu_perf_counter_ns','mode':'CPU_PLATFORM_TEST_ONLY',
            'sample_unit':'single_call_ms' if iterations==1 else 'batch_mean_call_ms',
            'iterations_per_sample':iterations,'samples':policy['samples'],'warmup':policy['warmup'],
            'stream':'none','cache':'reuse_no_flush','includes_output_allocation':True,
            'includes_input_reset':True,'includes_host_device_copy':False}}

    def metadata(self):
        from ..binding import flatten_tensors
        return {'device':'cpu','mode':'CPU_PLATFORM_TEST_ONLY',
                'inputs':[tensor_metadata(x) for x in flatten_tensors(self.inputs)]}

    def memory(self):
        return {'active_allocated_bytes':'unknown','cached_reserved_bytes':'unknown','device':'cpu'}

    def close(self):
        self.inputs=[]

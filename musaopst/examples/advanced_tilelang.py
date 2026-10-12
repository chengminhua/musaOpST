# SPDX-License-Identifier: MulanPSL-2.0
"""TileLang demonstration kernels, including explicit storage views and dynamic N.

Static examples report per-shape compilation; only axpby_dynamic declares a
compile-once rebind hook. A flattened storage view does not copy logical inputs.
"""
from hashlib import sha256


def _factory(case, inputs, workdir, operation):
    import torch
    import tilelang
    import tilelang.language as T
    from musaopst.backends.gpu import _flat
    from musaopst.errors import UnsupportedError
    values = list(_flat(inputs))
    attrs = case["attrs"]
    if values:
        a = values[0]
        device, dtype = a.device, str(a.dtype).replace("torch.", "")
    else:
        device = torch.device(f"musa:{torch.musa.current_device()}")
        dtype = attrs.get("dtype", case["outputs"][0]["dtype"])
    if dtype in ("complex64", "complex128"):
        raise UnsupportedError("Installed TileLang MUSA has no complex64/complex128 tensor dtype; this entry does not reinterpret complex storage as bytes")
    if operation != "identity" and dtype not in ("float32", "float16", "bfloat16"):
        raise UnsupportedError("TileLang numerical examples support float32/float16/bfloat16")
    def physical(tensor):
        if tensor.ndim > 2:
            raise UnsupportedError("TileLang examples support rank <= 2")
        if any(s < 0 for s in tensor.stride()):
            raise UnsupportedError("Negative input strides are unsupported")
        span = 1 + sum(max(n-1,0)*s for n,s in zip(tensor.shape,tensor.stride()))
        # For a zero-size tensor no device data is accessed, but its base storage
        # was allocated by the adapter with at least one element.
        return tensor.as_strided((max(span,1),), (1,), tensor.storage_offset())
    def layout(tensor):
        return (max(tensor.shape[1],1),tensor.stride(0),tensor.stride(1)) if tensor.ndim==2 else (max(tensor.numel(),1),0,tensor.stride(0) if tensor.ndim else 1)
    if operation == "fill":
        if values: raise UnsupportedError("fill needs no input")
        shape = attrs.get("shape",case["outputs"][0]["shape"])
        outputs = [torch.empty(shape, dtype=getattr(torch,dtype), device=device)]
        dummy = outputs[0] if outputs[0].numel() else torch.empty(1,dtype=getattr(torch,dtype),device=device)
        buffers = [physical(dummy),physical(dummy)]
        c,a0,a1=1,0,1; b0,b1=0,1
    else:
        if not values: raise UnsupportedError("operator requires tensor input")
        c,a0,a1=layout(a)
        if operation in ("axpby","matmul"):
            if len(values)!=2 or values[1].dtype!=a.dtype: raise UnsupportedError("two same-dtype tensors required")
            b=values[1]; _,b0,b1=layout(b)
            if operation=="axpby":
                if b.shape!=a.shape: raise UnsupportedError("axpby requires identical shapes")
                shapes=[a.shape]
            else:
                if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[0]: raise UnsupportedError("incompatible matrix shapes")
                if a.shape[1]>1024: raise UnsupportedError("demonstration matmul supports K<=1024")
                shapes=[(a.shape[0],b.shape[1])]
        else:
            b=a; b0,b1=a0,a1
            if operation=="sum_max":
                if a.numel()>4096: raise UnsupportedError("TileLang serial reduction example supports <=4096 elements")
                shapes=[(),()]
            elif operation=="softmax":
                if a.ndim!=2 or a.shape[1]>4096 or attrs.get("dim",-1) not in (-1,1): raise UnsupportedError("softmax requires rank2, cols<=4096, last axis")
                shapes=[a.shape]
            elif operation=="inplace_add":
                if any(n>1 and s==0 for n,s in zip(a.shape,a.stride())): raise UnsupportedError("Cannot mutate overlapping input")
                shapes=None
            else: shapes=[a.shape]
        outputs=[a] if shapes is None else [torch.empty(shape,dtype=a.dtype,device=device) for shape in shapes]
        buffers=[physical(a),physical(b)]
    n=outputs[0].numel() if operation=="fill" else a.numel()
    sizes=[value.numel() for value in buffers]
    output_buffers=[physical(out) if out.numel() else torch.empty(1,dtype=out.dtype,device=out.device) for out in outputs]
    if operation in ("identity","axpby","inplace_add","fill"):
        code={"identity":0,"axpby":1,"inplace_add":2,"fill":3}[operation]
        alpha=float(attrs.get("alpha",1.0)) if operation=="axpby" else float(attrs.get("value",1.0 if operation=="inplace_add" else 0.0))
        beta=float(attrs.get("beta",1.0))
        @tilelang.jit(out_idx=[],target="musa")
        def make_pointwise(N:int, C:int, AS:int, BS:int, YS:int, D:str):
            @T.prim_func
            def main(A:T.Tensor((AS,),D),B:T.Tensor((BS,),D),Y:T.Tensor((YS,),D)):
                with T.Kernel(T.ceildiv(N,128),threads=128) as bx:
                    for j in T.Parallel(128):
                        i=bx*128+j
                        if i<N:
                            if code==0:
                                Y[i]=A[(i//C)*a0+(i%C)*a1]
                            elif code==1:
                                Y[i]=T.cast(A[(i//C)*a0+(i%C)*a1],"float32")*alpha+T.cast(B[(i//C)*b0+(i%C)*b1],"float32")*beta
                            elif code==2:
                                Y[(i//C)*a0+(i%C)*a1]=T.cast(A[(i//C)*a0+(i%C)*a1],"float32")+alpha
                            else:
                                Y[i]=alpha
            return main
        compiled=make_pointwise(n,c,sizes[0],sizes[1],output_buffers[0].numel(),dtype) if n else None
        arguments=[*buffers,*output_buffers]
    elif operation=="sum_max":
        @tilelang.jit(out_idx=[],target="musa")
        def make_stats(N:int,C:int,AS:int,D:str):
            @T.prim_func
            def main(A:T.Tensor((AS,),D),S:T.Tensor((1,),D),M:T.Tensor((1,),D)):
                with T.Kernel(1,threads=1):
                    total=T.alloc_local((1,),"float32")
                    maximum=T.alloc_local((1,),"float32")
                    total[0]=0
                    maximum[0]=-3.4028234663852886e38 if N else 0
                    for i in T.serial(N):
                        total[0]=total[0]+T.cast(A[(i//C)*a0+(i%C)*a1],"float32")
                        maximum[0]=T.max(maximum[0],T.cast(A[(i//C)*a0+(i%C)*a1],"float32"))
                    S[0]=total[0]
                    M[0]=maximum[0]
            return main
        compiled=make_stats(n,c,sizes[0],dtype)
        arguments=[buffers[0],*output_buffers]
    elif operation=="softmax":
        rows,cols=a.shape
        @tilelang.jit(out_idx=[],target="musa")
        def make_softmax(R:int,C:int,AS:int,D:str):
            @T.prim_func
            def main(A:T.Tensor((AS,),D),Y:T.Tensor((R*C,),D)):
                with T.Kernel(R,threads=1) as row:
                    maximum=T.alloc_local((1,),"float32")
                    total=T.alloc_local((1,),"float32")
                    maximum[0]=-3.4028234663852886e38
                    total[0]=0
                    for i in T.serial(C):
                        maximum[0]=T.max(maximum[0],T.cast(A[row*a0+i*a1],"float32"))
                    for i in T.serial(C):
                        total[0]=total[0]+T.exp(T.cast(A[row*a0+i*a1],"float32")-maximum[0])
                    for i in T.serial(C):
                        Y[row*C+i]=T.exp(T.cast(A[row*a0+i*a1],"float32")-maximum[0])/total[0]
            return main
        compiled=make_softmax(rows,cols,sizes[0],dtype) if rows and cols else None
        arguments=[buffers[0],*output_buffers]
    elif operation=="matmul":
        m,k=a.shape; cols=b.shape[1]
        @tilelang.jit(out_idx=[],target="musa")
        def make_mm(M:int,N:int,K:int,AS:int,BS:int,D:str):
            @T.prim_func
            def main(A:T.Tensor((AS,),D),B:T.Tensor((BS,),D),Y:T.Tensor((M*N,),D)):
                with T.Kernel(T.ceildiv(M*N,128),threads=128) as bx:
                    total=T.alloc_local((1,),"float32")
                    for j in T.Parallel(128):
                        i=bx*128+j
                        if i<M*N:
                            total[0]=0
                            for q in T.serial(K):
                                total[0]=total[0]+T.cast(A[(i//N)*a0+q*a1],"float32")*T.cast(B[q*b0+(i%N)*b1],"float32")
                            Y[i]=total[0]
            return main
        compiled=make_mm(m,cols,k,sizes[0],sizes[1],dtype) if m and cols else None
        arguments=[*buffers,*output_buffers]
    else:
        raise UnsupportedError("Unknown TileLang example")
    def launch():
        if compiled is not None: compiled(*arguments)
        return outputs
    launch.allocation_included=False
    launch.runtime_shape_kind="specialize_per_shape"
    source=compiled.get_kernel_source() if compiled is not None and hasattr(compiled,"get_kernel_source") else None
    launch.launch_metadata={"operator":operation,"output":"preallocated","physical_storage_views_no_copy":True,
                            "artifact_id":sha256(source.encode()).hexdigest() if isinstance(source,str) else "unknown",
                            "runtime_shape":"not provided by this static entry",
                            "implementation":"simple demonstration kernels; not tuned"}
    launch()
    torch.musa.synchronize()
    return launch


def axpby_dynamic(case, inputs, workdir):
    """One compiled symbolic-N kernel; concrete tensors bind N at each call."""
    import torch
    import tilelang
    import tilelang.language as T
    from musaopst.errors import UnsupportedError
    symbolic=getattr(T,"dynamic",None) or getattr(T,"symbolic",None)
    if symbolic is None: raise UnsupportedError("Installed TileLang has no symbolic dimension API")
    N=symbolic("N")
    initial_dtype=inputs[0].dtype
    dtype=str(initial_dtype).replace("torch.","")
    alpha=float(case["attrs"].get("alpha",1.0)); beta=float(case["attrs"].get("beta",1.0))
    @tilelang.jit(out_idx=[],target="musa")
    def make():
        @T.prim_func
        def main(A:T.Tensor((N,),dtype),B:T.Tensor((N,),dtype),Y:T.Tensor((N,),dtype)):
            with T.Kernel(T.ceildiv(N,128),threads=128) as bx:
                for j in T.Parallel(128):
                    i=bx*128+j
                    if i<N:
                        Y[i]=T.cast(A[i],"float32")*alpha+T.cast(B[i],"float32")*beta
        return main
    compiled=make()
    state={}
    def bind(values, new_case):
        if len(values)!=2 or any(x.ndim!=1 or not x.is_contiguous() or x.dtype!=initial_dtype for x in values) or values[0].shape!=values[1].shape:
            raise UnsupportedError("Symbolic TileLang axpby needs two contiguous 1D tensors of the initial dtype")
        if new_case["attrs"]!=case["attrs"]: raise UnsupportedError("Symbolic TileLang entry cannot change compile-time attrs")
        state.update(inputs=values,output=torch.empty(values[0].shape,dtype=initial_dtype,device=values[0].device))
    bind(inputs,case)
    def launch():
        if state["output"].numel(): compiled(*state["inputs"],state["output"])
        return state["output"]
    launch.rebind=bind
    launch.allocation_included=False
    launch.runtime_shape_kind="compiled_symbolic_N"
    source=compiled.get_kernel_source() if hasattr(compiled,"get_kernel_source") else None
    launch.launch_metadata={"operator":"axpby","symbolic_dimension":"N","output":"preallocated",
                            "artifact_id":sha256(source.encode()).hexdigest() if isinstance(source,str) else "unknown"}
    launch(); torch.musa.synchronize()
    return launch


def axpby(case, inputs, workdir): return _factory(case, inputs, workdir, "axpby")
def sum_max(case, inputs, workdir): return _factory(case, inputs, workdir, "sum_max")
def fill(case, inputs, workdir): return _factory(case, inputs, workdir, "fill")
def identity(case, inputs, workdir): return _factory(case, inputs, workdir, "identity")
def matmul(case, inputs, workdir): return _factory(case, inputs, workdir, "matmul")
def softmax(case, inputs, workdir): return _factory(case, inputs, workdir, "softmax")
def inplace_add(case, inputs, workdir): return _factory(case, inputs, workdir, "inplace_add")


def optional_add(case, inputs, workdir):
    from musaopst.backends.gpu import _flat
    from musaopst.errors import UnsupportedError
    if len(inputs) != 2 or inputs[0] is None:
        raise UnsupportedError("optional_add expects tensor a and tensor/absent b")
    return _factory(case, list(_flat(inputs)), workdir, "identity" if inputs[1] is None else "axpby")


def tensor_list_sum(case, inputs, workdir):
    from musaopst.errors import UnsupportedError
    if len(inputs) != 1 or not isinstance(inputs[0], list) or len(inputs[0]) != 2:
        raise UnsupportedError("This TileLang TensorList example explicitly supports exactly two tensors")
    return _factory(case, inputs[0], workdir, "axpby")

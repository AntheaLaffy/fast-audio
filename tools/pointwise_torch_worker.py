#!/usr/bin/env python3
"""One persistent eager/compiled/captured pointwise worker with explicit output reuse."""
import argparse
from functools import partial
import json
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.pointwise_protocol import load_protocol, extent, storage, view, tensor_bytes, UNARY, BINARY


def respond(value):
    print(json.dumps(value,allow_nan=False),flush=True)


def compiled_writeback(name,arity):
    """Keep the caller-view store inside Inductor when direct out= tracing is unsupported."""
    if (name in UNARY and arity!=1) or (name in BINARY and arity!=2) or name not in UNARY+BINARY:
        raise ValueError("invalid compiled operation/arity")
    functional=getattr(torch.nn.functional,name) if name in ("silu","softplus") else getattr(torch,name)
    if arity==1:
        def call(source,unused,destination): return destination.copy_(functional(source))
    else:
        def call(left,right,destination): return destination.copy_(functional(left,right))
    return call


def configure_compiler():
    from torch._inductor import config
    from torch._inductor.cpp_builder import _get_ffast_math_flags,get_cpp_compiler
    config.use_fast_math=False
    config.cpp.enable_unsafe_math_opt_flag=False
    config.cpp.enable_floating_point_contract_flag="off"
    config.cuda.use_fast_math=False
    return {"policy":{"use_fast_math":config.use_fast_math,"cpp_unsafe_math":config.cpp.enable_unsafe_math_opt_flag,
                      "cpp_contract":config.cpp.enable_floating_point_contract_flag,"cuda_fast_math":config.cuda.use_fast_math},
            "cpp_math_flags":_get_ffast_math_flags(),"cpp_compiler":get_cpp_compiler()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol",type=Path,required=True)
    parser.add_argument("--platform",choices=["intel","nvidia"],required=True)
    parser.add_argument("--case",required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    p=load_protocol(args.protocol)
    spec=p["strata"][args.platform]
    device=spec["device"]
    if args.output_dir.exists(): parser.error("use a new worker directory")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    compiler=configure_compiler()
    case=next(c for c in p["cases"] if c["name"]==args.case)
    begin=time.perf_counter_ns()
    buffers=[storage(case,i).to(device) for i in range(len(case["inputs"]))]
    inputs=[view(x,g) for x,g in zip(buffers,case["inputs"])]
    output_storage=torch.full([extent(case["output"])],0x7fc12345,dtype=torch.int32,device=device).view(torch.float32)
    output=view(output_storage,case["output"])

    def sync():
        if device=="cuda": torch.cuda.synchronize()

    sync()
    payload=(time.perf_counter_ns()-begin)/1000
    args.output_dir.mkdir(parents=True)
    aten=getattr(torch.ops.aten,case["operation"]).out
    public=case["operation"] not in ("relu","silu","softplus")
    operation=getattr(torch,case["operation"]) if public else aten
    endpoint=("torch."+case["operation"]+"(out=)") if public else ("aten."+case["operation"]+".out")
    call=compiled_writeback(case["operation"],len(case["inputs"]))

    a=inputs[0]
    b=inputs[1] if len(inputs)==2 else None
    functions={}
    graphs=[]
    readiness={}
    modes=spec["modes"][1:]

    def dump(mode):
        sync()
        for i,buffer in enumerate(buffers): (args.output_dir/f"input-{i}.f32").write_bytes(tensor_bytes(buffer))
        (args.output_dir/f"{mode}-out.f32").write_bytes(tensor_bytes(output_storage))

    for mode in modes:
        output_storage.view(torch.int32).fill_(0x7fc12345)
        generated=None
        begin=time.perf_counter_ns()
        if mode=="eager":
            functions[mode]=partial(operation,a,out=output) if b is None else partial(operation,a,b,out=output)
        elif mode=="compiled":
            from torch._inductor import metrics
            metrics.reset()
            compiled=torch.compile(call,backend=p["compile"]["backend"],mode=p["compile"]["mode"],fullgraph=True,dynamic=False)
            functions[mode]=partial(compiled,a,b,output)
        else:
            original=functions["eager" if mode=="eager-cuda-graph" else "compiled"]
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(5): original()
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph): original()
            graphs.append(graph)
            functions[mode]=graph.replay
        sync()
        preparation=(time.perf_counter_ns()-begin)/1000
        begin=time.perf_counter_ns()
        functions[mode]()
        sync()
        first=(time.perf_counter_ns()-begin)/1000
        if mode=="compiled": generated=metrics.generated_kernel_count
        elif mode=="compiled-cuda-graph": generated=readiness["compiled"]["generated_kernel_count"]
        dump(mode)
        writeback=p["compile"]["writeback"] if mode in ("compiled","compiled-cuda-graph") else "out-parameter"
        actual_endpoint=("functional "+case["operation"]+" -> Tensor.copy_ (fullgraph)") if writeback=="functional-copy_" else endpoint
        readiness[mode]={"preparation_us":preparation,"first_call_us":first,"endpoint":actual_endpoint,
                         "writeback":writeback,"eager_out_schema":str(aten._schema),"generated_kernel_count":generated}
    window=spec["windows"][case["regime"]]
    respond({"status":"ready","engine":"torch","case":case,"modes":readiness,"device":device,
             "payload_prep_us":payload,"window":window,"torch":torch.__version__,"torch_config":torch.__config__.show(),
             "affinity":sorted(os.sched_getaffinity(0)),"threads":torch.get_num_threads(),"interop_threads":torch.get_num_interop_threads(),
             "cuda_version":torch.version.cuda,"compile":p["compile"],"compiler":compiler,"output_stride":list(output.stride()),
             "allocator":None if device=="cpu" else {"allocated":torch.cuda.memory_allocated(),"reserved":torch.cuda.memory_reserved()}})
    next_blocks={mode:0 for mode in modes}
    for line in sys.stdin:
        request=json.loads(line)
        if request["action"]=="quit":
            respond({"status":"stopped","measured_blocks":next_blocks})
            return
        mode=request.get("mode")
        block=request.get("block")
        if request["action"]!="measure" or mode not in functions or block!=next_blocks[mode] or block>=spec["blocks"]:
            raise ValueError("unexpected worker request or block sequence")
        run=functions[mode]
        for _ in range(window["warmup"]): run()
        sync()
        if device=="cuda":
            start=torch.cuda.Event(enable_timing=True)
            end=torch.cuda.Event(enable_timing=True)
            begin=time.perf_counter_ns()
            start.record()
            for _ in range(window["iterations"]): run()
            end.record()
            end.synchronize()
            wall=(time.perf_counter_ns()-begin)/1000/window["iterations"]
            stream_time=start.elapsed_time(end)*1000/window["iterations"]
        else:
            begin=time.perf_counter_ns()
            for _ in range(window["iterations"]): run()
            wall=(time.perf_counter_ns()-begin)/1000/window["iterations"]
            stream_time=None
        dump(mode)
        respond({"status":"measured","case":case["name"],"mode":mode,"block":block,"window":window,
                 "timing":{"wall_us":wall,"device_us":stream_time}})
        next_blocks[mode]+=1


if __name__=="__main__":
    with torch.inference_mode(): main()

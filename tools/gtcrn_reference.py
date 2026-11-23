#!/usr/bin/env python3
"""Pinned, complete GTCRN PyTorch reference and diagnostic operator inventory."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import tarfile
import time
import urllib.request

import torch

try:
    from .compare_torch import command_output, source_manifest
except ImportError:
    from compare_torch import command_output, source_manifest

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT/"models/gtcrn"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model(device, checkpoint):
    pin = json.loads((MODEL/"upstream.json").read_text())
    for name, record in pin["files"].items():
        if sha(MODEL/name) != record["sha256"]:
            raise ValueError(f"vendored upstream file changed: {name}")
    if not checkpoint.exists():
        checkpoint.parent.mkdir(parents=True,exist_ok=True)
        url = f"{pin['repository'].replace('github.com','raw.githubusercontent.com')}/{pin['revision']}/{pin['checkpoint']['path']}"
        content = urllib.request.urlopen(url,timeout=60).read()
        if hashlib.sha256(content).hexdigest() != pin["checkpoint"]["sha256"]:
            raise ValueError("downloaded checkpoint SHA-256 mismatch")
        checkpoint.write_bytes(content)
    if sha(checkpoint) != pin["checkpoint"]["sha256"]:
        raise ValueError("checkpoint SHA-256 mismatch")
    spec = importlib.util.spec_from_file_location("fast_audio_gtcrn_reference",MODEL/"gtcrn.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.GTCRN().eval()
    model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True)["model"])
    return model.to(device),pin


def fixture(batch,length):
    index = torch.arange(batch*length,dtype=torch.int64).reshape(batch,length)
    noise = ((index*37+11)%101).float()/51-1
    t = index.double()/16000
    return (0.12*torch.sin(2*torch.pi*220*t)+0.06*torch.sin(2*torch.pi*437*t)+0.04*noise).float()


def store_tensor(folder,name,value):
    value = value.detach().cpu().contiguous()
    path = folder/f"{name}.f32"
    path.write_bytes(value.numpy().tobytes())
    return {"file":path.name,"shape":list(value.shape),"sha256":sha(path)}


def inventory(model, spectrum, folder, capture_conv=False, capture_model=False):
    records=[]
    stages=[]
    handles=[]
    stage_names={"sfe","dpgrnn1","dpgrnn2",*(f"encoder.en_convs.{i}" for i in range(5)),*(f"decoder.de_convs.{i}" for i in range(5))}
    originals={}
    def stage(name):
        def capture(module,inputs,output):
            stages.append({"name":name,"output":store_tensor(folder,"stage-"+name.replace('.','-'),output)})
        return capture
    if capture_model:
        for method in ["bm","bs"]:
            original=getattr(model.erb,method);originals[method]=original
            def wrapped(x,original=original,name="erb."+method):
                output=original(x)
                stages.append({"name":name,"output":store_tensor(folder,"stage-"+name.replace('.','-'),output)})
                return output
            setattr(model.erb,method,wrapped)
    def hook(name):
        def capture(module,inputs,output):
            tensor_output = output[0] if isinstance(output,tuple) else output
            record={"name":name,"type":type(module).__name__,"input_shape":list(inputs[0].shape),
                "input_stride":list(inputs[0].stride()),"output_shape":list(tensor_output.shape)}
            if isinstance(module,torch.nn.GRU):
                key=name.replace('.','-')
                initial = inputs[1] if len(inputs)>1 and inputs[1] is not None else inputs[0].new_zeros((2 if module.bidirectional else 1,inputs[0].shape[0],module.hidden_size))
                record.update(input_size=module.input_size,hidden_size=module.hidden_size,
                    bidirectional=module.bidirectional,reset_after=True,
                    input=store_tensor(folder,key+"-input",inputs[0]),
                    initial=store_tensor(folder,key+"-initial",initial),
                    output=store_tensor(folder,key+"-output",output[0]),
                    final=store_tensor(folder,key+"-final",output[1]),
                    parameters={n:store_tensor(folder,key+"-"+n,p) for n,p in module.named_parameters()})
            if capture_conv and isinstance(module,(torch.nn.Conv2d,torch.nn.ConvTranspose2d)):
                key=name.replace('.','-')
                record.update(kernel=list(module.kernel_size),stride=list(module.stride),padding=list(module.padding),
                    dilation=list(module.dilation),groups=module.groups,transposed=isinstance(module,torch.nn.ConvTranspose2d),
                    output_padding=list(module.output_padding) if isinstance(module,torch.nn.ConvTranspose2d) else [0,0],
                    input=store_tensor(folder,key+"-input",inputs[0]),output=store_tensor(folder,key+"-output",output),
                    parameters={n:store_tensor(folder,key+"-"+n,p) for n,p in module.named_parameters()})
            records.append(record)
        return capture
    for name,module in model.named_modules():
        if not list(module.children()): handles.append(module.register_forward_hook(hook(name)))
        if capture_model and name in stage_names: handles.append(module.register_forward_hook(stage(name)))
    try:
        result=model(spectrum)
    finally:
        for handle in handles: handle.remove()
        for name,original in originals.items():setattr(model.erb,name,original)
    return result,records,stages


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    parser.add_argument("--threads",type=int,default=1)
    parser.add_argument("--warmup",type=int,default=3)
    parser.add_argument("--iterations",type=int,default=5)
    parser.add_argument("--samples",type=int,default=5)
    parser.add_argument("--capture-conv",action="store_true",help="also save all 22 convolution layers for native replay")
    parser.add_argument("--capture-model",action="store_true",help="export named FP32 weights and all 15 native-model comparison stages")
    parser.add_argument("--checkpoint",type=Path,default=ROOT/"target/models/gtcrn/model_trained_on_dns3.tar")
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists(): parser.error("use a new output directory")
    if min(args.threads,args.iterations,args.samples)<1 or args.warmup<0: parser.error("invalid timing counts")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=True
    model,pin=load_model(args.device,args.checkpoint)
    args.output_dir.mkdir(parents=True)
    report={"schema_version":1,"status":"running","model":pin,
        "protocol":{"scope":"PyTorch reference diagnostics, not a Rust performance acceptance campaign",
            "precision":"FP32; TF32 disabled","threads":args.threads,"warmup":args.warmup,
            "iterations":args.iterations,"samples":args.samples,"profile":"separate instrumented model invocation",
            "upstream_adaptations":"modern complex STFT API; explicit inverse waveform length"},
        "environment":{"timestamp_utc":time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            "platform":platform.platform(),"python":sys.version,"torch":torch.__version__,
            "torch_cuda":torch.version.cuda,"torch_cudnn":torch.backends.cudnn.version(),
            "torch_config":torch.__config__.show(),"device":args.device,
            "cpu_affinity":sorted(os.sched_getaffinity(0)),"cpu":command_output(["lscpu"]),
            "gpu":command_output(["nvidia-smi","--query-gpu=uuid,name,driver_version,memory.total","--format=csv,noheader"]),
            "source_sha256":source_manifest()},"cases":[]}
    destination=args.output_dir/"reference.json"
    archive=args.output_dir/"source.tar.gz"
    with tarfile.open(archive,"w:gz") as bundle:
        for name,digest in report["environment"]["source_sha256"].items():
            if sha(ROOT/name)!=digest: raise ValueError("source changed before archive creation")
            bundle.add(ROOT/name,arcname=name)
    report["environment"]["source_archive_sha256"]=sha(archive)
    if args.capture_model:
        weight_folder=args.output_dir/"weights";weight_folder.mkdir()
        parameters={}
        for name,value in model.state_dict().items():
            if value.dtype!=torch.float32:continue
            blob=store_tensor(weight_folder,name,value);blob["file"]="weights/"+blob["file"]
            parameters[name]=blob
        bundle=args.output_dir/"parameters.json"
        bundle.write_text(json.dumps({"format":"fast-audio-gtcrn-f32-v1","model":pin,"parameters":parameters},indent=2)+"\n")
        report["parameter_bundle"]={"file":bundle.name,"sha256":sha(bundle)}
    def save(): destination.write_text(json.dumps(report,indent=2)+"\n")
    def sync():
        if args.device=="cuda": torch.cuda.synchronize()
    window=torch.hann_window(512,periodic=True,device="cpu").sqrt().to(args.device)
    with torch.inference_mode():
        for batch,length in [(1,4096),(1,16000),(1,64000),(2,16000)]:
            folder=args.output_dir/f"b{batch}-l{length}";folder.mkdir()
            x=fixture(batch,length).to(args.device)
            stft=lambda:torch.stft(x,512,256,512,window,center=True,pad_mode="reflect",normalized=False,return_complex=True)
            def pipeline():
                z=model(torch.view_as_real(stft()))
                return torch.istft(torch.view_as_complex(z.contiguous()),512,256,512,window,center=True,length=length)
            spectrum=stft()
            enhanced,leaves,model_stages=inventory(model,torch.view_as_real(spectrum),folder,args.capture_conv,args.capture_model)
            waveform=pipeline()
            if not torch.isfinite(waveform).all(): raise RuntimeError("nonfinite reference waveform")
            tensors={name:store_tensor(folder,name,value) for name,value in
                [("input",x),("window",window),("spectrum",torch.view_as_real(spectrum)),("enhanced",enhanced),("waveform",waveform)]}
            stages={}
            for name,run in [("model",lambda:model(torch.view_as_real(spectrum))),("pipeline",pipeline)]:
                for _ in range(args.warmup): run()
                sync();samples=[]
                for _ in range(args.samples):
                    begin=time.perf_counter_ns()
                    for _ in range(args.iterations): result=run()
                    sync();samples.append((time.perf_counter_ns()-begin)/1000/args.iterations)
                stages[name]={"wall_us":samples,"median_wall_us":statistics.median(samples)}
            activities=[torch.profiler.ProfilerActivity.CPU]
            if args.device=="cuda": activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(activities=activities,record_shapes=True) as prof:
                result=model(torch.view_as_real(spectrum));sync()
            events=[{"operator":e.key,"shapes":e.input_shapes,"calls":e.count,
                "self_cpu_us":e.self_cpu_time_total,"total_cpu_us":e.cpu_time_total,
                "self_device_us":e.self_device_time_total} for e in prof.key_averages(group_by_input_shape=True)]
            report["cases"].append({"name":folder.name,"batch":batch,"input_len":length,"frames":spectrum.shape[-1],
                "tensors":tensors,"leaf_operators":leaves,"model_stages":model_stages,"timing":stages,"profile":events})
            save()
            print(folder.name,{name:round(v["median_wall_us"],2) for name,v in stages.items()},flush=True)
    report["status"]="verified_pytorch_reference";save()


if __name__=="__main__": main()

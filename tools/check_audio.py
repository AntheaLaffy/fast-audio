#!/usr/bin/env python3
"""Check complete centered spectra and inverse waveforms against eager PyTorch."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import torch

try:
    from .compare_torch import errors, source_manifest, ROOT
except ImportError:
    from compare_torch import errors, source_manifest, ROOT


def load(folder, description):
    content = bytearray((folder/description["file"]).read_bytes())
    return torch.frombuffer(content, dtype=torch.float32).reshape(description["shape"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--backend", choices=["reference", "mkl", "cuda"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("use a new output directory to preserve evidence")
    subprocess.run([str(args.binary.resolve()),args.backend,str(args.output_dir.resolve())],check=True)
    manifest = json.loads((args.output_dir/"audio.json").read_text())
    device = "cuda" if args.backend == "cuda" else "cpu"
    torch.set_num_threads(1)
    result = {"schema_version":1,"status":"running",
        "environment":{"torch":torch.__version__,"torch_cuda":torch.version.cuda,
            "binary_sha256":hashlib.sha256(args.binary.read_bytes()).hexdigest()},
        "protocol":{"backend":args.backend,"scope":"correctness only; no timing or performance promotion"},"cases":[]}
    sources = source_manifest()
    archive = args.output_dir/"source.tar.gz"
    with tarfile.open(archive,"w:gz") as bundle:
        for name in sources: bundle.add(ROOT/name,arcname=name)
    result["environment"]["source_sha256"] = sources
    result["environment"]["source_archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    # Existing measurement contract, applied to every element and both spectrum components.
    with torch.inference_mode():
        for case in manifest["cases"]:
            tensors = {name:load(args.output_dir,value) for name,value in case["files"].items()}
            n,hop,length = case["fft_size"],case["hop"],case["input_len"]
            window = torch.hann_window(n,periodic=True,dtype=torch.float64).float().to(device)
            spectrum = torch.stft(tensors["input"].to(device),n,hop_length=hop,window=window,
                center=True,pad_mode="reflect",normalized=False,onesided=True,return_complex=True)
            modified = torch.view_as_complex(tensors["modified"].contiguous()).transpose(-1,-2).to(device)
            inverse = lambda value:torch.istft(value,n,hop_length=hop,window=window,center=True,
                normalized=False,onesided=True,length=length).cpu()
            checks = {"spectrum":errors(tensors["spectrum"],torch.view_as_real(spectrum.transpose(-1,-2)).cpu()),
                "restored":errors(tensors["restored"],inverse(spectrum)),
                "processed":errors(tensors["processed"],inverse(modified)),
                "roundtrip":errors(tensors["restored"],tensors["input"])}
            result["cases"].append({"name":case["name"],"checks":checks,"inverse_workspace_bytes":case["inverse_workspace_bytes"]})
    result["status"] = "verified_audio_transforms" if all(check["passed"] for case in result["cases"] for check in case["checks"].values()) else "failed"
    (args.output_dir/"validation.json").write_text(json.dumps(result,indent=2)+"\n")
    print(f"{args.backend}: {result['status']}; {len(result['cases'])} complete-spectrum/waveform cases")
    if result["status"] == "failed": raise RuntimeError("audio transform numerical gate failed")


if __name__ == "__main__":
    main()

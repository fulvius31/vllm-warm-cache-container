#!/usr/bin/env python3
# pylint: disable=import-outside-toplevel, unused-argument
"""
vLLM / Triton environment & cache-key checker
vLLM cache generator
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

LOGGER = logging.getLogger(Path(__file__).stem)
try:
    import torch  # noqa: E402
except ImportError as exc:
    LOGGER.error("torch not installed: %s", exc)
    sys.exit(1)


MAX_LINE = 100

###############################################################################
# Helper functions
###############################################################################


def gpu_info() -> Dict[str, Any]:
    """Return a structured description of visible GPUs or a CPU-only marker."""
    if not torch.cuda.is_available():
        return {"gpu_type": "CPU-only", "gpu_count": 0}

    gpus: List[Dict[str, Any]] = []
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        entry: Dict[str, Any] = {
            "name": props.name,
            "total_memory_gb": round(props.total_memory / 2**30, 2),
        }
        # Optional vendor-specific fields
        if hasattr(props, "major"):
            entry["compute_capability"] = f"{props.major}.{props.minor}"
        if hasattr(props, "gcn_arch_name"):
            entry["architecture"] = props.gcn_arch_name
        gpus.append(entry)

    backend = "AMD ROCm" if getattr(torch.version, "hip", None) else "NVIDIA CUDA"
    return {"gpu_type": backend, "gpu_count": len(gpus), "gpus": gpus}


TRACKED_ENV: Tuple[str, ...] = (
    # vLLM
    "VLLM_CACHE_ROOT",
    "VLLM_ATTENTION_BACKEND",
    "VLLM_USE_TRITON",
    "VLLM_USE_XFORMERS",
    # Inductor
    "TORCHINDUCTOR_MAX_AUTOTUNE",
    "TORCHINDUCTOR_FORCE_DISABLE_CACHES",
    "TORCHINDUCTOR_DEBUG",
    "TORCHINDUCTOR_TRACE",
)


def collect_vllm_profile(model: str) -> Dict[str, Any]:
    """Return versions, hardware, and env-var snapshot relevant to vLLM."""
    try:
        import vllm  # noqa: E402
    except ImportError as exc:
        LOGGER.error("vLLM not installed: %s", exc)
        sys.exit(1)

    try:
        import triton  # noqa: E402
    except ImportError as exc:
        LOGGER.error("Triton not installed: %s", exc)
        sys.exit(1)

    versions: Dict[str, str] = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "torch_version": torch.__version__,
        "vllm_version": vllm.__version__,
        "triton_version": triton.__version__,
    }
    if torch.cuda.is_available() and torch.version.cuda:
        versions["cuda_version_torch"] = torch.version.cuda
    if getattr(torch.version, "hip", None):
        versions["rocm_version_torch"] = torch.version.hip
    try:
        versions["cudnn_version"] = str(torch.backends.cudnn.version())
    except (AttributeError, RuntimeError):
        versions["cudnn_version"] = "n/a"

    env_vars = {k: os.environ[k] for k in TRACKED_ENV if k in os.environ}

    return {
        "versions": versions,
        "hardware": gpu_info(),
        "environment_variables": env_vars,
        "vllm_engine_configuration": {
            "model": model,
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "max_model_len": 8192,
        },
    }


def _blk(title: str, mapping: Dict[str, Any]) -> str:
    """Pretty, aligned key→value block with stable ordering and multi-line values."""
    if not mapping:
        return f"{title}\n  <none>"

    items = sorted(mapping.items(), key=lambda kv: kv[0])
    pad = max(len(k) for k, _ in items)

    lines: List[str] = [title]
    for k, v in items:
        val_str = (
            json.dumps(v, separators=(",", ":"))
            if isinstance(v, (dict, list))
            else str(v)
        )
        first, *rest = val_str.splitlines()
        lines.append(f"  {k:<{pad}} : {first}")
        for line in rest:
            lines.append(" " * (pad + 5) + line)
    return "\n".join(lines)


###############################################################################
# Triton helpers
###############################################################################


def _lazy_triton_import():
    """Return the `triton` module or None (already logged) if missing."""
    try:
        import triton  # noqa: E402

        return triton
    except ImportError as exc:
        LOGGER.error("Triton required for cache-key generation: %s", exc)
        return None


def _triton_invalid_env(_triton) -> Dict[str, str]:
    """Return Triton env-vars that invalidate its kernel cache."""
    try:
        from triton._C.libtriton import get_cache_invalidating_env_vars  # noqa: E402

        return get_cache_invalidating_env_vars()
    except (ImportError, AttributeError) as exc:
        LOGGER.debug("Triton env-invalidators unavailable: %s", exc)
        return {}


def _triton_target() -> str:
    """Return backend-arch-warp descriptor of the active Triton target."""
    try:
        from triton.runtime.driver import driver  # noqa: E402

        tgt = driver.active.get_current_target()
        return f"{tgt.backend}-{tgt.arch}-{tgt.warp_size}"
    except (ImportError, RuntimeError) as exc:
        LOGGER.warning("Cannot query Triton target: %s", exc)
        return "unknown-backend-0-0"


def _triton_extern_libs() -> str:
    """Return a JSON string describing external libs linked into Triton kernels."""
    from triton.runtime.jit import JITFunction  # noqa: E402
    from triton.runtime.driver import driver  # noqa: E402

    kernels: List[JITFunction] = []
    for module in list(sys.modules.values()):
        if module is None:
            continue
        kernels.extend(
            val for val in module.__dict__.values() if isinstance(val, JITFunction)
        )

    if not kernels:
        return "{}"

    device = driver.active.get_current_device()
    cache = kernels[0].device_caches[device][2]
    opts = cache.parse_options({})
    return json.dumps(dict(opts.extern_libs), sort_keys=True, separators=(",", ":"))


def generate_triton_components(
    src_hash: str | None = None,
) -> Tuple[str, Dict[str, Any]]:
    """Return (cache_key, detailed_component_dict) for Triton’s kernel cache."""
    triton = _lazy_triton_import()
    if triton is None:
        return "n/a", {"note": "Triton not available; cache-key omitted"}

    comp: Dict[str, Any] = {}

    # noinspection PyProtectedMember
    comp["triton_key"] = triton.compiler.compiler.triton_key()  # type: ignore[attr-defined]

    src_digest = hashlib.sha256((src_hash or "dummy_source").encode()).hexdigest()
    comp["source"] = {"hash": src_digest}

    backend = _triton_target()
    backend_digest = hashlib.sha256(backend.encode()).hexdigest()
    comp["backend"] = {"info": backend, "hash": backend_digest}

    opts = {
        "debug": os.getenv("TRITON_DEBUG", "0") == "1",
        "extern_libs": _triton_extern_libs(),
    }
    opts_digest = hashlib.sha256(
        json.dumps(opts, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    comp["options"] = {"values": opts, "hash": opts_digest}

    env = _triton_invalid_env(triton)
    env_digest = hashlib.sha256(
        json.dumps(env, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    comp["environment"] = {"variables": env, "hash": env_digest}

    composite = "-".join(
        (
            comp["triton_key"],
            src_digest,
            backend_digest,
            opts_digest,
            env_digest,
        )
    )
    cache_key = hashlib.sha256(composite.encode()).hexdigest()
    comp["final_composite"] = composite
    LOGGER.debug("Triton final hash: %s", cache_key)
    return cache_key, comp


###############################################################################
# Hash helpers
###############################################################################


def vllm_hash(vllm_prof: Dict[str, Any]) -> str:
    """Return SHA-256 of the canonicalised vLLM profile."""
    blob = json.dumps(vllm_prof, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def global_hash(vllm_prof: Dict[str, Any], triton_comp: Dict[str, Any]) -> str:
    """Return a digest that flips if *any* vLLM or Triton determinant changes."""
    blob = json.dumps(
        vllm_prof, sort_keys=True, separators=(",", ":")
    ) + triton_comp.get("final_composite", "")
    return hashlib.sha256(blob.encode()).hexdigest()


###############################################################################
# CLI & I/O helpers
###############################################################################


def parse_cli() -> argparse.Namespace:
    """Build and parse the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Dump vLLM & Triton cache changers; optionally save JSON."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging")
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="Model name or path (e.g. ibm/granite-7B)",
    )
    return parser.parse_args()


def run_demo_generation(model: str) -> None:
    """Load a model with vLLM and run a tiny prompt set to populate caches."""
    try:
        from vllm import LLM, SamplingParams  # noqa: E402
    except ImportError as exc:
        LOGGER.error("vLLM not installed: %s", exc)
        sys.exit(1)

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    llm = LLM(model=model)
    outputs = llm.generate(prompts, SamplingParams(temperature=0.8, top_p=0.95))

    border = "-" * MAX_LINE
    LOGGER.info("\nGenerated Outputs:\n%s", border)
    for out in outputs:
        LOGGER.info("Prompt  : %r", out.prompt)
        LOGGER.info("Output  : %r", out.outputs[0].text)
        LOGGER.info(border)


def dump_summary(
    vllm_prof: Dict[str, Any], triton_comp: Dict[str, Any], triton_key: str
) -> None:
    """Print a human-readable summary of collected information."""
    sep = "=" * MAX_LINE
    LOGGER.info(sep)
    LOGGER.info("vLLM / Torch-Inductor Environment")
    LOGGER.info(sep)
    LOGGER.info(_blk("\n▶ Versions", vllm_prof["versions"]))
    LOGGER.info(_blk("\n▶ Hardware", vllm_prof["hardware"]))
    LOGGER.info(_blk("\n▶ Environment Variables", vllm_prof["environment_variables"]))
    LOGGER.info(sep)

    LOGGER.info("\nTriton Cache-Key")
    LOGGER.info("-" * MAX_LINE)
    if "note" in triton_comp:
        LOGGER.info(triton_comp["note"])
    else:
        LOGGER.info("cache_key : %s", triton_key)
    LOGGER.info("-" * MAX_LINE)


###############################################################################
# Main entry-point
###############################################################################


def main() -> None:
    """Script entry: parse CLI, collect data, run demo, emit results."""
    args = parse_cli()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if not torch.cuda.is_available():
        LOGGER.warning("CUDA/ROCm device NOT detected – proceeding on CPU.")

    vllm_prof = collect_vllm_profile(args.model)
    triton_key, triton_comp = generate_triton_components()

    result = {
        "vllm_profile": vllm_prof,
        "triton_components": triton_comp,
        "hashes": {
            "vllm_hash": vllm_hash(vllm_prof),
            "triton_cache_key": triton_key,
            "global_cache_hash": global_hash(vllm_prof, triton_comp),
        },
    }

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))

        cache_root = Path(os.getenv("VLLM_ROOT_CACHE", Path.home() / ".cache" / "vllm"))
        cache_root.mkdir(parents=True, exist_ok=True)
        file_path = cache_root / "metadata.json"
        file_path.write_text(json.dumps(result, indent=2, sort_keys=True))
        LOGGER.info("Successfully wrote JSON to: %s", file_path)
    else:
        dump_summary(vllm_prof, triton_comp, triton_key)
        LOGGER.info(
            "\nGLOBAL CACHE HASH\n%s\n%s",
            "=" * MAX_LINE,
            result["hashes"]["global_cache_hash"],
        )
        LOGGER.info("%s", "=" * MAX_LINE)

    run_demo_generation(args.model)


if __name__ == "__main__":
    main()

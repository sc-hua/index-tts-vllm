import inspect

from vllm.v1.worker.gpu_model_runner import GPUModelRunner

_PATCH_MODULE_NAME = (
    "patch_vllm_latest"
    if len(inspect.signature(GPUModelRunner._prepare_inputs).parameters) >= 4
    else "patch_vllm"
)

if _PATCH_MODULE_NAME == "patch_vllm_latest":
    from patch_vllm_latest import *  # noqa: F401,F403
else:
    from patch_vllm import *  # noqa: F401,F403

ACTIVE_PATCH_MODULE = _PATCH_MODULE_NAME
print(f"✅  Auto-selected {_PATCH_MODULE_NAME} for GPUModelRunner patching")

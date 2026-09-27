"""BSAI-ComfyUI-VRAM-Unlock — 统一解除 PyTorch 显存 fraction 限制。

为什么需要
----------
部分 ComfyUI 插件会在运行时调用 ``torch.cuda.set_per_process_memory_fraction(<1.0)``，
把 PyTorch 可用显存硬性限制到总显存的 60~70%（例如 24GB 卡被限到 16GiB），
导致大模型（MiniMax-H3 int8 19.5GB + 激活）直接 CUDA OOM。

已知触发源（无需逐个改它们）：
  * ComfyUI-QwenVL-Mod  ``AILab_QwenVL.set_pytorch_memory_fraction()``
  * RES4LYF  ``rk_sampler_beta.py``
  * 任何未来插件的新增调用

本插件做什么
------------
在 ComfyUI 加载 custom_nodes 的 import 阶段统一拦截
``torch.cuda.set_per_process_memory_fraction``：

  * 任何插件传入 < 1.0 的 fraction 一律强制为 1.0（用满物理显存）；
  * ``get_per_process_memory_fraction`` 统一返回 1.0，避免插件读到旧限制值；
  * 被拦截时打一条 warning 日志，标明触发来源调用值。

从此不需要在每个插件里各自加“检测 + 解除”代码——一个点，统一生效。

注意
----
强制 1.0 意味着本实例会用满物理显存；请勿与其它重型 GPU 进程
（其它 ComfyUI 实例 / 游戏 / 浏览器 GPU 加速）同时跑大模型。
"""
import logging

import torch

_LOG = logging.getLogger("BSAI.VRAMUnlock")

_ORIG_SET_FRACTION = torch.cuda.set_per_process_memory_fraction
_ORIG_GET_FRACTION = torch.cuda.get_per_process_memory_fraction


def _bsai_unlocked_set_fraction(fraction, device=None):
    """统一解除：拦截一切 <1.0 的 fraction 设置，强制 1.0。"""
    if fraction < 1.0:
        _LOG.warning(
            "intercepted set_per_process_memory_fraction(%.3f) from a plugin; "
            "forcing 1.0 (BSAI VRAM-Unlock unified policy)", float(fraction))
        fraction = 1.0
    return _ORIG_SET_FRACTION(fraction, device)


def _bsai_unlocked_get_fraction(device=None):
    """get 统一返回 1.0，避免插件读到旧的限制值。"""
    return 1.0


torch.cuda.set_per_process_memory_fraction = _bsai_unlocked_set_fraction
torch.cuda.get_per_process_memory_fraction = _bsai_unlocked_get_fraction

_LOG.info(
    "BSAI VRAM-Unlock active: torch.cuda memory fraction forced to 1.0; "
    "all plugin set_per_process_memory_fraction(<1.0) calls intercepted.")


class BSAIVRAMUnlockInfo:
    """只读诊断节点：显示统一解除状态（可拖入工作流查看）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "status"
    CATEGORY = "BSAI/VRAM"

    def status(self):
        return (f"VRAM-Unlock active: torch.cuda.set_per_process_memory_fraction "
                f"forced to 1.0; plugin sub-1.0 fraction calls intercepted.",)


NODE_CLASS_MAPPINGS = {"BSAIVRAMUnlockInfo": BSAIVRAMUnlockInfo}
NODE_DISPLAY_NAME_MAPPINGS = {"BSAIVRAMUnlockInfo": "BSAI VRAM Unlock (info)"}

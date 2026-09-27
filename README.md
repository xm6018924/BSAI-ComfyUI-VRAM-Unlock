# BSAI-ComfyUI-VRAM-Unlock

统一解除 ComfyUI 插件对 PyTorch 显存 fraction（`set_per_process_memory_fraction`）的限制。

## 问题

部分插件在运行时把 PyTorch 可用显存限制到总显存的 60~70%，例如：

- `ComfyUI-QwenVL-Mod` → `AILab_QwenVL.set_pytorch_memory_fraction()` 把 24GB 卡限到 **16GiB**
- `RES4LYF` → `rk_sampler_beta.py` 按显存上限计算 fraction

限制生效后，`MiniMax-H3` int8（19.5GB）+ 激活直接 CUDA OOM，报错形如：
`PyTorch limit (user-supplied memory fraction) 17179869184 (16GiB)`。

## 方案

不逐个改插件（插件更新会覆盖、且治标不治本）。本插件在 ComfyUI 加载时**统一拦截**
`torch.cuda.set_per_process_memory_fraction`：

- 任何插件传入 `< 1.0` 的 fraction → 强制 `1.0`（用满物理显存）
- `get_per_process_memory_fraction` → 统一返回 `1.0`
- 被拦截时打印 warning 日志（含触发值）

一个点，所有插件（现有 + 未来）统一生效。

## 安装

把整个 `BSAI-ComfyUI-VRAM-Unlock` 目录放入 ComfyUI 的 `custom_nodes/`，重启 ComfyUI 即可。
无需安装依赖（仅用 torch）。

## 验证

启动日志应出现：

```
BSAI VRAM-Unlock active: torch.cuda memory fraction forced to 1.0; ...
```

若某插件尝试设置 fraction < 1.0，日志出现：

```
intercepted set_per_process_memory_fraction(0.700) from a plugin; forcing 1.0 ...
```

## 注意

强制 1.0 = 本实例用满物理显存。**不要**与其它重型 GPU 进程（其它 ComfyUI 实例、
游戏、浏览器 GPU 加速）同时跑大模型，否则物理显存争抢仍会 OOM/驱动崩溃。

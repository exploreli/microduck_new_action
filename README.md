源自：https://github.com/pollen-robotics/microduck_rl

依据microduck_rl项目添加新的动作

## Quickstart

Requires a CUDA GPU (training runs through MuJoCo Warp) and [uv](https://docs.astral.sh/uv/).

> **On ARM boxes (DGX Spark / GB10, Jetson):** `uv sync` pulls ~2 GB of CUDA
> wheels on first run and uv's default 30 s HTTP timeout can abort mid-download.
> Export `UV_HTTP_TIMEOUT=600` for the first sync. 

```bash
git clone https://github.com/exploreli/microduck_new_action
cd microduck_new_action

# ① 冒烟测试——正式训练前必跑（64 envs / 5 iters，几毛钱捕获 ~95% 配置错误）
uv run train Mjlab-Backflip-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5

# ② 正式训练
uv run train Mjlab-Backflip-Flat-MicroDuck --env.scene.num-envs 4096

# ③ 断点续训
uv run train Mjlab-Backflip-Flat-MicroDuck --agent.load-checkpoint model_XXXX.pt --agent.resume True

# ④ 提交到 Hugging Face Jobs（远程）
uv run train Mjlab-Backflip-Flat-MicroDuck --env.scene.num-envs 4096 --hf-jobs
```


## 四、测试模型方式
** pt模型验证**
```bash
.venv/bin/play Mjlab-Backflip-Flat-MicroDuck \
  --checkpoint-file logs/rsl_rl/microduck_backflip/<run_id>/model_10000.pt
```

** 录视频 **

```bash
.venv/bin/python scripts/infer_policy.py \
  --standing logs/backflip.onnx \
  --new-cmd-obs \
  --headless \
  --output-video logs/backflip_demo.mp4 \
  --duration 10
## 如果环境中没有X11,可以前面加 MUJOCO_GL=egl 或者 MUJOCO_GL=osmesa
```

** 配置回归测试（CPU，无需 GPU/sim）**

```bash
uv run --with pytest pytest tests/test_backflip_cfg.py -q
```

** 查看策略效果（play）**

```bash
uv run play Mjlab-Backflip-Flat-MicroDuck --wandb-run-path <entity/project/run_id>
```

** 导出 ONNX（normalizer 必须烘焙，唯一正确路径）**

```bash
# 标准：从 wandb run
uv run scripts/export.py Mjlab-Backflip-Flat-MicroDuck --wandb-run-path <...> --checkpoint N

# 本机：从本地 checkpoint（绕过 wandb）
.venv/bin/python scripts/export.py Mjlab-Backflip-Flat-MicroDuck \
  --checkpoint-file logs/rsl_rl/microduck_backflip/<run>/model_4.pt \
  --onnx-file out.onnx
```

**⑤ CPU 部署彩排（上真机前）**

```bash
uv run scripts/infer_policy.py --bachflip out.onnx      # BAM M6 执行器（与训练一致）
uv run scripts/infer_policy.py --walking out.onnx --no-bam   # XML PD
```

**⑥ 发布到 HF Hub（供 runtime `robotctl policy add` 加载）**

```bash
uv run publish --task Mjlab-Backflip-Flat-MicroDuck --wandb-run-path <...> \
  --checkpoint N --repo <user>/microduck-backflip --kind episodic --duration-s 4.0
```

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.
3D model files are licensed under Creative Commons BY-SA-NC.

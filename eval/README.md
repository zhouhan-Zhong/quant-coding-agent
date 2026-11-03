# 在 GPU 服务器上本地评测（自建 House 模型 + 官方检查器）

目的：不占用 CodaBench 每天一次的提交，在自己的服务器上跑完 86 道公开题，拿到**逐题**对错和失败原因，
再据此改 agent。流程：vLLM 起一个和比赛同款的模型服务 → 用我们的镜像逐题解题（与平台相同的容器限制）→
用每道题自带的官方检查器打分。

## 0. 前提

- NVIDIA GPU：FP8 版官方最低要求 2×H100-80GB 或 1×B200。**1 张 H200（141 GB）偏紧**，有 2 张就用 2 张。
- 磁盘：模型权重约 120+ GB，另留 30 GB 给镜像和结果。
- Docker（跑 agent 容器和检查器）、Python 3.10+（跑 vLLM）、git。

## 1. 起模型服务（vLLM）

```bash
python -m venv ~/vllm-env && source ~/vllm-env/bin/activate
pip install "vllm>=0.17.1"                       # Nemotron 3 Super 需要 0.17.1 及以上
huggingface-cli download nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8   # 约 120+ GB
bash eval/serve_vllm.sh                          # 默认：有 ≥2 张卡就 TP=2，上下文 65536，端口 8000
```
- 只有 1 张 H200 时可试：`GPUS=1 MAXLEN=32768 GPU_UTIL=0.95 EXTRA="--enforce-eager" bash eval/serve_vllm.sh`；
  仍然 OOM 就只能用 2 张卡。
- 自测（应返回一段正常回答，没有 `<think>` 推理）：
```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "house", "max_tokens": 200, "temperature": 0,
  "chat_template_kwargs": {"enable_thinking": false},
  "messages": [{"role": "user", "content": "Write a one-line Python expression for the mean of [1,2,3]."}]}'
```

## 2. 准备题库和镜像

```bash
git clone https://github.com/Agenthon-2026/track1-coding-public.git kit
# 先构建主办方的基础镜像，再从本仓库构建 agent 镜像
docker build -t finance-bench-sandbox:latest -f kit/docker/sandbox.Dockerfile kit
docker build -t agent:dev .            # 精简版：docker build -f Dockerfile.lite -t agent:lite .
```

## 3. 跑评测

```bash
JOBS=3 bash eval/local_eval.sh agent:dev kit results-v2     # 全部 86 题
UNITS="t1-merton-jump-diffusion t1-momentum-backtest" bash eval/local_eval.sh agent:dev kit results-try   # 只跑几题
AGENT_ENV="-e AGENT_MEMORY=0" bash eval/local_eval.sh agent:dev kit results-nomem       # A/B 开关
```
- 每题：拷一份**不含 checks/** 的题目作为 /input（平台就是这样），容器以 uid 65534、只读根目录、64 MiB /tmp、
  16 CPU 配额运行，模型地址 `MODEL_ENDPOINT`（默认 http://127.0.0.1:8000）；之后在同一镜像里以 root 跑该题的
  `checks/test_outputs.py`，pytest 退出码 0 才算通过。
- 输出：`results-*/results.csv`（unit, pass, agent_exit, seconds, requests, checker_summary, output_files）和
  `results-*/logs/<unit>.agent.log|.check.log`。最后一行打印总通过数和平均用时。
- 时间：每题大约 2–8 分钟，JOBS=3 时 86 题约 2–3 小时。容器写出的文件属于 uid 65534，删除结果目录时可能需要 sudo。

## 4. 把结果带回来分析

```bash
tar czf results-v2.tgz results-v2/results.csv results-v2/logs
```
把 `results-v2.tgz` 放进 `E:\agenthon`（或直接发到对话里），就能逐题分析失败原因、决定 v3 先改什么。

## 注意

- 本地模型是公开的 FP8 检查点，和比赛用的快照、服务端设置不一定完全一致，分数用于**改动前后对比**，
  不要当作绝对分数；单次运行有约 ±2–3 题的波动，A/B 最好各跑 2–3 次。
- 自检：GitHub Actions 里的 `eval-selftest` 用模拟模型跑 5 道题，结果应为 `passed 4/5`。

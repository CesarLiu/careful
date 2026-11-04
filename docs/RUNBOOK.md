# 服务器 runbook：合并后的 `careful` 仓库

这份 runbook 用来在 GPU 服务器上确认：合并（catspond `gpudrive-ppo` + catk4test）没有改变已有结果，新的预测器接口在真实检查点和真实场景上能用。全部做完约半天。

- 原来的长实验命令还在各自的 runbook 里，在新仓库里同样可用：
  - DenseTNT / MetaDrive / RL：`responsibility/RUNBOOK_H200.md`
  - SMART：`responsibility/CATK_H200.md`、`docs/smart/RUNBOOK.md`（脚本路径已从 `scripts/responsibility/` 改成 `scripts/smart/`）
  - GPUDrive：`gpucat/PLAN.md`
- 本文只写新增的核对步骤。

带 ⏱ 的时间是按 `SMART_PLAN.md` S3 的吞吐量（500 个场景约 10 小时，每次模型调用 32 份场景副本）推算的估计，没有实测过。

## 0. 代码、数据和检查点

```bash
git clone git@github.com:CesarLiu/careful.git ~/careful
cd ~/careful
```

不在 git 里的文件，从原来的 catspond 目录链接过来（假设它在 `~/catspond`，CAT-K 在 `~/catk`）：

```bash
ln -s ~/catspond/raw_scenes_500 raw_scenes_500
ln -s ~/catspond/advgen/pretrained advgen/pretrained   # densetnt.bin
ln -s ~/catspond/metadrive metadrive                   # 只有 MetaDrive 闭环、UniTraj 的输入构造需要
export CATK_ROOT=~/catk                                # 带 src/smart、configs/、ckpts/ 的 CAT-K checkout
export UNITRAJ_ROOT=~/UniTraj                          # 只有 MTR / UniTraj 需要
```

确认：

```bash
ls raw_scenes_500 | wc -l                  # 500
ls -la advgen/pretrained/densetnt.bin      # ~125 MB
ls $CATK_ROOT/ckpts/                       # clsft_E9.ckpt、pre_bc_E31.ckpt
ls $CATK_ROOT/src/smart/tokens/*.pkl       # 3 个 token 文件（真实文件，不是 LFS 指针）
```

## 1. 环境：沿用原来的两个

新仓库不需要新环境：

| 环境 | 用于 | 来源 |
|---|---|---|
| `~/venvs/cat39`（Python 3.9） | DenseTNT、`scripts/responsibility/*`、MetaDrive TD3 | catspond 的 `scripts/responsibility/setup_env.sh` |
| `~/venvs/catk`（Python 3.11） | SMART、`scripts/smart/*`、`--model smart` | CAT-K 的 `install/setup_server.sh` |

`advgen/utils_cython.cpython-39-*.so` 已在 git 里。在别的 Python 版本下用 DenseTNT，要先运行 `python scripts/responsibility/build_cython.py`。

## 2. 单元测试（约 2 分钟）

```bash
source ~/venvs/cat39/bin/activate && python -m pytest tests -q
```

cat39 里没有 omegaconf 时，`tests/smart` 整个目录会被跳过（`tests/smart/conftest.py`），不会报错。它在 catk 环境里跑：

```bash
source ~/venvs/catk/bin/activate && python -m pytest tests/smart tests/responsibility/test_predictors.py -q
```

本机（Python 3.11，torch 2.4.1 CPU，PyG 2.6.1，TensorFlow 2.15）上，`pytest tests` 的结果是 450 passed、3 skipped，跳过的那 3 个需要 UniTraj、WOMD protos 或 MetaDrive；去掉 omegaconf 时是 241 passed、3 skipped（`tests/smart` 被跳过）。

## 3. DenseTNT：结果不变（约 20 分钟，GPU）

### 3a. 复现 CAT 的输入和候选

```bash
source ~/venvs/cat39/bin/activate
python -m scripts.responsibility.verify_densetnt --n 5 --device cuda
```

验收：末尾是 `ALL CHECKS PASSED`，和合并前一样。合并前在 3 个场景上，32 条对手轨迹的误差是 3.6e-12 m。

### 3b. 逐窗口复现旧运行

用 `RUNBOOK_H200.md` 第 3 步的运行（`run_h200.sh` 的默认设置）。新仓库重算前 5 个场景，和旧的 `windows.csv` 逐窗口比较：

```bash
for a in sdc adv; do
  python -m scripts.responsibility.regress_run prepare --old ~/catspond/logs/responsibility/$a --new logs/regress/$a
  python -m scripts.responsibility.compute_responsibility --scenes raw_scenes_500 --n 5 --agent $a \
      --out-dir logs/regress/$a --stride 5 --n-samples 40 --device cuda
  python -m scripts.responsibility.regress_run compare --old ~/catspond/logs/responsibility/$a --new logs/regress/$a
done
```

- 旧运行如果用了别的 `EXTRA` 标志（例如 `--use-ooi --motion-set nms --valid-counterfactuals`），这里要原样加上。`prepare` 会打印旧运行的设置。
- `compute_responsibility` 报 "holds results computed with other settings"，说明标志和旧运行不一致，或者新代码改变了设置的含义。后者必须查清楚。
- 验收：两个角色都是 `IDENTICAL`。DenseTNT 的随机流按场景播种，所以应该逐位相同，误差在 csv 的舍入（1e-5 m、1e-6 nats）以内。

## 4. SMART 适配器：真实检查点（约 10 分钟，GPU）

```bash
source ~/venvs/catk/bin/activate
python -m scripts.smart.verify_adapter --ckpt $CATK_ROOT/ckpts/clsft_E9.ckpt --scenes raw_scenes_500 --n 5 --device cuda
```

验收：每个场景 7 项全部 `PASS`，末尾是 `ALL CHECKS PASSED`。

| 检查 | 内容 | 本机（随机权重、合成场景）的值 |
|---|---|---|
| 0 | SMART 看得到地图（CAT-K 的预处理会丢掉点距大于 3 m 的折线） | 80 个地图 token |
| 1 | 批里正好是第 k 步可见的 agent | 3 / 3 |
| 2 | 样本的第一个点离日志位置不超过 3 m | 1.16 m |
| 3 | 同一个 generator 得到同样的样本 | 通过 |
| 4 | 去掉一个邻车会改变自由 agent 的 token 概率 | −7.606 → −7.628 |
| 5 | 推演在日志结束处截断 | k = 10 时 80 步，k = 20 时 70 步 |
| 6 | 指标能算出有限值 | 通过 |

训练好的检查点在检查 2 上应该更小（一般是 1 m 以内），检查 4 的平均 log p 应该远高于 log(1/2048) = −7.62。

## 5. SMART：通用适配器 vs CAT-K 原生管线 ⏱ 约 1–3 小时（GPU）

这一步回答两个问题：
- 合并后的通用路径（`--model smart`）和 catk4test 原来的原生管线，是否给出同样的 β？
- `SMART_PLAN.md` 里"SMART 给 SDC 的 β_c 中位数 0.54 nats，给对手的只有 0.06"这个异常，来自哪里？

```bash
source ~/venvs/catk/bin/activate
python -m scripts.smart.compare_adapter_native --ckpt $CATK_ROOT/ckpts/clsft_E9.ckpt \
    --scenes raw_scenes_500 --n 20 --agent sdc adv --device cuda \
    --out-dir logs/smart/compare_adapter_native 2>&1 | tee logs/smart/compare_adapter_native.log
```

- 中途断了重跑同一条命令即可续跑，已完成的场景会跳过。
- 先用 `--n 2` 量一下单个场景的耗时，再决定 `--n`。

**怎么比：**
- 每个窗口用原生管线自己选的邻车，两条路径算同一组（agent，邻车）的 β_s、β_c。
- 设置都取原生的：40 / 20 个样本，CVaR α 0.1，D_g 在 20 m 饱和，2 s，Boltzmann，`hidden`。
- 每条路径用两个种子各跑一次，种子间的差异就是蒙特卡洛噪声。两条路径的差异不超过这个噪声，才算一致。

**产出：** `summary.md`、`pairs.csv`、`windows.csv`、`config.json`。

**验收：** `summary.md` 末尾是 `RESULT: safety CONSISTENT, courtesy CONSISTENT`。判定条件是两条都满足：
- 两条路径之间的中位绝对差，不超过种子噪声的 1.5 倍；
- 两条路径之间的 Spearman，不比种子噪声的 Spearman 低 0.1 以上。

本机用随机权重 SMART 在 3 个合成场景上（78 对）的结果：

| 指标 | 适配器 vs 原生 Spearman | 种子噪声 Spearman | 适配器 vs 原生中位绝对差 | 种子噪声中位绝对差 | 判定 |
|---|---|---|---|---|---|
| safety | 0.960 | 0.951–0.970 | 0.241 m | 0.208–0.280 m | CONSISTENT |
| courtesy | 0.875 | 0.893–0.904 | 0.170 nats | 0.145–0.173 nats | CONSISTENT |

**SDC 的 β_c 异常：** 看 `summary.md` 的 "per role" 表。

| 看到的情况 | 含义 | 下一步 |
|---|---|---|
| 原生 β_c（全部邻车）远高于原生 β_c（只算车辆），而且只在 sdc 上 | 原生管线对行人、骑车人也算 courtesy（通用管线只算车辆），SDC 身边这类邻车多 | 原生结果改成只对车辆计算，或在论文里分开报告 |
| 原生和适配器的 sdc β_c 都高 | SMART 本身对 SDC 的反应不同 | 查 SMART 对 SDC 的条件（`role` 特征）；与合并无关 |
| 原生的 sdc β_c 高，适配器的不高，车辆过滤后也还高 | 来自两条路径的实现差异：场景构建、Boltzmann 打分的 agent 范围或地图点 | 加 `--adapter-context native` 再跑一次（见下文） |
| "distributions" 表里 courtesy 的最大值是几十上百 nats | 某条路径的道路偏离代价把 b 的方案拉开了几十 nats，最常见的原因是 SMART 的地图被预处理丢了 | 日志里搜 `SMART sees no map` |

**结果是 DIFFERENT 时，逐项排除：**
- `--adapter-context native`：适配器也用原生的 1.0 s 场景（1.0 s 时可见的 agent、1.0 s 的信号灯），排除场景构建的差别。
- `--neighbor-future logged`：换成 SMART_PLAN S3 用过的另一种设置，看差别是否只出现在 `hidden` 下。
- 看 `pairs.csv` 里差别集中在哪些场景或窗口。

## 6. （可选）预测模型依赖性：多个预测器，同一套指标 ⏱

评估报告（`PUBLICATION_ASSESSMENT.md`）把"β 的阈值依赖预测模型"写成局限。下面的运行可以把它变成一节有数据的分析：
- 用不懂交互的运动学下界说明学习模型贡献了多少；
- 用通用路径的 SMART 和 DenseTNT 在完全相同的代码下对照。

```bash
# 运动学下界：不需要 GPU 和模型，500 个场景约 1 小时（估计）
source ~/venvs/cat39/bin/activate
MODEL=constant_velocity OUT=logs/responsibility/cv AGENTS="sdc adv" SHARDS=16 bash scripts/responsibility/run_h200.sh

# SMART 走通用路径，设置与 DenseTNT 的主实验相同（第 3 步）
source ~/venvs/catk/bin/activate
MODEL=smart CHECKPOINT=$CATK_ROOT/ckpts/clsft_E9.ckpt OUT=logs/responsibility/smart AGENTS="sdc adv" SHARDS=8 \
    bash scripts/responsibility/run_h200.sh

# 两两对照
source ~/venvs/cat39/bin/activate
for m in cv smart; do for a in sdc adv; do
  python -m scripts.responsibility.compare_models --runs logs/responsibility/$a logs/responsibility/$m/$a \
      --labels DenseTNT $m --out-dir logs/compare/densetnt_vs_${m}_$a
done; done
```

- `run_h200.sh` 里的 `python` 是当前激活的环境，所以 SMART 那一行要在 catk 环境里运行。
- SMART 走通用路径时每个邻车单独调用模型，比原生管线的打包推演慢。先用 `N=5` 量一下速度。

UniTraj 的任一方法（有检查点时）：

```bash
MODEL=unitraj CHECKPOINT=ckpts/wayformer.ckpt EXTRA="--unitraj-method wayformer" OUT=logs/responsibility/wayformer \
    bash scripts/responsibility/run_h200.sh
```

## 7. 回传的结果

把下面这些发回来，我据此更新 `SMART_PLAN.md`、`RESEARCH_PLAN.md` 的状态块和 README 里的实测数字：

- 第 3a、3b 步的末尾几行（`ALL CHECKS PASSED`，两个 `IDENTICAL`）；
- 第 4 步的完整输出；
- 第 5 步的 `logs/smart/compare_adapter_native/summary.md`；
- 第 6 步做了的话，`logs/compare/*/` 里的 summary。

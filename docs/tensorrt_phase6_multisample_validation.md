# TensorRT Phase 6：Strict FP32 全测试集验证

更新时间：2026-07-17（Asia/Shanghai）

## 1. 结论

固定 test split 的 18 个样本均完成 PyTorch CUDA 与 TensorRT Strict FP32 推理。
TensorRT Runtime、VoxelUnique Plugin、输出有限性和分类结果全部正常，但 5 个样本的
logits 最大绝对误差超过本阶段逐样本严格条件 `max_abs_error < 1e-4`：

```text
total_samples=18
passed_samples=13
TensorRT ErrorRecorder errors=0
all sample label agreement=100%
aggregate mIoU delta=0
aggregate weld seam F1 delta=0
worst_case_sample=weld_14
worst_max_abs_error=1.230239868164e-04

TENSORRT_STRICT_FP32_MULTISAMPLE_VALIDATION_FAILED
```

失败状态只表示未满足预先规定的逐样本 logits 最大误差条件。没有调整容差，也没有修改
Engine、ONNX、Plugin、checkpoint、数据或图。

## 2. 固定对象与环境

- Engine：`artifacts/gcn_res_tensorrt/20260716_224643_531592_strict_fp32/strict_fp32.plan`
- Engine SHA-256：`b76563580089bbc0684e7a8e1edda6028c2d89fd91397c22b4efbd89ecd7fc2c`
- ONNX：`artifacts/gcn_res_tensorrt/20260716_190125_699274_dds_reshape_rewrite/dds_reshape_rewritten.onnx`
- ONNX SHA-256：`f71a585ded3f348a193e37395a124ab8a425ff72fc09ae92114d41872c460b98`
- VoxelUnique Plugin：name `VoxelUnique`、version `1`、namespace `com.tensorrt.ptv2`
- Plugin SHA-256：`60c48c400cc926f6aa183de62298a203916b1add6393a3483a332624da39d0ab`
- checkpoint SHA-256：`311bddf3607d76e6b7ded450b8419bf6ae98f34f50578608b3e6a1c1c3e58d21`
- Python：3.11.8
- PyTorch：2.7.1+cu128
- TensorRT：11.1.0.106
- CUDA Toolkit/Runtime：12.8
- cuDNN：9.7.1
- GPU：NVIDIA GeForce RTX 5060，SM 12.0

执行前后 Engine、ONNX、Plugin DLL 和 checkpoint 的 SHA-256 完全一致。

## 3. 数据与执行口径

划分数量为 train/val/test = `54/18/18`，三者无重叠。本轮只读取：

`data/weld/train_test_split/sub_shuffled_test_file_list.json`

test JSON SHA-256：
`d7464be1b9e0efc6e02bb2490e78e9cf13b10368a2be9bdf0bd86d64f40aff76`

每个样本严格复用历史评估入口的确定性预处理：

1. seed `42`，按 test split 与 JSON index 生成固定采样索引；
2. XYZ 去中心并按最大半径归一化；
3. 有放回采样为 2048 点；
4. 拼接常量 weld category，形成 `points [1,2048,4]`；
5. sklearn `kneighbors_graph(k=6)` 形成 `adj [1,2048,2048]`；
6. 同一 points/adj 依次送入 PyTorch deployment CUDA 和 TensorRT Engine；
7. 两套 logits 均逐样本保存。

PyTorch 侧显式设置 matmul/cuDNN TF32 为 false，并使用 `highest` float32 matmul
precision；结束后原设置已恢复。TensorRT 复用一个 execution context、固定 device
buffer 和一个 CUDA stream，按 JSON 顺序执行18次。没有预热、性能计时或 benchmark。

## 4. 逐样本结果

通过条件同时要求：有限输出、`max_abs < 1e-4`、cosine `> 0.99999`、point
agreement `> 99.9%`。

| Sample | Max abs | Mean abs | RMSE | Cosine | Agreement | Result |
|---|---:|---:|---:|---:|---:|---|
| weld_65 | `7.486343384e-05` | `1.737796720e-05` | `2.258445627e-05` | `0.999999999990` | `100%` | PASS |
| weld_30 | `6.914138794e-05` | `1.061991043e-05` | `1.490534343e-05` | `0.999999999992` | `100%` | PASS |
| weld_28 | `6.198883057e-05` | `1.003427224e-05` | `1.447665820e-05` | `0.999999999993` | `100%` | PASS |
| weld_81 | `7.534027100e-05` | `1.772374344e-05` | `2.241837155e-05` | `0.999999999989` | `100%` | PASS |
| weld_88 | `9.536743164e-05` | `2.159673750e-05` | `2.659047480e-05` | `0.999999999987` | `100%` | PASS |
| weld_5 | `1.168251038e-04` | `2.162653391e-05` | `2.853138473e-05` | `0.999999999984` | `100%` | **FAIL** |
| weld_55 | `8.082389832e-05` | `2.082565470e-05` | `2.561455518e-05` | `0.999999999990` | `100%` | PASS |
| weld_76 | `7.009506226e-05` | `1.789879525e-05` | `2.263252207e-05` | `0.999999999988` | `100%` | PASS |
| weld_12 | `1.113414764e-04` | `2.385641210e-05` | `3.113833271e-05` | `0.999999999982` | `100%` | **FAIL** |
| weld_70 | `6.604194641e-05` | `1.804751037e-05` | `2.253924398e-05` | `0.999999999989` | `100%` | PASS |
| weld_14 | `1.230239868e-04` | `2.243671406e-05` | `3.042986547e-05` | `0.999999999980` | `100%` | **FAIL** |
| weld_18 | `9.965896606e-05` | `2.101544533e-05` | `2.798842600e-05` | `0.999999999984` | `100%` | PASS |
| weld_29 | `5.960464478e-05` | `1.041218411e-05` | `1.500566978e-05` | `0.999999999992` | `100%` | PASS |
| weld_32 | `6.818771362e-05` | `1.123474794e-05` | `1.627104802e-05` | `0.999999999992` | `100%` | PASS |
| weld_36 | `6.055831909e-05` | `1.119354692e-05` | `1.547689418e-05` | `0.999999999991` | `100%` | PASS |
| weld_4 | `1.165866852e-04` | `2.271593894e-05` | `3.103115738e-05` | `0.999999999981` | `100%` | **FAIL** |
| weld_15 | `1.063346863e-04` | `2.196432160e-05` | `2.917100059e-05` | `0.999999999984` | `100%` | **FAIL** |
| weld_82 | `8.153915405e-05` | `1.956714732e-05` | `2.457064457e-05` | `0.999999999989` | `100%` | PASS |

汇总：

- worst max absolute error：`1.230239868164e-04`，样本 `weld_14`；
- average per-sample max absolute error：`8.540683322483e-05`；
- worst cosine similarity：`0.999999999979929`，样本 `weld_14`；
- mean point agreement：`1.0`；
- 超过 max-abs 条件的样本：`weld_5`、`weld_12`、`weld_14`、`weld_4`、`weld_15`。

## 5. 聚合分割指标

标签语义：class 0 = `weld_seam`，class 1 = `background`。混淆矩阵定义为
行 = ground truth，列 = prediction。

PyTorch 与 TensorRT 的 36,864 个点预测完全一致，二者聚合混淆矩阵均为：

```text
[[ 7003,   530],
 [  257, 29074]]
```

| Metric | PyTorch CUDA | TensorRT Strict FP32 | Absolute delta |
|---|---:|---:|---:|
| Overall accuracy | `0.978651258681` | `0.978651258681` | `0` |
| Weld seam IoU | `0.898973042362` | `0.898973042362` | `0` |
| Background IoU | `0.973644553096` | `0.973644553096` | `0` |
| mIoU | `0.936308797729` | `0.936308797729` | `0` |
| Weld seam precision | `0.964600550964` | `0.964600550964` | `0` |
| Weld seam recall | `0.929642904553` | `0.929642904553` | `0` |
| Weld seam F1 | `0.946799161766` | `0.946799161766` | `0` |

因此聚合 mIoU/F1 差值满足 `<1e-5`；最终失败只来自5个样本的逐样本 max-abs
条件。

## 6. 新增文件与产物

脚本：

`scripts/validate_gcn_res_tensorrt_strict_fp32_multisample.py`

产物目录：

`artifacts/gcn_res_tensorrt/20260717_110500_836041_strict_fp32_multisample/`

其中包含：

- `strict_fp32_validation_report.json`；
- `per_sample_results.json`；
- `worst_case_sample.md`；
- `environment.json`；
- `run.log`；
- `predictions/<index>_<sample>/pytorch_logits.npy`；
- `predictions/<index>_<sample>/tensorrt_logits.npy`。

## 7. 停止状态

本轮未执行 FP16、INT8、benchmark 或 C++ 部署。后续若需解释约
`1.0e-4 ~ 1.23e-4` 的残余误差，应另行开展只读归因；本报告不将分类完全一致作为
修改既定 logits 阈值的理由。

```text
TENSORRT_STRICT_FP32_MULTISAMPLE_VALIDATION_FAILED
```

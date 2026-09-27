# ⚠ 口径更正：echo 的「有序线性加权 κ」

发现于 2026-09-27（抽通用组件、跑 `07_通用方法组件/smoke_test.py` 时）。

## 结论

| 口径 | 值 | 说明 |
|---|---|---|
| 标称 κ（A/B/C 分类计数，不做加权） | **0.831** | ✅ 论文 §3.2 报的这个数正确，与 `sklearn.metrics.cohen_kappa_score` 一致 |
| 距离加权线性 κ（w = \|i−j\|/(K−1)） | **0.835** | ✅ 正确值；与 `sklearn` 的 `weights='linear'` 一致 |
| 原 `reliability.py` 输出、论文 §3.2 曾引用的「有序线性加权 κ」 | 0.890 | ❌ **错的**，见下 |

## 错在哪

`reliability.py` 的 `linear_weighted_kappa()` 里期望频数写成

```python
exp = p.sum(0) * p.sum(1)        # ← 得到的是长度 3 的向量，不是 3×3 的期望频数矩阵
pw = (w * exp).sum()             # ← (3,3) 权重矩阵与 (3,) 向量广播，分母被算错
```

正确写法是外积：`exp = np.outer(p.sum(1), p.sum(0))`。修好后同一批数据得 0.835。

复算（92 条双编码，`coding_validation_samples.csv`）：

```
unweighted           = 0.8312      ← 论文报的 0.831 ✔
sklearn linear       = 0.8353      ← 正确
distance-weighted    = 0.8353      ← 与 sklearn linear 一致（两种独立实现互证）
agreement-weighted   = -0.4114     ← 另一套权重约定，别混用（见 alignment_methods._kappa_point）
reliability.py 的错版 = 0.8901      ← 论文旧稿引用的就是这个
```

## 影响与处置

- **影响面**：只影响 §3.2 里"有序线性加权 κ = 0.890"这一处括注；标称 κ（0.831）与 function κ（0.937）、
  以及全部 bootstrap 区间都不受影响（那些走的是 `sklearn` 与 `revision_checks.py`）。
- **已处置**：
  1. 修回稿 §3.2 改为 "echo Cohen's κ = 0.831 (percent agreement .90; with linear disagreement
     weights, κ_w = 0.835)"。
  2. 通用组件 `alignment_methods.kappa_with_ci(how=...)` 显式区分三种权重约定，
     避免下次再混用（`_kappa_point` 的 docstring 写明了差别）。
  3. 复现包 `03_可复现包/coding/` 与本目录同放一份本说明。
- **未改动**：`reliability.py` 原文按"只复制、不改"保留，作为当时口径的证据；引用它时请连同本说明一起看。

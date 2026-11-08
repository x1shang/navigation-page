# 点球落点预测 · penalty forecast

`https://navi.x1shang.top/learning/penalty/`

输入罚球起点与 0.05 s、0.10 s 两个观测点的球心坐标，由论文里冻结的混合密度网络给出
球门平面（x = 11 m）上穿越点的概率密度：最可能落点、50% / 95% 高密度区域、落入门框的概率。
全部计算在浏览器里完成，不上传任何数据。

## 页面构成

| 文件 | 作用 |
|---|---|
| `index.html` | 结构（输入面板 / 画布 / 读数 / 方法说明） |
| `style.css` | 深色主题，沿用 `/learning/` 的配色 |
| `app.js` | 11 维特征 → 3 层 MLP（LayerNorm + GELU）→ 3 个二维高斯；密度网格、行进方格找等值线、Gauss–Legendre 求门内概率 |
| `model.json` | 特征名、三个种子的标准化参数与校准表、常量、权重张量表 |
| `weights.bin` | 三个训练种子的 float32 权重，451,800 字节 |

## 权重来源

来自 UPC 2026 B 题论文的 **P1 冻结模型**（`student-c/P1/final/checkpoints/phys_reduced_k3_{11,22,33}.pt`，
三个训练种子；权重里自带 `architecture = {depth: 3, activation: GELU, correlation: true}`，共 37,650 个参数）。
`model.json` 里记录了每个种子的 SHA-256，与论文 `student-c/P1/final/best_config.json` 的
`checkpoint_sha256` 一致：

```
phys_reduced_k3_11  0a5fc31a616c61b2a433193672d9761ee32367b532c540bea58355894a324c85
phys_reduced_k3_22  85f24a0c7f837e1182d4d31a62a90b14ae1fb5c96f55efdba01f29391ff2c00d
phys_reduced_k3_33  4e5ee62d01ebe9464c875928f14878db22641bf30be42dce4a98ca17002d3d42
```

网络结构：11 → 128 → 128 → 128 → 18（每层 Linear + LayerNorm + GELU），输出 3 个分量的
权重、二维均值、两个尺度与一个相关系数。`weights.bin` 中张量的顺序与偏移见 `model.json`
的 `tensorOrder` / `tensors`，每个种子按 `byteOffset` 顺序排列。

校准：每个噪声档（2 / 5 / 10 / 15 mm）有一个标准差缩放系数和两个 HDR 质量修正值，
取自 `best_config.json` 的 `calibration.phys_reduced_k3_*`。

## 与论文数值的对照

题面案例（r(0.05) = (1.50, 0.45, 0.44)、r(0.10) = (2.96, 0.86, 0.74)，σ = 5 mm，种子 11）：

| 量 | 本页 | 论文 `case.json` |
|---|---|---|
| 众数 (Y, Z) | (2.62, 2.15) m | (2.6161, 2.1535) m |
| 分布均值 | (2.70, 1.97) m | (2.6963, 1.9711) m |
| 入门概率 | 97.3% | 97.319% |
| 50% 区域 | 0.179 m² | 0.1772 m² |
| 95% 区域 | 0.769 m² | 0.7746 m² |

均值与概率是解析计算（混合高斯在门框矩形上的数值积分），与论文的 65,536 次抽样一致。
区域面积用行进方格在细网格上重新积分，论文用较粗网格计数，两者相差约 1%。
恒加速度基线 (2.495, 1.920) m 与论文一致。

## 维护

改模型时只需重新生成 `model.json` + `weights.bin`（保持 `tensorOrder` 与每种子
`byteOffset` 的约定），页面代码不用动。新增噪声档时要在 `model.json` 的 `calibration`
里补一项，并在 `index.html` 的 σ 单选里加一个值。

# 太极图形课 S1 - 基于 WCSPH 的流体模拟
## 背景简介
WCSPH（Weakly compressible SPH）[1] 是 SPH 算法的一个改进。[2] 的问题是压强的计算不科学，容易出现压缩（不保体积）的问题。针对这一问题，WCSPH 给出了一个压缩率很低的算法。

本来也写了 [2] 的，但是似乎因为参数问题调不出来，于是写了 [1]，实现有参考 https://github.com/erizmr/SPH_Taichi

## 成功效果展示

![wcsph](./src/image/wcsph.gif)

## 整体结构
```
-LICENSE
-|src
--|image
--main.py
--wcsph.py
--smooth.py
-README.MD
```

## 运行方式
`python3 src/main.py`

## 参考文献

[1] Becker, Markus, and Matthias Teschner. "Weakly compressible SPH for free surface flows." *Proceedings of the 2007 ACM SIGGRAPH/Eurographics symposium on Computer animation*. 2007.

[2] Müller, Matthias, David Charypar, and Markus Gross. "Particle-based fluid simulation for interactive applications." *Proceedings of the 2003 ACM SIGGRAPH/Eurographics symposium on Computer animation*. 2003.

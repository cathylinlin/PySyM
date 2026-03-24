# PySyM

PySyM (Python Symmetry) 是一个用于描述对称性及其在物理学中应用的 Python 库。本项目**目前**提供了完整的群论实现，支持抽象群论框架以及多种经典群的具体实现。

## 项目简介

PySyM 旨在为物理学、数学和晶体学研究者提供一个强大而灵活的对称性分析工具。通过抽象的群论框架和丰富的具体群实现，用户可以轻松地研究各种对称性结构。

### 目前实现

#### 抽象群论框架
- **群元素抽象基类** (GroupElement)：定义群元素的基本运算
- **群抽象基类** (Group)：提供完整的群抽象框架
- **有限群类** (FiniteGroup)：专门针对有限群的优化实现
- **群属性数据类** (GroupProperties)：描述群的各种代数性质

#### 群论核心概念
- **子群** (Subgroup)：子群的构造与性质分析
- **陪集** (Coset)：左陪集、右陪集及陪集空间
- **商群** (QuotientGroup)：通过正规子群构造商群
- **直积群** (DirectProductGroup)：群的直积运算
- **半直积群** (SemidirectProductGroup)：群的半直积运算
- **群同态** (GroupHomomorphism)：群之间的同态映射
- **群作用** (GroupAction)：群在集合上的作用
- **群工厂** (GroupFactory)：便捷的群构造工具
- **自由群** (FreeGroup)：自由群及其元素

#### 具体群实现
- **循环群** (CyclicGroup)：C_n
- **对称群** (SymmetricGroup)：S_n
- **交错群** (AlternatingGroup)：A_n
- **二面群** (DihedralGroup)：D_n
- **四元群** (QuaternionGroup)：Q_8
- **Klein群** (KleinGroup)：V_4

#### 矩阵群实现
- **一般线性群** (GeneralLinearGroup)：GL(n, F)
- **特殊线性群** (SpecialLinearGroup)：SL(n, F)
- **正交群** (OrthogonalGroup)：O(n)
- **特殊正交群** (SpecialOrthogonalGroup)：SO(n)

#### 代数结构模块
- **抽象代数基类** (AbstractAlgebra)：定义代数结构的基本接口
- **环** (Ring)：环的抽象和具体实现
- **域** (Field)：域的抽象和具体实现
- **模** (Module)：模的抽象和具体实现
- **代数关系** (AlgebraicRelations)：代数结构之间的关系

#### 表示论模块
- **抽象表示** (AbstractRepresentation)：表示论的抽象框架
- **矩阵表示** (MatrixRepresentation)：群的矩阵表示
- **不可约表示** (IrreducibleRepresentation)：不可约表示的实现
- **特征标** (Character)：群表示的特征标理论
- **诱导表示** (InducedRepresentation)：诱导表示的构造

#### 李代数理论模块
- **李代数** (LieAlgebra)：李代数的抽象框架

#### 工具模块
- **矩阵工具** (matrix_utils)：矩阵运算辅助函数
- **矩阵优化** (matrix_opt)：矩阵运算性能优化


### 技术特点

- 完全类型注解，提供良好的代码提示
- 遵循 Python 最佳实践和 PEP 8 规范
- 模块化设计，易于扩展新的群类型
- 高效的算法实现，支持大群的采样检查

## 安装

目前处于开发中，未上线PyPI。

## 快速开始

暂无

## 项目结构

```
PySyM/
├── src/
│   └── PySyM/
│       ├── __init__.py
│       └── core/
│           ├── __init__.py
│           ├── algebraic_structures/  # 代数结构模块
│           │   ├── __init__.py
│           │   ├── abstract_algebra.py  # 抽象代数基类
│           │   ├── algebraic_relations.py  # 代数关系
│           │   ├── field.py            # 域的实现
│           │   ├── module.py           # 模的实现
│           │   └── ring.py             # 环的实现
│           ├── group_theory/          # 群论模块
│           │   ├── __init__.py
│           │   ├── abstract_group.py   # 抽象群论基类
│           │   ├── specific_group.py   # 具体群实现
│           │   ├── subgroup.py         # 子群实现
│           │   ├── coset.py            # 陪集和商群
│           │   ├── product_group.py    # 直积群和半直积群
│           │   ├── group_factory.py    # 群工厂
│           │   └── group_func.py       # 群同态和群作用
│           ├── lie_theory/            # 李代数理论模块
│           │   └── __init__.py
│           ├── matrix_groups/          # 矩阵群模块
│           │   ├── __init__.py
│           │   ├── base.py             # 矩阵群基类
│           │   ├── general_linear.py   # 一般线性群
│           │   ├── special_linear.py   # 特殊线性群
│           │   └── orthogonal.py       # 正交群和特殊正交群
│           ├── representation/         # 表示论模块
│           │   ├── __init__.py
│           │   ├── abstract_representation.py  # 抽象表示
│           │   ├── character.py        # 特征标
│           │   ├── induced.py          # 诱导表示
│           │   ├── irreducible.py      # 不可约表示
│           │   └── matrix_representation.py  # 矩阵表示
│           └── utils/                  # 工具模块
│               ├── matrix_utils.py     # 矩阵工具函数
│               └── matrix_opt.py       # 矩阵运算优化
├── tests/                             # 测试模块
│   ├── __init__.py
│   ├── test_algebraic_structures.py    # 代数结构测试
│   ├── test_group_theory.py           # 群论测试
│   ├── test_matrix_group.py           # 矩阵群测试
│   └── test_representation.py         # 表示论测试
├── .vscode/                           # VS Code 配置
│   └── settings.json
├── pyproject.toml                     # 项目配置文件
├── LICENSE                            # MIT 许可证
└── README.md                          # 项目说明文档
```

## 开发状态

当前版本：0.1.0 (Alpha)

### 已完成功能
- ✅ 完整的抽象群论框架实现
- ✅ 6种经典群的具体实现（循环群、对称群、交错群、二面群、四元群、Klein群）
- ✅ 矩阵群基础框架（一般线性群、特殊线性群、正交群、特殊正交群）
- ✅ 群论核心概念（子群、陪集、商群、直积、半直积）
- ✅ 群同态和群作用
- ✅ 群工厂和自由群
- ✅ 矩阵运算工具和优化
- ✅ 完整的类型注解
- ✅ 基础测试框架
- ✅ 代数结构模块（环、域、模等）
- ✅ 表示论模块（抽象表示、矩阵表示、不可约表示、特征标、诱导表示）
- ✅ 李代数理论模块基础框架

### 开发中功能
- 🚧 更多矩阵群实现（酉群、辛群等）
- 🚧 李群和代数群完整支持
- 🚧 晶体学点群和空间群
- 🚧 可视化功能
- 🚧 完整的测试覆盖

### 计划中功能
- 📋 物理学应用案例（量子力学、晶体学、粒子物理）
- 📋 性能优化和并行计算
- 📋 交互式探索工具
- 📋 文档和教程完善

本项目正在积极开发中，欢迎提供反馈和建议。

## 欢迎合作者

我们热忱欢迎所有对对称性、群论和物理学应用感兴趣的开发者和研究者加入 PySyM 项目！

### 如何参与

1. **报告问题**：如果您发现了 bug 或有功能建议，请提交 issue
2. **贡献代码**：欢迎提交 pull request，改进现有功能或添加新功能
3. **完善文档**：帮助改进文档和示例代码
4. **分享想法**：在讨论区分享您的想法和使用经验

### 贡献方向

- 实现更多的群类型（如李群、晶体学点群等）
- 优化算法性能
- 添加可视化功能
- 扩展物理学应用案例
- 完善测试覆盖
- 改进文档和教程


## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 联系方式

- 作者：LDK（Cathylinlin;rrCathy）
- 邮箱：bluejam001@163.com

## 致谢

感谢所有为 PySyM 项目做出贡献的开发者和使用者！

---

让我们一起探索对称性的美妙世界！🎉

"""有限群表示论（矩阵表示、特征标、诱导与不可约分解）。

与物理应用的接口面：特征标内积、不可约分解、张量积与直和表示；具体哈密顿量或截面公式在更上层实现。
"""
from .abstract_representation import GroupRepresentation
from .character import Character
from .matrix_representation import MatrixRepresentation
from .induced import InducedRepresentation
from .irreducible import IrreducibleRepresentationFinder

__all__ = [
    'GroupRepresentation',
    'Character',
    'MatrixRepresentation',
    'InducedRepresentation',
    'IrreducibleRepresentationFinder'
]

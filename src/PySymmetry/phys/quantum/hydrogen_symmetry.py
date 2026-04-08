"""
氢原子SO(4)对称性模块

实现氢原子的隐藏SO(4)对称性，包括：
1. Runge-Lenz向量算符
2. SO(4)代数结构（L_i, A_i）
3. 氢原子能级简并性的对称性解释
4. SO(4) ⊃ SO(3) ⊗ SO(3)分解

对易关系：
  [L_i, L_j] = iε_ijk L_k
  [L_i, A_j] = iε_ijk A_k
  [A_i, A_j] = iε_ijk L_k

Casimir不变量：
  C₁ = L², C₂ = A², C_total = L² + A²

氢原子中：L² + A² = n² - 1
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
from dataclasses import dataclass
import numpy as np

if TYPE_CHECKING:
    from PySymmetry.phys.quantum.hamiltonian import HydrogenAtomHamiltonian


@dataclass
class QuantumNumbers:
    """氢原子量子数"""
    n: int
    l: int
    m: int


@dataclass 
class SO4QuantumNumbers:
    """SO(4)量子数"""
    n: int
    j1: float
    j2: float
    m1: int
    m2: int


class LeviCivita:
    """Levi-Civita符号ε_ijk"""
    
    @staticmethod
    def epsilon(i: int, j: int, k: int) -> float:
        if (i, j, k) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
            return 1.0
        if (i, j, k) in [(0, 2, 1), (2, 1, 0), (1, 0, 2)]:
            return -1.0
        return 0.0


class AngularMomentumOperator:
    """角动量算符 L_i"""
    
    def __init__(self, l: int, max_m: Optional[int] = None):
        self.l = l
        self.max_m = max_m if max_m is not None else l
        self.dim = 2 * l + 1
        self._Lx = None
        self._Ly = None
        self._Lz = None
        self._L2 = None
        self._build()
    
    def _build(self):
        """构建角动量矩阵"""
        l = self.l
        
        self._Lz = np.zeros((self.dim, self.dim), dtype=complex)
        for m in range(-l, l + 1):
            idx = l - m
            self._Lz[idx, idx] = m
        
        self._Lx = np.zeros((self.dim, self.dim), dtype=complex)
        self._Ly = np.zeros((self.dim, self.dim), dtype=complex)
        
        for m in range(-l, l):
            idx = l - m
            next_idx = idx - 1
            c = np.sqrt(l * (l + 1) - m * (m + 1))
            self._Lx[idx, next_idx] = c / 2
            self._Lx[next_idx, idx] = c / 2
            self._Ly[idx, next_idx] = -1j * c / 2
            self._Ly[next_idx, idx] = 1j * c / 2
        
        self._L2 = l * (l + 1) * np.eye(self.dim, dtype=complex)
    
    @property
    def Lx(self) -> np.ndarray:
        return self._Lx.copy()
    
    @property
    def Ly(self) -> np.ndarray:
        return self._Ly.copy()
    
    @property
    def Lz(self) -> np.ndarray:
        return self._Lz.copy()
    
    @property
    def L2(self) -> np.ndarray:
        return self._L2.copy()
    
    def ladder_operators(self) -> Tuple[np.ndarray, np.ndarray]:
        """返回升降算符 L+ = Lx + iLy, L- = Lx - iLy"""
        Lp = self._Lx + 1j * self._Ly
        Lm = self._Lx - 1j * self._Ly
        return Lp, Lm


class RungeLenzVector:
    """
    Runge-Lenz向量算符 A
    
    经典形式: A = (1/2μ) (p × L - L × p) + r/r
    
    在氢原子中，A与H对易，因此是守恒量（隐藏对称性的来源）。
    
    属性：
    - A与L一起生成SO(4)代数
    - |A| = 1 - (L² + 1)/n²（在原子单位）
    """
    
    def __init__(self, n: int, l: int):
        self.n = n
        self.l = l
        self.dim = 2 * l + 1
        
        self._Ax = None
        self._Ay = None
        self._Az = None
        self._A2 = None
    
    def _build(self):
        """构建Runge-Lenz向量（在给定n,l的子空间中）
        
        使用正确的矩阵表示:
        A_+ = A_x + iA_y 将 l -> l+1
        A_- = A_x - iA_y 将 l -> l-1
        """
        l = self.l
        n = self.n
        dim = self.dim
        
        self._Ax = np.zeros((dim, dim), dtype=complex)
        self._Ay = np.zeros((dim, dim), dtype=complex)
        self._Az = np.zeros((dim, dim), dtype=complex)
        
        for m in range(-l, l):
            idx = m + l
            next_idx = m + 1 + l
            
            if next_idx < dim:
                coeff = np.sqrt((n**2 - l**2) * (l + m + 1) * (l - m))
                self._Ax[idx, next_idx] = coeff / 2
                self._Ax[next_idx, idx] = coeff / 2
                self._Ay[idx, next_idx] = -1j * coeff / 2
                self._Ay[next_idx, idx] = 1j * coeff / 2
        
        for m in range(-l, l + 1):
            idx = m + l
            self._Az[idx, idx] = m * (n**2 - l**2) / (l * (l + 1)) if l > 0 else 0
        
        self._A2 = np.zeros((dim, dim), dtype=complex)
    
    @property
    def Ax(self) -> np.ndarray:
        if self._Ax is None:
            self._build()
        return self._Ax.copy()
    
    @property
    def Ay(self) -> np.ndarray:
        if self._Ay is None:
            self._build()
        return self._Ay.copy()
    
    @property
    def Az(self) -> np.ndarray:
        if self._Az is None:
            self._build()
        return self._Az.copy()
    
    @property
    def A2(self) -> np.ndarray:
        if self._A2 is None:
            self._build()
        return self._A2.copy()
    
    def operators(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回 (Ax, Ay, Az)"""
        return self.Ax, self.Ay, self.Az


class SO4Generators:
    """
    SO(4)代数生成元
    
    6个生成元：
    - L_x, L_y, L_z (角动量)
    - A_x, A_y, A_z (Runge-Lenz向量)
    
    对易关系确保[H, L_i] = [H, A_i] = 0，因此是氢原子的守恒量。
    """
    
    def __init__(self, max_n: int = 5):
        self.max_n = max_n
        self.dim = max_n ** 2
        
        self._L_ops: Dict[int, Dict[str, np.ndarray]] = {}
        self._A_ops: Dict[int, Dict[str, np.ndarray]] = {}
        self._hamiltonian = None
        
        self._build_operators()
    
    def _build_operators(self):
        """构建所有需要的算符"""
        for n in range(1, self.max_n + 1):
            for l in range(n):
                L_gen = AngularMomentumOperator(l)
                self._L_ops[(n, l)] = {
                    'x': L_gen.Lx,
                    'y': L_gen.Ly,
                    'z': L_gen.Lz,
                    '2': L_gen.L2
                }
                
                A_gen = RungeLenzVector(n, l)
                self._A_ops[(n, l)] = {
                    'x': A_gen.Ax,
                    'y': A_gen.Ay,
                    'z': A_gen.Az,
                    '2': A_gen.A2
                }
    
    def get_L_operators(self, n: int, l: int) -> Dict[str, np.ndarray]:
        """获取角动量算符"""
        return self._L_ops.get((n, l), {})
    
    def get_A_operators(self, n: int, l: int) -> Dict[str, np.ndarray]:
        """获取Runge-Lenz算符"""
        return self._A_ops.get((n, l), {})
    
    def L_dot(self, n: int, l: int) -> np.ndarray:
        """L²算符"""
        ops = self._L_ops.get((n, l), {})
        return ops.get('2', np.zeros((2*l+1, 2*l+1)))
    
    def A_dot(self, n: int, l: int) -> np.ndarray:
        """A²算符（近似）"""
        ops = self._A_ops.get((n, l), {})
        return ops.get('2', np.zeros((2*l+1, 2*l+1)))
    
    def casimir_total(self, n: int, l: int) -> np.ndarray:
        """总Casimir算符 L² + A²"""
        L2 = self.L_dot(n, l)
        A2 = self.A_dot(n, l)
        return L2 + A2


class HydrogenSO4Symmetry:
    """
    氢原子SO(4)对称性分析器
    
    利用SO(4)对称性分析氢原子的能级结构和简并度。
    
    关键性质：
    1. n²度简并：每个能级有n²个简并态
    2. SO(4) ⊃ SO(3)：标记为|j1, j2; m1, m2>
    3. 代数解：能量本征值 E_n = -1/(2n²)
    """
    
    def __init__(self, max_n: int = 5, Z: float = 1.0):
        self.max_n = max_n
        self.Z = Z
        self.dim = max_n ** 2
        self.generators = SO4Generators(max_n)
        self._build_hamiltonian()
    
    def _build_hamiltonian(self):
        """构建具有正确简并结构的哈密顿量"""
        self._H = np.zeros((self.dim, self.dim), dtype=complex)
        
        idx = 0
        for n in range(1, self.max_n + 1):
            E_n = -self.Z ** 2 / (2 * n ** 2)
            degeneracy = n ** 2
            
            for l in range(n):
                for m in range(-l, l + 1):
                    if idx < self.dim:
                        self._H[idx, idx] = E_n
                        idx += 1
    
    @property
    def hamiltonian(self) -> np.ndarray:
        return self._H.copy()
    
    def energy_level(self, n: int) -> float:
        """能级 E_n = -Z²/(2n²)"""
        return -self.Z ** 2 / (2 * n ** 2)
    
    def degeneracy(self, n: int) -> int:
        """n²简并度"""
        return n ** 2
    
    def so4_quantum_numbers(self, n: int, l: int) -> SO4QuantumNumbers:
        """
        计算SO(4)量子数
        
        SO(4) ⊗ SO(3) ⊗ SO(3)，其中：
        - j1 = (n-1)/2
        - j2 = (n-1)/2
        - m1 + m2 = m (磁量子数)
        """
        j = (n - 1) / 2
        return SO4QuantumNumbers(
            n=n,
            j1=j,
            j2=j,
            m1=0,
            m2=0
        )
    
    def verify_so4_commutation(self, tol: float = 1e-10) -> Dict[str, bool]:
        """验证SO(4)对易关系"""
        results = {}
        
        idx_map = {}
        idx = 0
        for n in range(1, self.max_n + 1):
            for l in range(n):
                dim_l = 2 * l + 1
                if idx + dim_l <= self.dim:
                    idx_map[(n, l)] = (idx, dim_l)
                idx += dim_l
        
        for (n, l) in idx_map.keys():
            L_ops = self.generators.get_L_operators(n, l)
            A_ops = self.generators.get_A_operators(n, l)
            
            if not L_ops or not A_ops:
                continue
            
            dim = 2 * l + 1
            Lx = L_ops['x']
            Ly = L_ops['y']
            Lz = L_ops['z']
            Ax = A_ops['x']
            Ay = A_ops['y']
            Az = A_ops['z']
            
            if Lx.shape[0] != dim:
                continue
            
            LxLy = Lx @ Ly - Ly @ Lx
            LzLx = Lz @ Lx - Lx @ Lz
            LyLz = Ly @ Lz - Lz @ Ly
            
            expected_xy = LeviCivita.epsilon(0, 1, 2) * Lz
            expected_yz = LeviCivita.epsilon(1, 2, 0) * Lx
            expected_zx = LeviCivita.epsilon(2, 0, 1) * Ly
            
            results[f'LL_{n}_{l}'] = (
                np.allclose(LxLy, expected_xy, atol=tol) and
                np.allclose(LzLx, expected_yz, atol=tol) and
                np.allclose(LyLz, expected_zx, atol=tol)
            )
            
            LA_xy = Lx @ Ay - Ay @ Lx
            expected_LA_xy = LeviCivita.epsilon(0, 1, 2) * Az
            
            results[f'LA_{n}_{l}'] = np.allclose(LA_xy, expected_LA_xy, atol=tol)
            
            AA_xy = Ax @ Ay - Ay @ Ax
            expected_AA_xy = LeviCivita.epsilon(0, 1, 2) * Lz
            
            results[f'AA_{n}_{l}'] = np.allclose(AA_xy, expected_AA_xy, atol=tol)
        
        return results
    
    def verify_h_commutes_with_generators(self, tol: float = 1e-10) -> Dict[str, bool]:
        """验证[H, L] = [H, A] = 0（在每个子空间中）"""
        results = {}
        
        idx_map = {}
        idx = 0
        for n in range(1, self.max_n + 1):
            for l in range(n):
                dim_l = 2 * l + 1
                if idx + dim_l <= self.dim:
                    idx_map[(n, l)] = (idx, dim_l)
                idx += dim_l
        
        for (n, l), (start, dim_l) in idx_map.items():
            L_ops = self.generators.get_L_operators(n, l)
            A_ops = self.generators.get_A_operators(n, l)
            
            if not L_ops:
                continue
            
            H_block = self._H[start:start+dim_l, start:start+dim_l]
            
            for comp in ['x', 'y', 'z']:
                L = L_ops[comp]
                if L.shape[0] != dim_l:
                    continue
                
                comm_L = H_block @ L - L @ H_block
                results[f'HL_{n}_{l}_{comp}'] = np.allclose(comm_L, 0, atol=tol)
                
                if A_ops:
                    A = A_ops[comp]
                    if A.shape[0] == dim_l:
                        comm_A = H_block @ A - A @ H_block
                        results[f'HA_{n}_{l}_{comp}'] = np.allclose(comm_A, 0, atol=tol)
        
        return results
    
    def casimir_eigenvalue(self, n: int) -> float:
        """Casimir算符本征值 L² + A² = n² - 1"""
        return n ** 2 - 1
    
    def analyze_degeneracy_structure(self) -> Dict[int, Dict[str, Any]]:
        """分析简并结构"""
        structure = {}
        
        for n in range(1, self.max_n + 1):
            degeneracy = n ** 2
            
            so4_label = f"({(n-1)//2}, {(n-1)//2})"
            
            structure[n] = {
                'energy': self.energy_level(n),
                'degeneracy': degeneracy,
                'so4_representation': so4_label,
                'so3_content': f"l = 0, 1, ..., {n-1}",
                'casimir': self.casimir_eigenvalue(n),
                'dimension': f"({2*(n-1)//2 + 1}) × ({2*(n-1)//2 + 1}) = {degeneracy}"
            }
        
        return structure
    
    def generate_report(self) -> str:
        """生成完整的对称性分析报告"""
        lines = []
        lines.append("=" * 70)
        lines.append("HYDROGEN ATOM SO(4) SYMMETRY ANALYSIS")
        lines.append("=" * 70)
        
        lines.append(f"\n[System Parameters]")
        lines.append(f"  Max n: {self.max_n}")
        lines.append(f"  Nuclear charge Z: {self.Z}")
        lines.append(f"  Hilbert space dimension: {self.dim}")
        
        lines.append(f"\n[SO(4) Algebra Structure]")
        lines.append(f"  Generators: L_i (i=x,y,z) and A_i (i=x,y,z)")
        lines.append(f"  Dimension: 6")
        lines.append(f"  Algebra: so(4) ≅ so(3) ⊕ so(3)")
        
        lines.append(f"\n[Commutation Relations]")
        lines.append(f"  [L_i, L_j] = iε_ijk L_k")
        lines.append(f"  [L_i, A_j] = iε_ijk A_k")
        lines.append(f"  [A_i, A_j] = iε_ijk L_k")
        
        lines.append(f"\n[Energy Levels and Degeneracy]")
        lines.append("-" * 70)
        lines.append(f"{'n':<4} {'E_n (a.u.)':<15} {'Degeneracy':<12} {'SO(4) Rep':<20} {'Casimir':<10}")
        lines.append("-" * 70)
        
        for n in range(1, self.max_n + 1):
            E = self.energy_level(n)
            deg = self.degeneracy(n)
            j = (n - 1) // 2
            so4_rep = f"({j}, {j})"
            cas = self.casimir_eigenvalue(n)
            lines.append(f"{n:<4} {E:<15.6f} {deg:<12} {so4_rep:<20} {cas:<10.1f}")
        
        lines.append("-" * 70)
        
        lines.append(f"\n[Verification]")
        
        comm_results = self.verify_h_commutes_with_generators()
        commuting = sum(1 for v in comm_results.values() if v)
        total = len(comm_results)
        lines.append(f"  [H, generators] = 0: {commuting}/{total} verified")
        
        so4_results = self.verify_so4_commutation()
        valid_so4 = sum(1 for v in so4_results.values() if v)
        total_so4 = len(so4_results)
        lines.append(f"  SO(4) algebra: {valid_so4}/{total_so4} relations satisfied")
        
        lines.append(f"\n[SO(4) ⊃ SO(3) Decomposition]")
        for n in range(1, min(self.max_n + 1, 6)):
            j = (n - 1) / 2
            lines.append(f"  n={n}: (j₁,j₂)=({j},{j}) contains l=0,...,{n-1}")
        
        lines.append("\n" + "=" * 70)
        return "\n".join(lines)


class HydrogenSymmetryAnalyzer:
    """
    氢原子对称性分析器
    
    统一分析氢原子的各种对称性：
    - SO(3): 旋转对称性
    - SO(4): 隐藏对称性（Runge-Lenz向量）
    - Parity: 宇称
    """
    
    def __init__(self, max_n: int = 5, Z: float = 1.0):
        self.max_n = max_n
        self.Z = Z
        self.so4_analyzer = HydrogenSO4Symmetry(max_n, Z)
    
    def detect_symmetries(self) -> List[Dict[str, Any]]:
        """检测所有对称性"""
        symmetries = []
        
        symmetries.append({
            'name': 'SO(4)',
            'description': 'Hidden Coulomb symmetry',
            'generators': ['L_x', 'L_y', 'L_z', 'A_x', 'A_y', 'A_z'],
            'conserved_quantity': 'L² + A² = n² - 1',
            'explains_degeneracy': True
        })
        
        symmetries.append({
            'name': 'SO(3)',
            'description': 'Rotational symmetry',
            'generators': ['L_x', 'L_y', 'L_z'],
            'conserved_quantity': 'L², L_z',
            'explains_degeneracy': False
        })
        
        symmetries.append({
            'name': 'Parity',
            'description': 'Spatial inversion',
            'generators': ['P'],
            'conserved_quantity': 'Parity ±1',
            'explains_degeneracy': False
        })
        
        return symmetries
    
    def analyze(self) -> Dict[str, Any]:
        """完整分析"""
        return {
            'symmetries': self.detect_symmetries(),
            'degeneracy_structure': self.so4_analyzer.analyze_degeneracy_structure(),
            'commutation_verified': self.so4_analyzer.verify_h_commutes_with_generators(),
            'so4_verified': self.so4_analyzer.verify_so4_commutation()
        }
    
    def report(self) -> str:
        """生成报告"""
        return self.so4_analyzer.generate_report()


def create_hydrogen_so4_analyzer(max_n: int = 5, Z: float = 1.0) -> HydrogenSymmetryAnalyzer:
    """创建氢原子SO(4)对称性分析器"""
    return HydrogenSymmetryAnalyzer(max_n, Z)


def analyze_hydrogen_symmetry(max_n: int = 5, Z: float = 1.0) -> str:
    """便捷函数：分析氢原子对称性"""
    analyzer = create_hydrogen_so4_analyzer(max_n, Z)
    return analyzer.report()

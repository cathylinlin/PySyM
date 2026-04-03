"""
薛定谔方程求解器模块

提供各种量子系统的求解方法：
1. 直接求解 - 精确对角化
2. 数值求解 - 有限差分、有限元等
3. 变分求解 - Ritz 变分法
4. 时间演化 - 分步傅里叶等方法

与 hamiltonian 模块和 states 模块集成。
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
from scipy import sparse, linalg
from scipy.sparse.linalg import expm_multiply, eigs

from .states import Ket, DensityMatrix, StateVector
from .hamiltonian import HamiltonianOperator, MatrixHamiltonian


class Solver(ABC):
    """
    求解器抽象基类
    
    定义量子系统求解的基本接口。
    """
    
    def __init__(self, hamiltonian: HamiltonianOperator):
        self._H = hamiltonian
    
    @abstractmethod
    def solve(self) -> Tuple[List[Ket], np.ndarray]:
        """
        求解本征问题
        
        Returns:
            (本征态列表, 本征能量列表)
        """
        pass
    
    def ground_state(self) -> Tuple[Ket, float]:
        """返回基态"""
        states, energies = self.solve()
        return states[0], energies[0]
    
    def excited_states(self, n: int = 1) -> List[Tuple[Ket, float]]:
        """返回前 n 个激发态"""
        states, energies = self.solve()
        return [(states[i], energies[i]) for i in range(min(n + 1, len(states)))]


class ExactDiagonalizationSolver(Solver):
    """
    精确对角化求解器
    
    通过数值对角化求解本征问题。
    适用于有限维希尔伯特空间。
    
    Args:
        hamiltonian: 哈密顿算符
        check_hermitian: 是否检查哈密顿量的厄米性
    """
    
    def __init__(self, hamiltonian: HamiltonianOperator, check_hermitian: bool = True):
        super().__init__(hamiltonian)
        self._check_hermitian = check_hermitian
    
    def solve(self) -> Tuple[List[Ket], np.ndarray]:
        """精确对角化"""
        H_matrix = self._H._to_matrix()
        
        if self._check_hermitian:
            if not np.allclose(H_matrix, H_matrix.conj().T, atol=1e-10):
                raise ValueError("哈密顿量不是厄米的")
        
        eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)
        
        states = [Ket(eigenvectors[:, i]) for i in range(len(eigenvalues))]
        energies = np.array([float(e) for e in eigenvalues])
        
        return states, energies


class SparseMatrixSolver(Solver):
    """
    稀疏矩阵求解器
    
    适用于大规模稀疏哈密顿矩阵。
    使用 ARPACK 迭代求解部分谱。
    
    Args:
        hamiltonian: 哈密顿算符
        num_eigenvalues: 要求解的本征值数量
        which: 求最小的还是最大的本征值 ('SM', 'LM', 'SA', 'LA')
    """
    
    def __init__(self, 
                 hamiltonian: HamiltonianOperator,
                 num_eigenvalues: int = 10,
                 which: str = 'SA'):
        super().__init__(hamiltonian)
        self._k = num_eigenvalues
        self._which = which
    
    def solve(self) -> Tuple[List[Ket], np.ndarray]:
        """稀疏矩阵求解"""
        H_matrix = self._H._to_matrix()
        H_sparse = sparse.csr_matrix(H_matrix)
        
        eigenvalues, eigenvectors = eigs(H_sparse, k=self._k, which=self._which)
        
        idx = np.argsort(eigenvalues.real)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        states = [Ket(eigenvectors[:, i].flatten()) for i in range(len(eigenvalues))]
        energies = np.array([float(e.real) for e in eigenvalues])
        
        return states, energies


class LanczosSolver(Solver):
    """
    Lanczos 迭代求解器
    
    用于求解大规模稀疏矩阵的极值本征值。
    
    Args:
        hamiltonian: 哈密顿算符
        num_eigenvalues: 要求解的本征值数量
        max_iterations: 最大迭代次数
        tolerance: 收敛容差
    """
    
    def __init__(self,
                 hamiltonian: HamiltonianOperator,
                 num_eigenvalues: int = 5,
                 max_iterations: int = 1000,
                 tolerance: float = 1e-8):
        super().__init__(hamiltonian)
        self._k = num_eigenvalues
        self._max_iter = max_iterations
        self._tol = tolerance
    
    def solve(self) -> Tuple[List[Ket], np.ndarray]:
        """Lanczos 迭代"""
        H_matrix = self._H._to_matrix()
        n = H_matrix.shape[0]
        
        v = np.random.randn(n) + 1j * np.random.randn(n)
        v = v / np.linalg.norm(v)
        
        V = [v]
        alpha = []
        beta = []
        
        w = H_matrix @ v
        
        for j in range(min(self._max_iter, n)):
            alpha_j = np.vdot(v, w).real
            alpha.append(alpha_j)
            
            w = w - alpha_j * v
            if j > 0:
                w = w - beta[j - 1] * V[j - 1]
            
            beta_j = np.linalg.norm(w)
            beta.append(beta_j)
            
            if beta_j < self._tol:
                break
            
            v_new = w / beta_j
            V.append(v_new)
            v = v_new
            w = H_matrix @ v
        
        T = self._build_tridiagonal(alpha, beta, len(alpha))
        eigenvalues, eigenvectors = np.linalg.eigh(T)
        
        states = []
        for i in range(min(self._k, len(eigenvalues))):
            vec = np.zeros(n, dtype=complex)
            for j in range(len(eigenvectors)):
                vec += eigenvectors[j, i] * V[j]
            states.append(Ket(vec / np.linalg.norm(vec)))
        
        energies = np.sort(eigenvalues)[:self._k]
        
        return states, energies
    
    def _build_tridiagonal(self, alpha: List[float], beta: List[float], n: int) -> np.ndarray:
        """构建三对角矩阵"""
        T = np.zeros((n, n))
        for i in range(n):
            T[i, i] = alpha[i]
            if i > 0:
                T[i, i - 1] = beta[i - 1]
                T[i - 1, i] = beta[i - 1]
        return T


class TimeEvolutionSolver:
    """
    时间演化求解器
    
    求解含时薛定谔方程 iℏ ∂|ψ⟩/∂t = H|ψ⟩
    
    Args:
        hamiltonian: 哈密顿算符（固定或含时）
        hbar: 约化普朗克常数
        dt: 时间步长
    """
    
    def __init__(self,
                 hamiltonian: HamiltonianOperator,
                 hbar: float = 1.0,
                 dt: float = 0.01):
        self._H = hamiltonian
        self._hbar = hbar
        self._dt = dt
        self._is_time_dependent = False
    
    def set_time_dependent_hamiltonian(self, H_func: Callable[[float], np.ndarray]) -> None:
        """设置含时哈密顿量"""
        self._H_func = H_func
        self._is_time_dependent = True
    
    def evolve(self, 
               initial_state: Ket,
               t0: float = 0.0,
               num_steps: int = 100) -> Tuple[List[Ket], np.ndarray]:
        """
        时间演化
        
        Args:
            initial_state: 初始态
            t0: 初始时间
            num_steps: 演化步数
            
        Returns:
            (演化路径上的态列表, 时间点列表)
        """
        states = [initial_state.copy()]
        times = [t0]
        
        current_state = initial_state.vector
        current_t = t0
        
        H_fixed = self._H._to_matrix()
        
        for step in range(num_steps):
            if self._is_time_dependent:
                H_t = self._H_func(current_t)
            else:
                H_t = H_fixed
            
            U = self._propagator(H_t, self._dt)
            current_state = U @ current_state
            
            current_t += self._dt
            times.append(current_t)
            states.append(Ket(current_state))
        
        return states, np.array(times)
    
    def _propagator(self, H: np.ndarray, dt: float) -> np.ndarray:
        """计算传播子 U = exp(-i H dt / ℏ)"""
        return linalg.expm(-1j * H * dt / self._hbar)


class SplitStepFourierSolver:
    """
    分步傅里叶法求解器
    
    用于求解含势场的薛定谔方程。
    适用于周期性边界条件。
    
    Args:
        potential: 势能函数 V(x)
        mass: 粒子质量
        x_range: 空间范围 [-L, L]
        num_points: 空间格点数
        dt: 时间步长
    """
    
    def __init__(self,
                 potential: Callable[[np.ndarray], np.ndarray],
                 mass: float = 1.0,
                 x_range: float = 10.0,
                 num_points: int = 512,
                 dt: float = 0.001):
        self._V = potential
        self._m = mass
        self._L = x_range
        self._N = num_points
        self._dt = dt
        
        self._setup_grid()
        self._setup_kinetic_operator()
    
    def _setup_grid(self) -> None:
        """设置空间和动量网格"""
        self._x = np.linspace(-self._L, self._L, self._N, endpoint=False)
        self._dx = 2 * self._L / self._N
        
        k = np.fft.fftfreq(self._N, d=self._dx) * 2 * np.pi
        self._k = k
        self._k_squared = k ** 2
    
    def _setup_kinetic_operator(self) -> None:
        """设置动能算符"""
        self._T = np.exp(-1j * self._dt * self._k_squared / (2 * self._m))
    
    def evolve(self,
               initial_wavefunction: np.ndarray,
               num_steps: int = 100,
               measure_interval: int = 10) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        演化波函数
        
        Args:
            initial_wavefunction: 初始波函数 ψ(x, 0)
            num_steps: 时间步数
            measure_interval: 测量间隔
            
        Returns:
            (波函数快照列表, 测量时间列表)
        """
        psi = initial_wavefunction.copy()
        
        snapshots = []
        times = []
        
        for step in range(num_steps):
            V_x = self._V(self._x)
            exp_V = np.exp(-1j * V_x * self._dt)
            
            psi = psi * exp_V
            
            psi = np.fft.fft(psi)
            psi = psi * self._T
            psi = np.fft.ifft(psi)
            
            psi = psi * exp_V
            
            if step % measure_interval == 0:
                snapshots.append(psi.copy())
                times.append(step * self._dt)
        
        return snapshots, np.array(times)
    
    def compute_energy(self, wavefunction: np.ndarray) -> Dict[str, float]:
        """计算能量期望值"""
        psi = wavefunction
        
        kinetic = np.sum(np.abs(np.fft.fft(psi))**2 * self._k_squared) * self._dx / (2 * self._m)
        
        potential = np.sum(np.abs(psi)**2 * self._V(self._x)) * self._dx
        
        return {
            'kinetic': float(kinetic.real),
            'potential': float(potential.real),
            'total': float((kinetic + potential).real)
        }


class VariationalSolver(Solver):
    """
    变分求解器
    
    使用变分原理求解量子系统。
    
    Args:
        hamiltonian: 哈密顿算符
        trial_function: 试验波函数生成器
        parameters: 变分参数初始值
    """
    
    def __init__(self,
                 hamiltonian: HamiltonianOperator,
                 trial_function: Callable[[np.ndarray], Ket],
                 parameters: np.ndarray):
        super().__init__(hamiltonian)
        self._trial = trial_function
        self._params = np.array(parameters, dtype=float)
        self._num_params = len(parameters)
    
    def solve(self, max_iterations: int = 1000, tolerance: float = 1e-8) -> Tuple[Ket, float]:
        """变分优化"""
        from scipy.optimize import minimize
        
        def energy_function(params: np.ndarray) -> float:
            ket = self._trial(params)
            return float(self._H.expectation(ket).real)
        
        result = minimize(
            energy_function,
            self._params,
            method='L-BFGS-B',
            options={'maxiter': max_iterations, 'ftol': tolerance}
        )
        
        self._params = result.x
        optimal_ket = self._trial(self._params)
        optimal_energy = result.fun
        
        return optimal_ket, optimal_energy


class NumerovSolver:
    """
    Numerov 数值求解器
    
    用于求解一维定态薛定谔方程。
    
    Args:
        potential: 势能函数 V(x)
        mass: 粒子质量
        energy: 猜测能量值
        x_range: 空间范围 [0, L]
        num_points: 空间格点数
        boundary_condition: 边界条件类型
    """
    
    def __init__(self,
                 potential: Callable[[float], float],
                 mass: float = 1.0,
                 energy: float = 1.0,
                 x_range: float = 10.0,
                 num_points: int = 1000,
                 boundary_condition: str = 'infinity'):
        self._V = potential
        self._m = mass
        self._E = energy
        self._L = x_range
        self._N = num_points
        self._bc = boundary_condition
        
        self._setup_grid()
    
    def _setup_grid(self) -> None:
        """设置网格"""
        self._x = np.linspace(0, self._L, self._N)
        self._dx = self._L / (self._N - 1)
        self._k_squared = 2 * self._m * (self._V(self._x) - self._E)
    
    def solve(self) -> Tuple[np.ndarray, float]:
        """
        Numerov 求解
        
        Returns:
            (波函数, 能量)
        """
        h2 = self._dx ** 2
        
        psi = np.zeros(self._N)
        g = 1 - self._k_squared * h2 / 12
        
        if self._bc == 'even':
            psi[0] = 1.0
            psi[1] = 1.0
        else:
            psi[0] = 0.0
            psi[1] = self._dx
        
        for i in range(1, self._N - 1):
            psi[i + 1] = (2 * psi[i] * g[i] - psi[i - 1] * g[i - 1]) / g[i + 1]
        
        return psi, self._E
    
    def find_eigenenergy(self, 
                         energy_range: Tuple[float, float],
                         num_trial: int = 50) -> float:
        """
        搜索本征能量
        
        Args:
            energy_range: 能量搜索范围
            num_trial: 试验能量点数
            
        Returns:
            找到的本征能量
        """
        from scipy.optimize import brentq
        
        energies = np.linspace(energy_range[0], energy_range[1], num_trial)
        
        previous_psi_end = None
        previous_E = None
        
        for E in energies:
            self._E = E
            self._k_squared = 2 * self._m * (self._V(self._x) - E)
            
            psi, _ = self.solve()
            
            if previous_psi_end is not None:
                if previous_psi_end * psi[-1] < 0:
                    def func(E_test):
                        self._E = E_test
                        self._k_squared = 2 * self._m * (self._V(self._x) - E_test)
                        psi_test, _ = self.solve()
                        return psi_test[-1]
                    
                    return brentq(func, previous_E, E)
            
            previous_psi_end = psi[-1]
            previous_E = E
        
        raise ValueError("未找到本征能量")


def solve_schrodinger(
    hamiltonian: HamiltonianOperator,
    method: str = 'exact',
    **kwargs
) -> Tuple[List[Ket], np.ndarray]:
    """
    便捷求解函数
    
    Args:
        hamiltonian: 哈密顿算符
        method: 求解方法 ('exact', 'sparse', 'lanczos')
        **kwargs: 其他参数
        
    Returns:
        (本征态列表, 本征能量列表)
    """
    if method == 'exact':
        solver = ExactDiagonalizationSolver(hamiltonian, **kwargs)
    elif method == 'sparse':
        solver = SparseMatrixSolver(hamiltonian, **kwargs)
    elif method == 'lanczos':
        solver = LanczosSolver(hamiltonian, **kwargs)
    else:
        raise ValueError(f"未知求解方法: {method}")
    
    return solver.solve()


def time_evolve(
    initial_state: Ket,
    hamiltonian: HamiltonianOperator,
    time: float,
    dt: float = 0.01
) -> Ket:
    """
    时间演化便捷函数
    
    Args:
        initial_state: 初始态
        hamiltonian: 哈密顿算符
        time: 演化时间
        dt: 时间步长
        
    Returns:
        演化后的态
    """
    solver = TimeEvolutionSolver(hamiltonian, dt=dt)
    states, _ = solver.evolve(initial_state, num_steps=int(time / dt))
    return states[-1]


def compute_spectrum(
    hamiltonian: HamiltonianOperator,
    num_levels: int = 10
) -> Dict[str, Any]:
    """
    计算能谱信息
    
    Args:
        hamiltonian: 哈密顿算符
        num_levels: 计算的能级数
        
    Returns:
        能谱信息字典
    """
    solver = ExactDiagonalizationSolver(hamiltonian)
    states, energies = solver.solve()
    
    spectrum = {
        'energies': energies[:num_levels],
        'ground_state_energy': energies[0],
        'excited_energies': energies[1:num_levels],
        'level_spacing': np.diff(energies[:num_levels]),
    }
    
    return spectrum

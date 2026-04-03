"""
结果解释器模块

提供量子计算结果的解释和分析：
1. 能谱分析 - 解读能级和跃迁
2. 态分析 - 分析量子态的性质
3. 动力学分析 - 解读时间演化
4. 测量解读 - 解释测量结果

与 states、hamiltonian、solver 模块集成。
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np

from .states import Ket, DensityMatrix, StateVector
from .hamiltonian import HamiltonianOperator


class ResultExplainer(ABC):
    """
    结果解释器抽象基类
    
    定义量子结果解释的基本接口。
    """
    
    def __init__(self, name: str = "Explainer"):
        self._name = name
    
    @abstractmethod
    def explain(self) -> str:
        """生成解释文本"""
        pass


class EnergySpectrumExplainer(ResultExplainer):
    """
    能谱解释器
    
    分析和解释量子系统的能谱信息。
    
    Args:
        hamiltonian: 哈密顿算符
    """
    
    def __init__(self, hamiltonian: HamiltonianOperator):
        super().__init__("EnergySpectrumExplainer")
        self._H = hamiltonian
    
    def compute_spectrum_info(self) -> Dict[str, Any]:
        """计算能谱信息"""
        energies = self._H.all_energy_levels()
        
        sorted_energies = np.sort(energies)
        ground_energy = sorted_energies[0] if len(sorted_energies) > 0 else 0
        excited_energies = sorted_energies[1:]
        
        gaps = np.diff(sorted_energies)
        
        return {
            'ground_energy': ground_energy,
            'first_excited_energy': excited_energies[0] if len(excited_energies) > 0 else None,
            'energy_gaps': gaps,
            'all_energies': sorted_energies,
            'degeneracies': self._compute_degeneracies(sorted_energies),
        }
    
    def _compute_degeneracies(self, energies: np.ndarray, tol: float = 1e-6) -> Dict[float, int]:
        """计算简并度"""
        degeneracies = {}
        for e in energies:
            found = False
            for key in degeneracies:
                if abs(e - key) < tol:
                    degeneracies[key] += 1
                    found = True
                    break
            if not found:
                degeneracies[e] = 1
        return degeneracies
    
    def explain(self, spectrum_data: Optional[Dict[str, Any]] = None) -> str:
        """生成能谱解释"""
        if spectrum_data is None:
            spectrum_data = self.compute_spectrum_info()
        
        lines = []
        lines.append("=== 能谱分析 ===")
        lines.append(f"基态能量: {spectrum_data['ground_energy']:.6f}")
        
        if spectrum_data['first_excited_energy'] is not None:
            gap = spectrum_data['first_excited_energy'] - spectrum_data['ground_energy']
            lines.append(f"第一激发态能量: {spectrum_data['first_excited_energy']:.6f}")
            lines.append(f"能隙: {gap:.6f}")
        
        lines.append(f"\n总能级数: {len(spectrum_data['all_energies'])}")
        
        degeneracies = spectrum_data['degeneracies']
        if len(degeneracies) < len(spectrum_data['all_energies']):
            lines.append("\n简并能级:")
            for energy, deg in sorted(degeneracies.items()):
                if deg > 1:
                    lines.append(f"  E = {energy:.6f}: 简并度 {deg}")
        
        return "\n".join(lines)


class QuantumStateExplainer(ResultExplainer):
    """
    量子态解释器
    
    分析和解释量子态的性质。
    
    Args:
        state: 量子态
    """
    
    def __init__(self, state: Union[Ket, DensityMatrix]):
        super().__init__("QuantumStateExplainer")
        self._state = state
    
    def compute_state_properties(self) -> Dict[str, Any]:
        """计算态的性质"""
        if isinstance(self._state, Ket):
            return self._explain_pure_state()
        else:
            return self._explain_mixed_state()
    
    def _explain_pure_state(self) -> Dict[str, Any]:
        """纯态分析"""
        vec = self._state.to_vector()
        
        probs = np.abs(vec) ** 2
        phases = np.angle(vec)
        
        return {
            'is_pure': True,
            'dimension': len(vec),
            'norm': float(np.linalg.norm(vec)),
            'probabilities': probs,
            'phases': phases,
            'entropy': 0.0,
            'max_probability': float(np.max(probs)),
            'participation_ratio': float(1 / np.sum(probs ** 2)),
        }
    
    def _explain_mixed_state(self) -> Dict[str, Any]:
        """混合态分析"""
        assert isinstance(self._state, DensityMatrix), "Mixed state must be a DensityMatrix"
        rho = self._state
        
        probs = rho.probabilities
        
        return {
            'is_pure': rho.is_pure,
            'dimension': rho.dimension,
            'purity': rho.purity,
            'entropy': rho.entropy(),
            'probabilities': probs,
            'max_probability': float(np.max(probs)),
            'participation_ratio': float(1 / np.sum(probs ** 2)),
        }
    
    def explain(self, state_data: Optional[Dict[str, Any]] = None) -> str:
        """生成态解释"""
        if state_data is None:
            state_data = self.compute_state_properties()
        
        lines = []
        state_type = "纯态" if state_data['is_pure'] else "混合态"
        lines.append(f"=== 量子态分析 ({state_type}) ===")
        lines.append(f"维度: {state_data['dimension']}")
        
        if state_data['is_pure']:
            lines.append(f"范数: {state_data['norm']:.6f}")
        else:
            lines.append(f"纯度: {state_data['purity']:.6f}")
            lines.append(f"von Neumann 熵: {state_data['entropy']:.6f}")
        
        lines.append(f"\n最大概率: {state_data['max_probability']:.6f}")
        lines.append(f"参与比: {state_data['participation_ratio']:.2f}")
        
        return "\n".join(lines)


class DynamicsExplainer(ResultExplainer):
    """
    动力学解释器
    
    分析和解释量子系统的动力学行为。
    
    Args:
        state_history: 态历史
        time_history: 时间历史
        hamiltonian: 哈密顿算符
    """
    
    def __init__(self,
                 state_history: List[Ket],
                 time_history: np.ndarray,
                 hamiltonian: HamiltonianOperator):
        super().__init__("DynamicsExplainer")
        self._states = state_history
        self._times = time_history
        self._H = hamiltonian
    
    def compute_dynamics_properties(self) -> Dict[str, Any]:
        """计算动力学性质"""
        if len(self._states) < 2:
            return {'evolution': 'insufficient_data'}
        
        energies = []
        overlaps = []
        variances = []
        
        initial_state = self._states[0]
        
        for state in self._states:
            E = float(self._H.expectation(state).real)
            energies.append(E)
            
            overlap = abs(np.vdot(initial_state.vector, state.vector)) ** 2
            overlaps.append(overlap)
            
            var = self._H.variance(state)
            variances.append(var)
        
        return {
            'times': self._times,
            'energy_evolution': np.array(energies),
            'fidelity_history': np.array(overlaps),
            'variance_history': np.array(variances),
            'energy_variance': float(np.var(energies)),
            'final_fidelity': overlaps[-1] if overlaps else 0.0,
        }
    
    def explain(self, dynamics_data: Optional[Dict[str, Any]] = None) -> str:
        """生成动力学解释"""
        if dynamics_data is None:
            dynamics_data = self.compute_dynamics_properties()
        
        lines = []
        lines.append("=== 动力学分析 ===")
        
        if dynamics_data.get('evolution') == 'insufficient_data':
            lines.append("数据不足，无法分析动力学行为")
            return "\n".join(lines)
        
        times = dynamics_data['times']
        energies = dynamics_data['energy_evolution']
        
        lines.append(f"模拟时长: {times[-1] - times[0]:.6f}")
        lines.append(f"时间步数: {len(times)}")
        
        lines.append(f"\n能量演化:")
        lines.append(f"  初始能量: {energies[0]:.6f}")
        lines.append(f"  最终能量: {energies[-1]:.6f}")
        lines.append(f"  能量方差: {dynamics_data['energy_variance']:.6e}")
        
        lines.append(f"\n态演化:")
        lines.append(f"  初始保真度: 1.000000")
        lines.append(f"  最终保真度: {dynamics_data['final_fidelity']:.6f}")
        
        return "\n".join(lines)


class MeasurementExplainer(ResultExplainer):
    """
    测量解释器
    
    分析和解释量子测量结果。
    
    Args:
        measurement_results: 测量结果列表
        basis_labels: 基底标签
    """
    
    def __init__(self,
                 measurement_results: List[int],
                 basis_labels: Optional[List[str]] = None):
        super().__init__("MeasurementExplainer")
        self._results = measurement_results
        self._labels = basis_labels
    
    def compute_statistics(self) -> Dict[str, Any]:
        """计算测量统计"""
        if len(self._results) == 0:
            return {'error': 'no_measurements'}
        
        counts = np.bincount(self._results, minlength=max(max(self._results) + 1, 2))
        frequencies = counts / len(self._results)
        
        mean_result = np.mean(self._results)
        var_result = np.var(self._results)
        
        return {
            'total_measurements': len(self._results),
            'counts': counts,
            'frequencies': frequencies,
            'mean': mean_result,
            'variance': var_result,
            'unique_outcomes': np.unique(self._results),
        }
    
    def explain(self, stats: Optional[Dict[str, Any]] = None) -> str:
        """生成测量解释"""
        if stats is None:
            stats = self.compute_statistics()
        
        lines = []
        lines.append("=== 测量分析 ===")
        
        if 'error' in stats:
            lines.append(f"错误: {stats['error']}")
            return "\n".join(lines)
        
        lines.append(f"总测量次数: {stats['total_measurements']}")
        lines.append(f"不同结果数: {len(stats['unique_outcomes'])}")
        
        lines.append(f"\n统计结果:")
        for i, (count, freq) in enumerate(zip(stats['counts'], stats['frequencies'])):
            if count > 0:
                label = self._labels[i] if self._labels and i < len(self._labels) else f"|{i}⟩"
                lines.append(f"  {label}: {count} 次 ({freq:.2%})")
        
        lines.append(f"\n均值: {stats['mean']:.4f}")
        lines.append(f"方差: {stats['variance']:.4f}")
        
        return "\n".join(lines)


class DecoherenceExplainer(ResultExplainer):
    """
    退相干解释器
    
    分析和解释退相干过程。
    
    Args:
        purity_history: 纯度历史
        entropy_history: 熵历史
        time_points: 时间点
    """
    
    def __init__(self,
                 purity_history: List[float],
                 entropy_history: List[float],
                 time_points: np.ndarray):
        super().__init__("DecoherenceExplainer")
        self._purity = purity_history
        self._entropy = entropy_history
        self._times = time_points
    
    def compute_decoherence_properties(self) -> Dict[str, Any]:
        """计算退相干性质"""
        if len(self._purity) < 2:
            return {'error': 'insufficient_data'}
        
        initial_purity = self._purity[0]
        final_purity = self._purity[-1]
        purity_decay = initial_purity - final_purity
        
        initial_entropy = self._entropy[0]
        final_entropy = self._entropy[-1]
        entropy_growth = final_entropy - initial_entropy
        
        decoherence_rate = purity_decay / self._times[-1] if self._times[-1] > 0 else 0
        
        return {
            'initial_purity': initial_purity,
            'final_purity': final_purity,
            'purity_decay': purity_decay,
            'initial_entropy': initial_entropy,
            'final_entropy': final_entropy,
            'entropy_growth': entropy_growth,
            'decoherence_rate': decoherence_rate,
            'times': self._times,
            'purity_history': np.array(self._purity),
            'entropy_history': np.array(self._entropy),
        }
    
    def explain(self, decoherence_data: Optional[Dict[str, Any]] = None) -> str:
        """生成退相干解释"""
        if decoherence_data is None:
            decoherence_data = self.compute_decoherence_properties()
        
        lines = []
        lines.append("=== 退相干分析 ===")
        
        if 'error' in decoherence_data:
            lines.append(f"错误: {decoherence_data['error']}")
            return "\n".join(lines)
        
        lines.append(f"初始纯度: {decoherence_data['initial_purity']:.6f}")
        lines.append(f"最终纯度: {decoherence_data['final_purity']:.6f}")
        lines.append(f"纯度衰减: {decoherence_data['purity_decay']:.6f}")
        
        lines.append(f"\n初始熵: {decoherence_data['initial_entropy']:.6f}")
        lines.append(f"最终熵: {decoherence_data['final_entropy']:.6f}")
        lines.append(f"熵增加: {decoherence_data['entropy_growth']:.6f}")
        
        lines.append(f"\n退相干率: {decoherence_data['decoherence_rate']:.6f}")
        
        if decoherence_data['final_purity'] < 0.5:
            lines.append("\n系统已高度退相干")
        elif decoherence_data['final_purity'] > 0.9:
            lines.append("\n系统仍保持较好相干性")
        else:
            lines.append("\n系统处于部分退相干状态")
        
        return "\n".join(lines)


class CompositeExplainer(ResultExplainer):
    """
    复合解释器
    
    综合多种分析结果。
    
    Args:
        explainers: 解释器列表
    """
    
    def __init__(self, explainers: List[ResultExplainer]):
        super().__init__("CompositeExplainer")
        self._explainers = explainers
    
    def explain(self) -> str:
        """生成综合解释"""
        return self.explain_all()
    
    def explain_all(self) -> str:
        """生成综合解释"""
        lines = []
        lines.append("=" * 50)
        lines.append("综合量子分析报告")
        lines.append("=" * 50)
        lines.append("")
        
        for explainer in self._explainers:
            if hasattr(explainer, 'explain'):
                explanation = explainer.explain()
                lines.append(explanation)
                lines.append("")
        
        return "\n".join(lines)


def explain_quantum_system(
    hamiltonian: HamiltonianOperator,
    state: Union[Ket, DensityMatrix],
    state_history: Optional[List[Ket]] = None,
    time_history: Optional[np.ndarray] = None,
) -> str:
    """
    综合量子系统解释
    
    Args:
        hamiltonian: 哈密顿算符
        state: 当前量子态
        state_history: 态历史（可选）
        time_history: 时间历史（可选）
        
    Returns:
        解释文本
    """
    explainers = []
    
    spectrum_explainer = EnergySpectrumExplainer(hamiltonian)
    explainers.append(spectrum_explainer)
    
    state_explainer = QuantumStateExplainer(state)
    explainers.append(state_explainer)
    
    if state_history is not None and time_history is not None:
        dynamics_explainer = DynamicsExplainer(state_history, time_history, hamiltonian)
        explainers.append(dynamics_explainer)
    
    composite = CompositeExplainer(explainers)
    return composite.explain_all()


def explain_measurement_results(
    measurement_results: List[int],
    basis_labels: Optional[List[str]] = None
) -> str:
    """
    解释测量结果
    
    Args:
        measurement_results: 测量结果列表
        basis_labels: 基底标签
        
    Returns:
        解释文本
    """
    explainer = MeasurementExplainer(measurement_results, basis_labels)
    return explainer.explain()

"""
物理相互作用实现
"""

from abc import ABC, abstractmethod

import numpy as np


class Interaction(ABC):
    """相互作用抽象基类"""

    @abstractmethod
    def calculate_force(self, particle1, particle2) -> np.ndarray:
        """计算两个粒子之间的力"""
        pass

    @abstractmethod
    def calculate_potential(self, particle1, particle2) -> float:
        """计算两个粒子之间的势能"""
        pass

    @abstractmethod
    def get_coupling_constant(self) -> float:
        """获取耦合常数"""
        pass

    def check_position_attribute(self, particle):
        """检查粒子是否有位置属性"""
        if not hasattr(particle, "position"):
            raise AttributeError(
                f"粒子 {particle.__class__.__name__} 缺少 position 属性"
            )
        return True


class ElectromagneticInteraction(Interaction):
    """电磁相互作用"""

    def __init__(self, coupling_constant=1.0):
        self._coupling_constant = coupling_constant

    def calculate_force(self, particle1, particle2):
        self.check_position_attribute(particle1)
        self.check_position_attribute(particle2)
        q1 = particle1.get_charge()
        q2 = particle2.get_charge()
        r_vec = particle1.position - particle2.position
        r = np.linalg.norm(r_vec)
        if r < 1e-10:
            return np.zeros_like(particle1.position)
        # 库仑力: F = k*q1*q2/r² * r̂
        # 同号电荷相斥(q1*q2>0)，异号相吸(q1*q2<0)
        force_magnitude = self._coupling_constant * q1 * q2 / (r**2)
        force_direction = r_vec / r  # 从 particle2 指向 particle1 的单位向量
        return force_magnitude * force_direction

    def calculate_potential(self, particle1, particle2):
        self.check_position_attribute(particle1)
        self.check_position_attribute(particle2)
        q1 = particle1.get_charge()
        q2 = particle2.get_charge()
        r = np.linalg.norm(particle1.position - particle2.position)
        if r < 1e-10:
            return 0.0
        return self._coupling_constant * q1 * q2 / r

    def get_coupling_constant(self):
        return self._coupling_constant


class GravitationalInteraction(Interaction):
    """引力相互作用"""

    def __init__(self, coupling_constant=6.674e-11):
        self._coupling_constant = coupling_constant

    def calculate_force(self, particle1, particle2):
        self.check_position_attribute(particle1)
        self.check_position_attribute(particle2)
        m1 = particle1.get_mass()
        m2 = particle2.get_mass()
        r = np.linalg.norm(particle1.position - particle2.position)
        if r < 1e-10:
            return np.zeros_like(particle1.position)
        force_magnitude = self._coupling_constant * m1 * m2 / (r**2)
        force_direction = (particle2.position - particle1.position) / r
        return force_magnitude * force_direction

    def calculate_potential(self, particle1, particle2):
        self.check_position_attribute(particle1)
        self.check_position_attribute(particle2)
        m1 = particle1.get_mass()
        m2 = particle2.get_mass()
        r = np.linalg.norm(particle1.position - particle2.position)
        if r < 1e-10:
            return 0.0
        return -self._coupling_constant * m1 * m2 / r

    def get_coupling_constant(self):
        return self._coupling_constant


class StrongInteraction(Interaction):
    """强相互作用"""

    def __init__(self, coupling_constant=1.0, range_parameter=1.0):
        self._coupling_constant = coupling_constant
        self._range_parameter = range_parameter

    def calculate_force(self, particle1, particle2):
        self.check_position_attribute(particle1)
        self.check_position_attribute(particle2)
        r = np.linalg.norm(particle1.position - particle2.position)
        if r < 1e-10:
            return np.zeros_like(particle1.position)
        force_magnitude = (
            -self._coupling_constant * np.exp(-r / self._range_parameter) / (r**2)
        )
        force_direction = (particle2.position - particle1.position) / r
        return force_magnitude * force_direction

    def calculate_potential(self, particle1, particle2):
        self.check_position_attribute(particle1)
        self.check_position_attribute(particle2)
        r = np.linalg.norm(particle1.position - particle2.position)
        if r < 1e-10:
            return 0.0
        return -self._coupling_constant * np.exp(-r / self._range_parameter) / r

    def get_coupling_constant(self):
        return self._coupling_constant


class WeakInteraction(Interaction):
    """弱相互作用"""

    def __init__(self, coupling_constant=1.0, range_parameter=1.0):
        self._coupling_constant = coupling_constant
        self._range_parameter = range_parameter

    def calculate_force(self, particle1, particle2):
        self.check_position_attribute(particle1)
        self.check_position_attribute(particle2)
        r = np.linalg.norm(particle1.position - particle2.position)
        if r < 1e-10:
            return np.zeros_like(particle1.position)
        force_magnitude = (
            -self._coupling_constant * np.exp(-r / self._range_parameter) / (r**2)
        )
        force_direction = (particle2.position - particle1.position) / r
        return force_magnitude * force_direction

    def calculate_potential(self, particle1, particle2):
        self.check_position_attribute(particle1)
        self.check_position_attribute(particle2)
        r = np.linalg.norm(particle1.position - particle2.position)
        if r < 1e-10:
            return 0.0
        return -self._coupling_constant * np.exp(-r / self._range_parameter) / r

    def get_coupling_constant(self):
        return self._coupling_constant


class CombinedInteraction(Interaction):
    """组合相互作用"""

    def __init__(self, interactions):
        self._interactions = interactions

    def calculate_force(self, particle1, particle2):
        self.check_position_attribute(particle1)
        self.check_position_attribute(particle2)
        total_force = np.zeros_like(particle1.position)
        for interaction in self._interactions:
            total_force += interaction.calculate_force(particle1, particle2)
        return total_force

    def calculate_potential(self, particle1, particle2):
        self.check_position_attribute(particle1)
        self.check_position_attribute(particle2)
        total_potential = 0.0
        for interaction in self._interactions:
            total_potential += interaction.calculate_potential(particle1, particle2)
        return total_potential

    def get_coupling_constant(self):
        return sum(
            interaction.get_coupling_constant() for interaction in self._interactions
        )

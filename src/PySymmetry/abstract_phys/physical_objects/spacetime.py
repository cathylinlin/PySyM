"""
时空结构实现
"""
from abc import ABC, abstractmethod
import numpy as np
from PySymmetry.core.matrix.factory import MatrixFactory

class Spacetime(ABC):
    """时空抽象基类"""
    
    @abstractmethod
    def metric_tensor(self, position):
        """计算度规张量"""
        pass
    
    @abstractmethod
    def christoffel_symbols(self, position):
        """计算克里斯托费尔符号"""
        pass
    
    @abstractmethod
    def geodesic_equation(self, position, velocity):
        """测地线方程"""
        pass
    
    @abstractmethod
    def proper_time(self, path):
        """计算固有时"""
        pass

class MinkowskiSpacetime(Spacetime):
    """闵可夫斯基时空（平直时空）"""
    
    def __init__(self, c=1.0):
        self._c = c
    
    def metric_tensor(self, position):
        dim = len(position)
        metric = MatrixFactory.zeros(dim, dim)
        metric[0, 0] = 1.0
        for i in range(1, dim):
            metric[i, i] = -1.0
        return metric
    
    def christoffel_symbols(self, position):
        dim = len(position)
        return np.zeros((dim, dim, dim))
    
    def geodesic_equation(self, position, velocity):
        return np.zeros_like(velocity)
    
    def proper_time(self, path):
        dt = np.diff(path[:, 0])
        dx = np.diff(path[:, 1:])
        ds = np.sqrt(dt**2 - np.sum(dx**2, axis=1))
        return np.sum(ds)

class SchwarzschildSpacetime(Spacetime):
    """史瓦西时空（球对称静态黑洞）"""
    
    def __init__(self, mass, G=1.0, c=1.0):
        self._mass = mass
        self._G = G
        self._c = c
        self._rs = 2 * G * mass / (c**2)
    
    def metric_tensor(self, position):
        """计算史瓦西度规张量
        
        使用标准球坐标 (t, r, θ, φ)
        ds² = (1-rs/r)c²dt² - (1-rs/r)⁻¹dr² - r²dθ² - r²sin²θ dφ²
        """
        if len(position) != 4:
            raise ValueError("史瓦西度规需要4维坐标 (t, r, θ, φ)")
        
        r = position[1]  # 径向坐标
        theta = position[2]  # 极角
        
        if r < 1e-10:
            raise ValueError("Radius cannot be zero")
        if r <= self._rs:
            raise ValueError("Inside or at event horizon")
        
        metric = MatrixFactory.zeros(4, 4)
        g_tt = 1.0 - self._rs / r  # 时间分量
        g_rr = -1.0 / (1.0 - self._rs / r)  # 径向分量
        g_thetatheta = -r**2  # 极角分量
        g_phiphi = -(r * np.sin(theta))**2  # 方位角分量
        
        metric[0, 0] = g_tt
        metric[1, 1] = g_rr
        metric[2, 2] = g_thetatheta
        metric[3, 3] = g_phiphi
        return metric
    
    def christoffel_symbols(self, position):
        if len(position) != 4:
            raise ValueError("史瓦西度规需要4维坐标 (t, r, θ, φ)")
        
        t, r, theta, phi = position
        if r < 1e-10:
            raise ValueError("Radius cannot be zero")
        if r <= self._rs:
            raise ValueError("Inside or at event horizon")
        
        gamma = np.zeros((4, 4, 4))
        
        # 计算非零的克里斯托费尔符号
        # gamma^t_rt = gamma^t_tr = (rs/(2r(r - rs)))
        gamma[0, 0, 1] = gamma[0, 1, 0] = self._rs / (2 * r * (r - self._rs))
        
        # gamma^r_tt = (rs (r - rs))/(2 r^3)
        gamma[1, 0, 0] = self._rs * (r - self._rs) / (2 * r**3)
        
        # gamma^r_rr = -rs/(2 r (r - rs))
        gamma[1, 1, 1] = -self._rs / (2 * r * (r - self._rs))
        
        # gamma^r_theta theta = -(r - rs)
        gamma[1, 2, 2] = -(r - self._rs)
        
        # gamma^r_phi phi = -(r - rs) sin^2(theta)
        gamma[1, 3, 3] = -(r - self._rs) * np.sin(theta)**2
        
        # gamma^theta_r theta = gamma^theta theta r = 1/r
        gamma[2, 1, 2] = gamma[2, 2, 1] = 1.0 / r
        
        # gamma^theta_phi phi = -sin(theta) cos(theta)
        gamma[2, 3, 3] = -np.sin(theta) * np.cos(theta)
        
        # gamma^phi_r phi = gamma^phi phi r = 1/r
        gamma[3, 1, 3] = gamma[3, 3, 1] = 1.0 / r
        
        # gamma^phi_theta phi = gamma^phi phi theta = cot(theta)
        gamma[3, 2, 3] = gamma[3, 3, 2] = np.cos(theta) / np.sin(theta)
        
        return gamma
    
    def geodesic_equation(self, position, velocity):
        if len(position) != 4 or len(velocity) != 4:
            raise ValueError("需要4维坐标和速度")
        
        t, r, theta, phi = position
        if r < 1e-10:
            raise ValueError("Radius cannot be zero")
        if r <= self._rs:
            raise ValueError("Inside or at event horizon")
        
        gamma = self.christoffel_symbols(position)
        acceleration = np.zeros_like(velocity)
        
        # 计算测地线方程: d^2 x^μ / dτ^2 = - Γ^μ_νσ dx^ν/dτ dx^σ/dτ
        for mu in range(4):
            for nu in range(4):
                for sigma in range(4):
                    acceleration[mu] -= gamma[mu, nu, sigma] * velocity[nu] * velocity[sigma]
        
        return acceleration
    
    def proper_time(self, path):
        """计算沿路径的固有时
        
        Args:
            path: 路径点数组，每行是 (t, r, θ, φ)
        """
        if path.shape[1] != 4:
            raise ValueError("路径必须是4维坐标 (t, r, θ, φ)")
        
        proper_time = 0.0
        
        for i in range(len(path) - 1):
            # 计算相邻两点间的间隔
            dt = path[i+1, 0] - path[i, 0]
            dr = path[i+1, 1] - path[i, 1]
            dtheta = path[i+1, 2] - path[i, 2]
            dphi = path[i+1, 3] - path[i, 3]
            
            # 使用平均半径
            r_avg = 0.5 * (path[i, 1] + path[i+1, 1])
            theta_avg = 0.5 * (path[i, 2] + path[i+1, 2])
            
            if r_avg <= self._rs:
                raise ValueError("Path goes inside or at event horizon")
            
            # 史瓦西度规下的线元
            # ds² = (1-rs/r)c²dt² - (1-rs/r)⁻¹dr² - r²dθ² - r²sin²θ dφ²
            g_tt = 1.0 - self._rs / r_avg
            g_rr = -1.0 / (1.0 - self._rs / r_avg)
            g_thetatheta = -r_avg**2
            g_phiphi = -(r_avg * np.sin(theta_avg))**2
            
            ds2 = (g_tt * dt**2 + g_rr * dr**2 + 
                   g_thetatheta * dtheta**2 + g_phiphi * dphi**2)
            
            if ds2 < 0:
                # 类空间隔
                pass
            
            proper_time += np.sqrt(np.abs(ds2))
        
        return proper_time

class FRWSpacetime(Spacetime):
    """FRW时空（宇宙学时空）"""
    
    def __init__(self, scale_function, k=0):
        self._scale_function = scale_function
        self._k = k
    
    def metric_tensor(self, position):
        t = position[0]
        a = self._scale_function(t)
        dim = len(position)
        
        metric = MatrixFactory.zeros(dim, dim)
        metric[0, 0] = 1.0
        for i in range(1, dim):
            metric[i, i] = -a**2
        return metric
    
    def christoffel_symbols(self, position):
        t = position[0]
        a = self._scale_function(t)
        da_dt = (self._scale_function(t + 1e-6) - a) / 1e-6
        
        dim = len(position)
        gamma = np.zeros((dim, dim, dim))
        for i in range(1, dim):
            gamma[0, i, i] = da_dt / a
            gamma[i, 0, i] = da_dt / a
            gamma[i, i, 0] = da_dt / a
        return gamma
    
    def geodesic_equation(self, position, velocity):
        t = position[0]
        a = self._scale_function(t)
        da_dt = (self._scale_function(t + 1e-6) - a) / 1e-6
        
        acceleration = np.zeros_like(velocity)
        acceleration[0] = 0.0
        for i in range(1, len(velocity)):
            acceleration[i] = -2 * (da_dt / a) * velocity[i]
        return acceleration
    
    def proper_time(self, path):
        dt = np.diff(path[:, 0])
        dx = np.diff(path[:, 1:])
        a_values = self._scale_function(path[:-1, 0])
        ds = np.sqrt(dt**2 - np.sum((a_values[:, np.newaxis] * dx)**2, axis=1))
        return np.sum(ds)

class CurvedSpacetime(Spacetime):
    """弯曲时空（通用）"""
    
    def __init__(self, metric_function):
        self._metric_function = metric_function
    
    def metric_tensor(self, position):
        return self._metric_function(position)
    
    def christoffel_symbols(self, position):
        epsilon = 1e-6
        dim = len(position)
        g = self.metric_tensor(position)
        g_inv = np.linalg.inv(g)
        
        gamma = np.zeros((dim, dim, dim))
        for mu in range(dim):
            for nu in range(dim):
                for sigma in range(dim):
                    dg_dlambda = (self.metric_tensor(position + epsilon * np.eye(dim)[sigma]) - 
                                 self.metric_tensor(position - epsilon * np.eye(dim)[sigma])) / (2 * epsilon)
                    gamma[mu, nu, sigma] = 0.5 * np.sum(g_inv[mu, :] * 
                                                         (dg_dlambda[:, nu] + dg_dlambda[:, sigma] - dg_dlambda[:, mu]))
        return gamma
    
    def geodesic_equation(self, position, velocity):
        gamma = self.christoffel_symbols(position)
        acceleration = np.zeros_like(velocity)
        for mu in range(len(velocity)):
            for nu in range(len(velocity)):
                for sigma in range(len(velocity)):
                    acceleration[mu] -= gamma[mu, nu, sigma] * velocity[nu] * velocity[sigma]
        return acceleration
    
    def proper_time(self, path):
        ds = 0.0
        for i in range(len(path) - 1):
            dx = path[i + 1] - path[i]
            g = self.metric_tensor(path[i])
            ds += np.sqrt(np.dot(dx, np.dot(g, dx)))
        return ds

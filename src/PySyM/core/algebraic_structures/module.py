"""模论和向量空间模块"""
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Optional, Union
from dataclasses import dataclass
from .ring import Ring, RingElement
from .field import Field, FieldElement

M = TypeVar('M')  # 模元素类型
R = TypeVar('R', bound=RingElement)  # 环元素类型
F = TypeVar('F', bound=FieldElement)  # 域元素类型


class ModuleElement(ABC):
    """模元素抽象基类"""
    
    @abstractmethod
    def __add__(self, other: 'ModuleElement') -> 'ModuleElement':
        """加法"""
        pass
    
    @abstractmethod
    def __sub__(self, other: 'ModuleElement') -> 'ModuleElement':
        """减法"""
        pass
    
    @abstractmethod
    def __mul__(self, scalar: R) -> 'ModuleElement':
        """标量乘法"""
        pass
    
    @abstractmethod
    def is_zero(self) -> bool:
        """是否为零元"""
        pass


class Module(ABC, Generic[M, R]):
    """模抽象基类"""
    
    def __init__(self, ring: Ring[R], name: str = ""):
        """
        初始化模
        
        Args:
            ring: 基础环
            name: 模的名称
        """
        self.ring = ring
        self.name = name
    
    @abstractmethod
    def add(self, a: M, b: M) -> M:
        """加法"""
        pass
    
    @abstractmethod
    def scalar_multiply(self, a: M, scalar: R) -> M:
        """标量乘法"""
        pass
    
    @abstractmethod
    def zero(self) -> M:
        """零元"""
        pass
    
    def subtract(self, a: M, b: M) -> M:
        """减法"""
        return self.add(a, self.additive_inverse(b))
    
    def additive_inverse(self, a: M) -> M:
        """加法逆元"""
        # 对于环上的模，加法逆元通常是标量-1乘以元素
        # 这里假设环有乘法单位元1，并且可以计算-1
        negative_one = self.ring.subtract(self.ring.zero(), self.ring.one())
        return self.scalar_multiply(a, negative_one)
    
    def __contains__(self, element: M) -> bool:
        """判断元素是否属于该模"""
        raise NotImplementedError("子类必须实现__contains__方法")
    
    def is_finite_dimensional(self) -> bool:
        """检查是否为有限维模"""
        return False
    
    def dimension(self) -> Optional[int]:
        """模的维数"""
        return None


class VectorSpaceElement(ModuleElement):
    """向量空间元素抽象基类"""
    
    @abstractmethod
    def __mul__(self, scalar: F) -> 'VectorSpaceElement':
        """标量乘法"""
        pass


class VectorSpace(Module[M, F]):
    """向量空间抽象基类"""
    
    def __init__(self, field: Field[F], name: str = ""):
        """
        初始化向量空间
        
        Args:
            field: 基础域
            name: 向量空间的名称
        """
        super().__init__(field, name)
        self.field = field
    
    def scalar_multiply(self, a: M, scalar: F) -> M:
        """标量乘法"""
        return a * scalar
    
    def is_vector_space(self) -> bool:
        """检查是否为向量空间"""
        return True
    
    def basis(self) -> Optional[List[M]]:
        """向量空间的基"""
        return None


@dataclass
class FiniteDimensionalVectorSpaceElement(VectorSpaceElement):
    """有限维向量空间元素"""
    components: List[F]
    field: Field[F]
    
    def __add__(self, other: 'FiniteDimensionalVectorSpaceElement') -> 'FiniteDimensionalVectorSpaceElement':
        if len(self.components) != len(other.components):
            raise ValueError("向量维度不匹配")
        if self.field != other.field:
            raise ValueError("向量空间基础域不匹配")
        new_components = [self.field.add(a, b) for a, b in zip(self.components, other.components)]
        return FiniteDimensionalVectorSpaceElement(new_components, self.field)
    
    def __sub__(self, other: 'FiniteDimensionalVectorSpaceElement') -> 'FiniteDimensionalVectorSpaceElement':
        if len(self.components) != len(other.components):
            raise ValueError("向量维度不匹配")
        if self.field != other.field:
            raise ValueError("向量空间基础域不匹配")
        new_components = [self.field.subtract(a, b) for a, b in zip(self.components, other.components)]
        return FiniteDimensionalVectorSpaceElement(new_components, self.field)
    
    def __mul__(self, scalar: F) -> 'FiniteDimensionalVectorSpaceElement':
        new_components = [self.field.multiply(a, scalar) for a in self.components]
        return FiniteDimensionalVectorSpaceElement(new_components, self.field)
    
    def is_zero(self) -> bool:
        return all(c.is_zero() for c in self.components)
    
    def __str__(self) -> str:
        components_str = [str(c) for c in self.components]
        return f"({', '.join(components_str)})"


class FiniteDimensionalVectorSpace(VectorSpace[FiniteDimensionalVectorSpaceElement, F]):
    """有限维向量空间"""
    
    def __init__(self, field: Field[F], dimension: int, name: str = ""):
        """
        初始化有限维向量空间
        
        Args:
            field: 基础域
            dimension: 向量空间的维度
            name: 向量空间的名称
        """
        super().__init__(field, name or f"{field.name} Vector Space of dimension {dimension}")
        self._dimension = dimension
    
    def add(self, a: FiniteDimensionalVectorSpaceElement, b: FiniteDimensionalVectorSpaceElement) -> FiniteDimensionalVectorSpaceElement:
        return a + b
    
    def zero(self) -> FiniteDimensionalVectorSpaceElement:
        zero_components = [self.field.zero() for _ in range(self._dimension)]
        return FiniteDimensionalVectorSpaceElement(zero_components, self.field)
    
    def __contains__(self, element: FiniteDimensionalVectorSpaceElement) -> bool:
        return (isinstance(element, FiniteDimensionalVectorSpaceElement) and 
                len(element.components) == self._dimension and 
                element.field == self.field)
    
    def is_finite_dimensional(self) -> bool:
        return True
    
    def dimension(self) -> int:
        return self._dimension
    
    def basis(self) -> List[FiniteDimensionalVectorSpaceElement]:
        """返回标准基"""
        basis = []
        for i in range(self._dimension):
            components = [self.field.zero() for _ in range(self._dimension)]
            components[i] = self.field.one()
            basis.append(FiniteDimensionalVectorSpaceElement(components, self.field))
        return basis


class LinearTransformation:
    """线性变换类"""
    
    def __init__(self, domain: VectorSpace[M, F], codomain: VectorSpace[M, F], matrix: List[List[F]]):
        """
        初始化线性变换
        
        Args:
            domain: 定义域向量空间
            codomain: 陪域向量空间
            matrix: 线性变换的矩阵表示（按列优先）
        """
        self.domain = domain
        self.codomain = codomain
        self.matrix = matrix
        
        # 验证矩阵维度
        domain_dim = domain.dimension()
        codomain_dim = codomain.dimension()
        
        if len(matrix) != codomain_dim:
            raise ValueError(f"矩阵行数必须等于陪域维度，期望{codomain_dim}，实际{len(matrix)}")
        for row in matrix:
            if len(row) != domain_dim:
                raise ValueError(f"矩阵列数必须等于定义域维度，期望{domain_dim}，实际{len(row)}")
    
    def __call__(self, vector: M) -> M:
        """应用线性变换到向量"""
        # 检查向量是否属于定义域
        if vector not in self.domain:
            raise ValueError("向量不属于定义域")
        
        # 计算变换后的向量
        components = []
        for row in self.matrix:
            component = self.codomain.field.zero()
            for i, scalar in enumerate(row):
                # 假设vector有components属性
                # 对于不同类型的向量空间元素，可能需要不同的处理方式
                if hasattr(vector, 'components'):
                    component = self.codomain.field.add(component, self.codomain.field.multiply(scalar, vector.components[i]))
                else:
                    raise NotImplementedError("不支持的向量类型")
            components.append(component)
        
        # 创建并返回变换后的向量
        return FiniteDimensionalVectorSpaceElement(components, self.codomain.field)
    
    def __add__(self, other: 'LinearTransformation') -> 'LinearTransformation':
        """线性变换加法"""
        if self.domain != other.domain or self.codomain != other.codomain:
            raise ValueError("线性变换的定义域和陪域必须相同")
        
        # 矩阵加法
        matrix = []
        for i in range(len(self.matrix)):
            row = []
            for j in range(len(self.matrix[0])):
                row.append(self.codomain.field.add(self.matrix[i][j], other.matrix[i][j]))
            matrix.append(row)
        
        return LinearTransformation(self.domain, self.codomain, matrix)
    
    def __mul__(self, scalar: F) -> 'LinearTransformation':
        """线性变换标量乘法"""
        # 矩阵标量乘法
        matrix = []
        for row in self.matrix:
            new_row = [self.codomain.field.multiply(scalar, elem) for elem in row]
            matrix.append(new_row)
        
        return LinearTransformation(self.domain, self.codomain, matrix)
    
    def compose(self, other: 'LinearTransformation') -> 'LinearTransformation':
        """线性变换复合"""
        if self.domain != other.codomain:
            raise ValueError("第一个变换的定义域必须等于第二个变换的陪域")
        
        # 矩阵乘法
        m = len(self.matrix)
        n = len(other.matrix[0])
        p = len(other.matrix)
        
        matrix = []
        for i in range(m):
            row = []
            for j in range(n):
                elem = self.codomain.field.zero()
                for k in range(p):
                    elem = self.codomain.field.add(elem, self.codomain.field.multiply(self.matrix[i][k], other.matrix[k][j]))
                row.append(elem)
            matrix.append(row)
        
        return LinearTransformation(other.domain, self.codomain, matrix)
    
    def __str__(self) -> str:
        """线性变换的字符串表示"""
        lines = ["Linear Transformation:", f"Domain: {self.domain.name}", f"Codomain: {self.codomain.name}", "Matrix:"]
        
        # 计算每列的最大宽度
        col_widths = [0] * len(self.matrix[0])
        for row in self.matrix:
            for j, elem in enumerate(row):
                col_widths[j] = max(col_widths[j], len(str(elem)))
        
        # 构建矩阵字符串
        for row in self.matrix:
            line = "[".ljust(2)
            for j, elem in enumerate(row):
                elem_str = str(elem)
                line += elem_str.ljust(col_widths[j] + 2)
            line += "]"
            lines.append(line)
        
        return "\n".join(lines)


@dataclass
class TensorProductElement(ModuleElement):
    """张量积元素"""
    factors: List[ModuleElement]
    module: 'TensorProduct'
    
    def __add__(self, other: 'TensorProductElement') -> 'TensorProductElement':
        if self.module != other.module:
            raise ValueError("张量积元素必须属于同一个张量积空间")
        # 简化处理，仅支持基本张量的线性组合
        # 实际实现中可能需要更复杂的处理
        # 这里只处理相同因子的情况
        if self.factors == other.factors:
            # 对于相同的纯张量，e + e = 2e
            two = self.module.ring.add(self.module.ring.one(), self.module.ring.one())
            new_factors = self.factors.copy()
            new_factors[0] = new_factors[0] * two
            return TensorProductElement(new_factors, self.module)
        else:
            # 对于不同的基本张量，需要更复杂的处理
            raise NotImplementedError("不同基本张量的加法尚未实现")
    
    def __sub__(self, other: 'TensorProductElement') -> 'TensorProductElement':
        if self.module != other.module:
            raise ValueError("张量积元素必须属于同一个张量积空间")
        # 简化处理，仅支持基本张量的线性组合
        # 实际实现中可能需要更复杂的处理
        # 这里只处理相同因子的情况
        if self.factors == other.factors:
            # 对于相同的基本张量，可以视为标量乘法后相减
            # 这里简化处理，返回零张量
            return self.module.zero()
        else:
            # 对于不同的基本张量，需要更复杂的处理
            raise NotImplementedError("不同基本张量的减法尚未实现")
    
    def __mul__(self, scalar: R) -> 'TensorProductElement':
        # 标量乘法
        new_factors = self.factors.copy()
        # 将标量乘到第一个因子上
        new_factors[0] = new_factors[0] * scalar
        return TensorProductElement(new_factors, self.module)
    
    def is_zero(self) -> bool:
        # 简化处理，仅当所有因子都是零元时才为零
        # 在张量积中，只要任一分量为零，纯张量就是零。
        return any(factor.is_zero() for factor in self.factors)
    
    def __str__(self) -> str:
        """张量积元素的字符串表示"""
        factors_str = [str(factor) for factor in self.factors]
        return " \\otimes ".join(factors_str)


class TensorProduct(Module[TensorProductElement, R]):
    """张量积空间"""
    
    def __init__(self, modules: List[Module], name: str = ""):
        """
        初始化张量积空间
        
        Args:
            modules: 要进行张量积的模块列表
            name: 张量积空间的名称
        """
        if not modules:
            raise ValueError("至少需要一个模块来构造张量积")
        
        # 所有模块必须在同一个环上
        ring = modules[0].ring
        for module in modules[1:]:
            if module.ring != ring:
                raise ValueError("所有模块必须在同一个环上")
        
        super().__init__(ring, name or f"Tensor Product of {[m.name for m in modules]}")
        self.modules = modules
    
    def add(self, a: TensorProductElement, b: TensorProductElement) -> TensorProductElement:
        return a + b
    
    def scalar_multiply(self, a: TensorProductElement, scalar: R) -> TensorProductElement:
        return a * scalar
    
    def zero(self) -> TensorProductElement:
        # 零张量
        zero_factors = [module.zero() for module in self.modules]
        return TensorProductElement(zero_factors, self)
    
    def __contains__(self, element: TensorProductElement) -> bool:
        return isinstance(element, TensorProductElement) and element.module == self
    
    def tensor(self, elements: List[ModuleElement]) -> TensorProductElement:
        """构造张量积元素"""
        if len(elements) != len(self.modules):
            raise ValueError(f"元素数量必须等于模块数量，期望{len(self.modules)}，实际{len(elements)}")
        
        # 检查每个元素是否属于对应的模块
        for elem, module in zip(elements, self.modules):
            if elem not in module:
                raise ValueError(f"元素不属于对应的模块")
        
        return TensorProductElement(elements, self)
    
    def dimension(self) -> Optional[int]:
        """张量积空间的维度"""
        # 如果所有模块都是有限维的，则张量积的维度是各模块维度的乘积
        dim = 1
        for module in self.modules:
            module_dim = module.dimension()
            if module_dim is None:
                return None
            dim *= module_dim
        return dim

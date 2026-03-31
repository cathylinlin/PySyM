"""离散群实现

该模块提供离散群的实现，包括：
- ParityGroup: 宇称群
- TimeReversalGroup: 时间反演群
"""

from .abstract_group import Group


class ParityGroup(Group):
    """宇称群 {1, P}"""
    
    def __init__(self):
        """初始化宇称群"""
        super().__init__("ParityGroup")
    
    def identity(self):
        """单位元：恒等操作"""
        return 1
    
    def multiply(self, a, b):
        """群乘法：宇称操作的乘法"""
        return a * b
    
    def inverse(self, element):
        """逆元：宇称操作的逆是自身"""
        return element
    
    def __contains__(self, element):
        """判断元素是否属于该群"""
        return element in {1, -1}
    
    def order(self):
        """群的阶"""
        return 2
    
    def elements(self):
        """所有群元素"""
        return [1, -1]
    
    def is_continuous(self) -> bool:
        """是否为连续群"""
        return False


class TimeReversalGroup(Group):
    """时间反演群 {1, T}"""
    
    def __init__(self):
        """初始化时间反演群"""
        super().__init__("TimeReversalGroup")
    
    def identity(self):
        """单位元：恒等操作"""
        return 1
    
    def multiply(self, a, b):
        """群乘法：时间反演操作的乘法"""
        return a * b
    
    def inverse(self, element):
        """逆元：时间反演操作的逆是自身"""
        return element
    
    def __contains__(self, element):
        """判断元素是否属于该群"""
        return element in {1, -1}
    
    def order(self):
        """群的阶"""
        return 2
    
    def elements(self):
        """所有群元素"""
        return [1, -1]
    
    def is_continuous(self) -> bool:
        """是否为连续群"""
        return False

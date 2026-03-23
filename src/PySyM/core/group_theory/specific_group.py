from typing import List, Tuple
from itertools import permutations
from .abstract_group import Group

class CyclicGroup(Group[int]):
    """循环群"""
    def __init__(self, n: int):
        if n <= 0:
            raise ValueError(f"循环群的阶必须为正整数， got {n}")
        super().__init__(f"C_{n}")
        self.n = n
    
    def identity(self) -> int:
        return 0
    
    def multiply(self, a: int, b: int) -> int:
        return (a + b) % self.n
    
    def inverse(self, a: int) -> int:
        return (-a) % self.n
    
    def __contains__(self, element: int) -> bool:
        return isinstance(element, int) and 0 <= element < self.n
    
    def order(self) -> int:
        return self.n
    
    def elements(self) -> List[int]:
        return list(range(self.n))

class SymmetricGroup(Group[Tuple[int, ...]]):
    """对称群"""
    def __init__(self, n: int):
        if n < 0:
            raise ValueError(f"对称群的次数必须为非负整数， got {n}")
        if n > 8:
            raise ValueError(f"对称群 S_{n} 太大，无法生成所有元素。最大支持 n=8")
        super().__init__(f"S_{n}")
        self.n = n
        # 延迟生成元素，只有在需要时才生成
        self._elements = None
    
    def _generate_permutations(self, elements: List) -> List[Tuple[int, ...]]:
        if len(elements) <= 1:
            return [tuple(elements)]
        permutations = []
        for i in range(len(elements)):
            first = elements[i]
            remaining = elements[:i] + elements[i+1:]
            for perm in self._generate_permutations(remaining):
                permutations.append((first,) + perm)
        return permutations
    
    def _ensure_elements_generated(self):
        """确保元素已生成"""
        if self._elements is None:
            self._elements = self._generate_permutations(list(range(self.n)))
    
    def identity(self) -> Tuple[int, ...]:
        return tuple(range(self.n))
    
    def multiply(self, a: Tuple[int, ...], b: Tuple[int, ...]) -> Tuple[int, ...]:
        # 置换乘法：a * b 表示先应用 b，再应用 a
        return tuple(a[b[i]] for i in range(self.n))
    
    def inverse(self, a: Tuple[int, ...]) -> Tuple[int, ...]:
        # 计算置换的逆
        inv = [0] * self.n
        for i in range(self.n):
            inv[a[i]] = i
        return tuple(inv)
    
    def __contains__(self, element: Tuple[int, ...]) -> bool:
        if not isinstance(element, tuple):
            return False
        if len(element) != self.n:
            return False
        return set(element) == set(range(self.n))
    
    def order(self) -> int:
        self._ensure_elements_generated()
        return len(self._elements)
    
    def elements(self) -> List[Tuple[int, ...]]:
        self._ensure_elements_generated()
        return self._elements
    
    def alternating_group(self) -> 'AlternatingGroup':
        """返回交错群 A_n"""
        return AlternatingGroup(self.n)
    
    def is_simple(self) -> bool:
        """检查是否为单群
        
        对称群 S_n 只有在 n >= 5 时，其交错子群 A_n 才是单群。
        S_n 本身（n >= 2）都有非平凡正规子群 A_n，所以不是单群。
        """
        # S_n 本身永远不是单群（当 n >= 2 时，A_n 是非平凡正规子群）
        if self.n >= 2:
            return False
        # S_1 是平凡群，也不是单群
        return False

class DihedralGroup(Group[Tuple[int, int]]):
    """二面群"""
    def __init__(self, n: int):
        if n <= 0:
            raise ValueError(f"二面群的边数必须为正整数， got {n}")
        super().__init__(f"D_{n}")
        self.n = n
        # 生成所有元素：(0, k) 表示旋转 k 步，(1, k) 表示翻转后旋转 k 步
        self._elements = []
        for r in [0, 1]:
            for k in range(n):
                self._elements.append((r, k))
    
    def identity(self) -> Tuple[int, int]:
        return (0, 0)
    
    def multiply(self, a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[int, int]:
        # 二面群乘法规则
        # (r1, k1) * (r2, k2) 表示先应用 b，再应用 a
        # r=0: 旋转, r=1: 翻转
        r1, k1 = a
        r2, k2 = b
        if r1 == 0 and r2 == 0:
            # 旋转 + 旋转 = 旋转
            return (0, (k1 + k2) % self.n)
        elif r1 == 0 and r2 == 1:
            # 旋转 + 翻转 = 翻转
            return (1, (k1 + k2) % self.n)
        elif r1 == 1 and r2 == 0:
            # 翻转 + 旋转 = 翻转（但方向相反）
            return (1, (k1 - k2) % self.n)
        else:  # r1 == 1 and r2 == 1
            # 翻转 + 翻转 = 旋转
            return (0, (k1 - k2) % self.n)
    
    def inverse(self, a: Tuple[int, int]) -> Tuple[int, int]:
        r, k = a
        if r == 0:
            # 旋转的逆是反向旋转
            return (0, (-k) % self.n)
        else:
            # 翻转的逆是自身
            return (1, k)
    
    def __contains__(self, element: Tuple[int, int]) -> bool:
        if not isinstance(element, tuple) or len(element) != 2:
            return False
        r, k = element
        return r in [0, 1] and 0 <= k < self.n
    
    def order(self) -> int:
        return 2 * self.n
    
    def elements(self) -> List[Tuple[int, int]]:
        return self._elements

class QuaternionGroup(Group[str]):
    """四元群"""
    def __init__(self):
        super().__init__("Q_8")
        # 四元群元素：1, -1, i, -i, j, -j, k, -k
        self._elements = ['1', '-1', 'i', '-i', 'j', '-j', 'k', '-k']
        # 乘法表
        self._mult_table = {
            ('1', '1'): '1', ('1', '-1'): '-1', ('1', 'i'): 'i', ('1', '-i'): '-i',
            ('1', 'j'): 'j', ('1', '-j'): '-j', ('1', 'k'): 'k', ('1', '-k'): '-k',
            ('-1', '1'): '-1', ('-1', '-1'): '1', ('-1', 'i'): '-i', ('-1', '-i'): 'i',
            ('-1', 'j'): '-j', ('-1', '-j'): 'j', ('-1', 'k'): '-k', ('-1', '-k'): 'k',
            ('i', '1'): 'i', ('i', '-1'): '-i', ('i', 'i'): '-1', ('i', '-i'): '1',
            ('i', 'j'): 'k', ('i', '-j'): '-k', ('i', 'k'): '-j', ('i', '-k'): 'j',
            ('-i', '1'): '-i', ('-i', '-1'): 'i', ('-i', 'i'): '1', ('-i', '-i'): '-1',
            ('-i', 'j'): '-k', ('-i', '-j'): 'k', ('-i', 'k'): 'j', ('-i', '-k'): '-j',
            ('j', '1'): 'j', ('j', '-1'): '-j', ('j', 'i'): '-k', ('j', '-i'): 'k',
            ('j', 'j'): '-1', ('j', '-j'): '1', ('j', 'k'): 'i', ('j', '-k'): '-i',
            ('-j', '1'): '-j', ('-j', '-1'): 'j', ('-j', 'i'): 'k', ('-j', '-i'): '-k',
            ('-j', 'j'): '1', ('-j', '-j'): '-1', ('-j', 'k'): '-i', ('-j', '-k'): 'i',
            ('k', '1'): 'k', ('k', '-1'): '-k', ('k', 'i'): 'j', ('k', '-i'): '-j',
            ('k', 'j'): '-i', ('k', '-j'): 'i', ('k', 'k'): '-1', ('k', '-k'): '1',
            ('-k', '1'): '-k', ('-k', '-1'): 'k', ('-k', 'i'): '-j', ('-k', '-i'): 'j',
            ('-k', 'j'): 'i', ('-k', '-j'): '-i', ('-k', 'k'): '1', ('-k', '-k'): '-1'
        }
    
    def identity(self) -> str:
        return '1'
    
    def multiply(self, a: str, b: str) -> str:
        return self._mult_table[(a, b)]
    
    def inverse(self, a: str) -> str:
        if a == '1':
            return '1'
        elif a == '-1':
            return '-1'
        else:
            # 其他元素的逆是其负数
            return '-' + a if a[0] != '-' else a[1:]
    
    def __contains__(self, element: str) -> bool:
        return element in self._elements
    
    def order(self) -> int:
        return 8
    
    def elements(self) -> List[str]:
        return self._elements

class KleinGroup(Group[tuple[int, int]]):
    """Klein群"""
    def __init__(self):
        super().__init__("V_4")
        # Klein群元素：(0,0), (0,1), (1,0), (1,1)
        self._elements = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    def identity(self) -> tuple[int, int]:
        return (0, 0)
    
    def multiply(self, a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
        # 按分量模2加法
        return ((a[0] + b[0]) % 2, (a[1] + b[1]) % 2)
    
    def inverse(self, a: tuple[int, int]) -> tuple[int, int]:
        # Klein群中每个元素都是自身的逆
        return a
    
    def __contains__(self, element: tuple[int, int]) -> bool:
        if not isinstance(element, tuple) or len(element) != 2:
            return False
        return element[0] in [0, 1] and element[1] in [0, 1]
    
    def order(self) -> int:
        return 4
    
    def elements(self) -> List[Tuple[int, int]]:
        return self._elements

class AlternatingGroup(Group[Tuple[int, ...]]):
    """交替群"""
    def __init__(self, n: int):
        if n < 0:
            raise ValueError(f"交错群的次数必须为非负整数， got {n}")
        if n > 8:
            raise ValueError(f"交错群 A_{n} 太大，无法生成所有元素。最大支持 n=8")
        super().__init__(f"A_{n}")
        self.n = n
        # 延迟生成元素，只有在需要时才生成
        self._elements = None
    
    def _ensure_elements_generated(self):
        """确保元素已生成"""
        if self._elements is None:
            # 生成所有偶置换
            self._elements = []
            for perm in permutations(range(self.n)):
                if self._is_even_permutation(perm):
                    self._elements.append(perm)
    
    def _is_even_permutation(self, perm: tuple[int, ...]) -> bool:
        # 计算置换的逆序数
        inversions = 0
        for i in range(len(perm)):
            for j in range(i + 1, len(perm)):
                if perm[i] > perm[j]:
                    inversions += 1
        return inversions % 2 == 0
    
    def identity(self) -> tuple[int, ...]:
        return tuple(range(self.n))
    
    def multiply(self, a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
        # 置换乘法：a * b 表示先应用 b，再应用 a
        return tuple(a[b[i]] for i in range(self.n))
    
    def inverse(self, a: tuple[int, ...]) -> tuple[int, ...]:
        # 计算置换的逆
        inv = [0] * self.n
        for i in range(self.n):
            inv[a[i]] = i
        return tuple(inv)
    
    def __contains__(self, element: tuple[int, ...]) -> bool:
        if not isinstance(element, tuple):
            return False
        if len(element) != self.n:
            return False
        if set(element) != set(range(self.n)):
            return False
        return self._is_even_permutation(element)
    
    def order(self) -> int:
        self._ensure_elements_generated()
        return len(self._elements)
    
    def elements(self) -> List[Tuple[int, ...]]:
        self._ensure_elements_generated()
        return self._elements



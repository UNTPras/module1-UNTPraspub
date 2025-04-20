"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable, List

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.

def mul(x: float, y: float) -> float:
    return x * y

def id(x: float) -> float:
    return x

def add(x: float, y: float) -> float:
    return x + y

def neg(x: float) -> float:
    return -x

def lt(x: float, y: float) -> bool:
    return x < y

def eq(x: float, y: float) -> bool:
    return x == y

def max(x: float, y: float) -> float:
    return x if x > y else y

def is_close(x: float, y: float) -> bool:
    return abs(x - y) < 1e-2

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))

def relu(x: float) -> float:
    return max(0.0, x)

def log(x: float) -> float:
    return math.log(x)

def exp(x: float) -> float:
    return math.exp(x)

def inv(x: float) -> float:
    return 1.0 / x

def log_back(x: float, d: float) -> float:
    return d / x

def inv_back(x: float, d: float) -> float:
    return -d / (x ** 2)

def relu_back(x: float, d: float) -> float:
    return d if x > 0 else 0.0

# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.


def map(fn: Callable[[float], float], lst: Iterable[float]) -> List[float]:
    """Applies fn to each element in lst."""
    return [fn(x) for x in lst]

def zipWith(fn: Callable[[float, float], float], lst1: Iterable[float], lst2: Iterable[float]) -> List[float]:
    """Applies fn to pairs of elements from lst1 and lst2."""
    return [fn(x, y) for x, y in zip(lst1, lst2)]

def reduce(fn: Callable[[float, float], float], lst: Iterable[float], start: float) -> float:
    """Reduces lst to a single value using fn starting from start."""
    result = start
    for x in lst:
        result = fn(result, x)
    return result

# Using higher-order functions to implement list operations

def negList(lst: Iterable[float]) -> List[float]:
    """Negates all elements in lst using map."""
    return map(neg, lst)

def addLists(lst1: Iterable[float], lst2: Iterable[float]) -> List[float]:
    """Adds corresponding elements from lst1 and lst2 using zipWith."""
    return zipWith(add, lst1, lst2)

def sum(lst: Iterable[float]) -> float:
    """Sums all elements in lst using reduce."""
    return reduce(add, lst, 0.0)

def prod(lst: Iterable[float]) -> float:
    """Calculates the product of all elements in lst using reduce."""
    return reduce(mul, lst, 1.0)
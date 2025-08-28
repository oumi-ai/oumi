#!/usr/bin/env python3
"""Sample Python file for testing code attachment functionality."""


def hello_world():
    """A simple hello world function."""
    print("Hello, World!")
    return "Hello, World!"


def fibonacci(n):
    """Generate fibonacci sequence up to n terms."""
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]

    sequence = [0, 1]
    for i in range(2, n):
        sequence.append(sequence[i - 1] + sequence[i - 2])
    return sequence


class Calculator:
    """A simple calculator class."""

    def __init__(self):
        self.result = 0

    def add(self, x, y):
        """Add two numbers."""
        return x + y

    def multiply(self, x, y):
        """Multiply two numbers."""
        return x * y


if __name__ == "__main__":
    hello_world()
    print("Fibonacci(10):", fibonacci(10))

    calc = Calculator()
    print("2 + 3 =", calc.add(2, 3))
    print("4 * 5 =", calc.multiply(4, 5))

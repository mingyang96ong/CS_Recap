import pytest

from random import randint

from sorting import *

@pytest.fixture
def generate_array_func():
    MAX_INT = (2**31-1)
    def generate_array(n: int, start: int = -MAX_INT, end: int = MAX_INT):
        return [randint(start, end) for _ in range(n)]
    return generate_array

def test_bubble_sort(generate_array_func):
    TESTCASES_COUNT = 10 # Bubble sort is a O(n^2) sorting algorithm. It runs very slow
    ARRAY_LENGTH = 1_000

    for i in range(TESTCASES_COUNT):
        arr = generate_array_func(ARRAY_LENGTH)
        expected = list(sorted(arr))
        output = bubble_sort(arr)
        check = all(exp == out for exp, out in zip(expected, output))
        assert check, f'\n{output=}\n {expected=}'

def test_insertion_sort(generate_array_func):
    TESTCASES_COUNT = 10 # Insertion sort is a O(n^2) sorting algorithm. It runs very slow
    ARRAY_LENGTH = 1_000

    for i in range(TESTCASES_COUNT):
        arr = generate_array_func(ARRAY_LENGTH)
        expected = list(sorted(arr))
        output = insertion_sort(arr)
        check = all(exp == out for exp, out in zip(expected, output))
        assert check, f'\n{output=}\n {expected=}'

def test_merge_sort(generate_array_func):
    TESTCASES_COUNT = 10 # Bubble sort is a O(n^2) sorting algorithm. It runs very slow
    ARRAY_LENGTH = 1_000
    for i in range(TESTCASES_COUNT):
        arr = generate_array_func(ARRAY_LENGTH)
        expected = list(sorted(arr))
        output = merge_sort(arr)
        check = all(exp == out for exp, out in zip(expected, output))
        assert check, f'\n{output=}\n {expected=}'

def test_merge_sort_inplace(generate_array_func):
    TESTCASES_COUNT = 10 # Bubble sort is a O(n^2) sorting algorithm. It runs very slow
    ARRAY_LENGTH = 1_000
    for i in range(TESTCASES_COUNT):
        arr = generate_array_func(ARRAY_LENGTH, start=0)
        expected = list(sorted(arr))
        output = merge_sort_inplace(arr)
        check = all(exp == out for exp, out in zip(expected, output))
        assert check, f'\n{output=}\n {expected=}'

def test_radix_sort(generate_array_func):
    TESTCASES_COUNT = 100
    ARRAY_LENGTH = 10_000
    for i in range(TESTCASES_COUNT):
        arr = generate_array_func(ARRAY_LENGTH, start = 0)
        expected = list(sorted(arr))
        output = radix_sort(arr)
        check = all(exp == out for exp, out in zip(expected, output))
        assert check, f'\n{output=}\n {expected=}'

def test_radix_sort_negative(generate_array_func):
    TESTCASES_COUNT = 100
    ARRAY_LENGTH = 10_000
    for i in range(TESTCASES_COUNT):
        arr = generate_array_func(ARRAY_LENGTH)
        expected = list(sorted(arr))
        output = radix_sort_negative(arr)
        check = all(exp == out for exp, out in zip(expected, output))
        assert check, f'\n{output=}\n {expected=}'

def test_quick_sort(generate_array_func):
    TESTCASES_COUNT = 100
    ARRAY_LENGTH = 10_000
    for i in range(TESTCASES_COUNT):
        arr = generate_array_func(ARRAY_LENGTH)
        expected = list(sorted(arr))
        output = quick_sort(arr)
        check = all(exp == out for exp, out in zip(expected, output))
        assert check, f'\n{output=}\n {expected=}'

def test_quick_sort_better(generate_array_func):
    TESTCASES_COUNT = 100
    ARRAY_LENGTH = 10_000
    for i in range(TESTCASES_COUNT):
        arr = generate_array_func(ARRAY_LENGTH)
        expected = list(sorted(arr))
        output = quick_sort_better(arr)
        check = all(exp == out for exp, out in zip(expected, output))
        assert check, f'\n{output=}\n {expected=}'

def test_heap_sort(generate_array_func):
    TESTCASES_COUNT = 100
    ARRAY_LENGTH = 100
    for i in range(TESTCASES_COUNT):
        arr = generate_array_func(ARRAY_LENGTH)
        expected = list(sorted(arr))
        output = heap_sort(arr)
        check = all(exp == out for exp, out in zip(expected, output))
        assert check, f'\n{output=}\n {expected=}'
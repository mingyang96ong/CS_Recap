import pytest

from random import randint, uniform
from math import isclose

from bits import *

def test_count_bits():
    COUNTS = 1000
    MAX_LONG = 2**63-1
    for _ in range(COUNTS):
        val = randint(0,MAX_LONG)
        expected = bin(val).count('1')
        output = count_bits(val)
        assert output == expected, f'Expected: {expected}. Received: {output} for \'{val}\''

def test_parity():
    COUNTS = 1000
    MAX_LONG = 2**63-1
    for _ in range(COUNTS):
        val = randint(0,MAX_LONG)
        expected = parity(val)
        output = cached_parity(val)
        assert output == expected, f'Expected: {expected}. Received: {output} for \'{val}\''

def test_swap_bits():
    COUNTS = 1000
    MAX_LONG = 2**63-1
    for _ in range(COUNTS):
        val = randint(0,MAX_LONG)
        # Brute force generation
        bit_length = val.bit_length()
        i, j = randint(0, bit_length-1), randint(0, bit_length-1)
        binary_str = list(bin(val)[2:])
        rightmost_i, rightmost_j = bit_length-i-1, bit_length-j-1
        binary_str[rightmost_i], binary_str[rightmost_j] = binary_str[rightmost_j], binary_str[rightmost_i]
        expected = int(''.join(binary_str), base=2)
        output = swap_bits(val, i, j)
        assert output == expected, f'Expected: {expected}. Received: {output} for \'{val=}\', \'{i=}\', \'{j=}\''

def test_reverse_bits():
    COUNTS = 1000
    MAX_LONG = 2**63-1
    for _ in range(COUNTS):
        val = randint(0,MAX_LONG)
        expected = reverse_bit(val)
        output = cached_reverse_bit(val)
        assert output == expected, f'Expected: {expected}. Received: {output} for \'{val}\''

def test_closest_int_same_bit_count():
    # Not going to brute force this. Consider worst case like 2^30. We need to check through (2^30-2^29).
    testcases = [
        (6, 5)
        , (8, 4)
        , (2**30, 2**29)
        , (7, 11)
    ]
    for (val, expected) in testcases:
        output = closest_int_same_bit_count(val)
        assert output == expected, f'Expected: {expected}. Received: {output} for \'{val}\''

def test_multiply():
    COUNTS = 1000
    MAX_INT = 2**32-1 # Reduced to 32 bit to prevent overflow
    for _ in range(COUNTS):
        val_1 = randint(0,MAX_INT)
        val_2 = randint(0,MAX_INT)
        expected = val_1 * val_2
        output = multiply(val_1, val_2)
        assert output == expected, f'Expected: {expected}. Received: {output} for \'{val_1=}\', \'{val_2=}\''

def test_divide():
    COUNTS = 1000
    MAX_LONG = 2**63-1
    for _ in range(COUNTS):
        val_1 = randint(0,MAX_LONG)
        val_2 = randint(0,MAX_LONG)
        expected = val_1 // val_2
        output = divide(val_1, val_2)
        assert output == expected, f'Expected: {expected}. Received: {output} for \'{val_1=}\', \'{val_2=}\''


def test_divide_with_no_overflow():
    COUNTS = 1000
    MAX_LONG = 2**63-1
    for _ in range(COUNTS):
        val_1 = randint(0,MAX_LONG)
        val_2 = randint(0,MAX_LONG)
        expected = val_1 // val_2
        output = divide_with_no_overflow(val_1, val_2)
        assert output == expected, f'Expected: {expected}. Received: {output} for \'{val_1=}\', \'{val_2=}\''


def test_power():
    COUNTS = 1000
    MAX_VALUE = 10
    MIN_VALUE = -10
    for _ in range(COUNTS):
        val_1 = uniform(MIN_VALUE, MAX_VALUE)
        val_2 = randint(MIN_VALUE, MAX_VALUE)
        expected = val_1 ** val_2
        output = power(val_1, val_2)
        assert isclose(expected, output), f'Expected: {expected}. Received: {output} for \'{val_1=}\', \'{val_2=}\''

def test_reverse_digits():
    COUNTS = 1000
    MAX_LONG = 2**63-1
    for _ in range(COUNTS):
        val = randint(-MAX_LONG,MAX_LONG)
        is_negative = val < 0
        tmp = -val if is_negative else val
        expected = int(str(tmp)[::-1])
        expected = -expected if is_negative else expected

        output_1 = reverse(val)
        output_2 = reverse_digits(val)
        assert expected == output_1, f'Expected: {expected}. Received: {output_1} for \'{val}\''
        assert expected == output_2, f'Expected: {expected}. Received: {output_2} for \'{val}\''

def test_is_palindrome_number():
    COUNTS = 1000
    MAX_LONG = 2**63-1
    for _ in range(COUNTS):
        val = randint(-MAX_LONG,MAX_LONG)
        expected = is_palindrome_number_expected(val)
        output = is_palindrome_number(val)
        assert output == expected, f'Expected: {expected}. Received: {output} for \'{val}\''

def test_generate_uniform_random_number():
    COUNTS = 1000
    MAX_LONG = 2**31-1
    for _ in range(COUNTS):
        val_1 = randint(-MAX_LONG,MAX_LONG)
        val_2 = randint(val_1, MAX_LONG) # Guaranteed val_2 to be greater than val_1
        output = generate_uniform_random_number(val_1, val_2)

        assert val_1 <= output <= val_2, f'Expected: {val_1} <= generated random number <= {val_2}. Received: {output} '
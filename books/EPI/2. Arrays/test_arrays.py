import pytest
from random import randint, shuffle, random

from arrays import *
from math import isclose

from copy import deepcopy
from functools import partial


@pytest.fixture
def even_odd_expected_func():
    def check(arr):
        is_odd = False
        n = len(arr)
        for i in range(n):
            if arr[i] & 1:
                is_odd = True
            elif is_odd: # is_even and previous element is odd
                return False
        return True
    return check

@pytest.fixture
def dutch_flag_partition_expected_func():
    def check(pivot_value,  arr):
        pivot = pivot_value
        n = len(arr)
        last_value = float('-inf')
        for i in range(n):
            num = arr[i]
            # Consider this case
            # [1,1,2,2,3,3]
            # 1 < 2 and -math.inf < 2
            # 1 < 2 and 1 < 2
            # 2 == 2 and 1 <= 2
            # 2 == 2 and 2 <= 2
            # 3 > 2 and 2 >= 2
            # 3 > 2 and 3 >= 2
            if (num < pivot and last_value <= pivot) \
                or (num == pivot and last_value <= pivot) \
                or (num > pivot and last_value >= pivot):
                last_value = num
            else:
                return False
                
        return True
    return check

@pytest.fixture
def generate_array_func():
    MAX_INT = (2**31-1)
    def generate_array(n: int, start: int = -MAX_INT, end: int = MAX_INT):
        return [randint(start, end) for _ in range(n)]
    return generate_array

@pytest.fixture
def parse_num_into_arr_func():
    def parse_num_into_arr(num):
            num_str = str(num)
            return [-int(num_str[i]) if i == 1 and num_str[0] == '-' else int(num_str[i]) for i in range(len(num_str)) if num_str[i] != '-']
    return parse_num_into_arr

@pytest.fixture
def can_reach_end_expected_func():
    def can_reach_end(A):
        furthest_reach_so_far, last_index = 0, len(A) - 1
        i=0
        while i <= furthest_reach_so_far and furthest_reach_so_far < last_index:
            furthest_reach_so_far = max(furthest_reach_so_far, A[i] + i)
            i+=1
        return furthest_reach_so_far >= last_index
    return can_reach_end

@pytest.fixture
def rearrange_expected_func():
    def rearrange_expected(A):
        if len(A) <= 1:
            return True
        valid = True
        n = len(A)
        for i in range(len(A)):
            if i == 0:
                valid = valid and A[i] < A[i+1]
            elif i == n-1:
                if i & 1:
                    valid = valid and A[i] > A[i-1]
                else:
                    valid = valid and A[i] < A[i-1]
            elif i & 1:
                valid = valid and A[i] > A[i-1] and A[i] > A[i+1]
            else:
                valid = valid and A[i] < A[i-1] and A[i] < A[i+1]
            if not valid:
                return valid
        return valid
    return rearrange_expected

@pytest.fixture
def generate_primes_expected_func():
    def generate_primes_expected(n):
        if n < 2:
            return []
        primes = []
        for i in range(2, n+1):
            for j in range(2, i):
                if i % j == 0:
                    break
            else:
                primes.append(i)
        return primes
    return generate_primes_expected

@pytest.fixture
def apply_permutation_expected_func():
    def apply_permutation_expected(perm, arr):
        n = len(arr)
        expected = [None] * n

        for i in range(n):
            if expected[i] is None: # It is not in an unvisited cycle
                prev_value = None
                nxt = i
                while expected[nxt] is None:
                    # Use expected[nxt] to hold current value
                    expected[nxt] = prev_value
                    tmp_nxt = perm[nxt]
                    prev_value = arr[nxt]
                    nxt = tmp_nxt
                expected[nxt] = prev_value
        return expected
    return apply_permutation_expected

@pytest.fixture
def sudoku_testcases():
    testcases = (
        (
            [
                [5, 3, 0, 0, 7, 0, 0, 0, 0]
                , [6, 0, 0, 1, 9, 5, 0, 0, 0]
                , [0, 9, 8, 0, 0, 0, 0, 6, 0]
                , [8, 0, 0, 0, 6, 0, 0, 0, 3]
                , [4, 0, 0, 8, 0, 3, 0, 0, 1]
                , [7, 0, 0, 0, 2, 0, 0, 0, 6]
                , [0, 6, 0, 0, 0, 0, 2, 8, 0]
                , [0, 0, 0, 4, 1, 9, 0, 0, 5]
                , [0, 0, 0, 0, 8, 0, 0, 7, 9]
            ], True
        )
        , (
            [
                [5, 3, 4, 6, 7, 8, 9, 1, 2]
                , [6, 7, 2, 1, 9, 5, 3, 4, 8]
                , [1, 9, 8, 3, 4, 2, 5, 6, 7]
                , [8, 5, 9, 7, 6, 1, 4, 2, 3]
                , [4, 2, 6, 8, 5, 3, 7, 9, 1]
                , [7, 1, 3, 9, 2, 4, 8, 5, 6]
                , [9, 6, 1, 5, 3, 7, 2, 8, 4]
                , [2, 8, 7, 4, 1, 9, 6, 3, 5]
                , [3, 4, 5, 2, 8, 6, 1, 7, 9]
            ], True
        )
        , (
            [
                [5, 3, 4, 6, 7, 8, 9, 1, 2],
                [5, 7, 2, 1, 9, 5, 3, 4, 8],
                [1, 9, 8, 3, 4, 2, 5, 6, 7],
                [8, 5, 9, 7, 6, 1, 4, 2, 3],
                [4, 2, 6, 8, 5, 3, 7, 9, 1],
                [7, 1, 3, 9, 2, 4, 8, 5, 6],
                [9, 6, 1, 5, 3, 7, 2, 8, 4],
                [2, 8, 7, 4, 1, 9, 6, 3, 5],
                [3, 4, 5, 2, 8, 6, 1, 7, 9]
            ], False
        )
        , (
            [
                [5, 3, 4, 6, 7, 8, 9, 1, 2],
                [6, 7, 2, 1, 9, 5, 3, 4, 8],
                [1, 9, 8, 5, 4, 2, 5, 6, 7],
                [8, 5, 9, 7, 6, 1, 4, 2, 3],
                [4, 2, 6, 8, 5, 3, 7, 9, 1],
                [7, 1, 3, 9, 2, 4, 8, 5, 6],
                [9, 6, 1, 5, 3, 7, 2, 8, 4],
                [2, 8, 7, 4, 1, 9, 6, 3, 5],
                [3, 4, 5, 2, 8, 6, 1, 7, 9]
            ], False
        )
        , (
            [
                [5, 3, 0, 9, 7, 0, 0, 0, 0]
                , [6, 0, 0, 1, 9, 5, 0, 0, 0]
                , [0, 9, 8, 0, 0, 0, 0, 6, 0]
                , [8, 0, 0, 0, 6, 0, 0, 0, 3]
                , [4, 0, 0, 8, 0, 3, 0, 0, 1]
                , [7, 0, 0, 0, 2, 0, 0, 0, 6]
                , [0, 6, 0, 0, 0, 0, 2, 8, 0]
                , [0, 0, 0, 4, 1, 9, 0, 0, 5]
                , [0, 0, 0, 0, 8, 0, 0, 7, 9]
            ], False
        )
    )
    return testcases

@pytest.fixture
def rotate_matrix_testcases():
    testcases = (
        (
            [[1, 2], [3, 4]],
            [[3, 1], [4, 2]]
        ),
        (
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[7, 4, 1], [8, 5, 2], [9, 6, 3]]
        ),
        (
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            [[13, 9, 5, 1], [14, 10, 6, 2], [15, 11, 7, 3], [16, 12, 8, 4]]
        ),
        (
            [[1, 2], [3, 4], [5, 6]],
            [[5, 3, 1], [6, 4, 2]]
        ),
        (
            [[1, 2, 3], [4, 5, 6]],
            [[4, 1], [5, 2], [6, 3]]
        ),
        (
            [[5]],
            [[5]]
        ),
        (
            [],
            []
        ),
        (
            [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]],
            [[21, 16, 11, 6, 1], [22, 17, 12, 7, 2], [23, 18, 13, 8, 3], [24, 19, 14, 9, 4], [25, 20, 15, 10, 5]]
        ),
        (
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
            [[10, 7, 4, 1], [11, 8, 5, 2], [12, 9, 6, 3]]
        ),
        (
            [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24], [25, 26, 27, 28, 29, 30], [31, 32, 33, 34, 35, 36]],
            [[31, 25, 19, 13, 7, 1], [32, 26, 20, 14, 8, 2], [33, 27, 21, 15, 9, 3], [34, 28, 22, 16, 10, 4], [35, 29, 23, 17, 11, 5], [36, 30, 24, 18, 12, 6]]
        )
    )
    return testcases

def test_even_odd(even_odd_expected_func, generate_array_func):
    COUNTS = 1000
    MAX_LENGTH = 1_000

    for _ in range(COUNTS):
        arr = generate_array_func(MAX_LENGTH)
        output = even_odd(arr)
        is_correct = even_odd_expected_func(output)
        assert is_correct, f'Failed for {arr}'
    
def test_dutch_flag_partition(dutch_flag_partition_expected_func, generate_array_func):
    COUNTS = 1000
    MAX_LENGTH = 1_000

    for _ in range(COUNTS):
        arr = generate_array_func(MAX_LENGTH)
        pivot_index = randint(0, len(arr)-1)
        pivot_value = arr[pivot_index]
        output = dutch_flag_partition(pivot_index, arr)
        is_correct = dutch_flag_partition_expected_func(pivot_value, output)
        assert is_correct, f'Failed for {arr}'

def test_plus_one(parse_num_into_arr_func):
    COUNTS = 1000

    for _ in range(COUNTS):
        number = randint(0, 2*31-1)
        arr = parse_num_into_arr_func(number)
        output = plus_one(arr)
        expected = list(map(int, str(number+1)))
        assert output == expected, f'Failed for {arr}'

def test_multiply(parse_num_into_arr_func):
    COUNTS = 1000

    for _ in range(COUNTS):
        num1 = randint(-(2**63-1), 2**63-1)
        num2 = randint(-(2**63-1), 2**63-1)

        nums1 = parse_num_into_arr_func(num1)
        nums2 = parse_num_into_arr_func(num2)
        
        output = multiply(nums1, nums2)

        expected = parse_num_into_arr_func(num1*num2)
        assert len(expected) == len(output) and all([expected[i]==output[i] for i in range(len(expected))]), f'\n{num1=}, {num2=}\nExpected: {expected}\nOutput:{output}'
   
def test_can_reach_end(generate_array_func, can_reach_end_expected_func):
    COUNTS = 1000
    MAX_LENGTH = 1_000
    for _ in range(COUNTS):
        arr = generate_array_func(MAX_LENGTH)

        output = can_reach_end(arr)
        expected = can_reach_end_expected_func(arr)

        assert expected == output, f'\nExpected: {expected}\nOutput:{output}\n{arr=}'

def test_delete_duplicates(generate_array_func):
    COUNTS = 1000
    MAX_LENGTH = 1_000
    for _ in range(COUNTS):
        arr = generate_array_func(MAX_LENGTH)
        arr.sort()

        output = delete_duplicates(arr)
        expected = len(set(arr))
        assert output == expected, f'\nExpected: {expected}\nOutput:{output}\n{arr=}'

def test_buy_and_sell_stock_once():
    testcases = (
        ([7, 1, 5, 3, 6, 4], 5)  # Buy at 1, sell at 6, profit = 6 - 1 = 5
        , ([7, 6, 4, 3, 1], 0)    # No profitable transactions, so profit = 0
        , ([1, 2], 1)             # Buy at 1, sell at 2, profit = 2 - 1 = 1
        , ([5, 4, 3, 2, 1], 0)    # No profitable transactions, so profit = 0
        , ([3, 2, 6, 5, 0, 3], 4) # Buy at 2, sell at 6, profit = 6 - 2 = 4
        , ([10, 22, 5, 75, 65, 80], 75)  # Buy at 5, sell at 80, profit = 80 - 5 = 75
        , ([2, 4, 1], 2)          # Buy at 2, sell at 4, profit = 4 - 2 = 2
        , ([1, 2, 3, 4, 5], 4)    # Buy at 1, sell at 5, profit = 5 - 1 = 4
        , ([3, 2, 6, 8, 5, 1], 6) # Buy at 2, sell at 8, profit = 8 - 2 = 6
        , ([1, 4, 2, 5, 7], 6)    # Buy at 1, sell at 7, profit = 7 - 1 = 6
    )

    for i, (arr, expected) in enumerate(testcases):
        output = buy_and_sell_stock_once(arr)
        assert output == expected, f'\nExpected: {expected}\nOutput:{output}\n{arr=}'

def test_buy_and_sell_stock_twice():
    testcases = [
        ([3,2,6,5,0,3], 7)
        , (list(range(10000, -1, -1)) + [0] * 90_000, 0)
        , ([1,2,3,4,5], 4)
    ]
    for arr, expected in testcases:
        output = buy_and_sell_stock_twice(arr)
        assert output == expected, f'\nExpected: {expected}\nOutput:{output}\n{arr=}'

def test_buy_and_sell_stock_twice_space_optimised():
    testcases = [
        ([3,2,6,5,0,3], 7)
        , (list(range(10000, -1, -1)) + [0] * 90_000, 0)
        , ([1,2,3,4,5], 4)
    ]
    for arr, expected in testcases:
        output = buy_and_sell_stock_twice_space_optimised(arr)
        assert output == expected, f'\nExpected: {expected}\nOutput:{output}\n{arr=}'

def test_buy_and_sell_stock_thrice_space_optimised():
    testcases = [
        ([3,2,6,5,0,3], 7)
        , (list(range(10000, -1, -1)) + [0] * 90_000, 0)
        , ([1,2,3,4,5], 4)
        , ([3, 2, 6, 5, 0, 3, 6, 4, 10], 16)
    ]
    for arr, expected in testcases:
        output = buy_and_sell_stock_thrice_space_optimised(arr)
        assert output == expected, f'\nExpected: {expected}\nOutput:{output}\n{arr=}'

def test_buy_and_sell_stock_at_most_k_space_optimised():
    testcases = [
        ([3,2,6,5,0,3], 2, 7)
        , (list(range(10000, -1, -1)) + [0] * 90_000, 2, 0)
        , ([1,2,3,4,5], 2, 4)
        , ([3, 2, 6, 5, 0, 3, 6, 4, 10], 3, 16)
        , ([3, 2, 6, 5, 0, 3, 6, 4, 10], 2, 14)
    ]
    for (arr, k, expected) in testcases:
        output = buy_and_sell_stock_at_most_k_space_optimised(arr, k)
        assert output == expected, f'\nExpected: {expected}\nOutput:{output}\n{arr=}'

def test_rearrange(generate_array_func, rearrange_expected_func):
    COUNTS = 1000
    MAX_LENGTH = 1_000
    for _ in range(COUNTS):
        arr = generate_array_func(MAX_LENGTH)
        output = rearrange(arr)
        assert rearrange_expected_func(output), f'\nExpected: {expected}\nOutput:{output}\n{arr=}'

def test_generate_primes(generate_primes_expected_func):
    COUNTS = 1000
    MAX_LENGTH = 1_000
    for _ in range(COUNTS):
        n = randint(0, MAX_LENGTH+1)

        output = generate_primes(n)
        expected = generate_primes_expected_func(n)

        assert len(output) == len(expected) and all([a == b for a,b in zip(output, expected)]), f'\nExpected: {expected}\nOutput:{output}\n{n=}'

def test_generate_primes_odd_optimised(generate_primes_expected_func):
    COUNTS = 1000
    MAX_LENGTH = 1_000
    for _ in range(COUNTS):
        n = randint(0, MAX_LENGTH+1)

        output = generate_primes_odd_optimised(n)
        expected = generate_primes_expected_func(n)

        assert len(output) == len(expected) and all([a == b for a,b in zip(output, expected)]), f'\nExpected: {expected}\nOutput:{output}\n{n=}'

def test_apply_permutation(generate_array_func, apply_permutation_expected_func):
    COUNTS = 1000
    MAX_LENGTH = 1_000
    perm = list(range(MAX_LENGTH))
    arr = generate_array_func(MAX_LENGTH)
    for _ in range(COUNTS):
        shuffle(perm)
        # Because apply_permutation is performing cyclic permutation
        # We need to compute the cyclic position
        expected = apply_permutation_expected_func(perm, arr)

        apply_permutation(perm, arr)
        assert all([exp == num for exp, num in zip(expected, arr)]), f'Expected:{expected}\nOutput:{arr}\nPerm: {perm}'

def test_apply_permutation_check_leftmost(generate_array_func, apply_permutation_expected_func):
    COUNTS = 1000
    MAX_LENGTH = 1_000
    perm = list(range(MAX_LENGTH))
    arr = generate_array_func(MAX_LENGTH)
    for _ in range(COUNTS):
        shuffle(perm)
        # Because apply_permutation is performing cyclic permutation
        # We need to compute the cyclic position
        expected = apply_permutation_expected_func(perm, arr)

        apply_permutation_check_leftmost(perm, arr)
        assert all([exp == num for exp, num in zip(expected, arr)]), f'Expected:{expected}\nOutput:{arr}\nPerm: {perm}'

def test_next_permutation():
    COUNTS = 100
    MAX_LENGTH = 10 # The brute force is n!. 10! = 3,628,800
    for _ in range(COUNTS):
        LENGTH = randint(1, MAX_LENGTH)
        perm = list(range(LENGTH))
        shuffle(perm)
        # Compute the next permutation expected
        from itertools import dropwhile, permutations, islice
        valid_next_permutations = list(islice(dropwhile(lambda p: list(p) <= perm, permutations(range(LENGTH))), 1))
        expected = [] if len(valid_next_permutations) == 0 else valid_next_permutations[0]

        output = next_permutation(perm)
        assert all([exp == num for exp, num in zip(expected, output)]), f'Expected:{expected}\nOutput:{output}\nPerm: {perm}'

def test_random_offline_sampling():
    n = 100

    counts = {}

    trials = 10000
    expected_value = 0

    for _ in range(trials):
        arr = list(range(n)) # This random offline sampling is in-place, we need to recreate arr
        k = randint(1, len(arr))
        random_offline_sampling(k, arr)

        for i in range(k):
            counts[arr[i]] = counts.get(arr[i], 0)+1
    
        expected_value += k/len(arr)

    expected_value /= trials
    for i in range(n):
        assert isclose(counts[i]/trials, expected_value, abs_tol=expected_value*0.1)

def test_random_online_sampling():
    n = 100
    arr = list(range(n))

    counts = {}

    trials = 10000
    expected_value = 0

    for _ in range(trials):
        k = randint(1, len(arr))
        sampled = random_online_sampling(k, arr)

        for value in sampled:
            counts[value] = counts.get(value, 0)+1
    
        expected_value += k/len(arr)

    expected_value /= trials
    for i in range(n):
        assert isclose(counts[i]/trials, expected_value, abs_tol=expected_value*0.05)

def test_compute_random_permutation():
    """
    This is pretty difficult to test
    This is almost always sampling k elements without replacement in range(list(arr))
    Means we can never be counting by just frequency of elements like we do in testing random offline sampling
    Because the counts of all elements will equal to the number of trials
    
    How about counting permutations?
    We can try count permutations which is tuple(permutation) and count this tuple
    10! = 3,628,800 
    This means if we only generate 10,000 trials, most of them will be 0
    
    How about we store the counts via position?
    pos[0] = 1/n
    pos[1] = (n-1)/n * 1/(n-1) = 1/n
    This means we need to assert for every position and every possible value which is n^2
    """
    n = 100
    arr = list(range(n))

    counts = {}
    freq = {}

    trials = 100000
    expected_value = 0

    for _ in range(trials):
        permutation = compute_random_permutation(arr)

        for pos, value in enumerate(permutation):
            counts[pos] = counts.get(pos, {})
            counts[pos][value] = counts[pos].get(value, 0) + 1
            freq[value] = freq.get(value, 0)+1
    
    expected_value = 1/n*trials
    
    for i in range(n):
        for j in range(n):
            assert isclose(counts[i].get(j, 0), expected_value, abs_tol = 0.2*expected_value, rel_tol = 0)
    # print(counts)

def test_random_subset():
    n = 100
    trials = 100_000

    counts = [0] * n

    arr = list(range(n))

    expected_value = 0

    for _ in range(trials):
        k = randint(1,n)
        subset = random_subset(arr, k)

        for value in subset:
            counts[value] += 1
    
        expected_value += k
    
    # Every element should have k/n probability of getting selected
    expected_value /= n
    for value in range(n):
        assert isclose(counts[value], expected_value, abs_tol = 0.1*expected_value, rel_tol=0)

def test_nonuniform_random_number_generator():
    n = 5
    trials = 100_000
    
    counts = [0] * n
    values = list(range(n))

    expected_value = [0]*n

    for _ in range(trials):
        probabilities = [0]*n
        accu = 0
        for i in range(n):
            if i < n-1:
                probabilities[i] += random()*0.2
                
                accu += probabilities[i]
            else:
                probabilities[i] = 1-accu
            expected_value[i] += probabilities[i]
        generated = nonuniform_random_number_generator(values, probabilities)
        counts[generated] += 1
    
    for value in range(n):
        assert isclose(counts[value], expected_value[value], abs_tol = 0.1*expected_value[value], rel_tol=0)
        
def test_is_valid_sudoku(sudoku_testcases):
    for i, (matrix, expected) in enumerate(sudoku_testcases, start=1):
        output = is_valid_sudoku(matrix)
        assert output == expected, f'Failed Testcase {i}'

def test_is_valid_sudoku_pythonic(sudoku_testcases):
    for i, (matrix, expected) in enumerate(sudoku_testcases, start=1):
        output = is_valid_sudoku_pythonic(matrix)
        assert output == expected, f'Failed Testcase {i}'

def test_matrix_in_spiral_order():
    testcases = (
        (
            [[1,2,3],[4,5,6],[7,8,9]]
            , [1,2,3,6,9,8,7,4,5]
        )
        , (
            [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
            , [1,2,3,4,8,12,11,10,9,5,6,7]
        )
        , (
            [[1]]
            , [1]
        )
        , (
            [
                [1, 2],
                [4, 3]
            ], [1, 2, 3, 4]
        )
        , (
            [
                [1, 2, 3],
                [8, 9, 4],
                [7, 6, 5]
            ], [1, 2, 3, 4, 5, 6, 7, 8, 9]
        )
        , (
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8]
            ], [1, 2, 3, 4, 8, 7, 6, 5]
        )
        , (
            [
                [1, 2, 3],
                [8, 9, 4],
                [7, 6, 5]
            ], [1, 2, 3, 4, 5, 6, 7, 8, 9]
        )
        , (
            [
                [1, 2, 3, 4],
                [12, 13, 14, 5],
                [11, 16, 15, 6],
                [10, 9, 8, 7]
            ], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        )
        , (
            [
                [1, 2],
                [4, 3],
                [5, 6]
            ], [1, 2, 3, 6, 5, 4]
        )
        , (
            [
                [1, 2, 3, 4, 5],
                [16, 17, 18, 19, 6],
                [15, 24, 25, 20, 7],
                [14, 23, 22, 21, 8],
                [13, 12, 11, 10, 9]
            ], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
        )
        , (
            [], []
        )
        , (
            [
                [1],
                [2],
                [3],
                [4],
                [5]
            ]
            , [1, 2, 3, 4, 5]
        )
        , (
            [
                [1, 2, 3, 4, 5]
            ], [1, 2, 3, 4, 5]
        )
        , (
            [
                [1, 2, 3],
                [8, 9, 4],
                [7, 6, 5],
                [12, 11, 10]
            ], [1, 2, 3, 4, 5, 10, 11, 12, 7, 8, 9, 6]
        )
    )

    for i, (testcase, expected) in enumerate(testcases):
        output = matrix_in_spiral_order(testcase)
        assert all([out==exp for out, exp in zip(output, expected)]), f'Failed Testcase {i}: {expected} {output}'

def test_rotate_matrix(rotate_matrix_testcases):
    for i, (matrix, expected) in enumerate(rotate_matrix_testcases):
        output = rotate_matrix(matrix)
        output_rows = len(output)
        output_cols = len(output[0]) if output_rows else 0

        expected_rows = len(expected)
        expected_cols = len(expected[0]) if expected_rows else 0

        output_shape = (output_rows, output_cols)
        expected_shape = (expected_rows, expected_cols)

        assert output_shape == expected_shape, f'Testcase {i}: Shapes of output ({output_shape}) and expected ({expected_shape}) is different'

        assert all([all([output[row][col] == expected[row][col] for row in range(output_rows)]) for col in range(output_cols)]), f'Testcase {i}: Different values {output}'

def test_rotate_matrix_inplace(rotate_matrix_testcases):
    is_valid_square_matrix = lambda matrix: len(matrix[0]) == 0 or len(matrix[0]) == len(matrix[0][0])
    keep_square_matrix = partial(filter, is_valid_square_matrix)
    # Inplace can only work on square matrix
    
    for i, (matrix, expected) in enumerate(keep_square_matrix(rotate_matrix_testcases)):

        assert len(matrix) == 0 or len(matrix) == len(matrix[0]), f'{is_valid_square_matrix(matrix)=}'
        output = deepcopy(matrix)
        rotate_matrix_inplace(output)

        output_rows = len(output)
        output_cols = len(output[0]) if output_rows else 0

        expected_rows = len(expected)
        expected_cols = len(expected[0]) if expected_rows else 0

        output_shape = (output_rows, output_cols)
        expected_shape = (expected_rows, expected_cols)

        
        assert all([all([output[row][col] == expected[row][col] for row in range(output_rows)]) for col in range(output_cols)]), f'Testcase {i}:\n{pformat(matrix)} \nOutput {pformat(output)}\nExpected:{pformat(expected)}'

def test_generate_pascal_triangle():
    testcases = (
        (0, [])
        , (1, [[1]])
        , (2, [[1], [1, 1]])
        , (5, [[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]])
        , (30, [[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1],[1,5,10,10,5,1],[1,6,15,20,15,6,1],[1,7,21,35,35,21,7,1],[1,8,28,56,70,56,28,8,1],[1,9,36,84,126,126,84,36,9,1],[1,10,45,120,210,252,210,120,45,10,1],[1,11,55,165,330,462,462,330,165,55,11,1],[1,12,66,220,495,792,924,792,495,220,66,12,1],[1,13,78,286,715,1287,1716,1716,1287,715,286,78,13,1],[1,14,91,364,1001,2002,3003,3432,3003,2002,1001,364,91,14,1],[1,15,105,455,1365,3003,5005,6435,6435,5005,3003,1365,455,105,15,1],[1,16,120,560,1820,4368,8008,11440,12870,11440,8008,4368,1820,560,120,16,1],[1,17,136,680,2380,6188,12376,19448,24310,24310,19448,12376,6188,2380,680,136,17,1],[1,18,153,816,3060,8568,18564,31824,43758,48620,43758,31824,18564,8568,3060,816,153,18,1],[1,19,171,969,3876,11628,27132,50388,75582,92378,92378,75582,50388,27132,11628,3876,969,171,19,1],[1,20,190,1140,4845,15504,38760,77520,125970,167960,184756,167960,125970,77520,38760,15504,4845,1140,190,20,1],[1,21,210,1330,5985,20349,54264,116280,203490,293930,352716,352716,293930,203490,116280,54264,20349,5985,1330,210,21,1],[1,22,231,1540,7315,26334,74613,170544,319770,497420,646646,705432,646646,497420,319770,170544,74613,26334,7315,1540,231,22,1],[1,23,253,1771,8855,33649,100947,245157,490314,817190,1144066,1352078,1352078,1144066,817190,490314,245157,100947,33649,8855,1771,253,23,1],[1,24,276,2024,10626,42504,134596,346104,735471,1307504,1961256,2496144,2704156,2496144,1961256,1307504,735471,346104,134596,42504,10626,2024,276,24,1],[1,25,300,2300,12650,53130,177100,480700,1081575,2042975,3268760,4457400,5200300,5200300,4457400,3268760,2042975,1081575,480700,177100,53130,12650,2300,300,25,1],[1,26,325,2600,14950,65780,230230,657800,1562275,3124550,5311735,7726160,9657700,10400600,9657700,7726160,5311735,3124550,1562275,657800,230230,65780,14950,2600,325,26,1],[1,27,351,2925,17550,80730,296010,888030,2220075,4686825,8436285,13037895,17383860,20058300,20058300,17383860,13037895,8436285,4686825,2220075,888030,296010,80730,17550,2925,351,27,1],[1,28,378,3276,20475,98280,376740,1184040,3108105,6906900,13123110,21474180,30421755,37442160,40116600,37442160,30421755,21474180,13123110,6906900,3108105,1184040,376740,98280,20475,3276,378,28,1],[1,29,406,3654,23751,118755,475020,1560780,4292145,10015005,20030010,34597290,51895935,67863915,77558760,77558760,67863915,51895935,34597290,20030010,10015005,4292145,1560780,475020,118755,23751,3654,406,29,1]])
    )

    def check(matrix1, matrix2):
        n1, n2 = len(matrix1), len(matrix2)
        if n1 != n2:
            return False
        
        for i in range(n1):
            m1, m2 = len(matrix1[i]), len(matrix2[i])
            if m1 != m2:
                return False
            valid = all(matrix1[i][j] == matrix2[i][j] for j in range(m1))
            if not valid:
                return False
        return True

    for i, (n, expected) in enumerate(testcases):
        output = generate_pascal_triangle(n)
        assert check(output, expected), f'Failed Testcase {i}: {n=}'

def test_generate_pascal_triangle_nth_row():
    testcases = (
        (0, [])
        , (1, [1])
        , (2, [1, 1])
        , (5, [1,4,6,4,1])
        , (30, [1,29,406,3654,23751,118755,475020,1560780,4292145,10015005,20030010,34597290,51895935,67863915,77558760,77558760,67863915,51895935,34597290,20030010,10015005,4292145,1560780,475020,118755,23751,3654,406,29,1])
    )

    for i, (n, expected) in enumerate(testcases):
        output = generate_pascal_triangle_nth_row(n)
        assert len(expected) == len(output) and all([exp == out for (exp, out) in zip(expected, output)]), f'Failed Testcase {i}: {n=}'
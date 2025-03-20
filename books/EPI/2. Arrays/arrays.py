from typing import List, Any, Iterator, Union
from random import randint, random

from itertools import accumulate
from bisect import bisect
from collections import Counter

from math import isclose

def even_odd(A: List[int]) -> List[int]:
    # Move all the even elements to the front in O(n) time and O(1) space complexity
    next_even, next_odd = 0, len(A) - 1
    while next_even < next_odd:
        if A[next_even] % 2 == 0:
            next_even += 1
        else :
            A[next_even], A[next_odd] = A[next_odd], A[next_even]
            next_odd -= 1
    return A

RED, WHITE, BLUE = range(3)

# This is usable for Quick Sort partitioning especially for equal elements array
def dutch_flag_partition(pivot_index: int, A: List[int]) -> List[int]:
    pivot = A[pivot_index]
    
    # Keep the following invariants during partitioning
    # bottom group: A[:smaller]
    # equal group: A[smaller:equal]
    # unclassified group: A[equal:larger]
    # top group: A[larger:]

    smaller = equal = 0
    larger = len(A)

    # Keep iterating as long as there is an unclassified element
    while equal < larger:
        if A[equal] < pivot:
            A[smaller], A[equal] = A[equal], A[smaller]
            smaller += 1
            equal += 1
        elif A[equal] == pivot:
            equal += 1
        else:
            larger -= 1
            A[equal], A[larger] = A[larger], A[equal]
    return A

def plus_one(A: List[int]) -> List[int]:
    n = len(A)
    A[-1] += 1
    for i in range(n-1, 0, -1):
        if A[i] >= 10:
            carry, A[i] = divmod(A[i], 10)
            A[i-1] += carry
        else:
            break
    # [9,9,9] -> [1,0,0,0] 
    # Notice the only time it will overflow the array is when all the values in the array is 9.
    if A[0] >= 10:
        A.append(0)
        A[0] //= 10
    return A

def multiply(num1: List[int], num2: List[int]) -> List[int]:
    # It will be negative if num1[0] XOR num2[0] is negative
    is_negative = (num1[0] < 0) ^ (num2[0] < 0)
    num1[0] = abs(num1[0])
    num2[0] = abs(num2[0])

    n1 = len(num1)
    n2 = len(num2)
    m = n1+n2
    # Multiplcation of 2 number will result in at most n1+n2 number of digits
    ans = [0] * m

    for i in range(n1):
        for j in range(n2):
            ans[m-i-j-1] += num1[n1-1-i] * num2[n2-1-j]
            carry, ans[m-i-j-1] = divmod(ans[m-i-j-1], 10)
            ans[m-i-j-2] += carry

    # Removing leading zeros
    if len(ans) > 1 and ans[0] == 0:
        cutoff = 0
        for i in range(m):
            if ans[i] == 0:
                cutoff = i
            else:
                break
        if cutoff+1 == m:
            return [0]
        ans = ans[cutoff+1:]

    if is_negative:
        ans[0] = -ans[0]
    return ans

def can_reach_end(A: List[int]) -> List[int]:
    furthest = 0
    for i in range(len(A)):
        if i > furthest:
            return False
        furthest = max(furthest, A[i] + i)
        if furthest >= len(A)-1:
            return True
    return True

def delete_duplicates(A: List[int]) -> int:
    count = 0
    prev = None
    n = len(A)
    for i in range(n):
        if A[i] == prev:
            continue
        A[count], A[i] = A[i], A[count]
        prev = A[count]
        count += 1
    return count

def buy_and_sell_stock_once(prices: List[int]) -> int:
    profit = 0
    min_price = float('inf')
    
    for price in prices:
        profit = max(profit, price-min_price)
        min_price = min(price, min_price)
    return profit

def buy_and_sell_stock_twice(prices: List[int]) -> int:
    n = len(prices)
    max_profit = [0] * (n+1)
    min_price = float('inf')

    for i in range(n):
        price = prices[i]
        min_price = min(price, min_price)
        max_profit[i+1] = max(max_profit[i], price-min_price )

    max_price = float('-inf')
    ans = 0
    for i in range(n-1, -1, -1):
        price = prices[i]
        max_price = max(max_price, price)
        ans = max(
            ans, max_price-price+max_profit[i]
        )
    return ans

def buy_and_sell_stock_twice_space_optimised(prices: List[int]) -> int:
    """
    Keep track of the gain from the previous profit and add the both profit together
    First Transaction -> Keep track of lowest price and find the maximum of the price - lowest_price
    Second Transaction -> Need to ensure the second transaction happens after first transaction
    The trick here is push the min price of the second transaction down based on the max_profit1.
    In this case, we can consider the profit of the second transaction is the pushed_down_min_price - price
    In this way, we are always considering the maximum profit after the first transaction
    A more intuitive way is that we bought at the 'pushed_down' price. 
    Consider this is that we use the profit from first transaction to buy the second lowest price

    In theory, you can extend this further to 3 or more transactions
    """

    min_price_1 = min_price_2 = float('inf')
    max_profit_1 = max_profit_2 = 0

    for i, price in enumerate(prices):
        min_price_1 = min(min_price_1, price) # Keep tracks of lowest price for first transaction
        max_profit_1 = max(max_profit_1, price-min_price_1) # Keep track of highest profit for first transaction
        min_price_2 = min(min_price_2, price-max_profit_1) # Keep track the next lowest price after highest profit
        max_profit_2 = max(max_profit_2, price-min_price_2) # current_price - lowest_price + max_profit_1 (previous highest price - previous lowest price)
    
    return max_profit_2

def buy_and_sell_stock_thrice_space_optimised(prices: List[int]) -> int:
    min_price_1 = min_price_2 = min_price_3 = float('inf')
    max_profit_1 = max_profit_2 = max_profit_3 = 0

    for i, price in enumerate(prices):
        min_price_1 = min(min_price_1, price)
        max_profit_1 = max(max_profit_1, price - min_price_1)

        min_price_2 = min(min_price_2, price - max_profit_1)
        max_profit_2 = max(max_profit_2, price - min_price_2)

        min_price_3 = min(min_price_3, price-max_profit_2)
        max_profit_3 = max(max_profit_3, price - min_price_3)
    
    return max_profit_3

# Extend this questions to at most k transactions
# Time Complexity: O(kn), Space Complexity: O(k)
def buy_and_sell_stock_at_most_k_space_optimised(prices: List[int], k: int) -> int:
    if k <= 0:
        return 0
    
    min_prices = [float('inf')] * k
    max_profits = [0] * k

    for i, price in enumerate(prices):
        for j in range(k):
            if j == 0:
                min_prices[j] = min(min_prices[j], price)
            else:
                min_prices[j] = min(min_prices[j], price - max_profits[j-1])
            
            max_profits[j] = max(max_profits[j], price - min_prices[j])
    return max_profits[k-1]

# Alternation
def rearrange(A: List[int]) -> List[int]:
    output = A[:] # Deepcopy
    for i in range(len(output)):
        output[i:i + 2] = sorted(output[i:i + 2], reverse=i % 2)
    return output

def rearrange(arr: List[int]) -> List[int]:
    n = len(arr)
    output = [0] * n

    for i in range(n):
        if i & 1: # i is odd, it needs to be larger
            if output[i-1] > arr[i]:
                output[i-1], output[i] = arr[i], output[i-1] 
            else:
                output[i] = arr[i]
        else: # i is even, it needs to be smaller
            if output[i-1] < arr[i]:
                output[i-1], output[i] = arr[i], output[i-1] 
            else:
                output[i] = arr[i]
    return output

def generate_primes(n: int) -> int:
    is_primes = [False, False] + [True] * (n-1) # 0 and 1 is not prime
    for prime in range(2, int(n**0.5)+1): # Run to sqrt of n, inner loop is running i*i, there is no need to run larger than sqrt(n) since it will always exceed n
        if is_primes[prime]: # Run if it is prime
            for j in range(prime*prime, n+1, prime): # Look for multiples of prime, set them to non-primes
                is_primes[j] = False
    return [p for p in range(n+1) if is_primes[p]]

# Given n, return all prines up to and including n.
def generate_primes_odd_optimised(n: int):
    """
    Run only for odd numbers
    1. We track is_prime for odd number only starting for 3
    2. If the current index is prime
        2a. Recreate the odd number based on the index -> i*2 + 3
            How we derive it? 
            We are only running for odd numbers so every index should be incremented by 2.
            We are also starting from 3.
            i*2 + 1 -> This give us odd number sequence but starts from 1
            i*2 + 1 + 2 -> This give us odd number sequence but starts from 3
        2b. We need to start marking non-primes from prime * prime (similar to generate_primes (Sieve of Eratosthenes))
            This is the tricky part.
            p = (i*2+3)
            p*p = (2*i+3)**2 = (4*i**2 + 12*i + 9)
            This p*p is the actual squared prime value and i is the index of the original prime value
            We need the index of the squared prime value
            (4*i**2 + 12*i + 9) = 2 * (index of p*p) + 3
            2*(index of p*p) = 4*i**2 + 12*i + 6
            index of p*p = 2*i**2 + 6*i + 3
            
            We will start with 2*i**2 + 6*i + 3 to size
        2c. We need to be increment by prime to mark multiple of prime
            This is harder to understand than 2b.
            You need mark multiples of prime while SKIPPING EVEN number multiple of primes
            i = (p-3)/2
            p*p = (4*i**2 + 12*i + 9)
            p = 2*i+3

            Desired next index should be incremented by 2p because we want to skip EVEN number of primes
            desired_index = ((p*p + 2p)-3) / 2 = (4*i**2 + 12*i + 9 + 4*i + 6 - 3) / 2 = 2*i**2 + 8*i + 6
            p*p index = 2*i**2 + 6*i + 3
            increment_index_value = 2*i**2 + 8*i + 6 - 2*i**2 + 6*i + 3 = 2*i + 3 = p

    """
    if n < 2:
        return []
    
    size=(n-3)//2+1 # Number of odd numbers from 3 to n
    primes = [2] # Stores the primes from 1 to n.

    # is_primes[i] represents (2i + 3) is prime or not.
    # Initially set each to True. Then use sieving to eliminate nonprimes.
    is_prime = [True] * size
    for i in range(size):
        if is_prime[i] :
            p = i*2+3
            primes.append(p)
            # Sieving from p^2, where p^2 = (4i^2 + 12i + 9). The index in is_prime
            # is (2i^2 + 6i + 3) because is_prime[i] represents 2i + 3.
            
            # Note that we need to use long for j because p^2 night overfTow.
            for j in range(2 * i**2 + 6 * i + 3, size, p):
                is_prime[j] = False
    return primes

def apply_permutation(perm: List[int], A: List[Any]) -> None:
    """
    Apply a cyclic permutation of copying values to index based on perm[index]
    As long as the index of the perm value are from 0 to len(A) and they are unique, we are guarenteed to have one or more closed loop.
    Ultimately, the algorithm is to copy values around within each closed loop.
    After the complete

    To illustrate it better, you can view the example below.
    Time Complexity: O(n), where n is the length of A
    Space Complexity: O(1), In place

    perm = [1, 2, 3, 0] (0 -> 1 -> 2 -> 3 -> 0) You can view this as arr[0] copy to arr[1], arr[1] copy to arr[2]... arr[3] copy to arr[0]
    arr = [0, 1, 2, 3]

    nxt = 0
    perm[nxt] = 1
    arr = [1, 0, 2, 3], perm = [-1, 2, 3, 0]

    nxt = 1
    perm[nxt] = 2
    arr = [2, 0, 1, 3], perm = [-1, -2, 3, 0]

    nxt = 2
    perm[nxt] = 3
    arr = [3, 0, 1, 2], perm = [-1, -2, -3, 0]

    nxt = 3
    perm[nxt] = 0
    arr = [3, 0, 1, 2], perm = [-1, -2, -3, 0]

    nxt = 0
    perm[nxt] = -1
    """
    for i in range(len(A)):
        # Check the index is already moved
        nxt = i
        while perm[nxt] >= 0:
            A[i], A[perm[nxt]] = A[perm[nxt]], A[i]
            temp = perm[nxt]
            # Subtracts len(perm) fron an entry in perm to make it negative,
            # which indicates the corresponding move has been performed.
            perm[nxt] -= len(perm)
            nxt = temp
    perm[:] = [a + len(perm) for a in perm]

def apply_permutation_check_leftmost(perm: List[int], A: List[Any]) -> None:
    def cyclic_permutation(start: int, perm: List[int], A: List[Any]) -> None:
        i, temp = start, A[start]
        while True: # Do While Loop
            next_temp = A[perm[i]]
            A[perm[i]] = temp
            temp = next_temp
            i = perm[i]
            if i == start: # Break once we get back to start
                break
    for i in range(len(A)):
        # Find the smallest index in the loop
        # Only perform the cyclic permutation if it is the smallest index in the loop
        j = perm[i]
        while j != i:
            if j < i:
                break
            j = perm[j]
        else:
            cyclic_permutation(i, perm, A)

def next_permutation(perm: List[int]) -> List[int]:
    """
    Get next permutation without brute force

    Return the next permutation array (if no more permutation returns [])

    Generally, we can find the decreasing suffix array
    The first smaller value we will need to swap with the next larger value in the suffix array
    After swapping them, we need to re-sort the suffix array. But notice that the suffix array is already sorted decreasing.
    We can simply use two pointer to swap the first and last element of the suffix array

    Example: [1,2,5,4,3,0] -> 5,4,3,0 is the decreasing suffix array
    Next larger number in the suffix array = 3
    We swap 2 and 3
    [1,3,5,4,2,0] -> Now we sort the decreasing suffix array
    [1,3,0,2,4,5] -> This is the expected next permutation
    """
    inversion_point = len(perm)-2

    # We check 0, so that perm[0] and perm[1] would be checked as well
    while inversion_point >= 0 and perm[inversion_point] >= perm[inversion_point+1]:
        inversion_point -= 1
    
    if inversion_point == -1:
        return []

    # Find the first value in the arr[inversion_point:] that is larger than perm[inversion_point]
    # Search from the back since arr[inversion_point:] is decreasingly sorted
    # You can do binary search if you wish to
    for i in reversed(range(inversion_point+1, len(perm))):
        if perm[i] > perm[inversion_point]:
            perm[i], perm[inversion_point] = perm[inversion_point], perm[i]
            break
    
    l = inversion_point+1
    r = len(perm)-1
    while l < r:
        perm[l], perm[r] = perm[r], perm[l]
        l += 1
        r -= 1
    return perm

def random_offline_sampling(k: int, A: List[Any]) -> None:
    """
    Sample subsets of elements in place
    Sample k elements with no replacements

    Since this is in-place, if k < len(A), only the first k elements in A is valid sampling
    """
    for i in range(k):
        if k > len(A):
            break
        selected = randint(i, len(A)-1) # Select from unselected elements
        A[i], A[selected] = A[selected], A[i] # Move selected elements to i and move it the unselected element to the back

def random_online_sampling(k: int, A: Union[Iterator[Any], List[Any]]) -> List[Any]:
    """
    Sample subsets of streaming elements
    Sample k streaming elements with no replacements

    Consider this we first store all k elements
    We can start sampling only when we have k+1 elements
    We decide to keep or drop any of the older elements in stored k array
    The difficult part would be the correct probability to keep or drop.
    Probability of remaining in sample = P(keep in k+1) * P(keep in k+2) * ... * P(keep in n)
    We know eventually the probability of remaining in sample should be k/n. Means every element has k/n chance being sample

    Instead of resampling for every new element coming into k+1, we can simply sample by 1/(number of elements seen)

    Mathematical Proof
    P(Stay in the sample in k+1 elements for those already in k) = 1/(k+1)
    P(Stay in the sample in k+1 elements for those not in k) = k/(k+1)

    The trick below are all cancelling out the numerators and denominators

    Elements at position i where i < k
    P(Elements to be sampled) = P(Stay in the sample) = 1 * (1-1/(k+1)) * (1-1/(k+2)) * ... (1-1/n) = k/(k+1) * (k+1)/(k+2) * ... (n-1)/n = k/n

    Element at position i where i == k
    P(Elements to be sampled) = k/k+1 * (1-1/(k+2)) * ... (1-1/n) = k/(k+1) * (k+1)/(k+2) * ... (n-1)/n = k/n

    Elements at some i after k where i > k
    P(Elements to be sampled) = k/i * (1-1/(i+1)) * ... (1-1/n) = k/i * i/(i+1) * ... (n-1)/n = k/n

    Example:
    k = 2
    n = 5
    P(Stay in the sample) = (1-1/3) * (1 - 1/4) * (1 - 1/5) = 2/3 * 3/4 * 4/5 = 24/60 = 2/5

    """
    if type(A) == list:
        A = iter(A)
    
    sampled = []

    for idx, elem in enumerate(A):
        if idx < k:
            sampled.append(idx)
        else:
            # Sample from 0 to idx. idx + 1 is the number of elements seen, idx = number of elements seen-1
            drop_index = randint(0, idx) 
            if drop_index < k:
                sampled[drop_index] = elem
    
    return sampled

def compute_random_permutation(n: Union[List[Any], int]) -> List[int]:
    """
    Compute a random permutation of integer size

    We can actually make use of random_offline_sampling.
    The idea is that we will first randomly select any index from 0 to n.
    Subsequently, we will randomly choose from the unselected index where index size reduced eventually to 0
    1. Randomly choose 0 to n-1, place the chosen number to 0 and move unchosen/original number at 0
    2. Repeat for any i index by randomly choosing i to n-1, and place the chosen number to i and move the unchosen number to be chosen i index.
    3. Repeat until we reach n
    """
    if type(n) == list:
        n = len(n)
    
    permutation = list(range(n))
    random_offline_sampling(n, permutation)
    return permutation

def random_subset(n: Union[List[Any], int], k: int) -> List[Any]:
    """
    Generate a random subset of size k with the equal probabilty of selecting them from n!/ ((n - k)!k!) (k-size subsets)

    The difference from compute_random_permutation is that k can be a lot smaller than n.
    If we were to use compute_random_permutation, the time and space complexity will be O(n).

    Instead, we shall use hashmap to store the index mapping and we shall then generate the subset based on the index mapping.
    Time Complexity: O(k)
    Space Complexity: O(k)
    """
    if type(n) == list:
        n = len(n)
    
    # We use a hashmap here as we assume that k is way smaller than n
    # We will only store at most 2*k elements -> where the first k index randomly selected all the indices after k
    changed_elements = {} 

    # We are simulating compute_random_permutation using hashmap instead of the original array
    for i in range(k):
        # Generate a random index between i and n-1, inclusive.
        rand_idx = randint(i, n-1)
        mapped_rand_idx = changed_elements.get(rand_idx, rand_idx)
        mapped_i = changed_elements.get(i, i)
        changed_elements[rand_idx] = mapped_i
        changed_elements[i] = mapped_rand_idx
    return [changed_elements.get(i, i) for i in range(k)]

def nonuniform_random_number_generator(values: List[Any], probabilities: List[Union[float, int]]) -> Any:
    """
    Make each of the probabilities into disjoint intervals
    i.e values = [1, 2, 3],  probabilities = [0.5,0.4, 0.1]
    We map probabilities into [0, 0.5), [0.5, 0.9), [0.9, 1) for [1, 2, 3] respectively
    We get a random value from 0 to 1.
    We then perform a binary search on the probabilities to find the expected index in the values.
    """
    assert 0 < min(probabilities) <= max(probabilities) < 1, 'Values in probabilities must be between 0 and 1'
    assert len(values) == len(probabilities), f'Length of values {len(values)} and length of probabilities {len(probabilities)} does not match'

    # We can handle these all by adjusting the values by probability/(sum(probabilities))
    assert isclose(sum(probabilities), 1), 'Values should sum to 1'

    prefix_sum = list(accumulate(probabilities))
    selected_idx = bisect(prefix_sum, random()) # Note that random is 0 <= x < 1
    return values[selected_idx]

def is_valid_sudoku(partial_assignment: List[List[int]]) -> bool:
    """
    Check for every square region, rows and cols that can not have duplicates

    Assume 0 as empty block
    """
    n = len(partial_assignment)

    square_size = int(n**(0.5))
    assert n%square_size == 0, 'Board size must be a perfect square'

    def has_duplicates(arr):
        elements = list(filter(lambda x: x!=0, arr))
        return len(elements) != len(set(elements))

    # 00 01 02
    # 10 11 12
    # 20 21 22
    for row in range(square_size):
        for col in range(square_size):
            if has_duplicates([partial_assignment[row*square_size+i][col*square_size+j] for i in range(square_size) for j in range(square_size)]):
                return False
    
    return all(not has_duplicates(partial_assignment[row]) for row in range(square_size)) \
    or all(not has_duplicates([partial_assignment[row][col] for row in range(square_size)]) for col in range(square_size))

def is_valid_sudoku_pythonic(partial_assignment: List[List[int]]) -> bool:
    """
    Quick explanation
    Count and store each entries of partial_assignment in this format (while skipping empty entry which is 0 here)
    (row_idx, entry_value_in_string)
    (entry_value_in_string, col_idx)
    (row_idx/region_size, col_idx/region_size, entry_value_in_string) -> Square region offset row and col
    """
    region_size = int(len(partial_assignment)**0.5)
    return max(
            Counter(k
            for i, row in enumerate(partial_assignment)
            for j, c in enumerate(row)
            if c != 0
            for k in ((i, str(c)), (str(c) , j),
                (i // region_size, j // region_size,
                str(c)))).values()
        , default=0) <= 1

def matrix_in_spiral_order(square_matrix: List[List[int]]) -> List[int]:
    """
    Time Complexity: O(nm), where n is number of the rows and m is number of columns
    Space Complexity: O(1)

    Idea:
    Notice that when we are navigating spirally, the steps that we are navigating is decreasing as it goes
    n = len(square_matrix)
    m = len(square_matrix[0])

    n -> m-1 -> n-1 -> m-2 -> n-2 -> .... until we walk n*m valid steps (n*m number of element)
    1 2 3
    4 5 6
    7 8 9

    1 2 3 (3) -> 6 9 (2) -> 8 7 (2) -> 4 (1) -> 5 (1)

    What we can do here is track the offset for n and m.
    Initialise n_offset as 1 and m_offset as 0
    This offset is dependent on the initial direction.
    We move horizontally across m direction first so m_offset starts from 0 and n_offset starts from 1

    Alternative Method:
    1. Keep a visited set and change direction when it is out of bound or we have visited it
        Time Complexity: O(nm)
        Space Complexity: O(nm)
    2. Modify the original matrix by setting it to a sentinel value like 0 to mark it as visited and navigate accordingly
        Time Complexity: O(nm)
        Space Complexity: O(1)
        Modifying the original matrix may not be great.
    
    """
    n = len(square_matrix)
    if n <= 0:
        return []
    m = len(square_matrix[0])

    expected_size = n*m

    results = [0]*expected_size
    output_i = d = 0
    i, j = 0, -1
    DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))
    # n -> m-1 -> n-1 -> m-2  -> n-2 

    n_offset = 1
    m_offset = 0

    while output_i < expected_size:
        di, dj = DIRECTIONS[d]
        if d % 2: # Going horizontally
            for _ in range(n-n_offset):
                i, j = i+di, j+dj
                results[output_i] = square_matrix[i][j]
                output_i += 1
            d += 1
            n_offset += 1
        else:
            for _ in range(m-m_offset):
                i, j = i+di, j+dj
                results[output_i] = square_matrix[i][j]
                output_i += 1
            d += 1
            m_offset += 1
        d %= len(DIRECTIONS)
    return results

# Works on rectangular matrix, O(nm) space where n and w is the width and height
def rotate_matrix(square_matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the matrix in the clockwise direction

    Transpose and Flip Horizontally

    1 2 3 4                    1 5 9 13                         13 9 5 1
    5 6 7 8      -> Transpose  2 6 10 14 -> Flip Horizontally   14 10 6 2
    9 10 11 12                 3 7 11 15                        15 11 7 3
    13 14 15 16                4 8 12 16                        16 12 8 4
        
    """

    # Map the original index to rotated index
    # (0, 0) -> (0, 3)
    # (1, 3) -> (3, 1) -> (3, m-1-1)

    n = len(square_matrix)
    if n < 1:
        return []
    m = len(square_matrix[0])

    # When we rotate the 90 or 270, it will swap the shape
    output = [
        [0] * n
        for _ in range(m)
    ]

    # 1 2 -> 3 1
    # 3 4    4 2

    # 1 3 i = 0, j = 0, mapped_i, mapped_j = 0, 0 mapped_j = 2-0-1 = 1, 
    # 2 4
    for i in range(n):
        for j in range(m):
            # Transpose and Flip Horizonally
            mapped_i, mapped_j = j, n-i-1

            output[mapped_i][mapped_j] = square_matrix[i][j]
    return output

# Only works on square matrix, O(1) Space
def rotate_matrix_inplace(square_matrix: List[List[int]]) -> None:
    """
    Rotate the matrix in the clockwise direction in place

    Rotate in 4 ways swap from outermost into inner square
    Repeatedly perform transpose and flip horizontal will result in a closed loop of values
    (0, 0) 
    -> (0, n-0-1)
    -> (n-0-1, m-0-1) 
    -> (m-0-1, n - (n-0-1)-1) = (m-0-1, 0) 
    -> (0, m - (m-0-1)-1) = (0, 0)
    From the above equations, you can see that we are rotating the moving around these values

    We need to map the index of the location to values
    To find its value from the index, we need to work the operation (transpose + flip horizontally) backwards
    To find value at index, flip horizontally + transpose

    Notice one thing is that we are moving lesser steps in the inner square

    Outermost rotate
    1 2 3 4                    13 2 3 1                 13 9 3 1                   13 9 5 1
    5 6 7 8      -> Rotate 4   5 6 7 8     -> Rotate 4  5 6 7  2      -> Rotate 4  14 6 7 2
    9 10 11 12                 9 10 11 12               15 10 11 12                15 10 11 3
    13 14 15 16                16 14 15 4               16 14 8 4                  16 12 8 4

    Inner rotate
    13 9 5 1                   13 9 5 1
    14 6 7 2     -> Rotate 4   14 10 6 2
    15 10 11 3                 15 11 7 3
    16 12 8 4                  16 12 8 4
    """
    n = len(square_matrix)
    if n < 1:
        return
    m = len(square_matrix[0])

    # The number of steps to reach center
    # 1 -> 0, 2 -> 1, 3 -> 1 (Why? We only need to swap the outermost matrix), 4 -> 2 ..
    for i in range(n//2):
        for j in range(i, n-i-1): # 0, 2-0 = 
            # (i, j) is belong to the index after transpose and flipping
            # To assign the value to an index, we do the operation in the reverse manner. Means we need to flip horizonally and transpose
            # (i, j) -> flip horizontally (i, m-j-1) -> (m-j-1, i)
            square_matrix[i][j], square_matrix[j][n-i-1], square_matrix[n-i-1][m-j-1], square_matrix[m-j-1][i] = (
                square_matrix[m-j-1][i], square_matrix[i][j], square_matrix[j][n-i-1], square_matrix[n-i-1][m-j-1]
            )

def generate_pascal_triangle(n: int) -> List[int]:
    """
    Generate first n rows of pascal triangle

    Time Complexity: O(n^2)
    Space Complexity: O(n^2)
    """
    if n < 1:
        return []
    
    results = [[1] * (i+1) for i in range(n)]

    for i in range(2, n): # i refers to index of the row
        for j in range(1, i): # j refers to the position we need iterate from 1 to the last index of the row excluded
            results[i][j] = results[i-1][j-1] + results[i-1][j]
    return results

def generate_pascal_triangle_nth_row(n: int) -> List[int]:
    """
    Generate the nth row of pascal triangle

    Time Complexity: O(n^2)
    Space Complexity: O(n)
    """

    if n < 1:
        return []
    
    results = [1] * (n)

    for i in range(2, n):
        to_be_replaced = results[:2]
        for j in range(1, i):
            to_be_replaced[1] = results[j]
            results[j] = sum(to_be_replaced)
            to_be_replaced[0] = to_be_replaced[1]
    
    return results
import math # Used in is_palindrome_number
from random import randint # Used in generate_uniform_random_number

import collections

# Important trick of bits: num & (num-1) will erasing the rightmost set bit
# num & ~(num-1) will get the rightmost set bit
def count_bits(num: int) -> int:
    bits = 0
    while num > 0:
        bits += 1
        num &= num-1
    return bits

def parity(num: int) -> int:
    # This is the better solution without caching any precomputed parity subsolution
    result = 0
    while num:
        result ^= 1
        num &= num-1
    return result

def parity(num: int) -> int:
    """
    Upper bits XOR with lower bits
    This is good if you know what is the fixed bits in the num itself
    """
    num ^= num >> 32
    num ^= num >> 16
    num ^= num >> 8
    num ^= num >> 4
    num ^= num >> 2
    num ^= num >> 1
    return num & 1

PRECOMPUTED_PARITY = {v:parity(v) for v in range(2**16)}
def cached_parity(num: int) -> int:
    MASK_SIZE = 16
    BIT_MASK = 0xFFFF # set 16 rightmost bits
    return PRECOMPUTED_PARITY[num >> (3*MASK_SIZE)] \
        ^ PRECOMPUTED_PARITY[(num >> (2*MASK_SIZE)) & BIT_MASK] \
        ^ PRECOMPUTED_PARITY[(num >>MASK_SIZE) & BIT_MASK]\
        ^ PRECOMPUTED_PARITY[num & BIT_MASK]

def swap_bits(num: int, i: int, j: int) -> int:
    if (num >> i) & 1 != (num >> j) & 1: # If the bit at i and j are different, we create a mask and swap
        bit_mask = 1 << i | 1 << j
        num ^= bit_mask
    return num 

def reverse_bit(num: int) -> int:
    bit_length = max(16, num.bit_length())
    ans = 0
    for i in range(bit_length):
        if num & 1:
            ans |= 1 << (bit_length-1-i)
        num >>= 1
    return ans

PRECOMPUTED_REVERSED = {v: reverse_bit(v) for v in range(2**16)}
def cached_reverse_bit(num: int) -> int:
    """ The original_reverse_bit does not work in Python. Because bits length is not exactly fixed in python

    Lets run through a concrete example here.
    num = 4053730604815070667
    binary representation = 0011100001000001 1011111010000001 1110101100100001 1011010111001011
    Split into segments [S1, S2, S3, S4], the bits in the parenthesis are the reversed bits
    S1 = 0011 1000 0100 0001 (1000 0010 0001 1100) (num >> (MASK_SIZE*3)) & BIT_MASK)
    S2 = 1011 1110 1000 0001 (1000 0001 0111 1101) (num >> (MASK_SIZE*2)) & BIT_MASK)
    S3 = 1110 1011 0010 0001 (1000 0100 1101 0111) (num >> (MASK_SIZE)) & BIT_MASK)
    S4 = 1011 0101 1100 1011 (1101 0011 1010 1101) (num & BIT_MASK)

    Expected should be: 1101 0011 1010 1101 1000 0100 1101 0111 1000 0001 0111 1101 1000 0010 0001 11
    How we get the expected? -> bin(int(bin(num)[2:][::-1], base=2))[2:]

    So what is the problem in this example?
    We cannot simply right shift S4 by (MASK_SIZE*3). Notice its position is not exactly at 48 bits away. It is instead at 46 bits away.
    This means the offset from the position is dependent on the original bit length.
    We need to offset the position based off the bit_length.
    Notice for S1 as well, we need to right shift the bits away to remove excess bits on the right.
    """
    MASK_SIZE = 16
    BIT_MASK = 0xFFFF # set 16 rightmost bits

    ORIGINAL_BIT_LENGTH = num.bit_length()
    RIGHT_SHIFT = 64-ORIGINAL_BIT_LENGTH

    return (PRECOMPUTED_REVERSED[num & BIT_MASK]) << max(0, (ORIGINAL_BIT_LENGTH-MASK_SIZE)) \
        | PRECOMPUTED_REVERSED[(num >> MASK_SIZE) & BIT_MASK] << max(0, (ORIGINAL_BIT_LENGTH-2*MASK_SIZE)) \
        | PRECOMPUTED_REVERSED[(num >> (2*MASK_SIZE)) & BIT_MASK] << max(0, (ORIGINAL_BIT_LENGTH-3*MASK_SIZE)) \
        | PRECOMPUTED_REVERSED[(num >> (3*MASK_SIZE)) & BIT_MASK] >> RIGHT_SHIFT

def original_reverse_bit(num: int) -> int:
    """
    Cannot really work in Python.
    Refer to the explanation in cached_reverse_bit
    """
    MASK_SIZE = 16
    BIT_MASK = 0xFFFF # set 16 rightmost bits

    return (PRECOMPUTED_REVERSED[num & BIT_MASK]) << (3*MASK_SIZE) \
        | PRECOMPUTED_REVERSED[(num >> MASK_SIZE) & BIT_MASK] << (2*MASK_SIZE) \
        | PRECOMPUTED_REVERSED[(num >> (2*MASK_SIZE)) & BIT_MASK] << MASK_SIZE \
        | PRECOMPUTED_REVERSED[(num >> (3*MASK_SIZE)) & BIT_MASK]

def closest_int_same_bit_count(x: int) -> int:
    """
    Same bit weight with the smallest absolute difference
    Same bit weight hints that we need to swap the set bit with the unset bit.
    What is difference in value after swapping? 
    The absolute difference after swapping i, j = abs(1 << i - 1 << j) -> 2^i-2^j(where i>j)
    If i > j, the absolute smallest difference is determined by 1 << i - 1 << j
    To make the difference to be smallest (2^i-2^j), we need to find the smallest i and j needs to closest to i.
    """
    NUM_UNSIGNED_BITS = 64
    for i in range(NUM_UNSIGNED_BITS - 1):
        if (x >> i) & 1 != (x >> (i + 1)) & 1:
            x ^= (1 << i) | (1 << (i + 1)) # Swaps bit-i and bit-(i + 1)
            return x
        # Raise error if all bits of x are 0 or 1
    raise ValueError('A1l bits are 0 or 1')

def multiply(x: int, y: int) :
    """
    Multiply without using +, -, *, /
    Works like long multiplication, but it is performing using binary multiplication
    15 x 10
      15
    x 10
    ----
       0
    +150
    ---
    150

         1111
    x    1010
    ---------
         0000
        1111
       0000
    + 1111
    ---------
     10010110 (150 in decimal)
     """
    def add(a: int, b: int) -> int:
        running_sum, carryin, k, temp_a, temp_b = 0, 0, 1, a, b
        while temp_a or temp_b:
            ak, bk = a & k, b & k
            carryout = (ak & bk) | (ak & carryin) | (bk & carryin)
            running_sum |= ak ^ bk ^ carryin
            carryin, k, temp_a, temp_b = (carryout << 1, k << 1, temp_a >> 1, temp_b >> 1)
        return running_sum | carryin
    running_sum = 0
    while x: # Examines each bit of x.
        if x & 1:
            running_sum = add(running_sum, y)
        x, y = x >> 1, y << 1
    return running_sum

    # x / y ->  get the largest multiple of y closest to x
    # x - yn = remainder_x if remainder_x < y then remainder_x * 10
    # x = ay + r

def divide(x: int, y: int) -> int:
    """
    Divide with only addition, substraction and shifting operators
    1. Find the largest multple of y (Since we cannot use multiplication operator, we can only use multiple of 2 with bit shifting)
    2. Once found, substract off from x and add the multiple that y is increased by
    3. Repeat Step 1 to 2 until x < y. Basically we are going to floor it.
    This looks like it has a very high chance of overflowing.
    Consider this, x = 5, y = 2912083091273. If we left shift y by 64, this is very likely to overflow the 64-bit integer range.
    """
    result, power = 0, 64
    y_power = y << power
    
    while x >= y:
        while y_power > x:
            y_power >>= 1
            power -= 1
        result += 1 << power
        x -= y_power
    return result

def divide_with_no_overflow(x: int, y: int) -> int:
    """
    Divide with only addition, substraction and shifting operators
    We grow y from the bottom to the larger value.
    """
    quotient = 0
    while x - y >= 0:
        ny = y
        count = 1
        while ny << 1 <= x:
            ny <<= 1
            count <<= 1
        x -= ny
        quotient += count
    return quotient

def power(x: int, y: int) -> int:
    """
    We can represent 4**5 = 4**4 * 4**1
    For negative number, we can rewrite 4**(-5) = (1/4)**5
    5 = 101 in binary
    5 = 2**2 + 2**0 (in binary we can rewrite the equation based on the position of set bits in the binary representation)
    By replacing 5 in the original equation, we can get 4 ** 5 = 4 ** (2**2 + 2**0) 
    We further split up the equation, we can get 4**(2**2) * 4**(2**0)
    4 bit shift left by 2 position we can get 4**(2**2)
    Notice this are bit set in the power
    Hence the code below are leftshifting the x value (4 here) and check the y/power value's bit is set.
    We need multiply them into the result because of power rule.
    """
    result, power = 1.0, y
    if y < 0 :
        power, x = -power, 1.0/x
    while power: 
        if power & 1:
            result *= x
        x, power = x * x, power >> 1
    return result

def reverse(x: int) -> int:
    # Reverse digit
    result, x_remaining = 0, abs(x)
    while x_remaining:
        result = result * 10 + x_remaining % 10
        x_remaining //= 10
    return -result if x < 0 else result

def reverse_digits(num: int) -> int:
    is_negative = num < 0
    num = -num if is_negative else num
    result = 0
    while num:
        num, digit = divmod(num, 10)
        result *= 10
        result += digit
    return -result if is_negative else result

def is_palindrome_number_expected(num: int) -> bool: # Used in test
    if num < 0:
        return False
    num_digits = 1
    tmp = num
    while tmp >= 10:
        num_digits += 1
        tmp //= 10
    leftmost = (num // (10**(num_digits-1))) % 10 # 191, num_digits = 3
    rightmost = num % 10
    for i in range(num_digits//2):
        leftmost = (num // (10**(num_digits-1-i))) % 10
        rightmost = (num // (10**i)) % 10
        if leftmost != rightmost:
            return False
    return True

def is_palindrome_number(x: int) -> bool:
    if x <= 0:
        return x == 0
    num_digits = math.floor(math.log10(x)) + 1
    msd_mask = 10**(num_digits-1)
    for i in range(num_digits // 2):
        if x // msd_mask != x * 10:
            return False
        x %= msd_mask # Remove the most significant digit of x
        x //= 10 # Remove the least significant digit of x.
        msd_mask //= 100
    return True

def generate_uniform_random_number(lower_bound: int, upper_bound: int) -> int:
    """
    Given a random number generator that only produce 0 and 1 at an equal probability,
    Generate a random number between lower_bound and upper_bound
    Consider the time complexity,
    O (number of tries * number of bits in number of outcomes)
    Can we calculate number of tries?
    1. P(succeed in first tries) = number_of_outcomes/2**(number of bits in number of outcomes)
    Why number_of_outcomes = succeed, because when we are generating the number we are generating from 0 <= x < 2**(number of bits in number of outcomes)
    Any number of x <= number_of_outcomes will be the successfully generated number
    2. We can compare 2**(number of bits in number of outcomes) with 2 * number of outcomes
    2 ** (number of bits in number of outcomes) can only be greater than or equal 2 * number of outcomes
    Consider that (number of bits in number of outcomes) is computed from number of outcomes, 
    Let number of outcomes = 5, number of bits in number of outcomes=3, 2**number of bits in number of outcomes = 2**3 = 8
    Because 2**number of bits in number of outcomes assumes all bits are set to 1.
    Yet, number of outcomes' bits may not be entirely set as 1.
    This means number_of_outcomes/2**(number of bits in number of outcomes) >= number of outcomes/(2 * number of outcomes) (1/2)
    3. We can get that probabilty of P(succeed in first tries) >= 1/2
    4. Each tries are independent and P(Fail) < 1/2
    Number of tries = (1/2)^(number_of_tries) which converges very quickly to zero. We can take O(Number of tries) as O(1)
    Final Time Complexity: O(number_of_bits_in_number_of_outcomes)
    """
    
    result = 0
    number_of_outcomes = upper_bound-lower_bound+1
    zero_or_one_random = lambda : randint(0, 1)
    while True:
        result, i = 0, 0
        while (1 << i) < number_of_outcomes:
            result = (result << 1) | zero_or_one_random()
            i += 1
        if result < number_of_outcomes:
            break
    return result + lower_bound

Rectangle = collections.namedtuple('Rectangle', ('x', 'y', 'width', 'height'))

def intersect_rectangle(R1: Rectangle, R2: Rectangle) -> Rectangle:
    """
    R2.x + R2.width = R2's Rightmost x
    R1.x + R1.width = R1's Rightmost x
    R2.y + R2.height = R2's Top y
    R1.y + R1.height = R1's Top y

    R1 Leftmost X <= R2 Rightmost X and R1 Rightmost X >= R2 Leftmost X and
    R1 Bottom Y <= R2 Top Y and R1 Top Y >= R2 Bottom Y

            ----------|
           |          |
      ----------      |
     |     |          |
     |     |          |
     |     -----------|
     |         |
      ----------
    If you try to visualise the above diagram, you will realise whichever rectangle you assign as R1 or R2 will work as expected
    """
    def is_intersect(R1, R2):
        return (R1.x <= R2.x + R2.width and R1.x + R1.width >= R2.x \
            and R1.y <= R2.y + R2.height and R1.y + R1.height >= R2.y)

    if not is_intersect(R1, R2):
        return Rectangle(0, 0, -1, -1) # No intersection.
    
    # Leftmost points of the intersected rectangle should be the larger x and y -> leftmost is the keyword
    # Intersected rectangle width/Height = Smaller of Rightmost (x/y) - (Larger Leftmost (x/y) / Leftmost points of intersected rectangle)
    return Rectangle( max(R1.x , R2.x) \
        , max(R1.y, R2.y) \
        , min(R1.x + R1.width, R2.x + R2.width) - max(R1.x, R2.x) \
        , min(R1.y + R1.height, R2.y + R2.height) - max(R1.y, R2.y))
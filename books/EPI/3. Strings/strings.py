from functools import reduce
from itertools import groupby

from typing import List
import string

def is_palindromic(s: str) -> bool:
    # Note that s[~i] for i in [0,len(s) - 1] is s[-(i + 1)]
    # ~ means invert all bits, - means invert all bits + 1 (2's complement)
    # -i = ~i+1
    # ~i = -i-1 = -(i+1)
    # i = 0, ~i = -1
    # i = 1, ~i = -2
    # This works because negative indexing works in python
    # If your language do not work with negative indexing, just change ~i into len(s)-i-1
    return all(s[i] == s[~i] for i in range(len(s)//2))

def int_to_string(n: int) -> str:
    is_negative = False
    if n < 0:
        # Invert the negative value and mark that it is negative
        n, is_negative = -n, True
    
    # Use array since it is more efficient than concatenating every step
    s = []
    while n:
        s.append(chr(ord('0') + n%10))
        n //= 10
    
    # Add the negative sign if n is negative
    if is_negative and len(s):
        s.append('-')
    
    return ''.join(reversed(s))

def string_to_int(s: str) -> int:
    """
    It is short just that it is less readable.
    Harder to come up of during interview
    reduce takes in a function, iterable and Optional initial value
    """
    return reduce(
        lambda running_sum, c: running_sum * 10 + string.digits.index(c), s[s[0] == '-':], 0
    ) * (-1 if s[0] == '-' else 1) if len(s) else 0

def string_to_int_unrolled(s: str) -> int:
    result = 0
    is_negative = False
    for i in range(len(s)):
        if i == 0 and s[i] == '-':
            is_negative = True
            continue
        result *= 10
        result += ord(s[i])-ord('0')
    return (-1 if is_negative else 1) * result

def convert_to_base(s: str, b1: int, b2: int) -> str:
    """
    Convert s (string) from b1 base to b2 base
    Return the converted b2 base string
    Constraint:
    1 <= b1 <= 16
    1 <= b2 <= 16

    String Conversion from b1
    1. For each character in string
        1a. Shift accumulated decimal values by b1 first
        1b. Convert the character into the respective base number be it using indexing or hashmap of character to value
        1c. In the code, we use built-in c function index function.
        1d. Time comparison
            i. hashmap -> sum(timeit.repeat("d={v:k for k,v in enumerate('0123456789abcdefABCDEF')};d['f']", number=5)) / 5 -> 2.2883199926582164e-05
            ii. built-in index -> sum(timeit.repeat("import string;string.hexdigits.index('f')", number=5))/5 -> 4.933400123263709e-06
        1e. built in c index function is faster, building the hashmap is O(n) and index the hashmap is O(n) with the advantage using c which is likely to be faster
    
    Decimal Integer Conversion back to b2
    Consider a decimal to binary number 
    4 -> 100
    4 // 2 = 2, (remainder = 0)
    2 // 2 = 1, (remainder = 0)
    1 // 2 = 0, (remainder = 1)
    We know the remainder will determine the representation in b2.
    Also, we know it will be in reversed order.
    
    We can repeatedly divide the decimal number by base b2 and store the representation in a list.
    Join them in the reversed order later.
    """

    # 1. Convert string in base b1 to decimal
    decimal = 0
    baseset = string.hexdigits #'0123456789abcdefABCDEF'
    is_negative = s and s[0] == '-'
    for c in s:
        decimal *= b1
        if c == '-':
            continue
        decimal += baseset.index(c.lower())
    
    # 2. Convert decimal into base b2 str
    ans = []
    while decimal:
        decimal, remainder = divmod(decimal, b2)
        ans.append(baseset[remainder].upper())

    if is_negative and ans:
        ans.append('-')
    
    return ''.join(reversed(ans)) if ans else '0'

def convert_to_base_with_functools(num_as_string: str, b1: int, b2: int) -> str:
    # Solution from the book
    def construct_from_base(num_as_int: int, base: int) -> str:
        return ('' if num_as_int == 0 else
                construct_from_base(num_as_int // base, base) +
                string.hexdigits[num_as_int % base].upper())
    is_negative = num_as_string[0] == '-'
    num_as_int = reduce(lambda x, c: x*b1 + string.hexdigits.index(c.lower()), num_as_string[is_negative:], 0)
    return ('-' if is_negative else '') + ('0' if num_as_int == 0 else construct_from_base(num_as_int, b2)) 

def spreadsheet_decode_col_id(col: str) -> int:
    """
    It will be more intuitive to examine the examples first
    We would notice that we can represent them with a range of values from 0 to 26.
    However, A in this case is not the same as 0 in decimal or any integer representation.
    0001 = 1 in all integer representation, hence the 0 can be ignored.
    In spreadsheet, preceding A cannot be ignored. Hence it must be a value larger than 0.
    Can we adjust the range from 0 to 25 to another range that is suitable for the example below?
    A -> 1
    Z -> 26
    AA -> 27
    Since A is mapped to 1, can we simply make the new range from 1 to 26
    In fact, we can. AA = 1 * 26**1 + 1 * 26**0

    Variant Question: "A" corresponding to 0. 
    The idea is simply minus 1 off the current answer. Like this, we can move the entire range of values down by one.
    """
    return reduce(lambda accu, c: accu*26 + ord(c)-ord('A')+1, col, 0)

def replace_and_delete(size: int, s: List[str]) -> int:
    """
    You need to do two operations here
    1. Replace every 'a' with 2 'd'
    2. Remove every 'b'

    We are given the size of array. This assumes that s always have enough spaces for the true solution
    Time Complexity: O(size)
    Space Complexity: O(1) -> As it assumes that the s array has enough space and we can modify the array
    """
    # 1. Count the number of 'a' and remove 'b'
    # ['a','b','a','c','d']
    # expected output size = ['a','b','a','c','d', _]
    # if size == 0:
    #     return 0
    
    a_count = write_idx = 0
    for i in range(size):
        char = s[i]
        if char != 'b':
            # print(char, write_idx, i)

            s[write_idx] = s[i] # Cannot do swapping here, otherwise you will recount 'b' char
            write_idx += 1
        if char == 'a':
            a_count += 1
    
    read_idx = write_idx-1
    write_idx += a_count-1
    final_size = write_idx+1
    
    for i in range(read_idx, -1, -1):
        char = s[i]
        if char == 'a':
            s[write_idx] = 'd'
            s[write_idx-1] = 'd'
            write_idx -= 1
        else:
            s[write_idx] = char
        write_idx -= 1
    
    return final_size

def is_palindrome(s: str) -> bool:
    """
    Return the s string is palindrome (reading front and back are the same) 
    While ignoring the 2 following conditions
    1. non-alphanumeric characters are ignored
    2. upper/lower case are ignored
    """

    l = 0
    r = len(s)-1
    while l < r:

        while not s[l].isalnum() and l < r:
            l += 1
        
        while not s[r].isalnum() and l < r:
            r -= 1
        
        if s[l].lower() != s[r].lower():
            return False
        l += 1
        r -= 1

    return True

def reverse_words(s: bytearray) -> bytearray:
    """
    Reverse the words in the sentence string.
    Assumption: string is in bytearray format -> Allow you to directly access and swap the values in the bytearray

    Algorithm (in place with bytearray): (Fun fact: The order of step 1 and step 2 does not matter)
    1. Find the space character position and start reversing every word.
    2. Reverse the entire string

    Quick explanation:
    As every word may not be same length, swapping two words to their respective position (i.e word_a first character to word_b first character) may be tedious.
    Hence, you can first reverse within the same word and reverse the entire string
    """
    def reverse_chars(s: bytearray, start: int, end: int):
        while start < end:
            s[start], s[end] = s[end], s[start]
            start += 1
            end -= 1

    start = 0
    while True:
        end = s.find(b' ', start)
        if end < 0:
            break
        reverse_chars(s, start, end-1) # Do not reverse the space
        start = end+1
    
    # Reverse the last word
    reverse_chars(s, start, len(s)-1)

    s.reverse()
    return s

PHONE_MAPPING = ('0', '1', 'ABC', 'DEF', 'GHI', 'JKL', 'MNO', 'PQRS', 'TUV', 'WXYZ')
def phone_mnemonic(phone_number: str) -> List[str]:
    n = len(phone_number)
    ans = []
    def helper(digit_count: int, memorised: List[str]):
        if digit_count >= n:
            ans.append(''.join(memorised))
            return
        
        num = phone_number[digit_count]
        num = ord(num) - ord('0')

        if num < len(PHONE_MAPPING):
            for char in PHONE_MAPPING[num]:
                memorised.append(char)
                helper(digit_count+1, memorised)
                memorised.pop()
    helper(0, [])
    return ans

def look_and_say(n: int) -> str:
    """
    Get the nth sequence of look and say sequence

    A look and say sequence begins with '1' and proceed the next sequence by having the count of the each consecutive element of the previous sequence prepended on them
    1 -> 11 -> 21 -> 1211 -> 111221 -> 312211 -> 13112221 -> 1113213211 ...

    When n = 1, we will return 1 as str. This is 1st indexed.
    """

    def helper(sequence: List[str]) -> List[str]:
        prev = None
        count = 0
        res = []

        for c in sequence:
            if c != prev and prev is not None:
                res.append(str(count))
                res.append(str(prev))
                count = 1
                prev = c
            else:
                prev = c
                count += 1
        if prev is not None and count:
            res.append(str(count))
            res.append(str(prev))
        return res
    
    if n <= 0:
        return ''

    cur = ['1']

    for _ in range(1, n):
        cur = helper(cur)
    return ''.join(cur)

def look_and_say_pythonic(n: int) -> str:
    # In python groupby returns key/element and a grouper iterator of the elements
    return reduce(lambda prev_sequence, _: ''.join(str(len(list(v)))+k for k, v in groupby(prev_sequence)), range(1, n), '1') if n > 0 else ''

ROMAN_MAPPING = {
    'I': 1
    , 'V': 5
    , 'X': 10
    , 'L': 50
    , 'C': 100
    , 'D': 500
    , 'M': 1000
}
def roman_to_integer(s: str) -> int:
    """
    Convert roman string to integer
    I = 1, IV = 4, V = 5, IX = 9, X = 10
    XL = 40, L = 50, XC = 90, C = 100
    CD = 400, D = 500, CM = 900, M = 1000
    """

    n = len(s)
    ans = 0
    prev = ''
    for i in range(n-1, -1, -1):
        char = s[i]
        if (char == 'I' and prev in ('V', 'X')) \
            or (char == 'X' and prev in ('L', 'C')) \
            or (char == 'C' and prev in ('D', 'M')):
            ans -= ROMAN_MAPPING[char]
            prev = ''
        else:
            ans += ROMAN_MAPPING[char]
            prev = char
    return ans

def roman_to_integer_pythonic(s: str) -> int:
    return reduce(
        lambda result, i: result + ROMAN_MAPPING[s[i]] *
        (
            -1 if (s[i] == 'I' and s[i+1] in ('V', 'X')) or (s[i] == 'X' and s[i+1] in ('L', 'C')) or (s[i] == 'C' and s[i+1] in ('D', 'M'))
            else 1
        ), reversed(range(len(s)-1)), ROMAN_MAPPING[s[-1]]
    ) if len(s) > 0 else 0

def get_valid_ip_addresses(s: str) -> List[str]:
    ips = []

    def form_valid_ip(pos: int, mem: List[str]):
        if mem and mem[-1] and ((int(mem[-1]) < 0 or int(mem[-1]) > 255) or (mem[-1][0] == '0' and len(mem[-1]) > 1)):
            return
        
        if pos >= len(s) and len(mem) == 4:
            ips.append('.'.join(mem))
            return
        elif pos >= len(s):
            return
        elif len(mem) > 4:
            return
        
        cur = []
        for j in range(min(5, len(s)-pos)):
            cur.append(s[pos+j]) 
            mem.append(''.join(cur))
            form_valid_ip(pos+j+1, mem)
            mem.pop()
    
    form_valid_ip(0, [])

    return ips

def get_valid_ip_addresses_iterative(s: str) -> List[str]:
    ips = []
    n = len(s)
    for i in range(1, min(5, n)):
        part1 = s[:i]

        if (part1[0] == '0' and i > 1) or int(part1) > 255: # Prune the path if it is leading 0 and length greater than 1
            break
        for j in range(1, min(5, n-i)):
            part2 = s[i:i+j]
            if (part2[0] == '0' and j > 1) or int(part2) > 255:
                break
            
            for k in range(1, min(5, n-i-j)):
                part3 = s[i+j:i+j+k]
                if (part3[0] == '0' and k > 1) or int(part3) > 255:
                    break
                
                part4 = s[i+j+k:]
                if (part4[0] == '0' and n-i-j-k > 1) or int(part4) > 255:
                    continue
                
                ips.append(
                    '.'.join([part1, part2, part3, part4])
                )
    return ips

def snake_string(s: str) -> str:
    """
    Given a string s, return the string in the sinusoidal way
    i.e s = 'Hello World'
    snakestring = 
    "  e               l 
     H   l   o   W   r   d
           l       o 
    "
    merge back "e lHloWrdlo"

    Programming
     r   a   n
    P o r m i g
       g   m
    ranPormiggm
    """

    ans = []
    n = len(s)

    # First row offset by 1 and recurrence every 4 steps
    for i in range(1, n, 4):
        ans.append(s[i])
    
    # Second row no offset and recurrence every 2 steps
    for j in range(0, n, 2):
        ans.append(s[j])
    
    # Third row offset by 3 and recurrence every 4 steps
    for k in range(3, n, 4):
        ans.append(s[k])

    return ''.join(ans)

def snake_string_pythonic(s: str) -> str:
    return s[1::4] + s[::2] + s[3::4]

def decode_run_length_encoding(s: str) -> str:
    """
    What is run length encoding?
    It converts a string "aaaabbc" -> 4a2b1c

    In this program, it assumes the encoded string does not consist of any digits
    """
    count, result = 0, []
    
    for c in s:
        if c.isdigit():
            count = count*10 + int(c)
        else:
            result.append(c*count)
            count = 0
    if count:
        result.append(s[-1] * count)
    return ''.join(result)

def encode_run_length_encoding(s: str) -> str:
    """
    Encode the string into run length encoding (Refer to decode_run_length_encoding function for details)

    In this program, it assumes the to be encoded string does not consist of any digits
    """
    encoded, count = [], 0
    prev = ''

    for c in s:
        if prev != c and count:
            encoded.append(str(count)+prev)
            count = 1
        else:
            count += 1
        prev = c
    if count:
        encoded.append(str(count)+prev)
    return ''.join(encoded)

def rabin_karp_naive(s: str, p: str) -> int:
    """
    String matching algorithm
    Preconditions: You need to know all the possible characters first

    Idea: Make use of hashing to reduce the amount of time required to build and compare each of the substring character by character
    Consider we have already computed the hash for the original substring up to the length of substring
    we only have to compare the two hashes first to check if it is worth the effort to make the string comparison.

    The hash can be rolled over to the next substring by performing rolling hash

    In this case, most of the comparison would be O(1) except the same hash substring.

    Why compare the substring even after hashes are the same? -> In rare cases, there may be hash key collision.

    This would have a big problem once we hit a very long length of p and s with very big base.

    maximum hash_value = BASE ** len(p)
    This number can get very big
    """

    if len(s) < len(p):
        return -1

    cand_s = {v: k for k, v in enumerate(set(s).union(set(p)))} # Char -> index

    BASE = len(cand_s)
    
    # Precompute the hash of substring of s up to length of p
    hash_s = reduce(lambda h, c: h*BASE + cand_s[c], s[:len(p)], 0)
    hash_m = reduce(lambda h, c: h*BASE + cand_s[c], p, 0) # Precompute the hash of p
    hash_mod = BASE ** max(len(p), 0) # Quick way to remove the hash value longer than length of p

    for i in range(len(p), len(s)):
        if hash_s == hash_m and s[i-len(p):i] == p: # Check hash and compare
            return i - len(p)
        # This is rolling hash, we add in the current hash value and remove the out of bound hash value with hash_mod
        hash_s = (hash_s * BASE + cand_s[s[i]]) % hash_mod

    # Final check on the last possible substring
    if hash_s == hash_m and s[-len(p):] == p:
        return len(s)-len(p)
    return -1

def rabin_karp(s: str, p: str) -> int:
    """
    For rabin_karp algorithm, read rabin_karp_naive comments.
    Optimisation:
    Modulus all the values of the hash to reduce possible very large hash values

    Idea:
    1. Set MOD as a prime number (The smaller it is, the more hash collision, the slower the algorithm)
    2. We will modulus all the hash value with MOD
    3. For the rolling hash, we need to do the following steps
        3a. Minus the rolling hash of s by (index value of start character of the previous substring MULTIPLIED by the special hash mod)
        3b. What is the special hash mod? We cannot simply minus the start character index value from the rolling hash of s
        3c. The rolling hash of s has already been modulus, directly substracting from it does consider how the hash is shifted for every base
        3d. The idea is that you need to create the modulus value of BASE**(p-1) and multiple to start index value of rolling has of s.
        3e. Will then add the index value of the new character to be included
        3f. Finally modulus them again
    
    Long explanation on the rolling hash (Particularly on special_hash_mod)
    hash = (s[0] * BASE**(length of p-1) + s[1] * BASE**(length of p-2) ... ) % MOD
    (s[0] * BASE**(length of p-1)) % MOD are not equal to s[0] % MOD, assuming s[0] is the index value of character at 0th position of s.
    s[0] can be expressed as q_1 * MOD + r_1
    BASE**(length of m_1) can be expressed as q_2 * MOD + r_2

    s[0] % MOD = r_1
    BASE**(length of m_1) % MOD = r_2

    (s[0] * BASE**(length of m_1)) = ((q_1 * MOD) + r_1) * ((q_2 * MOD) + r_2) = q_1 q_2 MOD**2 + q_1 r_2 MOD + q_2 r_1 MOD + r_1 r_2
    If we modulus (s[0] * BASE**(length of m_1)) by MOD, we get r_1 r_2 which is r_1 * r_2.
    This means we cannot simply minus s[0] % MOD without considering BASE**(length of m_1) % MOD
    
    hash is the sum of all the remainder of each character index with their base position weight.
    We need minus the base position weight with the leftmost character index.
    (s[0] * BASE**(length of m_1)) % MOD = (r_1 * r_2) % MOD (r_1*r_2 might be bigger than MOD)

    """
    if len(s) < len(p):
        return -1

    cand_s = {v: k for k, v in enumerate(set(s).union(set(p)))} # Char -> index

    BASE = len(cand_s)

    MOD = 1_000_000_007

    hash_s = hash_m = 0
    special_hash_mod = 1

    for i in range(len(p)):
        hash_s = ((hash_s * BASE) + cand_s[s[i]]) % MOD
        hash_m = ((hash_m * BASE) + cand_s[p[i]]) % MOD
        if i < len(p)-1: # Because our base position is 0-indexed and we started from 1 for easy multiplication.
            special_hash_mod = (special_hash_mod * BASE) % MOD
    
    for i in range(len(p), len(s)):
        if hash_s == hash_m and s[i-len(p):i] == p:
            return i - len(p)
        hash_s = ((hash_s - cand_s[s[i-len(p)]]*special_hash_mod)*BASE + cand_s[s[i]]) % MOD
    
    if hash_s == hash_m and s[-len(p):] == p:
        return len(s)-len(p)
    return -1

def kmp(s: str, p: str) -> int:
    """
    KMP -> Knuth Morris Pratt Algorithm
    Input:
        s (str): original string
        p (str): matching string
    Output:
        matched_idx (str): the first occurrence index of the matching string in the original string. Return -1 if not found

    General Idea:
    1. Precompute the longest prefix substring pattern before comparing with the original string
    2. In this way, we would never need to move back the pointer of s since we know where we should backtrack to in p.

    Time Complexity: O(n+k) where n is the length of s (original string) and k is the length of p (matching substring)
    1. Building LPS O(m)
    2. Matching the string O(n) -> In the worst case, i will never move back and j can only move back n times. Hence: O(n)

    A More Intuitive Explanation of Longest Prefix Suffix Array
    Let index of s be i and index of p be j.
    Assume that now s[i] != p[j], we want to avoid moving back i because length of s is greater or equal to length of p.
    We know:
    s[i-j:i] == p[:j] because they are matched up to this point until s[i] != p[j]
    If we do not want to move back i, we need to consider if there is any longest valid suffix in s[i-j:i] that is same as prefix of p.
    Consider that we have some length k that is the longest suffix of s[:i] that is equal to the longest prefix of p
    s[i-k:i] == p[:k] where 0 < k < j
    Note that k is length, because s[i] != p[j] and we want to compare the next index of valid longest prefix which will be k = (some longest valid index+1).

    In this sense, if we do this recursively, we only have to move back to j instead of i until j == 0 as there is no more j can be moved back to.

    We need to find the index where p[:some index] == s[i-some_index:i]. Then we continue on the comparison starting p[some_index] with s[i].

    If we cannot find any of such index, it means that there is no similar suffix of s that is same as prefix of p. 
    It also means moving back i to the matched length will not be useful since no suffix of the s[i-matched_length:i] matches any of the possible prefix in p.
    """
    
    if len(s) < len(p):
        return -1
    
    if len(p) == 0: # If your match substring is empty, it can match anything. Hence it should return 0
        return 0
    
    prefix = [0] * len(p)
    l, i = 0, 1

    while i < len(p):
        if p[l] == p[i]:
            l += 1
            prefix[i] = l # We add 1 to l before assigning i to i. Because we want the next index for comparison and not the current index.
            i += 1
        elif l > 0:
            l = prefix[l-1] # Move back to the previous longest valid prefix
        else:
            i += 1
    
    # Check if the remaining length of s is greater or equal to remaining length of p
    i = j = 0
    while i < len(s) and (len(s)-i) >= (len(p)-j):
        if s[i] == p[j]:
            i += 1
            j += 1
        elif j > 0: # Backtrack to the previous valid position in p
            j = prefix[j-1]
        else: # Cannot backtrack anymore j = 0 here
            i += 1
        
        if j == len(p):
            return i-j # Current position - length of p, current position = last correct index + 1
    return -1

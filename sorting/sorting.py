from typing import List

from random import randint

COMMENTS = """
Although some code can be split into reusable function, for the sake easier reading it is better to keep relevant code together.  
Generally most of them can be adjusted to perform the sort descendingly.  
But I only keep ascending implementation since I just want to recap the algorithm
"""

# TC: O(n^2), SC: O(1) -> Bubble the largest elements to right side of the array
def bubble_sort(arr: List[int]) -> List[int]:
    """General idea:
    1. Repeat n times where n equals to size of arr
      1a. Compare the current and next element
      1b. If current element is greater than next element, swap.
    After one iteration, we are guaranteed to have the largest element to be on the rightmost.
    Because we start from the left most and regardless where the largest element is at, it will be bubbled to the right
    Time Complexity: O(n^2)
    """
    n = len(arr)
    for i in range(n):
        for j in range(n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

# TC: O(n^2), SC: O(1) -> Perform copying which result in the slower TC
def merge_sort_inplace_slow(arr: List[int]) -> List[int]:
    def merge_sort_helper(arr: List[int], left: int, right: int) -> None:
        """
        General Idea:
        1. Split the array into halves recursively until the size of the smaller arrays is less than or equal to 1.
        2. We will start merging the smaller arrays together first, starting from right-left == 2 until right-left == size of array
        Merge Algorithm
        1. left is the start of the left sorted array
        2. mid is the start of the right sorted array and end (exclusive index) of left sorted array 
        3. right is the end (exclusive index) of the right sorted array
        Time Complexity: O(n^2)
        Master Theorem:
        T(n) = 2(T(n/2)) + O(n)
        Recurrence Time Complexity: n**(log2(2)) = O(n)
        Non-Recurrence Time Complexity: O(n^2)
        According to Master Theorem, if Recurrence Time Complexity is polynomially lesser to Non-Recurrence Time Complexity.
        Overall Time Complexity: O(n^2)
        Space Complexity (Excluding Stack): O(1)
        """
        if left+1 >= right:
            return
        mid = left + (right-left)//2
        merge_sort_helper(arr, left, mid)
        merge_sort_helper(arr, mid, right)
        # [2, 4, 6, 1, 3, 5]
        # I need to insert 1 to the left index. 
        # Why cannot swap? 
        # Because the current left index can be larger than the right index next element which will destroy the sorted ordering
        while left < mid < right:
            if arr[left] < arr[mid]:
                left += 1
            else:
                if arr[left] == arr[mid]:
                    left += 1
                tmp = arr[mid]
                for i in range(mid, left, -1):
                    arr[i] = arr[i-1]
                arr[left] = tmp
                left += 1
                mid += 1
        return arr

    return merge_sort_helper(arr, 0, len(arr))

# TC: O(nlog(n)), SC: O(1) -> Values has to be small enough to perform this sort
def merge_sort_inplace(arr: List[int]) -> List[int]:
    """
    Time Complexity: O(n(log(n)), where n is the length of array
    Space Complexity (Without Recursion Stack): O(1)

    However, it is only usable in certain scenario.
    Constraints:
    1. It cannot work on arrays with large values as it will overflow
    2. It cannot work on arrays with negative values

    General Idea:
    1. Recursively split by halves until the array reaches size of 1
    2. Start merging two halves together
    Dive straight into the merging algorithm
    Quick Explanation of the Merge
    It is using a mathematical way of marking the values around in the array.
    The trick of this algorithm is to use buckets to store your original array value and use the overflowing part to store your 'new' array value
    This comes with a downside when your maximum element is very large. It will overflow the integer value.
    Another constraint is that it should not contain negative values.
    
    Merge Algorithm
    1. Find the largest number first and we add 1 to key to make the BUCKET SIZE KEY.
    2. Iterating from left to right
        2a. We get the original value in the entry left_index and right_index by modulus the arr[left_index]%BUCKET_SIZE_KEY and arr[right_index]%BUCKET_SIZE_KEY
        2b. We compare them and if the original value is lesser, we will set them. The way we set it is the trick.
        2c. Assume arr[left_index]%BUCKET_SIZE_KEY is smaller, we will set it as arr[left_index]%BUCKET_SIZE_KEY * BUCKET_SIZE_KEY + arr[left_index]%BUCKET_SIZE_KEY
        2d. In this way, we can still get the original value by modulus BUCKET_SIZE_KEY and get the new value by floor dividing BUCKET_SIZE_KEY

    Time Complexity: O(n(log(n))) (Master Theorem: 2(T(n/2)) + O(n)) -> O(nlog(n))
    Space Complexity (Excluding Recursion Stack): O(1)
    """
    def merge_sort_helper(arr: List[int], left: int, right: int):
        if left+1 >= right:
            return
        mid = left + (right-left)//2
        merge_sort_helper(arr, left, mid)
        merge_sort_helper(arr, mid, right)

        BUCKET_SIZE_KEY = max(arr[i] for i in range(left, right)) + 1
        left_index, right_index = left, mid

        for i in range(right-left):
            if right_index == right or \
                (left_index < mid and (arr[left_index] % BUCKET_SIZE_KEY) < arr[right_index] % BUCKET_SIZE_KEY):
                arr[i+left] += (arr[left_index] % BUCKET_SIZE_KEY) * BUCKET_SIZE_KEY
                left_index += 1
            else:
                arr[i+left] += (arr[right_index] % BUCKET_SIZE_KEY) * BUCKET_SIZE_KEY
                right_index += 1
        
        for i in range(left, right):
            arr[i] //= BUCKET_SIZE_KEY
        return arr
    return merge_sort_helper(arr, 0, len(arr))

# TC: O(nlog(n)), SC: O(n) -> Storing values into a temporary array and achieve better TC
def merge_sort(arr: List[int]) -> List[int]:
    """
    Time Complexity: O(n(log(n))), where n is the length of arr
    Recurrence Part: 2(T(n/2)) = O(n)
    Non-Recurrence Part: O(n)
    
    According to Master Theorem, if recurrence part is polynomially equal to non-recurrence part, the time complexity = O(Non-Recurrence-Part/Recurence Part * log(n))
    (Since recurrence part is polynomially equal to non-recurrence part)
    Overall Time Complexity: O(n(log(n))
    Space Complexity (Excluding Stack): O(n)
    """
    def merge_sort_helper(arr, left, right):
        if left + 1 >= right:
            return
        mid = left + (right-left)//2
        merge_sort_helper(arr, left, mid)
        merge_sort_helper(arr, mid, right)

        current_length = (right-left)
        tmp = [0] * current_length
        left_index, right_index = left, mid

        for i in range(current_length): # right = 2, left = 0, 2
            if right_index == right or (left_index < mid and arr[left_index] < arr[right_index]): # l = 0, m = 1
                tmp[i] = arr[left_index]
                left_index += 1
            else:
                tmp[i] = arr[right_index]
                right_index += 1
        
        for i in range(current_length):
            arr[left+i] = tmp[i]
        return arr
    return merge_sort_helper(arr, 0, len(arr))

# TC: O(n), SC: O(n) -> Storing values in buckets but has a problem handling negative values
def radix_sort(arr: List[int]) -> List[int]:
    """
    General Idea of Radix Sort:
    Put each value into buckets digit by digit
    [734,127, 374, 324]
    1. Put them into their last digit bucket
    [[], [], [324, 127], [734], [], [], [], [374], [], []]
    2. Iterate from 0 to 10 and put them back into the original arr. If you iterate from 10 to 0, you can sort by descending.
    [734, 374, 324, 127] 
    3. Repeat for 2nd last digit to last digits
    2nd last digit bucket (After step 1 and 2): 
    [324, 127, 734, 374]
    first digit bucket (After step 1 and 2):
    [127, 324, 374, 734]

    734,127, 374, 324]
    [127, 734, 374, 324]
    [374, 734, 324, 127]
    [734, 374, 324, 127]

    This Radix Sort cannot work on negative values.
    To handle negative values,
    1. You need to split up the negative values
    2. Perform radix sort in their absolute value and reverse the list. This is the negative sorted array.
    3. Perform radix sort on the positive values.
    
    """

    if len(arr) < 1:
        return arr
    
    digits = 1
    largest_num = max(arr) # Get the largest num for calculating the number of iterations needed
    smallest_num = min(arr) # Get the smallest num to check for negative values
    DIGIT_BASE = 10
    if smallest_num < 0:
        raise Exception(f'Arr contains negative values: {smallest_num}')
    
    while largest_num >= DIGIT_BASE:
        largest_num //= 10
        digits += 1
    

    buckets = [[] for _ in range(DIGIT_BASE)]
    digit_mask = 0
    n = len(arr)
    for digit in range(digits):
        digit_mask = 1 if digit_mask == 0 else digit_mask * 10
        for num in arr:
            bucket_key = (num//digit_mask) % 10
            buckets[bucket_key].append(num)
        
        i = 0
        for j in range(DIGIT_BASE):
            for value in buckets[j]:
                arr[i] = value
                i += 1
            buckets[j].clear()
            if i == n:
                break
    return arr

# TC: O(n), SC: O(n) -> Perform 3-pass (1 for splitting negative values up, 1 for radix sort on negative values, 1 for radix sort on positive values)
def radix_sort_negative(arr: List[int]) -> List[int]:
    """
    This implementation handles negative values in the arr.
    Algorithm:
    1. Check if there is any negative value.
        1a. If there is negative values, move the negative values to the front of the array. The order does not matter here so we can just swap.
        1b. Meanwhile, we keep track of the size of the negative array.
        1c. Set the negative value into its positive form
    2. Run radix sort from 0 to size of the negative array.
    3. Run radix sort from size of negative array to size of full array

    Some optimisation
    We can set the negative value back from positive negative in the last digit iteration

    For radix sort algorithm, read the comments in radix_sort function above.
    """
    n = len(arr)
    if n < 1:
        return arr
    
    # We can check if there is negative values in the arr
    smallest_num = min(arr)
    negative_start_index = negative_end_index = 0
    n = len(arr)
    
    if smallest_num < 0:
        # Shift the negative values to the front
        for i in range(n):
            if arr[i] < 0:
                arr[i], arr[negative_end_index] = arr[negative_end_index], arr[i]
                arr[negative_end_index] = -arr[negative_end_index] # Make it positive
                negative_end_index += 1
        # We know negative values are from 0 to negative_end_index (excluded)
    
    def radix_sort_helper(arr: List[int], start: int, end: int, desc: bool =False, is_orig_neg: bool=False):
        if start >= end:
            return
        
        largest_num = max(arr[i] for i in range(start, end))
        
        digits = 1
        DIGIT_BASE = 10
        while largest_num >= DIGIT_BASE:
            largest_num //= DIGIT_BASE
            digits += 1
        
        buckets = [[] for _ in range(DIGIT_BASE)]
        digit_mask = 0
        for digit in range(digits):
            digit_mask = 1 if digit_mask == 0 else digit_mask * 10
            for i in range(start, end):
                num = arr[i]
                bucket_key = (num//digit_mask) % 10
                if digit == digits-1 and is_orig_neg:
                    num = -num
                buckets[bucket_key].append(num)
        
            i = start

            bucket_traverse = range(DIGIT_BASE) if not desc else reversed(range(DIGIT_BASE))
            
            for j in bucket_traverse:
                for value in buckets[j]:
                    arr[i] = value
                    i += 1
                buckets[j].clear()
                if i == end:
                    break
    
    radix_sort_helper(arr, negative_start_index, negative_end_index, desc=True, is_orig_neg=True)
    radix_sort_helper(arr, negative_end_index, n, desc=False)
    return arr

# TC: O(n^2), SC: O(1) -> Swapping in values one by one towards the left sorted array
def insertion_sort(arr: List[int]) -> List[int]:
    n = len(arr)
    # Insertion is like looking for the position to slot in the value
    for i in range(n-1):
        for j in range(i+1, 0, -1):
            if arr[j] < arr[j-1]: # If the right value is smaller than left value, we swap inwards like bubble sort
                arr[j], arr[j-1] = arr[j-1], arr[j]
            else:
                break
    return arr

# TC: O(n^2), SC: O(n) -> Generally faster than most algorithm as it is inplace (Save time on allocating memory)
def quick_sort(arr: List[int]) -> List[int]:
    """
    General Idea:
    1. Randomly choose a pivot point from start to end range
    2. Split the array by moving anything smaller than pivot to the left and anything bigger to the right
    3. Based on the pivot, we will split the array from start to pivot_index and pivot_index+1 to end.

    Splitting Logic (quick_sort_helper):
    1. Move the pivot to the start position
    2. After step 1, we should NOT MOVE the pivot
    3. What we need to track is the FINAL position where the pivot should be at
    4. pivot_index = start, since we move it to the start
    5. If we find anything that is smaller than or equal to pivot, we will move to pivot_index + 1.
    6. In this case, we will not move the pivot until we finish iterating the arr from start to end
    7. Notice that the pivot index is the largest index that contain anything that is smaller or EQUAL to pivot.
    8. At the end of the iteration from start to end, we can move pivot back to the FINAL pivot_index

    Worst Case: O(n^2) when values are sorted descendingly at the start, therefore pivot are randomly chose for better split

    It has a problem when we have a lot of values that are equal. 
    One way to fix this is to return pivot left and pivot right.
    Of course the algorithm is a little more tougher to write

    """
    
    def quick_sort_helper(arr: List[int], start: int, end: int):
        if start+1 >= end: # Array only has at most 1 value
            return start
        pivot_index = randint(start, end-1)
        pivot = arr[pivot_index]

        # Move the pivot to the start (You can also move to the end but you need to change the implementation)
        arr[start], arr[pivot_index] = arr[pivot_index], arr[start]

        # Idea: 2 subarray [values smaller than or equal to pivot ] [values greater than pivot]
        # Return the final pivot location
        pivot_index = start

        for i in range(start+1, end):
            if arr[i] <= pivot:
                pivot_index += 1
                arr[i], arr[pivot_index] = arr[pivot_index], arr[i]
        arr[pivot_index], arr[start] = arr[start], arr[pivot_index]
        quick_sort_helper(arr, start, pivot_index)
        quick_sort_helper(arr, pivot_index+1, end)
    
    quick_sort_helper(arr, 0, len(arr))
    return arr

# TC: O(n^2), SC: O(n) -> Handles equal elements cases better
def quick_sort_better(arr: List[int]) -> List[int]:
    """
    General Idea:
    1. Randomly choose a pivot point from start to end range
    2. Split the array into 3 parts, [anything that is smaller] [anything that is equal to pivot] [anything that is greater than pivot]

    Worst Case: O(n^2) when values are sorted descendingly at the start, therefore pivot are randomly chose for better split
    
    It handles the equal cases better. All elements are equal -> It will result in O(n^2)
    However, the implementation is more confusing to implement.

    There is a better and cleaner implementation in EPI (Python) Arrays Chapter on dutch_flag_partition.
    You can look for the implementation.
    Because the below implementation does the confusing 3 way swapping.
    """
    
    def quick_sort_helper(arr: List[int], start: int, end: int):
        if start+1 >= end: # Array only has at most 1 value
            return start
        pivot_index = randint(start, end-1)
        pivot = arr[pivot_index]

        # Move the pivot to the start (You can also move to the end but you need to change the implementation)
        arr[start], arr[pivot_index] = arr[pivot_index], arr[start]

        # Idea: 3 subarray [anything smaller] [anything equal] [anything larger]
        # pivot_index refers to last value of pivot equal array
        # smaller_index refers to first value of pivot equal array
        # [anything smaller] = [:smaller_index], anything equal = [smaller_index:pivot_index+1], anything larger = [pivot_index+1: i]
        # We need to perform 3 way swap
        # Consider this
        # [1,1,2,5,5,5,8,9,3] # i = 8, pivot_index = 5, smaller_index = 3
        # [1,1,2] [5,5,5] [8,9,3] Notice 8 needs to move to arr[i] 3's position, 3 need to move to arr[pivot_index+1] 8's position
        # [1,1,2] [5,5,5] [3,8,9] Notice that 3 at arr[pivot_index+1] need to move to arr[smaller_index(3)] 5's position 
        # [1,1,2] [3,5,5] [5,8,9] If we move the pivot_index and smaller_index, we will then partition them correctly
        # [1,1,2,3] [5,5,5] [8,9] pivot_index = 6 and smaller_index= 4
        pivot_index = start
        smaller_index = start

        for i in range(start+1, end):
            # pivot_index refers to last value of pivot equal array
            # smaller_index refer to start value of pivot equal array
            if arr[i] < pivot: # Three way swap. 
                # arr[i] swap to arr[pivot_index+1] (next position of equal element). We further move the current value to arr[smaller_index]
                # Where smaller index is expected next position for new smaller than pivot value
                pivot_index += 1
                arr[pivot_index], arr[i] = arr[i], arr[pivot_index]
                arr[smaller_index], arr[pivot_index] = arr[pivot_index], arr[smaller_index]
                smaller_index += 1
            elif arr[i] == pivot:
                pivot_index += 1
                arr[pivot_index], arr[i] = arr[i], arr[pivot_index]
        
        quick_sort_helper(arr, start, smaller_index)
        quick_sort_helper(arr, pivot_index+1, end)
    
    quick_sort_helper(arr, 0, len(arr))
    return arr

# TC: O(n(log(n))), SC: O(n) -> Use builtin heapq to build heap
def heap_sort_with_builtin(arr: List[int]) -> List[int]:
    """
    General Idea:
    1. Build a heap by adding in element by element
    2. Once the heap is built, we will remove the root element and rebalance the heap
    3. Add the root element one by one back to the arr

    To sort ascendingly/descending, build the min-heap and max-heap accordingly.
    """

    import heapq

    heap = list(arr)
    heapq.heapify(heap)
    n = len(arr)
    for i in range(n):
        arr[i] = heapq.heappop(heap)
    
    return arr

# TC: O(n(log(n))), SC: O(n) -> Create a heap class (Just for fun)
def heap_sort(arr: List[int]) -> List[int]:
    """
    We can implement heap on our own (Just for fun)
    heap is usually stored in an arr
    Using array to implement, we can access the parents of ith node by using (i-1)//2 and its child by 2*i+1 and 2*i+2
    [0, 1, 2, 3, 4]
    """
    class Heap:
        def __init__(self, arr: List[int]) -> None:
            self.heapify(arr)
        
        def heapify(self, arr: List[int]) -> None:
            self.heap = arr
            n = len(arr)
            # Starting from the last non-leaf node to root node
            # if i is the last index leaf node, parent of last index leaf node = (i-1)//2
            # last index node i = n-1, n = i+1, parent of last index leaf node = (n-1-1)//2 = n//2-1
            for i in reversed(range(n//2)):
                self.rebalance_down(i)

        def rebalance_down(self, node_index: int) -> None:
            n = len(self.heap) # [4, 1, 2]
            while node_index < n:
                left_child_index, right_child_index = 2*node_index+1, 2*node_index+2
                if left_child_index < n or right_child_index < n:
                    left_child = self.heap[left_child_index] if left_child_index < n else float('inf')
                    right_child = self.heap[right_child_index] if right_child_index < n else float('inf')
                    smallest_child = min(left_child, right_child)
                    if smallest_child < self.heap[node_index]: # Parent > child
                        smallest_child_idx = left_child_index if left_child == smallest_child else right_child_index
                        self.heap[smallest_child_idx], self.heap[node_index] = self.heap[node_index], self.heap[smallest_child_idx]
                        node_index = smallest_child_idx
                    else:
                        break
                else:
                    break

        def heappush(self, value: int) -> None:
            self.heap.append(value)
            current = len(self.heap)-1
            while current > 0:
                parent_idx = (current-1)//2
                if self.heap[parent_idx] > self.heap[current]: # Build min-heap
                    self.heap[parent_idx], self.heap[current] = self.heap[current], self.heap[parent_idx]
                    current = parent_idx
                else: # No more swapping required
                    break
        
        def heappop(self) -> int:
            # Find the new smallest value
            self.heap[0], self.heap[-1] = self.heap[-1], self.heap[0]
            element_to_be_popped = self.heap.pop()
            current = 0
            n = len(self.heap)
            self.rebalance_down(0)
            return element_to_be_popped


    heap = list(arr)
    heap = Heap(heap)

    n = len(arr)
    for i in range(n):
        arr[i] = heap.heappop()
    
    return arr

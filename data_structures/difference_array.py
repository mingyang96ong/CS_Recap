from typing import List

class DifferenceArray: # Zero Array Transformation II (Can attempt this question to practice using this structure)
    def __init__(self, arr: List[int]):
        self.n = len(arr)
        self.difference = [0] * (self.n+1)
        self.arr = arr
    
    def update(self, l: int, r: int, val: int): # O(1) Update
        assert 0 <= l < self.n and 0 <= r < self.n , f'Invalid range {r} or {l}'

        self.difference[l] += val
        self.difference[r+1] -= val
    
    def get_array(self) -> List[int]: # O(n) Time Complexity
        output = [0] * self.n
        accumulate_diff = 0
        for i in range(self.n):
            accumulate_diff += self.difference[i]
            output[i] = accumulate_diff + arr[i]
        return output

if __name__ == '__main__':
    arr = [ 10, 5, 20, 40 ] 

    difference_array = DifferenceArray(arr)

    updates = [(0, 1, 10)]
    for update in updates:
        difference_array.update(*update)
    
    print(difference_array.get_array())

    updates = [(1, 3, 20), (2, 2, 30)]
    for update in updates:
        difference_array.update(*update)
    print(difference_array.get_array())

    # Expected:
    # 20 15 20 40 
    # 20 35 70 60
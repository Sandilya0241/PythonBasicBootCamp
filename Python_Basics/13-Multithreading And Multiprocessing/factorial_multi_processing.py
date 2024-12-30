'''
Real-world Example: Multiprocessing for CPU-bound Tasks
Scenario : Factorial Calculation
Factorial calculations, especially for large number, involve significant computational work. Multiprocessing can be used to distribute the work. Multiprocessing can be used to distribute the workload across multiple CPU cores, improving performance.
'''

import multiprocessing
from math import factorial
from sys import set_int_max_str_digits
from time import time, sleep


## Increase the maximum number of digits for integer conversion
set_int_max_str_digits(100000)

## Function to calculate the factorial of a given number
def find_factorial(number):
    print(f'Finding factorial of number {number}')
    result = factorial(number)
    print(f'Factorial of {number} is {result}')
    return result

if __name__ == "__main__":

    numbers = [5000,6000,7000,8000]

    st_tm = time()

    with multiprocessing.Pool() as pool:
        results = pool.map(find_factorial, numbers)

    end_tm = time()
    print(f"results are {results}")
    print(f'Time taken to calculate factorial {end_tm-st_tm}')
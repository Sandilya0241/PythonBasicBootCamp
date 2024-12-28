### Multithreading with Thread pool executor

from concurrent.futures import ThreadPoolExecutor
from time import time, sleep

def print_number(number):
    sleep(1)
    return f'Number : {number}'

numbers = [1,2,3,4,5,6,7,8,9,0,1,2,3,4]

with ThreadPoolExecutor(max_workers=3) as executor:
    results = executor.map(print_number, numbers)

for result in results:
    print(result)
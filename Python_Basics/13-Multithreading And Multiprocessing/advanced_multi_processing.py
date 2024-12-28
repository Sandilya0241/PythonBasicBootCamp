## Multiprocessing with ProcessPoolExecutor

from concurrent.futures import ProcessPoolExecutor
from time import time, sleep

def square_number(num):
    sleep(2)
    return f'Number : {num * num}'

numbers = [1,2,3,4,5,6,7,8,9,10,11,2,3,12,14]

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=3) as executor:
        results = executor.map(square_number, numbers)


    for result in results:
        print(result)

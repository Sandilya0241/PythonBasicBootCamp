## It allows you to create processes that run in parallel
## When to use:
## 1.) CPU-Boubd Tasks - That are heavy on CPU Usage(e.g., mathemetical computations, data processing).
## 2.) Parallel execution - Multiple cores of the CPU.

import multiprocessing
from time import time, sleep

def sqr_nums():
    for i in range(5):
        sleep(1)
        print(f'Square is : {i*i}')


def cube_nums():
    for i in range(5):
        sleep(1.5)
        print(f'Cube is : {i*i*i}')

if __name__ == "__main__":
    strt_tm = time()
    ## Create 2 processes
    p1 = multiprocessing.Process(target=sqr_nums)
    p2 = multiprocessing.Process(target=cube_nums)

    ## Start processes
    p1.start()
    p2.start()

    ## Join processes
    p1.join()
    p2.join()
    fnshd_tm = time() - strt_tm
    print(f'Time elapsed : {fnshd_tm}')
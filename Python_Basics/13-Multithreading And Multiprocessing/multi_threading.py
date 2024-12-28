## Multithreading
## When to use Multithreading - 
## 1.) I/O-bound tasks : Tasks that spends more time waiting for I/O operations (e.g., file operations, network requests).
## 2.) Concurrent execution : When we want to improve the throughput of your application by performing multiple operations concurrently.

import threading
from time import time,sleep

def print_number():
    for i in range(5):
        sleep(2)
        print(f"Number : {i}\n")


def print_letter():
    for letter in 'abcde':
        sleep(2)
        print(f'Letter : {letter}\n')


## Create 2 threads
t1=threading.Thread(target=print_number)
t2=threading.Thread(target=print_letter)

st_time = time()
t1.start()
t2.start()

## Wait for threads to join
t1.join()
t2.join()

fnsh_time = time() - st_time
print(fnsh_time)
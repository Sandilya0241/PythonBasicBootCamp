import os

def count_upto_without_generator(n):
    '''
    Characteristics of Generators are
        - yield keyword
        - Lazy evaluation
        - Memory efficiency

    to invoke this function
    # print("count with generators")
    # for val in count_upto_with_generator(10):
    #     print(val)
    '''
    count=1
    while count<=n:
        print(count)
        count=count+1

def count_upto_with_generator(n):
    '''
    to invoke this funtion

    # my_gen = count_upto_with_generator(10)
    # print(next(my_gen))
    '''
    count=1
    while count<=n:
        yield(count)
        count=count+1


def read_large_file(file_path):
    with open(file_path,mode='r') as fp:
        for line in fp.readlines():
            yield(line)


if __name__ == "__main__":
    my_gen = read_large_file("simple.txt")
    for line in my_gen:
        print(line)
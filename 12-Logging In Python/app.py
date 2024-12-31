import logging as LOG

## Configure logging settings
LOG.basicConfig(
    level = LOG.DEBUG,
    format='%(asctime)s-%(name)s-%(levelname)s-%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        LOG.FileHandler('app1.log'),
        LOG.StreamHandler()
    ]
)

LOGGER = LOG.getLogger("ArithmeticApp")

def add(num1, num2):
    res = num1 + num2
    LOGGER.debug(f'Adding {num1} + {num2} = {res}')
    return res


def subtract(num1, num2):
    res = num1 - num2
    LOGGER.debug(f'Subtracting {num1} - {num2} = {res}')
    return res


def multiply(num1, num2):
    res = num1 * num2
    LOGGER.debug(f'Multiplying {num1} * {num2} = {res}')
    return res


def divide(num1, num2):
    try:
        res = num1 / num2
        LOGGER.debug(f'Dividing {num1} / {num2} = {res}')
        return res
    except ZeroDivisionError as ze:
        LOGGER.error(f'Cannot divide {num1} with {num2}')
        return None
    


add(10,15)
subtract(15,10)
multiply(10,20)
divide(20,10)
divide(20,0)
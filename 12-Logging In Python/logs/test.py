from logger import logging as log

def add(num1, num2):
    log.debug("Addition operation started")
    return num1 + num2

log.debug("Addition function is called")
add(2,3)
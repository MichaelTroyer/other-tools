# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 19:48:03 2017

@author: michael
"""


import math
import timeit
import logging

from functools import wraps
from random import randint as rI


# Constants
LOG_FORM = '%(asctime)s %(levelname)s %(message)s'
LOG_PATH = r'C:\Users\michael\Documents\_logs\log_decorator.log'

# Logging
logging.basicConfig(filename=LOG_PATH, level=logging.DEBUG, format=LOG_FORM)


def log(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        msg = '%s (%s, %s) -> %s' % (func.__name__, args, kwargs, result)
        logging.debug(msg)
        return result
    return wrapper


def trace(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print '%s (%s, %s) -> %s' % (func.__name__, args, kwargs, result)
        return result
    return wrapper


def fn_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = timeit.default_timer()
        result = func(*args, **kwargs)
        t1 = timeit.default_timer()
        print '%s (%s, %s) -> %s sec' % (func.__name__, args, kwargs, t1 - t0)
        return result
    return wrapper


if __name__ == '__main__':

    @log
    @trace
    @fn_time
    def p1p2Dist(pnt1, pnt2):
        """
        Returns the distance between two different points
        """
        dY = pnt2[1] - pnt1[1]
        dX = pnt2[0] - pnt1[0]
        dist = math.hypot(dX, dY)
        return dist

    jobs = [p1p2Dist(
            pnt1=[rI(1, 10), rI(1, 10)],
            pnt2=[rI(1, 10), rI(1, 10)])
            for _ in range(10)]

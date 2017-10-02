# -*- coding: utf-8 -*-
"""
Created on Tue Apr 04 09:20:50 2017

@author: mtroyer
"""

#class Bunch:
#    def __init__(self, **kwds):
#        self.__dict__.update(kwds)
#
#    
## decorators
def uppercase(func):
    def wrapper():
        original_result = func()
        modified_result = original_result.upper()
        return original_result + "  -->  " + modified_result
    return wrapper

def strong(func):
    def wrapper():
        return '<strong>' + func() + '</strong>'
    return wrapper

def emphasis(func):
    def wrapper():
        return '<em>' + func() + '</em>'
    return wrapper   

@strong
@uppercase 
@emphasis    
def greet():
    return "Hello World"


def trace(func):
    def trace_wrapper(*args, **kwargs):
        original_result = func(*args, **kwargs)
        print
        print 'TRACE: calling {}() with {}, {}'.format(func.__name__, args, kwargs)
        print 'TRACE: {}() returned {}'.format(func.__name__, original_result)
        print
        return original_result
    return trace_wrapper

@uppercase
@trace 
def print_x_y(x,y):
    return str(x) + ' ' + str(y)
    
print print_x_y(x='test', y=55)

    
    
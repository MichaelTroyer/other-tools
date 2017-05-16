#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Michael D. Troyer

Date:

Purpose:

Comments:

TODO: timeit

"""


import multiprocessing
import os

def init(l):  # Create and share a multiprocessing lock globally..
    global lock
    lock = l

def f(x):
    pid = os.getpid()

    l.acquire()
    print 'starting process {}'.format(pid)
    l.release()

    result = x*x

    l.acquire()
    print 'process {}\tresult {}'.format(pid, result)
    print 'ending process {}'.format(pid)
    l.release()


if __name__ == '__main__':
    l = multiprocessing.Lock()

    pool = multiprocessing.Pool(processes=4, initializer=init, initargs=(l,))              # start 4 worker processes

    pool.map(f, range(20))

    pool.close()
    pool.join()
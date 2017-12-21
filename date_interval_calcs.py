# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 13:34:08 2017

@author: mtroyer
"""


import datetime
from pprint import pprint
import numpy as np


dates = [datetime.date.today() - datetime.timedelta(i)
         for i in np.random.randint(5, 300, size=20)]

prices = np.random.normal(loc=100, size=20)

periods = zip(dates, prices)
periods.sort()

buy_tuples = periods[0::2]
sell_tuples = periods[1::2]

def period_metrics(buy_date_price_tuples, sell_date_price_tuples):

    wins = []
    loss = []

    periods = zip(buy_date_price_tuples, sell_date_price_tuples)

    for (buy_date, buy_price), (sell_date, sell_price) in periods:
        period_length = (sell_date - buy_date).days
        period_return = (sell_price - buy_price)
        if period_return > 0:
            wins.append((period_length, period_return))
        else:
            loss.append((period_length, period_return))

    results = [wins, loss]
    n_wins = len(wins)
    n_loss = len(loss)
    sum_wins = sum([w[1] for w in wins])
    sum_loss = sum([w[1] for w in loss])
    average_wins = sum_wins / n_wins
    average_loss = sum_loss / n_loss
    net_returns = sum_wins + sum_loss

    print 'N Wins        : {}'.format(n_wins)
    print 'N Loss        : {}'.format(n_loss)
    print 'Average Win   : {}'.format(average_wins)
    print 'Average Loss  : {}'.format(average_loss)
    print 'Net Returns   : {}'.format(net_returns)

    print 'Wins:'
    pprint(results[0])
    print
    print 'Losses:'
    pprint(results[1])

    return {
            'Results': results,
            'N Wins': n_wins,
            'N Loss': n_loss,
            'Average Win': average_wins,
            'Average Loss': average_loss,
            'Net Returns': net_returns
            }

metrics = period_metrics(buy_tuples, sell_tuples)

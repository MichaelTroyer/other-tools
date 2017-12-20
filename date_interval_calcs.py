# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 13:34:08 2017

@author: mtroyer
"""


import datetime
from collections import OrderedDict
from pprint import pprint

import numpy as np


dates = [datetime.date.today() - datetime.timedelta(i) for i in range(10)]
prices = np.random.normal(loc=100, size=10)

periods = zip(dates, prices)
periods.sort()


def period_metrics(periods):
    data = []
    for i, (start_date, start_value) in enumerate(periods[:-1]):
        period_len = (periods[i+1][0] - start_date).days
        period_ret = periods[i+1][1] - start_value
        data.append((period_len, period_ret))
    wins = [m for _, m in data if m > 0]
    losses = [m for _, m in data if m < 0]
    average_win = np.mean(wins)
    averege_loss = np.mean(losses)

    return OrderedDict(
        [('Results', data), ('Wins', len(wins)), ('Losses', len(losses)),
         ('Average_Win', average_win), ('Average_Loss', averege_loss),
         ('Net_Returns', sum(wins + losses))])

metrics = period_metrics(periods)
for item in metrics.items():
    pprint(item)

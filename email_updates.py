# -*- coding: utf-8 -*-
"""
Created on Sat Nov 04 07:39:58 2017

@author: michael
"""

import os
import gmail
import datetime
import holidays


us_holidays = holidays.UnitedStates()

login_info = {
        'email': 'tailwatertroyer@gmail.com',
        'passx': 'GTroyer3?'
        }

src_path = r'C:\Users\michael\Desktop\stock-trading\strategies'
attch = [os.path.join(src_path, f)
         for f in os.listdir(src_path)
         if os.path.splitext(f)[1] == '.html']

today = datetime.date.today()
weekday = True if today.isoweekday() not in (6, 7) else False


def emailResults():
    mail = gmail.GMail(login_info['email'], login_info['passx'])
    mail.connect()

    mail.send(gmail.Message(
              subject='Trading strategy updates - [{}]'.format(str(datetime.date.today())),
              to=login_info['email'],
              text="Don't lose all our money..",
              attachments=attch))
    mail.close()


if __name__ == '__main__':

    if today not in us_holidays and weekday:
        emailResults()

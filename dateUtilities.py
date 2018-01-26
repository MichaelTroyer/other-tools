import dateutil.parser


def tryParseDate(date):
    # dateutil.parser expects a string
    inPut = date
    kwargs = {}
    if isinstance(date, (list, tuple)):
        date = ' '.join([str(x) for x in date])
    if isinstance(date, int):
        date = str(date)
    if isinstance(date, dict):
        kwargs = date
        date = kwargs.pop('date')

    try:
        try:
            parsedate = dateutil.parser.parse(date, **kwargs)
#            print 'Sharp {} -> {}'.format(repr(inPut), parsedate)
            return parsedate
        except ValueError:
            parsedate = dateutil.parser.parse(date, fuzzy=True, **kwargs)
#            print 'Fuzzy {} -> {}'.format(repr(inPut), parsedate)
            return parsedate
    except Exception, err:
        print 'Cannot parse {}: {}'.format(date, err)


if __name__ == '__main__':
    tests = [
        'January 3, 2017',
        ('5', 'Oct', '09'),
        ['Monday', 'Jan', 1, 2017],
        'Thursday, November 18',
        'Monday, October 12th, 2012',
        '7-12-12',
        '8/10/2010',
        '01/21/01',
        19950317,
        'Tuesday'
        ]

    for test in tests:
        tryParseDate(test)

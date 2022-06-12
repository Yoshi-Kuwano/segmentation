def sum_(a, b):
    return a+b

def times_(a, b):
    return a*b

def sum_times(a, b, c):
    return times_(sum_(a,b),c)


class SumTime():

    def sum_times_(self, a, b, c):
        return sum_times(a, b, c)

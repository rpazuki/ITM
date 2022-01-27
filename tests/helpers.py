from itertools import zip_longest


def compare(arr_1, arr_2, operation=lambda x, y: x == y):
    # return [operation(x, y) for x, y in zip(arr_1, arr_2)]
    return [operation(x, y) for x, y in zip_longest(arr_1, arr_2)]


def compare_in(arr_1, arr_2):
    return [x in arr_2 for x in arr_1]

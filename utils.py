import time


def timeit(method):
    def timer(*args, **kwargs):


        result = method(*args, **kwargs)

        return result
    return timer


def print_progress(current_idx, entire_list):
    print('Progress: {} out of {}'.format(int(current_idx), len(entire_list)))

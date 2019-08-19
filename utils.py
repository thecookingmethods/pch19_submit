import time


def timeit(method):
    def timer(*args, **kwargs):
        print('Running {}()...'.format(method.__name__))
        start = time.time()
        result = method(*args, **kwargs)
        end = time.time()

        print('{}() finished, took {:0.2f} seconds\n'.format(method.__name__, end - start))
        return result
    return timer


def print_progress(current_idx, entire_list):
    print('Progress: {} out of {}'.format(int(current_idx), len(entire_list)))

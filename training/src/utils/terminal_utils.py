import sys

def progressbar(count, total, status='', name=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '\u2588' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('PROGRESS: %s [%s] %s%s ...%s\r' % (name, bar, percents, '%', status))
    sys.stdout.flush()


def add_break(ch='*'):
        print_break = ch
        for i in range(50):
            print_break += ch

        print(print_break)
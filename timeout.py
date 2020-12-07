import signal
import time

# https://code-maven.com/python-timeout?fbclid=IwAR1ZHSjtBAQOZKJpi5lOrvMvTlnznnpCz7LIhhMTPSxw8LBIqdc1qtlk-w8


class TimeoutException(Exception):
    pass


def alarm_handler(signum, frame):
    raise TimeoutException()


signal.signal(signal.SIGALRM, alarm_handler)
signal.alarm(8)

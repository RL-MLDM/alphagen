import os
from contextlib import redirect_stdout, contextmanager
import baostock as bs


def baostock_login():
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull):
            bs.login()


def baostock_logout():
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull):
            bs.logout()


def baostock_relogin():
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull):
            bs.logout()
            bs.login()


@contextmanager
def baostock_login_context():
    baostock_login()
    try:
        yield None
    finally:
        baostock_logout()

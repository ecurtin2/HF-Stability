"""Utilities for printing human readable output."""


def header(string_, n_cols=80, symbol='-'):
    """Return a header with a string centered and a border.

    :param string_: The string to be centered
    :param n_cols: The width of the output
    :param symbol: The symbol to be used for the border
    :return: A string containing the bordered text.
    """
    hr = '\n' + n_cols * symbol + '\n'
    blank = symbol + (n_cols - 2*len(symbol)) * ' ' + symbol

    start = (n_cols // 2) - (len(string_) // 2)
    stop = start + len(string_)
    centered = blank[:start] + string_ + blank[stop:]

    return hr + centered + hr

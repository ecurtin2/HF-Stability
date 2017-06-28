
def header(string_, n_cols=80, symbol='-'):
    hr = '\n' + n_cols * symbol + '\n'
    blank = symbol + (n_cols - 2*len(symbol)) * ' ' + symbol

    start = (n_cols // 2) - (len(string_) // 2)
    stop = start + len(string_)
    centered = blank[:start] + string_ + blank[stop:]

    return hr + centered + hr

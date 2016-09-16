
def cpp_head_to_pyxhead(cpp_head, breaker=None):
    f = cpp_head + '.h'
    with open(f) as fyle:
        content = fyle.readlines()

    newlines = []
    public = False
    for line in content:
        if not line.lstrip().startswith('#'):
            line = line.lstrip().rstrip() # remove leading & trailing whitespace
            line = line.strip(";")           # remove end of line semicolon
            line = line.replace("//", "#")   # comments
            line = line.replace("arma::", "")
            line = line.replace(":", "")
            line = line.replace("uint64_t", "long long unsigned int")
            line = line.replace("uword", "long long unsigned int")
            if breaker in line:
                break
            words = line.split()
            if words:
                if words[0] == 'namespace':
                    my_ns = words[1]
                elif words[0] == 'class':
                    my_class = words[1]
                elif words[0] == 'public':
                    public = True
                elif words[0] == 'private':
                    public = False
                else:
                    if public and words[0] != '}':
                        newlines.append(line)
    line1 = 'cdef extern from "' + f + '"'
    if my_ns:
        line1 += (' namespace "' + my_ns + '"')
    line1 += ':'
    tab = '    ' # tab users rekt
    indentlevel = 1

    line2 = ''
    if my_class:
        line2 += tab + 'cdef cppclass ' + my_class + ':'
        indentlevel = 2

    newlines = [indentlevel*tab + i for i in newlines]

    comment = '#This file was automatically generated from ' + f + 'using cpp_header_to_pyxheader!'
    newlines.insert(0, line2)
    newlines.insert(0, line1)
    newlines.insert(0, comment)
    outfile = open(cpp_head + "_h.pyx", 'w')
    for line in newlines:
        outfile.write("%s\n" % line)
    outfile.close()


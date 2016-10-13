import os

class Wrapper(object):

    def __init__(self, pyx_header=None, pyx_lines=None, cpp_header=None, cpp_class=None, func_blacklist=None, pyx_is_class=False):
        self.pyx_header_file = pyx_header
        self.pyx_lines_file = pyx_lines
        self.cpp_header_file = cpp_header
        self.cpp_class_file = cpp_class
        self.py_funcs = None
        self.pyx_is_class = pyx_is_class
        self.func_blacklist = func_blacklist
        self.set_defaults()
        # Keep this order
        if pyx_header:
            self.get_pyx_header()

        self.get_cpp_pyx_header()
        if pyx_lines:
            self.get_pyx_lines()
        self.get_properties()
        self.read_funcs()
        self.write_pyfunc()
        
    def set_defaults(self):
        self.section_number = 0
        if self.func_blacklist == None:
            self.func_blacklist = ['2398caeaoxlkmdzmzjaalwk39damalkqk29mgziao']
        self.tab = '    '
        self.out_fname = 'CppClassWrap.pyx'
        self.output = ['# This File was automatically generated by CppClassWrapper\n']            
        self.cpp_to_py_type = {'double' : 'float',
                  'bool'   : 'bool',
                  'float'  : 'float',
                  'int'    : 'int',
                'uint64_t' : 'int',
                'uword' : 'int',
                'string' : 'str',
                '' : '',
                'double[]' : 'double&',
                'vec'  : ['numpy_to_vec_d' , 'vec_to_numpy' , 'np.ndarray[double, ndim=1]'], 
                'mat'  : ['numpy_to_mat_d' , 'mat_to_numpy' , 'np.ndarray[double, ndim=2, mode="fortran"]'], 
                'uvec' : ['numpy_to_uvec_d', 'uvec_to_numpy', 'np.ndarray[long long unsigned int, ndim=1]'], 
                'umat' : ['numpy_to_umat_d', 'umat_to_numpy', 'np.ndarray[long long unsigned int, ndim=2, mode="fortran"]'], 
                'vec&'  : ['numpy_to_vec_d' , 'vec_to_numpy' , 'np.ndarray[double, ndim=1]'], 
                'mat&'  : ['numpy_to_mat_d' , 'mat_to_numpy' , 'np.ndarray[double, ndim=2, mode="fortran"]'], 
                'uvec&' : ['numpy_to_uvec_d', 'uvec_to_numpy', 'np.ndarray[long long unsigned int, ndim=1]'], 
                'umat&' : ['numpy_to_umat_d', 'umat_to_numpy', 'np.ndarray[long long unsigned int, ndim=2, mode="fortran"]'] 
                              }

    def write_section(self, section, label):
        self.section_number += 1
        poundline = 72 * '#' + '\n'
        if len(label) % 2 != 0:
            label += ' '
        pad = ((71 - len(label)) / 2) * ' '
        sectionlabel = '#' + pad + label + pad + '#' + '\n'
        vspacing = [2 * '\n']
        section = [poundline, sectionlabel, poundline] + section + vspacing     
        return section

    def combine_sections(self):
        if self.pyx_header_file:
            self.output += self.write_section(self.pyx_header, '.pyx Header')
        
        self.output += self.write_section(self.cpp_pyx_header, 'Generated cpp->pyx Header')
        if self.pyx_lines_file:
            self.output += self.write_section(self.pyx_lines, 'pyx class')
        if self.pyfuncs:
            self.output += self.write_section(self.pyfuncs, 'Py Funcs')
        self.output += self.write_section(self.properties, 'Generated Properties')

    def get_pyx_header(self):
        with open(self.pyx_header_file) as f:
            lines = f.readlines()
        self.pyx_header =  lines

    def get_pyx_lines(self, classdocstring=None):
        lines = []
        if self.pyx_is_class: 
            line1 = 'cdef class Py' + self.my_class + ':' + '\n'
            if classdocstring:
                lines += self.tab + '"""' + classdocstring + '"""'
            line2 = self.tab + 'cdef ' + self.my_class + '* c_' + self.my_class + '\n'
            lines += ['def __cinit__(self):']
            lines += [self.tab + 'self.c_' + self.my_class + ' = new ' + self.my_class + '()']
            lines += ['def __dealloc__(self):']
            lines += [self.tab + 'del self.c_' + self.my_class]
            lines = [self.tab + line + '\n' for line in lines]
            lines = [line1] + [line2] + lines
         
        with open(self.pyx_lines_file) as f:
            pyx_lines = f.readlines()
        if self.pyx_is_class:
            tab = self.tab
        else:
            tab = ''
        pyx_lines = [tab + line for line in pyx_lines]
        self.pyx_lines = lines + [2*'\n'] + pyx_lines

    def get_property(self, c_type, var):
        py_type = self.cpp_to_py_type[c_type]
        is_ary = len(py_type[0]) > 1
        lines = []
        lines += ['def get_' + var + '():']
        
        # Getter
        if is_ary: 
            lines += [self.tab + '"""(' + py_type[2] + ') Get ' + var + '"""']
            lines += [self.tab + 'global ' + var]
            lines += [self.tab + 'return ' + py_type[1] + '(' + var + ')']
        else:
            lines += [self.tab + '"""(' + py_type + ') Get ' + var + '"""']
            lines += [self.tab + 'global ' + var]
            lines += [self.tab + 'return ' +  var]
        
        # Setter
        if is_ary:
            lines += ['def set_' + var + '('+ py_type[2] + ' \n ' + 5*self.tab + 'value not None):']
            lines += [self.tab + '"""(' + py_type[2] + ') Set ' + var + '"""']
            lines += [self.tab + 'global ' + var]
            lines += [self.tab + var + ' = ' + py_type[0] + '(value)']
        else:
            lines += ['def set_' + var + '(value):']
            lines += [self.tab + '"""(' + py_type[2] + ') Set ' + var + '"""']
            lines += [self.tab + 'global ' + var]
            lines += [self.tab + var + ' = ' + py_type + '(value)']
#lines += ['py_' + var + ' = property(get_'+ var + ', set_' + var + ')']
        lines += ['\n']
        return [line + '\n' for line in lines]
        
    def get_properties(self):
        variables = []
        for line in self.var_lines:
            if '(' not in line and ')' not in line:
                words = line.split()
                c_type = words[0]
                for i in range(1,len(words)):
                    variables.append([c_type, words[i].replace(',', '')])
        properties = []
        for x in variables:
            properties += self.get_property(x[0], x[1])            
        self.properties = [x for x in properties]

    def get_cpp_pyx_header(self, breaker=None):
        with open(self.cpp_header_file) as fyle:
            lines = fyle.readlines()
        self.out_fname = os.path.splitext(self.cpp_header_file)[0] + "_h.pyx"
        self.my_class = None
        

        self.var_lines = []
        public = True
        for line in lines:
            if (not '#' in line.lstrip() and not '<' in line.lstrip()):
                line = line.lstrip().rstrip() # remove leading & trailing whitespace
                line = line.strip(";")           # remove end of line semicolon
                line = line.replace("//", "#")   # comments
                line = line.replace("arma::", "")
                line = line.replace("std::", "")
                line = line.replace(":", "")
                line = line.replace("extern", "")
                if breaker:
                    if breaker in line:
                        break
                words = line.split()
                if words:
                    if words[0] == 'namespace':
                        self.my_ns = words[1]
                    elif words[0] == 'class':
                        self.my_class = words[1]        
                    elif words[0] == 'public':
                        public = True
                    elif words[0] == 'private':
                        public = False
                    else:
                        if public and words[0] != '}':
                            self.var_lines.append(line)
        line1 = 'cdef extern from "' + self.cpp_header_file + '"'
        if self.my_ns:
            line1 += (' namespace "' + self.my_ns + '"')
        line1 += ':'
        indentlevel = 1

        line2 = ''
        if self.my_class:
            line2 += self.tab + 'cdef cppclass ' + self.my_class + ':'
            indentlevel = 2

        cpp_pyx_header = [indentlevel*self.tab + i + '\n' for i in self.var_lines]
        cpp_pyx_header = [line2 + '\n'] + cpp_pyx_header
        cpp_pyx_header = [line1 + '\n'] + cpp_pyx_header
        new_header = []
        for line in cpp_pyx_header:
            line = line.replace("uint64_t", "long long unsigned int")
            line = line.replace("uword", "long long unsigned int")
            new_header.append(line)
        self.cpp_pyx_header = new_header
        
    def read_funcs(self):
        self.returntypes = []
        self.fnames = []
        self.argtypes = []
        for line in self.var_lines:
            if "(" in line and ")" in line:
                words = line.split()
                fname = words[1].rpartition('(')[0]
                if fname not in self.func_blacklist: 
                    self.returntypes.append(words[0])
                    self.fnames.append(fname)
                    self.argtypes.append((line.split('(')[1]).split(')')[0])

    def write_pyfunc(self):
        lines = []
        for i in range(len(self.returntypes)):
            returntype = self.returntypes[i]
            fname = self.fnames[i]
            argtypes = self.argtypes[i]
            argtypes = [x.strip() for x in argtypes.split(',')]

            args = []
            conv_funcs = ['' for i in range(len(argtypes))]
            conv_idx = []
            armalist = ['mat', 'umat', 'vec', 'uvec', 'mat&', 'umat&', 'vec&', 'uvec&']
            for idx, j in enumerate(argtypes):
                if any([i in j for i in armalist]):
                    conv_idx.append(idx)

                args.append(self.cpp_to_py_type[j])
            count = 1
            if len(args) == 1 and isinstance(args[0], basestring):
                if str(args[0]) != '':
                    argstring = args[0] + ' val'
                else:
                    argstring = args[0]       
            else:
                argstring = ''
                for arg in args:
                    if count > 1:
                        argstring += ', \n' + (len(fname) + 8) * ' '
                    if type(arg) == list:
                        t = arg[2]
                        conv_funcs[count - 1] = arg[0]                        
                    else:
                        t = arg
                    argstring += str(t) + ' val' + str(count)

                    count += 1


            lines += ['def py_' + fname + '(' + argstring + '):\n']

            ## calling the c level function
            c_func_args = []
            if count == 1 and args[0] != '':
                c_func_args.append('val')
            for i in range(count - 1):
                if i > 0:
                    c_func_args.append(', val' + str(i+1))
                else:
                    c_func_args.append('val' +  str(i+1))
                if i in conv_idx: 
                    lines += [self.tab + 'VAL' + str(i+1) + ' = '+ conv_funcs[i]  + '(val' + str(i+1) + ')\n']
                    if i > 0:
                        c_func_args[i] = ', VAL' + str(i+1)
                    else:
                        c_func_args[i] = 'VAL' +  str(i+1)
            c_func_arg_str = ''
            for i in c_func_args:
                c_func_arg_str += i

            if returntype == 'void':
                returnstr = ''
                returnclose = ''
            elif any([i in returntype for i in armalist]):
                returnstr = 'return ' + self.cpp_to_py_type[returntype][1] + '('
                returnclose = ')'
            else:
                returnstr = 'return '
                returnclose = ''
            lines += [self.tab + returnstr + fname + '(' + c_func_arg_str +  ')'  + returnclose + 2*'\n' ]
            
        self.pyfuncs = lines

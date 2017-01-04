import os  

DIR = '../Archive/'

for File in os.listdir(DIR):
    filename, file_extension = os.path.splitext(File)
    if file_extension == '.log':
        print "Changing: ", filename
        f = open(DIR+File, "r")
        contents = f.readlines()
        f.close()
        
        count = 0
        for line in contents:
            count += 1
            if "rs = " in line:
                break

        contents.insert(count, "mycase = cRHF2cUHF\n")
        f = open(DIR+File, "w")
        contents = "".join(contents)
        f.write(contents)
        f.close()

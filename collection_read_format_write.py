import os
def get_all_filenames(path):

    file_names=[]
    file_names1 =os.listdir('C:/Users/spj/Desktop/text grid')
    for name in file_names1:
        file_names.append(path+'/' + str(name))#err
    return file_names
files = get_all_filenames('C:/Users/spj/Desktop/text grid')

for file in files:
    text = open(file,'r')
    line = text.readline()
    lines = []
    while line:
        line = line.strip()
        if(line == 'item [2]:'):
            lines = text.readlines()
            break
        line = text.readline()
    if len(lines) == 0:
        continue
    lines.insert(0,'File type = "ooTextFile"\nObject ')
    text = open(file,'w')
    text.writelines(lines)
    text.close()
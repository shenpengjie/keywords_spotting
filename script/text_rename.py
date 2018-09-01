import os
g_res_path = 'C:/Users/spj/Desktop/labeled' #本地标签路径

def get_all_filenames(path):
     file_names = os.listdir(path)
     return file_names

def read_file(path,filename):
    text = open(os.path.join(path,filename))
    res = []
    res1 = []
    for i in range(10):
        res.append(text.readline())
    text.readline()
    res.append('        name = \"neidatongxue\"\n ')
    res1=text.readlines()
    text = open(os.path.join(path, filename),'w')
    for i in res:
        text.write(str(i))
    for i in res1:
        text.write(str(i))


files = get_all_filenames(g_res_path)

for filename in files:
    read_file(g_res_path, filename)

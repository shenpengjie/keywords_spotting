import os
g_res_path = 'C:/Users/spj/Desktop/labeled'

def get_all_filenames(path):
     file_names = os.listdir(path)
     return file_names

def read_file(path,filename):
    text = open(os.path.join(path,filename))
    text.readline()
    text.readline()
    nameline = text.readline()
    text.close()
    nameline = nameline.split('=')[1]
    nameline = nameline.strip()
    nameline = nameline.strip('\"')
    os.rename(os.path.join(path,filename),os.path.join(path,nameline + '.textgrid'))

files = get_all_filenames(g_res_path)

for filename in files:
    read_file(g_res_path, filename)
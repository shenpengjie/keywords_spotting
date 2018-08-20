import os;
from pydub import AudioSegment
def rename():
        i=0
        path="C:/Users/spj/Desktop/text grid";
        filelist=os.listdir(path)
        for files in filelist:
            i=i+1
            Olddir=os.path.join(path,files);
            if os.path.isdir(Olddir):
                    continue;
            filename=os.path.splitext(files)[0];
            filetype=os.path.splitext(files)[1];
            Newdir=os.path.join(path,str(i)+filetype);
            os.rename(Olddir,Newdir)
rename()






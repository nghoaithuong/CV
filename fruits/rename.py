import os
os.chdir('/home/hoaithuong/PycharmProjects/fruits/Plum')
i=1
for file in os.listdir():
      src=file
      dst="plum."+str(i)+".jpg"
      os.rename(src,dst)
      i+=1

import os
os.chdir('/home/hoaithuong/CV/fruits/Train/Train_Tomato')
i=1
for file in os.listdir():
      src=file
      dst="tomato."+str(i)+".jpg"
      os.rename(src,dst)
      i+=1
os.chdir('/home/hoaithuong/CV/fruits/Train/Train_Apple')
i=1
for file in os.listdir():
      src=file
      dst="apple."+str(i)+".jpg"
      os.rename(src,dst)
      i+=1

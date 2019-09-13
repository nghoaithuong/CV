import os
os.chdir('/home/hoaithuong/CV/detect_tomato/Train_Tomato')
i=1
for file in os.listdir():
      src=file
      dst="tomato"+str(i)+".jpg"
      os.rename(src,dst)
      i+=1

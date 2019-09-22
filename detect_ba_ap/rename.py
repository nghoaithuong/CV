import os
os.chdir('/home/hoaithuong/CV/detect_ba_ap/Lemon')
i=1
for file in os.listdir():
      src=file
      dst="lemon."+str(i)+".jpg"
      os.rename(src,dst)
      i+=1

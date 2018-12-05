f=open("./2000landmarks.txt","r")
d=open("./r2000landmarks.txt","w")

for i in range(2000):
    a=f.readline()[0:-1]
    d.write(a+', \n')

f.close()
d.close()
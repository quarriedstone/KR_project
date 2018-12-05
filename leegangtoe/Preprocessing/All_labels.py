f=open("./All_labels.txt","r")
d=open("./ratings.txt","w")
dict={}
for i in range(5500):
    a=f.readline().split(" ")
    dict[a[0]]=a[1]
    print(a[1][0:-1])

for i in range(2000):
    key="AF"+str(i+1)+".jpg"
    if(key=="AF1112.jpg"):
        key="AF1111.jpg"
    d.write(format(float(dict[key][0:-1]),'.2f')+'\n')

f.close()
d.close()
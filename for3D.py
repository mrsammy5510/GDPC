f = open('3D_spatial_network.txt')
after = open('3D_spatial_network.csv',"w")

for i in range(434874):
    k = f.readline()
    comma = k.rfind(',')
    l = list(k)
    del k[0:k]
    data = ''.join(l)
    ans = data.replace(' ','')
    after.write(ans)

print(ans)
f = open('S2.txt')
after = open('S2.csv',"w")

for i in range(5000):
    k = f.readline()
    comma = k.find(',')
    l = list(k)
    l[comma] = ','
    data = ''.join(l)   
    ans = data.replace(' ','')
    after.write(ans)

print(ans)
def find_prev_spike(s,t,id):
    for j in range(t,0,-1):
        if s[id][j] == 0:
            return j
    return 0

d = [[0,1,2,3,4,0,1,2,3,4,5,6,0],[1,2,0,1,2,3,0, 1,2,3,4,5,6]]
print(d[0][1])
print(find_prev_spike(s=d,t=6-1,id=0))
print(find_prev_spike(s=d,t=6-1,id=1))
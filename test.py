import random


fp = open('points.txt','w')
N = 100
fp.write('100\n')
for i in range(N):
    x = random.randrange(0, 1000)
    y = random.randrange(0, 1000)
    fp.write('{} {}\n'.format(x,y))
fp.write('10\n')
for i in range(10):
    x = random.randrange(0, 1000)
    y = random.randrange(0, 1000)
    fp.write('{} {}\n'.format(x,y))

fp.close()

import numpy as np

# nStart = 12
# nEnd = 23
#
# psi = np.load(f'ICs/month_{nStart}.npy')
# print(np.shape(psi))
# psi = psi[0:150]
#
#
# for i in range(nStart+1,nEnd+1):
#     next_psi = np.load(f'ICs/month_{i}.npy')
#     psi = np.concatenate((psi, next_psi[0:150]), axis=0)
#     print("==>",i,psi.shape)
#     #print(psi[0])
#
# np.save('ICs/oneYear.npy',psi)



nStart = 1
nEnd = 12

psi = np.load(f'Data-LR/areza_{nStart}.npy')
print(np.shape(psi),"***")
psi = psi[0:150]

for i in range(nStart+1,nEnd+1):
    next_psi = np.load(f'Data-LR/areza_{i}.npy')
    psi = np.concatenate((psi, next_psi[0:150]), axis=0)
    print("==>",i,psi.shape)
    #print(psi[0])

np.save('Data-LR/oneYearLR.npy',psi)
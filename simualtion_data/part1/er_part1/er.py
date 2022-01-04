import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

name = 'part1 er'
cols = ['<c>', 'PC', 'PD', "F"]

df1 = pd.read_csv(name+'0.csv')

c_m1 = df1['<c>'].to_numpy()
pc1 = df1['PC'].to_numpy()
pds1 = df1['PD'].to_numpy()
f1 = df1['F'].to_numpy()

# for i in range(45, 47):
for i in range(1, 3):
    df_t1 = pd.read_csv(name+str(i)+'.csv')

    c_m1 += df_t1[cols[0]].to_numpy()

    pc1 += df_t1[cols[1]].to_numpy()

    pds1 += df_t1[cols[2]].to_numpy()

    f1 += df_t1[cols[3]].to_numpy()

bs1 = np.arange(1, 2.2, 0.1)
N1 = 4000 * 3

# bs = np.arange(2.2, 3.8, 0.1)
# N = 4000 * 3

plt.plot(bs1, c_m1/3, 'o-', label='<c> for ER')#, c='#00035b')
plt.plot(bs1, np.asarray(pc1) / N1, '^-', label='PC for ER')#, c='r')
plt.plot(bs1, np.asarray(pds1) / N1, 's-', label='PD')
plt.plot(bs1, np.asarray(f1) / N1, '*-', label='F')
# plt.legend()
# plt.xlabel('b')
# plt.savefig('normal_pic02_2_2.png')


# name0 = 'part1 data_comp0_'
# name1 = 'part1 data_comp1_'
# name2 = 'part1 data_comp2_'
# cols = ['<c>', 'PC', 'PD', "F"]

# df = pd.read_csv('part1 data_comp1_0.csv')

# c_m = df['<c>'].to_numpy()
# pc = df['PC'].to_numpy()
# pds = df['PD'].to_numpy()
# f = df['F'].to_numpy()

# for i in range(1, 10):
#     df_t1 = pd.read_csv(name1+str(i)+'.csv')
#     df_t2 = pd.read_csv(name2+str(i)+'.csv')

#     c_m += df_t1[cols[0]].to_numpy()
#     c_m += df_t2[cols[0]].to_numpy()

#     pc += df_t1[cols[1]].to_numpy()
#     pc += df_t2[cols[1]].to_numpy()

#     pds += df_t1[cols[2]].to_numpy()
#     pds += df_t2[cols[2]].to_numpy()

#     f += df_t1[cols[3]].to_numpy()
#     f += df_t2[cols[3]].to_numpy()

# for i in range(10, 20):
#     df_t0 = pd.read_csv(name0 + str(i) + '.csv')
#     c_m += df_t0[cols[0]].to_numpy()

#     pc += df_t0[cols[1]].to_numpy()

#     pds += df_t0[cols[2]].to_numpy()

#     f += df_t0[cols[3]].to_numpy()

# for i in range(10, 20):
#     df_t0 = pd.read_csv(name1 + str(i) + '.csv')
#     c_m += df_t0[cols[0]].to_numpy()

#     pc += df_t0[cols[1]].to_numpy()

#     pds += df_t0[cols[2]].to_numpy()

#     f += df_t0[cols[3]].to_numpy()

# c_m += pd.read_csv(name2+"0.csv")[cols[0]].to_numpy()
# pc += pd.read_csv(name2+"0.csv")[cols[1]].to_numpy()
# pds += pd.read_csv(name2+"0.csv")[cols[2]].to_numpy()
# f += pd.read_csv(name2+"0.csv")[cols[3]].to_numpy()

# bs = np.arange(1, 5.8, 0.1)
# N = 4000 * 40
# # plt.figure(figsize=(5, 3.5))
# plt.plot(bs[:20], (c_m/40)[:20], 'o-', label='<c> for BA', markersize=5, clip_on=False, c='blue')
# plt.plot(bs[:20], (np.asarray(pc) / N)[:20], '^-', label='PC for BA', markersize=5, clip_on=False, c='#8A2BE2')

plt.legend()
plt.show()
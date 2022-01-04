import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

name = 'normal_part1 data_comp_'
cols = ['<c>', 'PC', 'PD', "F"]

df = pd.read_csv(name+'30.csv')

c_m = df['<c>'].to_numpy()
pc = df['PC'].to_numpy()
pds = df['PD'].to_numpy()
f = df['F'].to_numpy()

# for i in range(45, 47):
for i in range(31, 42):
    df_t1 = pd.read_csv(name+str(i)+'.csv')

    c_m += df_t1[cols[0]].to_numpy()

    pc += df_t1[cols[1]].to_numpy()

    pds += df_t1[cols[2]].to_numpy()

    f += df_t1[cols[3]].to_numpy()

bs = np.arange(1, 4.2, 0.1)
N = 4000 * 30

# bs = np.arange(2.2, 3.8, 0.1)
# N = 4000 * 3

plt.plot(bs, c_m/30, 'o-', label='<c>')
plt.plot(bs, np.asarray(pc) / N, '^-', label='PC')
plt.plot(bs, np.asarray(pds) / N, 's-', label='PD')
plt.plot(bs, np.asarray(f) / N, '*-', label='F')
plt.legend()
plt.xlabel('b')
# plt.savefig('normal_pic02_2_2.png')
plt.show()
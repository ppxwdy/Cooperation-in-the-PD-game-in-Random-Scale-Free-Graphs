import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

name0 = 'part1 data_comp0_'
name1 = 'part1 data_comp1_'
name2 = 'part1 data_comp2_'
cols = ['<c>', 'PC', 'PD', "F"]

df = pd.read_csv('part1 data_comp1_0.csv')

c_m = df['<c>'].to_numpy()
pc = df['PC'].to_numpy()
pds = df['PD'].to_numpy()
f = df['F'].to_numpy()

for i in range(1, 10):
    df_t1 = pd.read_csv(name1+str(i)+'.csv')
    df_t2 = pd.read_csv(name2+str(i)+'.csv')

    c_m += df_t1[cols[0]].to_numpy()
    c_m += df_t2[cols[0]].to_numpy()

    pc += df_t1[cols[1]].to_numpy()
    pc += df_t2[cols[1]].to_numpy()

    pds += df_t1[cols[2]].to_numpy()
    pds += df_t2[cols[2]].to_numpy()

    f += df_t1[cols[3]].to_numpy()
    f += df_t2[cols[3]].to_numpy()

for i in range(10, 20):
    df_t0 = pd.read_csv(name0 + str(i) + '.csv')
    c_m += df_t0[cols[0]].to_numpy()

    pc += df_t0[cols[1]].to_numpy()

    pds += df_t0[cols[2]].to_numpy()

    f += df_t0[cols[3]].to_numpy()

for i in range(10, 20):
    df_t0 = pd.read_csv(name1 + str(i) + '.csv')
    c_m += df_t0[cols[0]].to_numpy()

    pc += df_t0[cols[1]].to_numpy()

    pds += df_t0[cols[2]].to_numpy()

    f += df_t0[cols[3]].to_numpy()

c_m += pd.read_csv(name2+"0.csv")[cols[0]].to_numpy()
pc += pd.read_csv(name2+"0.csv")[cols[1]].to_numpy()
pds += pd.read_csv(name2+"0.csv")[cols[2]].to_numpy()
f += pd.read_csv(name2+"0.csv")[cols[3]].to_numpy()

bs = np.arange(1, 5.8, 0.1)
N = 4000 * 40
plt.figure(figsize=(5, 3.5))
plt.plot(bs, c_m/40, 'o-', label='<c>', markersize=3, clip_on=False)
plt.plot(bs, np.asarray(pc) / N, '^-', label='PC', markersize=3, clip_on=False)
plt.plot(bs, np.asarray(pds) / N, 's-', label='PD',markersize=3,  clip_on=False)
plt.plot(bs, np.asarray(f) / N, '*-', label='F', markersize=3, clip_on=False)


plt.legend(loc=1)
plt.xlabel('b')
plt.ylabel('ratio')

plt.xlim(1, 5.7)
plt.ylim(0, 1)

plt.savefig('ba_part1.png', bbox_inches='tight')
plt.show()
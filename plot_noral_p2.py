import pandas as pd
import matplotlib.pyplot as plt

k2 = pd.read_csv('normal_part2_k2.csv')
k3 = pd.read_csv('normal_part2_k3.csv')


record_k2 = dict()
record_k3 = dict()
for col in k2.columns:
    record_k2[col] = k2[col].to_list()
    record_k3[col] = k3[col].to_list()



fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharex='all')
t = [i for i in range(1, k2.shape[0]+1)]
b_s = list(record_k2.keys())
bs = []
for i in range(len(b_s)):




for b in bs:
    ax[0].plot(t, record_k2[b], 'o-', markersize=0.5, lw=1, label='b is '+str(b))
    ax[1].plot(t, record_k3[b], 'o-', markersize=0.5, lw=1, label='b is '+str(b))

ax[0].set_yscale('log')
ax[1].set_yscale('log')

# ax[1].legend(loc=(.05, -.4), ncol=3)
ax[0].set_ylabel('b')
ax[1].set_ylabel('b')

ax[0].set_title('the time evolution of the fraction of cooperators c(t) when k = 2')
ax[1].set_title('the time evolution of the fraction of cooperators c(t) when k = 3')

plt.xlim(1)
plt.xscale('log')
plt.xlabel('t')
# plt.savefig('normal_pic_part2_horizontal.png')
plt.show()
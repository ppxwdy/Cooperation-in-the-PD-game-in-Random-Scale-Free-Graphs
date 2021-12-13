import pandas as pd
import matplotlib.pyplot as plt

k2 = pd.read_csv('normal_part2_k2.csv')
k3 = pd.read_csv('normal_part2_k3.csv')


record_k2 = dict()
record_k3 = dict()
for col in k2.columns:
    record_k2[col] = k2[col].to_list()
    record_k3[col] = k3[col].to_list()


# horizontal
# fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey='all')

# vertical
fig, ax = plt.subplots(2, 1, figsize=(8, 12), sharex='all')


t = [i for i in range(1, k2.shape[0]+1)]
b_s = list(record_k2.keys())
bs = []
for b_ in b_s:
    bs.append(str(b_)[:3])


for i in range(len(bs)):
    ax[0].plot(t, record_k2[b_s[i]], 'o-', markersize=0.5, lw=1, label='b is '+bs[i], clip_on=False)
    ax[1].plot(t, record_k3[b_s[i]], 'o-', markersize=0.5, lw=1, label='b is '+bs[i], clip_on=False)


# horizontal
# ax[0].set_yscale('log')
# ax[1].set_yscale('log')

# # ax[1].legend(loc=(.05, -.4), ncol=3)
# ax[0].set_ylabel('<c>', size=15)
# ax[1].set_ylabel('<c>', size=15)

# ax[0].set_title('The time evolution of the fraction of cooperators c(t)\n in Random SF when k = 2', size=15)
# ax[1].set_title('The time evolution of the fraction of cooperators c(t)\n in Random SF when k = 3', size=15)

# ax[0].set_xscale('log')
# ax[1].set_xscale('log')

# ax[0].set_xlim(1, 10000)
# ax[1].set_xlim(1, 10000)

# ax[0].set_ylim(0, 1)
# ax[1].set_ylim(0, 1)


# ax[0].set_xlabel('t', size=15)
# ax[1].set_xlabel('t', size=15)

# ax[1].legend(loc=2, bbox_to_anchor=(1.05, 1), borderaxespad=0.)
# plt.savefig('random_pic_part2_horizontal.png')

# vertical
ax[0].set_yscale('log')
ax[1].set_yscale('log')

# ax[1].legend(loc=(.05, -.4), ncol=3)
ax[0].set_ylabel('<c>')
ax[1].set_ylabel('<c>')

ax[0].set_title('The time evolution of the fraction of cooperators c(t)\n in Random SF when k = 2')
ax[1].set_title('The time evolution of the fraction of cooperators c(t)\n in Random SF when k = 3')

ax[0].set_xscale('log')
ax[1].set_xscale('log')

ax[0].set_xlim(1, 10000)
ax[1].set_xlim(1, 10000)

ax[0].set_ylim(0, 1)
ax[1].set_ylim(0, 1)

ax[0].set_xlabel('t')
ax[1].set_xlabel('t')

ax[1].legend(bbox_to_anchor=(1, -.1), borderaxespad=0., ncol=6)
plt.savefig('random_pic_part2_vertical.png')


plt.show()
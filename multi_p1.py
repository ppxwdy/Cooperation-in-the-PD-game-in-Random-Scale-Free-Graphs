from multiprocessing import Process, Manager
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd


def generator(N, m):
    """
    generate the network
    :param N: number of nodes
    :param m: new node has m links
    :return:
    """

    # randomly generate the C and D
    identity = np.ones(N, dtype=int)
    Cs = np.random.choice(N, N // 2, replace=False)
    for i in Cs:
        identity[i] = 0  # 0 - C, 1 - D

    # generate a BA network with 4000 nodes and <k> = 2m = 4
    ba = nx.barabasi_albert_graph(N, m, seed=666)
    # ba = nx.scale_free_graph(N)

    # nodes
    nodes = [i for i in range(N)]

    # get the neighbours for each vert [(0, [1, 2]), (1, [0, 3]), (2, [0, 3]), (3, [1, 2])]
    adj = [(n, list(nbrdict.keys())) for n, nbrdict in ba.adjacency()]
    # get the degrees [(0, [1, 2]), (1, [0, 3]), (2, [0, 3]), (3, [1, 2])]
    degrees = list(ba.degree())

    return ba, adj, identity, degrees, nodes


# ba, adj, identity, degrees, nodes = generator(4000, 2)


def cal_gain(node, b, adj, identity):
    """
    calculate gain for the given node
    :param node: the node i
    """
    neighbours = adj[node][1]
    gain = 0
    idi = identity[node]
    for n in neighbours:
        idj = identity[n]
        if idi == idj and idi == 0:
            gain += 1
        elif idi == 1 and idj == 0:
            gain += b
    return gain


def update(nodei, nodej, b, gains, records, identity, degrees):
    """
    compare nodei's gain with its neighbour nodej, and find out
    how to udpate nodei's od
    :param records: records of the numebr of pc, pd, f
    :param nodei: nodei
    :param nodej: nodej
    :return:
    """
    pc = records[0]
    pd = records[1]
    f = records[2]
    if gains[nodei] < gains[nodej]:
        beta = 1 / (max(degrees[nodei][1], degrees[nodei][1]) * b)
        dice = np.random.rand()
        # dice <= beta means accept
        if dice <= beta:
            idi = identity[nodei]
            idj = identity[nodej]
            # if i and j have different id and i will change
            if idi != idj:
                if idi == 0:
                    pc[nodei] = 0
                elif idi == 1:
                    pd[nodei] = 0
                f[nodei] = 1
            identity[nodei] = identity[nodej]


def choose_neighbor(nodei, adj):
    """
    choose the node j for the node compare the gain
    :param nodei: the node we need to find a neighbour for
    :return: the neighbour chosen randomly
    """
    neighbors = adj[nodei][1]
    dice = np.random.randint(0, len(neighbors))
    return neighbors[dice]


# generate the b values
bs = np.arange(4.5, 6, 0.1)

# transient time
t0 = 500
# get steady
t1 = 1000
# after steady
ts = 10000

N = 4000
m = 2

s_all = time.time()


# for b in bs:

def iter(N, m, c_means, pcs, pds, fs, b):
    sb = time.time()
    # generate the graph
    ba, adj, identity, degrees, nodes = generator(N, m)
    # pc, pd, f will all have N elements
    # if born as c or d, in pc, pd the elements on the same pos will be 1, 0 otherwise
    # if change id, the pos in f will be 1
    # calculate the sum of these list will give the number of node in these classes
    pc = np.zeros(N)
    pd = np.zeros(N)
    f = np.zeros(N)
    for node in range(N):
        if identity[node] == 0:
            pc[node] = 1
        else:
            pd[node] = 1
    records = [pc, pd, f]
    # pre-evo
    for t in range(t0):
        # calculate gain
        gains = dict()
        for node in nodes:
            gains[node] = cal_gain(node, b, adj, identity)
        # update
        for node in nodes:
            nodej = choose_neighbor(node, adj)
            update(node, nodej, b, gains, records, identity, degrees)

    # get steady
    key = True
    for t in range(t1):
        oldc = N - np.sum(identity)
        # calculate gain
        gains = dict()
        for node in nodes:
            gains[node] = cal_gain(node, b, adj, identity)
        # update
        for node in nodes:
            nodej = choose_neighbor(node, adj)
            update(node, nodej, b, gains, records, identity, degrees)
        newc = N - np.sum(identity)
        # find out reach steady state or not
        if key:
            if abs(newc - oldc) < 1 / np.sqrt(N):
                print(f'we have reached the steady state after {t} times evolution.')
                key = False
    eb = time.time()
    c_means.append((b, (N - np.sum(identity)) / N))
    pcs.append((b, np.sum(records[0])))
    pds.append((b, np.sum(records[1])))
    fs.append((b, np.sum(records[2])))
    print(f'The time we used for b = {b} is {eb - sb}s.')


# ea = time.time()
# print(f'The time we used for all bs is {s_all-ea}s.')


def takefirst(x):
    return x[0]


if __name__ == '__main__':
    ma = Manager()

    # record the <c> for each b
    c_means = ma.list()
    # record for the number of PC
    pcs = ma.list()
    # record for the number of PD
    pds = ma.list()
    # record for the number of F
    fs = ma.list()
    s = time.time()
    processes = []
    for b in bs:
        p = Process(target=iter, args=(N, m, c_means, pcs, pds, fs, b,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    print(f'The whole process include t0 and t1 need {time.time() - s}s')

    pcs.sort(key=takefirst)
    c_means.sort(key=takefirst)
    pds.sort(key=takefirst)
    fs.sort(key=takefirst)

    pcs_ = [pcs[i][1] for i in range(len(pcs))]
    pds_ = [pds[i][1] for i in range(len(pds))]
    c_means_ = [c_means[i][1] for i in range(len(c_means))]
    fs_ = [fs[i][1] for i in range(len(fs))]

    # save the data
    # data = pd.DataFrame({'<c>': c_means_, 'PC': pcs_, 'PD': pds_, 'F': fs_})
    # data.to_csv('part1 data.csv', index=False, sep=',')

    plt.plot(bs, c_means_, 'o-', label='<c>')
    plt.plot(bs, np.asarray(pcs_) / 2000, '^-', label='PC')
    plt.plot(bs, np.asarray(pds_) / 2000, 's-', label='PD')
    plt.plot(bs, np.asarray(fs_) / N, '*-', label='F')
    plt.legend()
    plt.xlabel('b')
    # plt.savefig('pic1.png')
    plt.show()

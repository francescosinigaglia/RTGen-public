import numpy as np
import matplotlib.pyplot as plt

raw = np.genfromtxt('sed.txt')

lam = raw[:,0]
sed = raw[:,1]

plt.plot(lam[2:], sed[2:])
plt.xscale('log')
plt.yscale('log')
plt.savefig('sed_test.pdf', bbox_inches='tight')

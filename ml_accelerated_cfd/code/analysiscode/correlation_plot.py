import matplotlib.pyplot as plt

# Exact 
nx_exact = 256
order_exact = 2

# order = 0

nxs_0   = [64,    128,   256,  512]
times_0 = [0.026, 0.293, 3.0,  28.5]
t_95_0  = [6.2,   8.6,   11.5, 14.5]


nxs_1   = [32,    64,   128,  192,  256,  384]
times_1 = [0.019, 0.19, 1.48, 5.77, 12.9, 48.0]
t_95_1  = []

nxs_2   = [32,    64,  96,   128,  192]
times_2 = [0.118, 1.0, 3.64, 10.7, 39.5]
t_95_2  = []


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_yscale('log')
line, = ax.plot(t_95_0, times_0, color='blue', lw=2)
ax.set_xlim([0.0, 15.0])

plt.show()
import matplotlib.pyplot as plt
import numpy as onp
import h5py

from arguments import get_args

orders = [0, 1, 2]
nxs = [[32, 64, 128], [32, 48, 64], [16, 32, 48]]
nxs_fv_baseline = [32, 64, 128]
baseline_dt_reductions = [8.0]
Tf = 10.0
Np = int(Tf * 10)
T_chunk = Tf / Np
N_test = 1
T_runtime = 1.0
device = 'cpu'

args = get_args()


### read runtimes
f1 = h5py.File(
	"{}/data/{}_fv.hdf5".format(args.read_write_dir, device),
	"r",
)
runtime_fv = []

for nx in nxs_fv_baseline:
	runtime_fv.append(f1[str(nx)][0] / T_runtime)
f1.close()

runtime_dg = []
for o, order in enumerate(orders):
	f2 = h5py.File("{}/data/{}_order{}.hdf5".format(args.read_write_dir, device, order), "r",)
	runtime_dg.append([])
	for nx in nxs[o]:
		runtime_dg[o].append(f2[str(nx)][0] / T_runtime)
	f2.close()

### read correlation coefficients

fv_corr = onp.zeros((len(nxs_fv_baseline), Np+1))
order0_corr = onp.zeros((len(nxs[0]), Np+1))
order1_corr = onp.zeros((len(nxs[1]), Np+1))
order2_corr = onp.zeros((len(nxs[2]), Np+1))

for n in range(N_test):

	f_fv = h5py.File("{}/data/corr_run{}_fv.hdf5".format(args.read_write_dir, n),"r",)
	for i, nx in enumerate(nxs_fv_baseline):
		fv_corr[i] += f_fv[str(nx)][:] / N_test
	f_fv.close()

	f0 = h5py.File("{}/data/corr_run{}_order0.hdf5".format(args.read_write_dir, n),"r")
	for i, nx in enumerate(nxs[0]):
		order0_corr[i] += f0[str(nx)][:] / N_test
	f0.close()

	f1 = h5py.File("{}/data/corr_run{}_order1.hdf5".format(args.read_write_dir, n),"r")
	for i, nx in enumerate(nxs[1]):
		order1_corr[i] += f1[str(nx)][:] / N_test
	f1.close()

	f2 = h5py.File("{}/data/corr_run{}_order2.hdf5".format(args.read_write_dir, n),"r")
	for i, nx in enumerate(nxs[2]):
		order2_corr[i] += f2[str(nx)][:] / N_test
	f2.close()


### find time until correlation < 0.95

t95_fv = [0.] * len(nxs_fv_baseline)
t95_order0 = [0.] * len(nxs[0])
t95_order1 = [0.] * len(nxs[1])
t95_order2 = [0.] * len(nxs[2])

for n in range(N_test):

	f_fv = h5py.File("{}/data/corr_run{}_fv.hdf5".format(args.read_write_dir, n),"r",)
	for i, nx in enumerate(nxs_fv_baseline):
		j = 0
		while j < (Np+1):
			if f_fv[str(nx)][j] < 0.95:
				break
			else:
				j += 1
		t95_fv[i] += ((j-1) * T_chunk + (0.95 - f_fv[str(nx)][j-1]) / (f_fv[str(nx)][j] - f_fv[str(nx)][j-1]) * T_chunk) / N_test
	f_fv.close()

	f0 = h5py.File("{}/data/corr_run{}_order0.hdf5".format(args.read_write_dir, n),"r")
	for i, nx in enumerate(nxs[0]):
		j = 0
		while j < (Np+1):
			if f0[str(nx)][j] < 0.95:
				break
			else:
				j += 1
		t95_order0[i] += ((j-1) * T_chunk + (0.95 - f0[str(nx)][j-1]) / (f0[str(nx)][j] - f0[str(nx)][j-1]) * T_chunk) / N_test
	f0.close()

	f1 = h5py.File("{}/data/corr_run{}_order1.hdf5".format(args.read_write_dir, n),"r")
	for i, nx in enumerate(nxs[1]):
		j = 0
		while j < (Np+1):
			if f1[str(nx)][j] < 0.95:
				break
			else:
				j += 1
		t95_order1[i] += ((j-1) * T_chunk + (0.95 - f1[str(nx)][j-1]) / (f1[str(nx)][j] - f1[str(nx)][j-1]) * T_chunk) / N_test
	f1.close()

	f2 = h5py.File("{}/data/corr_run{}_order2.hdf5".format(args.read_write_dir, n),"r")
	for i, nx in enumerate(nxs[2]):
		j = 0
		while j < (Np+1):
			if f2[str(nx)][j] < 0.95:
				break
			else:
				j += 1
		t95_order2[i] += ((j-1) * T_chunk + (0.95 - f2[str(nx)][j-1]) / (f2[str(nx)][j] - f2[str(nx)][j-1]) * T_chunk) / N_test
	f2.close()

#### Plot 1: time vs correlation
fig1, axs1 = plt.subplots()
fig2, axs2 = plt.subplots()
fig3, axs3 = plt.subplots()

T = onp.arange(0, Np+1) * T_chunk

for i, nx in enumerate(nxs_fv_baseline):
	axs1.plot(T, fv_corr[i], label="FV nx={}".format(nx))
axs1.plot(T, order2_corr[-1], label="DG nx={}".format(nxs[2][0]))
axs1.set_xlabel("Time")
axs1.set_ylabel("Correlation")

fig1.legend()

#### Plot 2: grid resolution vs t95

axs2.plot(nxs_fv_baseline, t95_fv, label="FV")
axs2.plot(nxs[0], t95_order0, label="DG p=0")
axs2.plot(nxs[1], t95_order1, label="DG p=1")
axs2.plot(nxs[2], t95_order2, label="DG p=2")
axs2.set_xlabel("grid resolution")
axs2.set_xscale('log')
axs2.set_ylabel("T95")
axs2.set_xticks([])
axs2.set_xticks([16, 32, 64, 128])
axs2.minorticks_off()
axs2.set_xticklabels(['16', '32', '64', '128'])
fig2.legend()



#### Plot 3: t95 vs runtime

axs3.plot(t95_fv, runtime_fv, label="FV")
axs3.plot(t95_order0, runtime_dg[0], label="DG p=0")
axs3.plot(t95_order1, runtime_dg[1], label="DG p=1")
axs3.plot(t95_order2, runtime_dg[2], label="DG p=2")
axs3.set_xlabel("T95")
axs3.set_ylabel("Runtime")
axs3.set_yscale('log')
axs3.set_xlim([0.,10.0])
fig3.legend()




plt.show()

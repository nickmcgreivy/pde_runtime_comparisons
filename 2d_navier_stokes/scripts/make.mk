# SCRIPTS
COMPUTE_RUNTIME_SCRIPT = $(BASEDIR)/scripts/test_runtime.py
COMPUTE_CORRCOEF_SCRIPT = $(BASEDIR)/scripts/compute_corrcoef.py
DG_PLOT_SIM = $(BASEDIR)/scripts/plot_dg_demo.py
PLOT_ACCURACY_RUNTIME_SCRIPT = $(BASEDIR)/scripts/plot_accuracy_vs_runtime.py
PRINT_FNO_STATISTICS_SCRIPT = $(BASEDIR)/scripts/print_fno_statistics_script.py

ARGS = --poisson_dir $(POISSON_DIR) --read_write_dir $(READWRITE_DIR)

demo_dg_code:
	python $(DG_PLOT_SIM) $(ARGS)

compute_runtime :
	python $(COMPUTE_RUNTIME_SCRIPT) $(ARGS)

compute_corrcoef :
	python $(COMPUTE_CORRCOEF_SCRIPT) $(ARGS)

plot_accuracy_runtime: 
	python $(PLOT_ACCURACY_RUNTIME_SCRIPT) $(ARGS)

print_fno_statistics:
	python $(PRINT_FNO_STATISTICS_SCRIPT) $(ARGS)
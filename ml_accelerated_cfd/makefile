BASEDIR=/Users/nmcgreiv/research/thesis/pde_runtime_comparisons/ml_accelerated_cfd
READWRITE_DIR=/Users/nmcgreiv/research/thesis/pde_runtime_comparisons/ml_accelerated_cfd
SCRIPT_DIR=$(BASEDIR)/code/scripts
POISSON_DIR=$(BASEDIR)
MAKE_SCRIPT=$(SCRIPT_DIR)/make.mk
PYTHONPATH=$(BASEDIR)/code/simcode:$(BASEDIR)/code/analysiscode:$(BASEDIR)/code/datagencode:$(BASEDIR)/code/scripts:$(BASEDIR)

ADVECTION_ARGS_FILE=$(BASEDIR)/data/args/2d_advection_args.txt
EULER_ARGS_FILE=$(BASEDIR)/data/args/2d_euler_args.txt
HW_ARGS_FILE=$(BASEDIR)/data/args/2d_hw_args.txt

EQ=EULER

UNIQUE_ID = $(EQ)_2d_test
LABELS=label
ARGS_FILE = $($(EQ)_ARGS_FILE)

export SCRIPT_DIR READWRITE_DIR POISSON_DIR UNIQUE_ID ARGS_FILE PYTHONPATH

train_data:
	make -f $(MAKE_SCRIPT) train_data

test_data:
	make -f $(MAKE_SCRIPT) test_data

train_model:
	make -f $(MAKE_SCRIPT) train_model

plot_training_losses:
	make -f $(MAKE_SCRIPT) plot_training_losses

test_losses:
	make -f $(MAKE_SCRIPT) test_losses

plot_losses:
	make -f $(MAKE_SCRIPT) plot_losses

test_stability:
	make -f $(MAKE_SCRIPT) test_stability

plot_stability:
	make -f $(MAKE_SCRIPT) plot_stability

analyze_test_data:
	make -f $(MAKE_SCRIPT) analyze_test_data

eval_data:
	make -f $(MAKE_SCRIPT) eval_data

analyze_eval_data:
	make -f $(MAKE_SCRIPT) analyze_eval_data

compute_runtime:
	make -f $(MAKE_SCRIPT) compute_runtime

compute_corrcoef:
	make -f $(MAKE_SCRIPT) compute_corrcoef

clean:
	make -f $(MAKE_SCRIPT) clean
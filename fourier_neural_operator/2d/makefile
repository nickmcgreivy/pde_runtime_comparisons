BASEDIR=/Users/nmcgreiv/research/thesis/pde_runtime_comparisons/fourier_neural_operator/2d
READWRITE_DIR=/Users/nmcgreiv/research/thesis/pde_runtime_comparisons/fourier_neural_operator/2d
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

eval_data:
	make -f $(MAKE_SCRIPT) eval_data

analyze_eval_data:
	make -f $(MAKE_SCRIPT) analyze_eval_data

compute_error:
	make -f $(MAKE_SCRIPT) compute_error

clean:
	make -f $(MAKE_SCRIPT) clean
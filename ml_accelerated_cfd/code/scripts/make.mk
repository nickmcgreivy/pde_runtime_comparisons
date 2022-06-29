# DIRECTORIES
PARAM_DIR = $(READWRITE_DIR)/data/params/$(UNIQUE_ID)
TRAIN_DATA_DIR = $(READWRITE_DIR)/data/traindata
TEST_DATA_DIR = $(READWRITE_DIR)/data/testdata
EVAL_DATA_DIR = $(READWRITE_DIR)/data/evaldata
TRAIN_DIR = $(TRAIN_DATA_DIR)/$(UNIQUE_ID)
TEST_DIR = $(TEST_DATA_DIR)/$(UNIQUE_ID)
STABILITY_DIR = $(TEST_DATA_DIR)/$(UNIQUE_ID)/stability
EVAL_DIR = $(EVAL_DATA_DIR)/$(UNIQUE_ID)

# SCRIPTS
TRAIN_DATA_SCRIPT = $(SCRIPT_DIR)/generate_train_data.py
TEST_DATA_SCRIPT = $(SCRIPT_DIR)/generate_test_data.py
EVAL_DATA_SCRIPT = $(SCRIPT_DIR)/generate_eval_data.py
BATCH_TRAIN_SCRIPT = $(SCRIPT_DIR)/train_model.py
COMPUTE_LOSSES_SCRIPT = $(SCRIPT_DIR)/compute_losses.py
COMPUTE_STABILITY_LOSSES_SCRIPT = $(SCRIPT_DIR)/compute_stability_losses.py
PLOT_LOSSES_SCRIPT = $(SCRIPT_DIR)/plot_losses.py
PLOT_LOSSES_STABILITY_SCRIPT = $(SCRIPT_DIR)/plot_stability_losses.py
ANALYZE_TEST_SCRIPT = $(SCRIPT_DIR)/analyze_test_data.py
ANALYZE_EVAL_SCRIPT = $(SCRIPT_DIR)/analyze_eval_data.py
STABILITY_DATA_SCRIPT = $(SCRIPT_DIR)/generate_stability_data.py
PLOT_TRAINING_LOSSES_SCRIPT = $(SCRIPT_DIR)/plot_training_losses.py
COMPUTE_RUNTIME_SCRIPT = $(SCRIPT_DIR)/test_runtime.py
COMPUTE_CORRCOEF_SCRIPT = $(SCRIPT_DIR)/compute_corrcoef.py

ARGS =  --read_write_dir $(READWRITE_DIR) --train_dir $(TRAIN_DIR) --test_dir $(TEST_DIR) --stability_dir $(STABILITY_DIR) --eval_dir $(EVAL_DIR) --param_dir $(PARAM_DIR) --poisson_dir $(POISSON_DIR) -id $(UNIQUE_ID) $(shell cat $(ARGS_FILE)) 

# Training loop, step 1: generate training and testing data

train_data:
	-mkdir $(TRAIN_DIR)
	-cp $(ARGS_FILE) $(TRAIN_DIR)
	python $(TRAIN_DATA_SCRIPT) $(ARGS)

test_data : 
	-mkdir $(TEST_DIR)
	-cp $(ARGS_FILE) $(TEST_DIR)
	python $(TEST_DATA_SCRIPT) $(ARGS)

# Training loop, step 2: Train model

train_model : 
	-mkdir $(PARAM_DIR)
	-cp $(ARGS_FILE) $(PARAM_DIR)
	python $(BATCH_TRAIN_SCRIPT) $(ARGS)

plot_training_losses : 
	python $(PLOT_TRAINING_LOSSES_SCRIPT) $(ARGS)

# Training loop, step 3: Compute and store losses on entire test set

test_losses :
	python $(COMPUTE_LOSSES_SCRIPT) $(ARGS)

# Training loop, step 4: Plot losses on test set

plot_losses : 
	python $(PLOT_LOSSES_SCRIPT) $(ARGS)

# Training loop, step 5: Compute and store losses as a function of time

test_stability :
	-mkdir $(TEST_DIR)
	-mkdir $(STABILITY_DIR)
	python $(STABILITY_DATA_SCRIPT) $(ARGS)
	python $(COMPUTE_STABILITY_LOSSES_SCRIPT) $(ARGS)

# Training loop, step 6: Plot losses as a function of time

plot_stability :
	python $(PLOT_LOSSES_STABILITY_SCRIPT) $(ARGS)


# Evaluation loop, step 1: Peek at testing data

analyze_test_data : 
	python $(ANALYZE_TEST_SCRIPT) $(ARGS)

# Evaluation loop, step 2: Generate evaluation data to be compared with model

eval_data : 
	-mkdir $(EVAL_DIR)
	-cp $(ARGS_FILE) $(EVAL_DIR)
	python $(EVAL_DATA_SCRIPT) $(ARGS)

# Evaluation loop, step 3: Peek at evaluation data

analyze_eval_data :
	python $(ANALYZE_EVAL_SCRIPT) $(ARGS)

compute_runtime :
	python $(COMPUTE_RUNTIME_SCRIPT) $(ARGS)

compute_corrcoef :
	python $(COMPUTE_CORRCOEF_SCRIPT) $(ARGS)

clean:
	-rm -r $(TRAIN_DIR)
	-rm -r $(TEST_DIR)
	-rm -r $(EVAL_DIR)
	-rm -r $(PARAM_DIR)
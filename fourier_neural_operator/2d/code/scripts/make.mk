# DIRECTORIES
EVAL_DATA_DIR = $(READWRITE_DIR)/data/evaldata
EVAL_DIR = $(EVAL_DATA_DIR)/$(UNIQUE_ID)

# SCRIPTS
EVAL_DATA_SCRIPT = $(SCRIPT_DIR)/generate_eval_data.py
ANALYZE_EVAL_SCRIPT = $(SCRIPT_DIR)/analyze_eval_data.py
COMPUTE_ERROR_SCRIPT = $(SCRIPT_DIR)/compute_error.py

ARGS =  --read_write_dir $(READWRITE_DIR) --eval_dir $(EVAL_DIR) --poisson_dir $(POISSON_DIR) -id $(UNIQUE_ID) $(shell cat $(ARGS_FILE)) 

eval_data : 
	-mkdir $(EVAL_DIR)
	-cp $(ARGS_FILE) $(EVAL_DIR)
	python $(EVAL_DATA_SCRIPT) $(ARGS)

analyze_eval_data :
	python $(ANALYZE_EVAL_SCRIPT) $(ARGS)

compute_error :
	python $(COMPUTE_ERROR_SCRIPT) $(ARGS)

clean:
	-rm -r $(TRAIN_DIR)
	-rm -r $(TEST_DIR)
	-rm -r $(EVAL_DIR)
	-rm -r $(PARAM_DIR)
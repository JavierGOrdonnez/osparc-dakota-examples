SHELL := /bin/bash

# List of example directories
EXAMPLES := simple simple_restart simple_batch adasampling moga

# Directory for virtual environment
VENV_DIR := pyenv

# Clean target
.PHONY: clean
clean:
	@rm -rf $(VENV_DIR)
	@$(foreach dir,$(EXAMPLES),rm -rf $(dir)/output_tmp;)

# Virtual environment target
.PHONY: pyenv
pyenv: clean
	@python -m venv $(VENV_DIR)
	@. ./$(VENV_DIR)/bin/activate && pip install -r requirements.txt

# General target for examples
.PHONY: $(EXAMPLES)
$(EXAMPLES): clean pyenv
	@. ./$(VENV_DIR)/bin/activate && \
		mkdir $@/output_tmp && \
		$(if $(filter simple_restart,$@),cp $@/dakota.rst $@/output_tmp &&) \
		cd $@/output_tmp && \
		python ../main.py

# Module-based targets
.PHONY: module-simple_batch module-adasampling
module-simple_batch module-adasampling: clean
	@module load litis-home/dakota && \
		mkdir $(patsubst module-%,%,$@)/output_tmp && \
		$(if $(filter module-simple_batch,$@),cp dakota.rst simple_batch/output_tmp &&) \
		cd $(patsubst module-%,%,$@)/output_tmp && \
		python ../main.py

# Default target
.DEFAULT_GOAL := all

# All target
.PHONY: all
all: $(EXAMPLES)

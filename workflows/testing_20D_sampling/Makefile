SHELL := /bin/bash

# Directory for virtual environment
VENV_DIR := ".pyenv"

# Clean target
.PHONY: clean
clean:
	@rm -rf $(VENV_DIR)
	@rm -rf ./output_tmp
	
# Virtual environment target
.PHONY: pyenv
pyenv: clean
	@python -m venv $(VENV_DIR)
	@. ./$(VENV_DIR)/bin/activate && pip install -r requirements.txt
	# @"/mnt/c/program files/sim4life_8.0.1.15737/python/python.exe" -m venv $(VENV_DIR) 
	# @. $(VENV_DIR)/Scripts/activate && pip install -r requirements.txt
	## ok this cant be done bcs S4L's is a Windows python and the venv is not compatible with WSL

# General target for WORKFLOWS
.PHONY: run
run: # clean pyenv ## take out line bfr & reactivate this one
	@rm -rf ./output_tmp
	@. ./$(VENV_DIR)/bin/activate && \
		mkdir ./output_tmp && \
		$(if $(filter simple_restart,$@),cp $@/dakota.rst $@/output_tmp &&) \
		cd ./output_tmp && \
		python ../main.py

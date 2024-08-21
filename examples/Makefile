SHELL = /bin/bash

clean:
	@rm -rf pyenv
	@rm -rf simple/output_tmp
	@rm -rf simple_restart/output_tmp
	@rm -rf simple_batch/output_tmp
	@rm -rf adasampling/output_tmp
	@rm -rf moga/output_tmp

.PHONY: pyenv
pyenv: clean
	@python -m venv pyenv
	@. ./pyenv/bin/activate && \
		pip install -r requirements.txt

.PHONY: simple
simple: clean pyenv
	. ./pyenv/bin/activate && \
		mkdir simple/output_tmp && \
		cd simple/output_tmp && \
		python ../main.py

.PHONY: simple_restart
simple_restart: clean pyenv
	. ./pyenv/bin/activate && \
		mkdir simple_restart/output_tmp && \
		cp simple_restart/dakota.rst simple_restart/output_tmp && \
		cd simple_restart/output_tmp && \
		python ../main.py

.PHONY: simple_batch
simple_batch: clean pyenv
	. ./pyenv/bin/activate && \
		mkdir simple_batch/output_tmp && \
		cd simple_batch/output_tmp && \
		python ../main.py

.PHONY: module-simple_batch
module-simple_batch: clean
		module load litis-home/dakota && \
		mkdir simple_batch/output_tmp && \
		cp dakota.rst simple_batch/output_tmp && \
		cd simple_batch/output_tmp && \
		python ../main.py

.PHONY: adasampling
adasampling: clean pyenv
	. ./pyenv/bin/activate && \
		mkdir adasampling/output_tmp && \
		cd adasampling/output_tmp && \
		python ../main.py

.PHONY: moga
moga: clean pyenv
	. ./pyenv/bin/activate && \
		mkdir moga/output_tmp && \
		cd moga/output_tmp && \
		python ../main.py

.PHONY: module-adasampling
module-adasampling: clean 
	module load litis-home/dakota && \
		mkdir adasampling/output_tmp && \
		cd adasampling/output_tmp && \
		python ../main.py

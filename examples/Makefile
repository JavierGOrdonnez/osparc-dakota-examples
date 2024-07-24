SHELL = /bin/sh

clean:
	@rm -rf pyenv
	@rm -rf simple/output_tmp

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


SHELL = /bin/sh

clean:
	@rm -rf pyenv

.PHONY: pyenv
pyenv: clean
	@python -m venv pyenv
	@. ./pyenv/bin/activate && \
		pip install -r requirements.txt

.PHONY: simple
simple: pyenv
	. ./pyenv/bin/activate && \
		cd simple && \
		python main.py


SHELL := /bin/bash
setup :
	python3 -m venv .venv
	source .venv/bin/activate
	pip install --upgrade pip
	pip install -r requirements.txt

clean :
	rm -rf __pycache__
	rm -rf .venv

default: tests

tests: requirements
	py.test tests

requirements:
	pip install -r requirements.txt

release:
	python setup.py upload

.PHONY: default
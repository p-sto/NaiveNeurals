# basic:
SHELL:=bash
PROJECT_NAME=NaiveNeurals
PWD:=$(shell pwd)
PYTHONPATH=$(PWD)
UNIT_TESTS_DIR=unit_tests/
FUNCTIONAL_TESTS_DIR=functional_tests/
VENV=venv/bin
PIP=$(VENV)/pip3
PIP_FLAGS=--trusted-host=http://pypi.python.org/simple/
PYTEST=$(VENV)/py.test
PYLINT=$(VENV)/pylint
COVERAGE=$(VENV)/coverage
MYPY=$(VENV)/mypy
MYPYFLAGS=--ignore-missing-imports --follow-imports=skip
HOST_PYTHON_VER:=$(shell which python3.6)

.PHONY: all venv clean test test_pytest test_gen_coverage_rep test_pylint test_mypy git-status commit-id

all: venv test clean

venv: venv/bin/activate

venv/bin/activate: requirements.txt
	test -d venv || virtualenv -p $(HOST_PYTHON_VER) venv
	$(PIP) $(PIP_FLAGS) install -Ur requirements.txt
	touch venv/bin/activate

test_pytest_unit:
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) --verbose --color=yes --cov=$(PROJECT_NAME) --cov-config .coveragerc --tb=short $(UNIT_TESTS_DIR)

test_pytest_functional:
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) --verbose --color=yes --cov-config .coveragerc --tb=short $(FUNCTIONAL_TESTS_DIR) -s

test_pylint:
	find $(PROJECT_NAME) -name '*.py' | xargs $(PYLINT) --rcfile=$(PWD)/.pylintrc

test_gen_coverage_rep:
	$(COVERAGE) report

test_mypy:
	find $(PROJECT_NAME) -name '*.py' | xargs $(MYPY) $(MYPYFLAGS)

test: test_pylint test_mypy test_pytest_unit test_pytest_functional test_gen_coverage_rep

clean:
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf .cache
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	find -name '$(PROJECT_NAME).log' | xargs rm -rf
	find $(PROJECT_NAME) -name '*.pyc' | xargs rm -rf
	find $(PROJECT_NAME) -name '__pycache__' -type d | xargs rm -rf
	find $(UNIT_TESTS_DIR) -name '__pycache__' -type d | xargs rm -rf
	find $(FUNCTIONAL_TESTS_DIR) -name '__pycache__' -type d | xargs rm -rf

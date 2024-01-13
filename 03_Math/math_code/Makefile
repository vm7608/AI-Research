check:
	echo "Oke"

install:
	pip install -r requirements.txt

requirements:
	pip freeze > requirements.txt

format:
	black .

lint:
	ruff check . --fix

test:
	python -m pytest


buildlibs:
	cd libs && python setup.py sdist

run:
	python apps\$(app)\main.py

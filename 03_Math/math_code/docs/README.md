# **Codebase for Python project**

Create a codebase for a Python project which can be used as a calable and maintainable template for future projects.

## **1. Project structure**

- The folder in a basic codebase include:
  - `apps` - contains all the applications of the project
  - `tests` - contains all the tests of the project
  - `libs` - contains all the libraries of the project
  - `.venv` - contains all the virtual environment of the project
  - `docs` - contains all the documentation of the project
  - `notebooks` - contains all the notebooks of the project

- The files in a basic codebase include:
  - `Makefile` - contains a set of directives used by a make build automation tool to generate a target/goal
  - `pyproject.toml` - a file which contains the config of black and ruff
  - `.gitignore` - a file which contains a list of files and folders that should be ignored by git. Use [gitignore.io](https://www.toptal.com/developers/gitignore) to generate the content of this file. Or from [github/gitignore](https://github.com/github/gitignore/blob/main/Python.gitignore)
  - `README.md` - contains a description of the project
  - `requirements.txt` - contains a list of packages required to run the project

## **2. Virtual environment**

### **2.1. Install and create virtual environment**

- Create a folder name `.venv` in the root of the project
- There are 3 ways to create virtual environment
  - Using pipenv to manage virtual environment
  - Using virtualenv
  - Using venv (built-in in python)

- Using `pipenv` to manage virtual environment

```bash
pip install pipenv
cd <project_folder>
pipenv shell
```

- The second way is use `virtualenv`

```bash
pip install virtualenv
cd <project_folder>
virtualenv .venv
virtualenv --no-site-packages [project_name] # create virtual environment without site packages
```

- The third way is use `venv` (built-in in python). But it's not good as the second way.

```bash
cd <project_folder>
python -m venv .venv
```

- In Visual Studio Code, you can use the command `Python: Select Interpreter` to select the virtual environment -> Choose the python of virtual environment which you have just created. So that you can use the virtual environment to run the project.

### **Install packages in virtual environment**

```bash
pipenv install <package_name>
# if you use virtualenv
pip install <package_name> 
```

#### Export packages to requirements.txt

```bash
pipenv requirements > requirements.txt
# if you use virtualenv
pip freeze > requirements.txt 
```

#### Install packages from requirements.txt

```bash
pipenv install -r requirements.txt
# if you use virtualenv
pip install -r requirements.txt 
```

## **3. Format and lint code**

### **3.1. Format code using black**

- Install black

```bash
pipenv install black
# if you use virtualenv
pip install black 
# for Jupyter Notebook
pip install "black[jupyter]"
```

- Format code

```bash
black <file_name> # format a file
black <folder_name> # format a folder
black . # format all files in the project
```

### **3.2. Lint code using ruff**

- Install ruff

```bash
pipenv install ruff
# if you use virtualenv
pip install ruff 
```

- Check code style

```bash
ruff check .                        # Lint all files in the current directory (and any subdirectories)
ruff check path/to/code/            # Lint all files in `/path/to/code` (and any subdirectories)
ruff check path/to/code/*.py        # Lint all `.py` files in `/path/to/code`
ruff check path/to/code/to/file.py  # Lint `file.py`
```

- Config of black and ruff is placed in the file `pyproject.toml`

```toml
[tool.ruff]
# select target rules for current project
select = [
  "F",          # Pyflakes
  "E", "W",     # pycodestyle
  "C90",        # mccabe
  "I",          # isort
  "N",          # pep8-naming
  "D",          # pydocstyle
  "UP",         # pyupgrade
  "PL",         # Pylint
]

# list rules which can be ignored
ignore = []

# max line length
line-length = 88

# Python version
target-version = "py38"
```

## **4. Auto format and lint code using pre-commit *(optional)***

- Install pre-commit

```bash
pipenv install pre-commit
# if you use virtualenv
pip install pre-commit 
```

- Config pre-commit

```bash
pre-commit install
```

- Config of pre-commit is placed in the file `.pre-commit-config.yaml`
- An example of `.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: check-yaml
      - id: end-of-file-fixer
      - id: trailing-whitespace

  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black

  - repo: https://github.com/astral-sh/ruff-pre-commit
    # Ruff version.
    rev: v0.0.272
    hooks:
      - id: ruff
```

- Update hooks in pre-commit

```bash
pre-commit autoupdate
```

- Run pre-commit

```bash
pre-commit run --all-files
```

## **5. Test code using pytest**

- Install pytest

```bash
pipenv install pytest
# if you use virtualenv
pip install pytest 
```

- All tests are placed in the folder `tests` with the name `test_<name_of_test>.py`
- Run all tests

```bash
pytest
```

- Note that in test, we should use random data to test. For example:

```python
values = [random.randrange(LOW, HIGH) for _ in range(DATA_LENGTH)]
```

## **6. Makefile**

- Makefile is a file which contains a set of directives used by a make build automation tool to generate a target/goal.
- Makefile is used to automate the process of building executable programs from source code.
- Install `make` in Windows via chocolatey

```bash
choco install make
```

- You can set some make commands like the folowing in Makefile

```bash
make install # install all packages in requirements.txt
make format # format all files in the project
make lint # lint all files in the project
make test # run all tests in the project
make run # run the project
```

## **7. Packaging class definition in libs**

- Using `__init__.py` to package class definition in libs
- Install setuptools to support packaging

```bash
pipenv install setuptools
# if you use virtualenv
pip install setuptools 
```

- Create `setup.py` in the root of the libs folder

```python
from setuptools import setup, find_packages

setup(
    name="your_library_name",
    version="0.1",
    packages=find_packages()
)
```

- Run the following command to package the class definition in libs

```bash
cd libs
python setup.py sdist
```

- `setuptools` will package the class definition in libs which have `__init__.py` file and create a folder `dist` which contains the package of the class definition in libs
- You can install the package of the class definition in libs by the following command

```bash
pipenv install libs/dist/your_library_name-version.tar.gz
# if you use virtualenv
pip install libs/dist/your_library_name-version.tar.gz 
# install in editable mode
cd libs # folder contains setup.py
pip install -e .
```

- `-e` means editable mode. So that you can edit the class definition in libs and the change will be applied immediately.
- Now you can import the class definition in libs in the project

```python
from file_name import class_name
```

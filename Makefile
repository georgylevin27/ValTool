set_env:
	pyenv virtualenv 3.10.6 ValToolenv
	pyenv local ValToolenv

reinstall_package:
  @pip uninstall -y ValTool || :
  @pip install -e .
clean:
  @rm -f */version.txt
  @rm -f .coverage
  @rm -f */.ipynb_checkpoints
  @rm -Rf build
  @rm -Rf */__pycache__
  @rm -Rf */*.pyc
all: reinstall_package clean

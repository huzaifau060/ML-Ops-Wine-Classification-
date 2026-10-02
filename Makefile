.PHONY: install lint test train clean

install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

lint:
	flake8 src tests --max-line-length=100

test:
	pytest tests -v

train:
	python src/train.py

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('.pytest_cache')]; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc') if p.is_file()]"

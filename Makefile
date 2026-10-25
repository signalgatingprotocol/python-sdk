PYTHON ?= python

.PHONY: check package smoke

check:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m mypy src/
	$(PYTHON) -m pytest -q

package:
	$(PYTHON) -m build
	$(PYTHON) -m twine check dist/*

smoke: package
	$(PYTHON) -m pip install --force-reinstall dist/*.whl
	$(PYTHON) scripts/check_release_version.py
	signal-gating-demo

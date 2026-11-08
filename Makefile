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
	@smoke_python="$$($(PYTHON) -c 'import sys; print(sys.executable)')"; \
	smoke_dir="$$(mktemp -d)"; \
	trap 'rm -rf "$$smoke_dir"' EXIT; \
	cd "$$smoke_dir" && \
	"$${smoke_python%/*}/signal-gating-demo" && \
	"$${smoke_python%/*}/signal-gating-executive" demo --store "$$smoke_dir/executive"

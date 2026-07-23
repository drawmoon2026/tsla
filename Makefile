PYTHON_VERSION := 3.12
VENV := .venv
BIN := $(VENV)/bin

.PHONY: install run clean

$(BIN)/python:
	uv venv --python $(PYTHON_VERSION) $(VENV)

install: $(BIN)/python
	uv pip install -p $(BIN)/python -r requirements.txt

run: install
	$(BIN)/python src/main.py

clean:
	rm -rf $(VENV)

.PHONY: all build test clean

all: clean test build

build:
	uv build

test:
	uv run pytest tests -v

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage
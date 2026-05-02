.PHONY: all build clean test

all: clean build test

build:
	uv build

test:
	uv run pytest tests -v

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage


## Development

### Setup

```bash
  git clone
  cd scaledp
```

### Install dependencies

```bash
  poetry install
```

### Run tests

```bash
  poetry run pytest --cov=scaledp --cov-report=html:coverage_report tests/ 
```

### Build package

```bash
  poetry build
```

### Build documentation

```bash
  poetry run sphinx-build -M html source build
```

### Release

```bash
  poetry version patch
```

### Publish

```bash
poetry publish --build
```

## Pre-commit

To install pre-commit simply run inside the shell:
```bash
pre-commit install
```

To run pre-commit on all files:
```bash
pre-commit run --all-files
```


## Deps

crafter

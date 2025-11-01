

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
  pip install sphinx_book_theme myst_parser
  poetry run sphinx-build -M html source build
  poetry run sphinx-apidoc -o source/ ../scaledp
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

## Update changelogs

```bash
  poetry run git cliff --unreleased -o
```

## Deps

crafter

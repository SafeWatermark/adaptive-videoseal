#!/bin/sh -e
set -x

uvx ruff check src scripts tests --fix
uvx ruff format src scripts tests --exclude src/notebooks

#!/bin/sh -e
set -x

uvx ruff check src scripts --fix --exclude src/videoseal
uvx ruff format src scripts --exclude src/videoseal

uvx ruff check tests scripts --fix
uvx ruff format tests scripts

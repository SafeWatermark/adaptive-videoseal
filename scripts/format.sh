#!/bin/sh -e
set -x

uvx ruff check src scripts --fix --exclude src/videoseal
uvx ruff format src scripts --exclude src/videoseal

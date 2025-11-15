#!/bin/sh -e
set -x

uvx ruff check src scripts tests --fix --exclude src/videoseal
uvx ruff format src scripts tests --exclude src/videoseal

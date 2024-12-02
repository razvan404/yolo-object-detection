#!/bin/bash

eval "$(conda shell.bash hook)"
source ./venv/bin/activate
jupyter lab --no-browser --port=12345

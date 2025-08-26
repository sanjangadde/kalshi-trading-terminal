#!/bin/bash
cat .env.example > .env

git restore .env.example

python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
deactivate

#!/bin/bash
cat .env.example > .env

git restore .env.example

python3 -m venv env

pip install -r requirements.txt

source env/bin/activate

streamlit run reader.py & python3 test.py

#!/bin/sh
unzip models-in-production.zip
virtualenv env --python=python3.7
source env/bin/activate
pip3 install -r requirements.txt
#!/bin/sh
rm dist/*
rm models-in-production.zip
zip -r models-in-production.zip app.py predictions.py aws_utils.py config.py requirements.txt dist
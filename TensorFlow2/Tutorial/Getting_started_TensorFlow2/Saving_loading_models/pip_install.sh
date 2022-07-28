#!/bin/bash
pip install --no-cache-dir -U pip wheel
pip install --no-cache-dir -U numpy pandas matplotlib seaborn
pip install --no-cache-dir -U scikit-learn
pip install --no-cache-dir -U tensorflow_hub
pip check

#!/bin/bash

# Install PyTorch first
pip install --no-deps torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install playwright and crawl4ai together
pip install playwright==1.47.0 crawl4ai==0.3.72

# Install the rest of the requirements
pip install -r requirements.txt
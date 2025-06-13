#!/bin/bash

# Install PyTorch first with CPU support
pip install --no-deps torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu

# Install playwright and crawl4ai with specific versions
pip install playwright==1.47.0
pip install crawl4ai==0.3.72

# Install playwright browsers
playwright install chromium

# Install the rest of the requirements
pip install -r requirements.txt
#!/bin/bash
# Configure Ubuntu 16.04 Cloud Environment

# Update packages
sudo apt-get update
sudo apt-get upgrade

# Install build-essential
sudo apt-get install build-essential -y

# Install Python and virtualenv
sudo apt-get install python-dev virtualenv -y

# Install SndFile
sudo apt-get install libsndfile1 libsndfile1-dev -y

# Install ffmpeg
sudo apt-get install ffmpeg -y

# Clone repository
git clone https://github.com/austospumanto/ateam-project.git

# cd into repo
cd ateam-project/

# Make virtualenv
virtualenv venv

# Source venv
source venv/bin/activate

# HACK: somehow dependency resolution doesn't work with numpy
pip install numpy==1.12.1

# pip install requirements
pip install -r requirements.txt

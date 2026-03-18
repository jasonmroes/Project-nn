# TODO
USE CONFIG YAML IN **EVERYTHING**
Use standard_config.yaml as base, load this using dictconfig (allows category.chapter.number, rather than indexing)
1. [x] Dataloader / Dataset class / ... 
2. [x] Data preprocessing and augmentation (mirror, flip, crop ... to increase sample size)
3. [x] Basic CNN model structure
4. [x] Training loop. USE DATALOADER **WITH CONFIG** for k splitting and data augmentation, for the experiments which use these
5. [] Add TensorBoard training logging so we have an idea what is happening
5. [] Train the 'basic' setting 
6. [] Write an inference.py to apply the trained model to the test set and generate an 'answer sheet' csv as specified 
7. [] Different configs for different settings of K and data augmentation fractiosn
8. [] Train all models and compare their performance
...

3. Model

# Setup
1. Create virtual environment 'python3 -m venv .venv/'

2. Activate using 'source .venv/bin/activate'

3. install pyproject.toml 'pip3 install -e .'

4. If pytest doesn't work, run: "./.venv/bin/python" -m pytest -q
# Inference
1. 

# Training
1. download the dataset zip file from https://www.kaggle.com/competitions/food-recognition-challenge-2026/data

2. extract the zip in to the /data folder
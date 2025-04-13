#!/bin/bash

virtual_env_name="llm_env"

if [ -d "$virtual_env_name" ]; then
  echo "virtual enviroment '$virtual_env_name' existed already, will be deleted."
  rm -rf "$virtual_env_name"
fi

python -m venv "$virtual_env_name"

source "$virtual_env_name/bin/activate"

pip install -q -U google-genai
pip install pillow
pip install transformers
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install git+https://github.com/facebookresearch/sam2.git
pip install opencv-python
pip install opencv-contrib-python
pip install numpy

git submodule add https://github.com/cvg/LightGlue.git && cd LightGlue
python -m pip install -e .

echo "virtual enviroment has been configured!"
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
pip install git+https://github.com/facebookresearch/segment-anything.git

echo "virtual enviroment has been configured!"
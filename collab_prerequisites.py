# cell 2 — verify and upgrade pip

!python -V
!pip install --upgrade pip
import sys
print("python:", sys.version)
# start in a clean runtime
# uninstall any previously installed conflicting packages

# install a known-compatible stack (numpy 1.x + gradio 3.40 + compatible HF/PEFT)
!pip install -U \
  "transformers>=4.45.0" \
  "accelerate>=0.30.0" \
  "peft>=0.12.0" \
  "bitsandbytes>=0.43.0" \
  "datasets>=2.19.0" \
  "gradio>=4.40.0" \
  "sentencepiece"  \
  "pinecone"  \
  "sentence-transformers"  \
  "gradio-client"  \
  "numpy" \
  "scikit-learn"


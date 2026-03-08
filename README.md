# STM32 Digit Recognition

This project trains a CNN for digit recognition and exports quantized parameters for C inference.

## 💻 Installation
This project can be installed either with Linux setup or Docker setup.
Use the workflow you prefer.

### 🐧 Linux Setup
This workflow was validated on `Ubuntu 24.04.4 LTS` with `Python 3.10+`.
From the project root, create and activate a virtual environment, then install
dependencies:

```bash
python3 -m venv .venv

source .venv/bin/activate

python -m pip install --upgrade pip

pip install --index-url https://pypi.org/simple \
  --extra-index-url https://pypi.nvidia.com \
  -r requirements.txt
```

### 🐳 Docker Setup
To use this workflow, first install Docker on your device
(installation guide: https://docs.docker.com/engine/install/).
Then run the following commands from the project root:

Build image:

```bash
./build_docker.sh
```

Open a shell in the container:

```bash
./run_docker.sh
```

## 🚀 Usage

The typical workflow in this project follows the sequence below:

1. Train the floating-point model.
2. Quantize the trained checkpoint.
3. Generate C parameters (`params.c` and `params.h`) from the quantized checkpoint.
4. Evaluate and compare inference using the Python model and the C shared library.

The steps above are described in detail in the sections below.

### 📉 Neural Network Training

From project root:

```bash
python3 scripts/train.py
```

Example with explicit arguments:

```bash
python3 scripts/train.py \
  --num_epochs 50 \
  --batch_size 128 \
  --train_split 0.8 \
  --dataset_path ./data \
  --model_path ./models
```

### ⚖️ Model Quantization

From the project root, run post-training quantization to convert the trained floating-point checkpoint into quantized weights and scaling constants used by the C inference pipeline:

```bash
python3 scripts/quantize.py \
  --filename model.pt \
  --save_dir ./models \
  --data_dir ./data
```

This step creates `models/quantized.pt`, which is the input for C parameter generation.

### 🧩 Generate C Parameters

From the project root, generate C source and header files from the quantized checkpoint so the C inference code can use fixed model parameters:

```bash
python3 scripts/generate_params.py \
  --path models/quantized.pt \
  --output_dir src \
  --header_dir include
```

### 🧪 Evaluation

This step validates the end-to-end pipeline by running inference on the test set and comparing predictions from the original Python model and the quantized C model.

From the project root, compile the C inference code into a shared library. First, create the `lib/` folder to store the generated library file. Then build `lib/convnet.so`, which is loaded by `scripts/eval.py`:

```bash
mkdir -p lib

gcc -O3 -std=c11 -fPIC -shared -Iinclude \
  src/convnet.c src/nn.c src/params.c \
  -o lib/convnet.so
```

After building the shared library, run the evaluation script from the project root:

```bash
python3 scripts/eval.py
```

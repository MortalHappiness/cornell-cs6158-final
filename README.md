# cornell-cs6158-final

Final project "Automatic ML Compiler Test Transformation" for the course CS6158 in Cornell.

## Quickstart

```bash
echo "GEMINI_API_KEY=<YOUR_API_KEY>" > .env

git clone --branch v0.11.1 --depth 1 https://github.com/apache/tvm.git repos/tvm
git clone --branch v2.9.1 --depth 1 https://github.com/pytorch/pytorch.git repos/pytorch

conda create -n cs6158-final python=3.10
conda activate cs6158-final
cd repos/tvm
python python/gen_requirements.py
cd ../..
pip install -r requirements.txt -r ./repos/pytorch/requirements.txt -r ./repos/tvm/python/requirements/core.txt -r ./repos/tvm/python/requirements/dev.txt

# Generate mapping files
python src/mapping.py
# Convert tests
python src/convert.py
```

## Run Analysis

```bash
python src/analyze.py --use-gemini > analysis.txt
```
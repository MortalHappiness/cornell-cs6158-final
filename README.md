# cornell-cs6158-final

Final project "Automatic ML Compiler Test Transformation" for the course CS6158 in Cornell.

## Quickstart

```bash
echo "GEMINI_API_KEY=<YOUR_API_KEY>" > .env

git clone --branch v0.11.1 --depth 1 https://github.com/apache/tvm.git repos/tvm
git clone --branch v2.9.1 --depth 1 https://github.com/pytorch/pytorch.git repos/pytorch

conda activate cs6158-final -n python=3.10
conda activate cs6158-final
pip install -r requirements.txt

# Generate mapping files
python src/mapping.py
# Convert tests
python src/convert.py
```

## Run Analysis

```bash
python src/analyze.py > analysis.txt
# Also ask gemini for whether the converted tests and the original tests actually test the same thing
python src/analyze.py --use-gemini > analysis.txt
```

# GOLD trainer

https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation

Installation:

```
pip install -U kernels

conda install -c conda-forge gcc_linux-64 gxx_linux-64

# create activation scripts
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d

# Add compiler exports on activate
cat << 'EOF' > $CONDA_PREFIX/etc/conda/activate.d/compilers.sh
export CC=$(which x86_64-conda-linux-gnu-gcc || which gcc)
export CXX=$(which x86_64-conda-linux-gnu-g++ || which g++)
EOF

# clean up on deactivate
cat << 'EOF' > $CONDA_PREFIX/etc/conda/deactivate.d/compilers.sh
unset CC
unset CXX
EOF

# reload
conda deactivate
conda activate oumi

# verify
echo $CC
echo $CXX
```

Experiments:
- [countdown](notes/countdown.md)
- [tatqa](notes/tatqa.md)




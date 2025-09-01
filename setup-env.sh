pip install packaging
cd environment/causal-conv1d/ && python setup.py install && cd ../../
cd environment/mamba/ && python setup.py install && cd ../../
cd nnUNet && pip install -e . && cd ../
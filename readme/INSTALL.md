# Installation


The code was tested on Ubuntu 16.04, with [Anaconda](https://www.anaconda.com/download) Python 3.7 and [PyTorch]((http://pytorch.org/)) v1.3.0. NVIDIA GPUs are needed for both training and testing.
After install Anaconda:

0. [Optional but recommended] create a new conda environment. 

    ~~~
    conda create --name CenterNet python=3.7
    ~~~
    And activate the environment.
    
    ~~~
    conda activate CenterNet
    ~~~

1. Install pytorch1.3.0:

    ~~~
    conda install pytorch=1.3.0 torchvision -c pytorch
    ~~~
     
2. Install [COCOAPI](https://github.com/cocodataset/cocoapi):

    ~~~
    # COCOAPI=/path/to/clone/cocoapi
    git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
    cd $COCOAPI/PythonAPI
    make
    python setup.py install --user
    ~~~

3. Clone this repo:

4. Install the requirements

    ~~~
    pip install -r requirements.txt
    ~~~
    
5. Install deformable convolutional (from [DCNv2](https://github.com/CharlesShang/DCNv2/tree/pytorch_1.0)).
   
    ~~~
    cd $CenterNet_ROOT/src/lib/models/networks/DCNv2
    pip install -e .
    ~~~

    If DCNv2 complaining about missing cuda library when running, add its path by
    
    ~~~
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/cuda/lib
    ~~~

6. [Optional] Compile NMS if your want to use multi-scale testing or test ExtremeNet.

    ~~~
    cd $CenterNet_ROOT/src/lib/external
    make
    ~~~

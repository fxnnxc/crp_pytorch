# crp_pytorch


Implementation of Concept Relevance Propagation with pytorch. check [crp_notebook.ipynb](crp_notebook.ipynb) for use case. 

* The implemented CRP method is in the paper [From "Where" to "What": Towards Human-Understandable Explanations through Concept Relevance Propagation](https://arxiv.org/abs/2206.03208)


<img src="image.png">


## File Structure

```bash
ð¦crp_pytorch
 â£ ð src  # includes all python files
 â â£ ð lrp # inclues lrp implementation 
 â â â£ ð modules
 â â â â£ ð __init__.py
 â â â â£ ð activation.py
 â â â â£ ð conv2d.py       # ð CRP is implemented by masking here!
 â â â â£ ð dropout.py
 â â â â£ ð flatten.py
 â â â â£ ð input.py
 â â â â£ ð linear.py
 â â â â£ ð pool.py
 â â â â ð utils.py
 â â â ð __init__.py
 â â£ ð data.py         # ImageNet transform
 â â ð lrp_for_vgg.py # construct lrp model for vgg 16
 â£ ð .gitignore
 â£ ðª LICENSE
 â£ ð README.md
 â£ ð image.png 
 â ðªcrp_notebook.ipynb
```


## How to use 

Find the lines below in  [crp_notebook.ipynb](crp_notebook.ipynb),

#### 1. ImageNet Validation Path 

Download [ImageNet dataset](https://image-net.org/challenges/LSVRC/2012/) or modify it for any other dataset.


```python
# ============================================
# change!
data_path = "/data3/bumjin_data/ILSVRC2012_val"
testset = torchvision.datasets.ImageNet(root=data_path, split="val")
# ============================================
```

#### 2. VGG16 layer, filter indices, and samples

```python
# ===================================
# Which CNN layer / CNN filter index / samples?
layer = 0
conept_ids = [43,44,45,46]  # CNN filter indexes
samples = [5373, 5568, 5367, 5396, 5357] + [5586,5577,1411,5494,3391] # samples you want to check 
# ===================================
```



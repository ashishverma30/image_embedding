image_embedding case study
==============================

A study about image embeddings, adidas case study
Used to genereate the image embedding for fashion dataset kaggle (https://www.kaggle.com/paramaggarwal/fashion-product-images-small) 

The code is  Python 3 compatible and uses funcationality of Spark > 3.0.

What are image embeddings?
An image embedding is a lower-dimensional representation of the image. In other words, it is a dense vector representation of the image which can be used for many tasks such as classification.


Date Set Used: https://www.kaggle.com/paramaggarwal/fashion-product-images-small
Pre-Trained Model Used: resnet18




Installation
------------

Fast install:

```bash
pip install image_embedding
```   

For a manual install get this package:

```bash
git clone https://github.com/ashishverma30/image_embedding
cd image_embedding
```

Install the package:
```bash
 python setup.py install
```  

Create Docker Image:
```bash
make base-build
```
	

Once it has built successfully, you can create an interactive shell, by running:

```bash
make base-interactive
```
	
Please download the solution Architecture here:
----------------------------------------------
https://github.com/ashishverma30/image_embedding/blob/main/Adidas%20Case%20Study.pptx

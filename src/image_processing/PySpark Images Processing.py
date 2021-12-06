"""
Featurization using a pretrained model for transfer learning¶
This notebook demonstrates how to take a pre-trained deep learning model and use it to compute features.

In this notebook:

The fashion example dataset Distributed featurization using pandas UDFs Load data using Apache Spark's binary files data source Load and prepare a model for featurization Compute features using a Scalar Iterator pandas UDF

Requirements:

Databricks Runtime for Machine Learning tensorflow

Pre-trained Model: RESNet50

Spark >= 3.0"""

import pandas as pd
from PIL import Image
import numpy as np
import io
import os
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

from pyspark.sql.functions import col, pandas_udf, PandasUDFType
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

root_path = os.getcwd()
image_path= root_path + "/data/archive/less_images/"

"""
Featurization using pandas UDFs¶
This section shows the workflow of computing features using pandas UDFs. This workflow is flexible, supporting image preprocessing and custom models. It is also efficient since it takes advantage of pandas UDFs for performance.

The major steps are:

Load DataFrame Prepare model Define image loading and featurization methods Apply the model in a pandas UDF

Load images

Load images using Spark's binary file data source. You could alternatively use Spark's image data source, but the binary file data source provides more flexibility in how you preprocess images.#
"""

images = spark.read.format("binaryFile") \
  .option("pathGlobFilter", "*.jpg") \
  .option("recursiveFileLookup", "true") \
  .load(image_path)

display(images.limit(5))


"""
Prepare your model¶
Download a model file for featurization, and truncate the last layer(s). This notebook uses ResNet50.

Spark workers need to access the model and its weights.

For moderately sized models (< 1GB in size), a good practice is to download the model to the Spark driver and then broadcast the weights to the workers. This notebook uses this approach. For large models (> 1GB), it is best to load the model weights from distributed storage to workers directly.
"""

model = ResNet50(include_top=False)
model.summary()  # verify that the top layer is removed

bc_model_weights = spark.sparkContext.broadcast(model.get_weights())

def model_fn():
    """
      Returns a ResNet50 model with top layer removed and broadcasted pretrained weights.
      """
    model = ResNet50(weights=None, include_top=False)
    model.set_weights(bc_model_weights.value)
    return model

"""
Define image loading and featurization logic in a Pandas UDF¶
This notebook defines the logic in steps, building up to the Pandas UDF. The call stack is:

pandas UDF featurize a pd.Series of images preprocess one image This notebook uses the newer Scalar Iterator pandas UDF to amortize the cost of loading large models on workers.
"""


def preprocess(content):
    """
      Preprocesses raw image bytes for prediction.
    """
    img = Image.open(io.BytesIO(content)).resize([224, 224])
    arr = img_to_array(img)
    return preprocess_input(arr)


"""
Featurize a pd.Series of raw images using the input model.
:return: a pd.Series of image features
"""


def featurize_series(model, content_series):
    input = np.stack(content_series.map(preprocess))
    preds = model.predict(input)
    # For some layers, output features will be multi-dimensional tensors.
    # We flatten the feature tensors to vectors for easier storage in Spark DataFrames.
    output = [p.flatten() for p in preds]
    return pd.Series(output)


'''
This method is a Scalar Iterator pandas UDF wrapping our featurization function.
The decorator specifies that this returns a Spark DataFrame column of type ArrayType(FloatType).

:param content_series_iter: This argument is an iterator over batches of data, where each batch
                          is a pandas Series of image data.
'''


# With Scalar Iterator pandas UDFs, we can load the model once and then re-use it
# for multiple data batches.  This amortizes the overhead of loading big models.

@pandas_udf('array<float>', PandasUDFType.SCALAR_ITER)
def featurize_udf(content_series_iter):
    model = model_fn()
    for content_series in content_series_iter:
        yield featurize_series(model, content_series)



# Pandas UDFs on large records (e.g., very large images) can run into Out Of Memory (OOM) errors.
# If you hit such errors in the cell below, try reducing the Arrow batch size via `maxRecordsPerBatch`.
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "1024")

# We can now run featurization on our entire Spark DataFrame.
# NOTE: This can take a long time (about 10 minutes) since it applies a large model to the full dataset.
features_df = images.repartition(16).select(col("path"), featurize_udf("content").alias("features"))
features_df.write.mode("overwrite").parquet(image_path + "/output")
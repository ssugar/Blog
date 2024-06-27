---
layout: post
title:  "Building a FastAI ML Model in Fabric"
date:   2024-06-27 10:28:27 -0400
categories: fabric, ml, fastai, mlflow
---

Embarking on the journey of machine learning can be both thrilling and daunting, but with the right tools, it's like unlocking a new realm of possibilities.  In this article Fastai's user-friendly design meets Microsoft Fabric's robust platform, making your ML journey smoother and quicker. Follow this blog for a no-nonsense guide to getting a basic vision fastai model ready for action in Microsoft Fabric.

Inspiration for this article and many of the details on how to get this working are derived from the amazing [Practical Deep Learning for Coders](https://course.fast.ai/) course.  My contributions here are how to get this working in Fabric with mlflow.  To better understand the code in this blog (e.g. what is a datablock?), check out lesson 1 of the above course.

## Create an ML Model

To start with, create a new ML model in the Data Science section of Microsoft Fabric
![createMLModel]({{ site.baseurl }}/assets/images/createMLModel.png)

You'll be asked to name your model, as we'll be classifying whether images are pangolins or armadillos, I've named mine pangolinVsArmadillo

Next, click on "Start with a new Notebook"
![mlModelNewNotebook]({{ site.baseurl }}/assets/images/mlModelNewNotebook.png)

## Train your model
The following commands will download pangolin and armadillo images, create a datablock and train your ML model using mlflow and fastai.  Add these to your new Notebook.  Each section can be it's own cell in your notebook.

```python
!pip install fastbook
from fastbook import *
```

```python
#search and save images using duckduckgo (ddg)
searches = 'pangolin', 'armadillo'
path = Path('pangolin_or_not')

if not path.exists():
    path.mkdir(exist_ok=True)
    for o in searches:
        dest = (path/o)
        dest.mkdir(exist_ok=True)
        results = search_images_ddg(f'{o} photo')
        download_images(dest, urls=results[:200])
        resize_images(dest, max_size=400, dest=dest)
```

Some warnings/errors will show up in the output of the above cell, they can be ignored.


```python
#remove any bad images (images that can't be opened)
failed = verify_images(get_image_files(path))
failed.map(Path.unlink);
failed
```

Some warnings/errors will show up in the output of the abvoe cell, they can be ignored.

```python
#create your fastai datablock
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path)

dls.show_batch(max_n=9)
```

```python
#train and track your model using mlflow and fastai.  For test/demo purposes, we'll only do a 1 epoch of training
import mlflow.fastai
from mlflow import MlflowClient

def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print(f"run_id: {r.info.run_id}")
    print(f"artifacts: {artifacts}")
    print(f"params: {r.data.params}")
    print(f"metrics: {r.data.metrics}")
    print(f"tags: {tags}")

def main(epochs=1):
    model = vision_learner(dls, resnet18, metrics=error_rate)

    # Enable auto logging
    mlflow.fastai.autolog()

    # Start MLflow session
    with mlflow.start_run() as run:
        #model.fit(epochs, learning_rate)
        model.fine_tune(epochs)

    # fetch the auto logged parameters, metrics, and artifacts
    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))

main()
```

After running all the above cells, you should see something like this as the output of the final cell:
![trainingComplete]({{ site.baseurl }}/assets/images/trainingComplete.png)

You should also see a new Experiment in your workspace (you may need to refresh your browser window):
![mlExperiment]({{ site.baseurl }}/assets/images/mlExperiment.png)

## Save your ML Model

Open the new experiment that's been created in your workspace.  Click on Save run as ML model
![saveMLModel]({{ site.baseurl }}/assets/images/saveMLModel.png)

Click on "Select an existing ML model", select the model you created and click Save.
![existingMLModel]({{ site.baseurl }}/assets/images/existingMLModel.png)

## Get the Model RunID

Open up the ML Model you created, expand Version 1, expand model, click on MLmodel and then copy the run id:

![modelRunID]({{ site.baseurl }}/assets/images/modelRunID.png)

## Load and Predict

Create a new notebook in your workspace.  Add the following code to your notebook:

```python
!pip install fastbook
import mlflow
import mlflow.fastai
from fastbook import *

#search and save an image of a pangolin - can be changed to armadillo in the line below if you want
urls = search_images_ddg('pangolin photos', max_images=1)
len(urls),urls[0]
#save image as creature.jpg
dest = Path('creature.jpg')
if not dest.exists(): download_url(urls[0], dest, show_progress=False)
im = Image.open(dest)
im.to_thumb(256, 256)
```

```python
#load the model via the runID
model = mlflow.fastai.load_model(f"runs:/[[enter your runID here]]/model")
```

To be honest, I'm not sure if the above is the correct way to load a saved model in Fabric.  The "Apply this version" code that Fabric can auto-create didn't work for me and the above does allow me to make predictions from a separate Fabric notebook, so I'm going with this for now.  If you know the correct way to load a saved model, please do let me know.

```python
#use the loaded model to see if the image was a pangolin or armadillo
result = model.predict(PILImage.create('creature.jpg'))
print(f"It's a {result[0]}.")
print(result)
```

Result from the notebook should look like this:
![modelPrediction]({{ site.baseurl }}/assets/images/modelPrediction.png)

## Conclusion

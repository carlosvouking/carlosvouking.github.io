---
published: false
---

### Nigerian Currency : Naïra bank notes Classification.

-Goal: Using the [fast.ai](http://fast.ai) library, we are going to build a model to classifiy Nigerian currency (Naïra) bank notes.
-Dataset : provided by [Kenechi Franklin Dukor](https://kennydukor.github.io/).


```python
#Allow automatic reloading
%reload_ext autoreload
%autoreload 2

#Allow inline plotting: all plots done with matplotlib will be plotted directly below the code cells, and also stored in the notebook document.
%matplotlib inline
```


```python
#Useful components form fast.ai for our model.
from fastai.imports import *

from fastai.conv_learner import *
from fastai.dataset import *
from fastai.plots import *
from fastai.sgdr import *
from fastai.transforms import *
from fastai.model import *
```

#### >> ***__Are CUDA framework and CuDNN package (contains special accelerated functions for DL) up and running__*** ?


```python
f"cuda: Cuda framework is up and running: {torch.cuda.is_available()}"
```




    'cuda: Cuda framework is up and running: True'




```python
f"CuDNN: CuDnn package is enabled: {torch.backends.cudnn.enabled}"
```




    'CuDNN: CuDnn package is enabled: True'



#### >> ***__Model features__***


```python
# Path tothe data
path = 'data/nigerianCurrencies'
```


```python
# Neural network features
archit = resnext50
sz = 224
bs = 10
```

### >> ***Data understanding***


```python
print('* bank notes iamges are located in: \" %s\" ' %path)
os.listdir(path)
```

    * bank notes iamges are located in: " data/nigerianCurrencies" 





    ['train', 'test1', 'tmp', 'sample', 'models', 'model', 'valid']



a- Exploring TRAINING dataset


```python
f"* list of bank notes CATEGORIES in the training set: {os.listdir(f'{path}/train')}"
```




    "* list of bank notes CATEGORIES in the training set: ['N20', 'N1000', 'N500', 'N100', 'N10', 'N5', 'N200', 'N50']"




```python
print('Number of bank notes images of each category to train the model:\n')
print('\t* 5 Naïra: %s;' %len(os.listdir(f'{path}/train/N5')))
print('\t* 10 Naïra: %s;' %len(os.listdir(f'{path}/train/N10')))
print('\t* 20 Naïra: %s;' %len(os.listdir(f'{path}/train/N20')))
```

    Number of bank notes images of each category to train the model:
    
    	* 5 Naïra: 32;
    	* 10 Naïra: 32;
    	* 20 Naïra: 32;



```python
f"first 5 bank notes of 5 Naïra for training purposes: {os.listdir(f'{path}/train/N5')[:5]}"
```




    "first 5 bank notes of 5 Naïra for training purposes: ['09.jpg', '32.jpg', '23.jpg', '26.jpg', '03.jpg']"



b- Exploring VALIDATION dataset


```python
print('* list of bank notes CATEGORIES in the validation set: %s' %(os.listdir(f'{path}/valid')))
```

    * list of bank notes CATEGORIES in the validation set: ['N20', 'N1000', 'N500', 'N100', 'N10', 'N5', 'N200', 'N50']



```python
print('Number of bank notes images of each category to validate the model: \n')
print('\t* 5 Naïra: %s;' %len(os.listdir(f'{path}/valid/N5')))
print('\t* 10 Naïra: %s;' %len(os.listdir(f'{path}/valid/N10')))
print('\t* 1000 Naïra: %s' %len(os.listdir(f'{path}/valid/N1000')))
```

    Number of bank notes images of each category to validate the model: 
    
    	* 5 Naïra: 15;
    	* 10 Naïra: 15;
    	* 1000 Naïra: 15



```python
print('first 5 bank notes images of 5 Naïra for validation purposes: %s' %(os.listdir(f'{path}/valid/N5'))[:5])
```

    first 5 bank notes images of 5 Naïra for validation purposes: ['09.jpg', '32.jpg', '23.jpg', '26.jpg', '31.jpg']



```python
files = os.listdir(f'{path}/valid/N5')[:5]
files
```




    ['09.jpg', '32.jpg', '23.jpg', '26.jpg', '31.jpg']




```python
#read the second image in the validation set
val_img_0 = plt.imread(f'{path}/valid/N5/{files[0]}')
plt.imshow(val_img_0);
```


![png](/images/naira_classification_files/naira_classification_21_0.png)



```python
val_img_0.size
```




    262500




```python
val_img_0.shape
```




    (250, 350, 3)




```python
val_img_0[:2, :3]
```




    array([[[129,  36,  29],
            [112,  19,  14],
            [100,   6,   4]],
    
           [[139,  44,  38],
            [124,  29,  25],
            [109,  14,  12]]], dtype=uint8)



 **Notes on above: ** Our currency dataset shows a ratio of 15 / 32 ... the validation set is  50 % of the training set.

### ***Let us now build our Classifier !***

### ***  1/- Without Data Augmentation***


```python
tfms = tfms_from_model(archit, sz)
data = ImageClassifierData.from_paths(path, bs=bs, tfms=tfms)
learn = ConvLearner.pretrained(archit, data)
```


```python
# Optimal learning rate through Learning rate finder
lrf = learn.lr_find()
```


    HBox(children=(IntProgress(value=0, description='Epoch', max=1), HTML(value='')))


     92%|█████████▏| 24/26 [00:04<00:00,  5.95it/s, loss=4.36]



```python
learn.sched.plot_loss()
```


![png](/images/naira_classification_files/naira_classification_30_0.png)



```python
learn.sched.plot_lr()
```


![png](/images/naira_classification_files/naira_classification_31_0.png)



```python
learn.sched.plot()
```


![png](/images/naira_classification_files/naira_classification_32_0.png)



```python
# our learning rate shoul be somewhere between 1e-2 and 1e-1
lr = 0.01
```


```python
learn.fit(lr, 3)
```


    HBox(children=(IntProgress(value=0, description='Epoch', max=3), HTML(value='')))


    epoch      trn_loss   val_loss   accuracy   
        0      1.715736   0.41925    0.875     
        1      1.030992   0.175984   0.95      
        2      0.757789   0.083566   0.975     
    





    [array([0.08357]), 0.9749999940395355]




```python
learn.fit(lr, 3, cycle_len=1)
```


    HBox(children=(IntProgress(value=0, description='Epoch', max=3), HTML(value='')))


    epoch      trn_loss   val_loss   accuracy   
        0      0.377057   0.076735   0.975     
        1      0.259685   0.040686   0.991667  
        2      0.22681    0.037616   0.991667  
    





    [array([0.03762]), 0.9916666646798452]




```python
learn.unfreeze()
```


```python
#set differential learning rates
dlr = np.array([lr/100 ,lr/10 ,lr])
dlr
```




    array([0.0001, 0.001 , 0.01  ])




```python
learn.fit(dlr, 3)
```


    HBox(children=(IntProgress(value=0, description='Epoch', max=3), HTML(value='')))


    epoch      trn_loss   val_loss   accuracy   
        0      0.890019   0.181679   0.925     
        1      0.622748   0.046668   0.983333  
        2      0.551292   0.056624   0.983333  
    





    [array([0.05662]), 0.9833333293596903]




```python
learn.fit(dlr, 3, cycle_len=1)
```


    HBox(children=(IntProgress(value=0, description='Epoch', max=3), HTML(value='')))


    epoch      trn_loss   val_loss   accuracy   
        0      0.10045    0.018287   0.991667  
        1      0.086283   0.014844   0.991667  
        2      0.098118   0.02244    0.991667  
    





    [array([0.02244]), 0.9916666646798452]




```python
learn.sched.plot_lr()
```


![png](/images/naira_classification_files/naira_classification_40_0.png)



```python
learn.sched.plot_loss()
```


![png](/images/naira_classification_files/naira_classification_41_0.png)



```python
learn.save('224_r50_lalay_noTfms')
```


```python
learn.load('224_r50_lalay_noTfms')
```

### ***2/- Adding Transformations hoping for model improvment***


```python
tfms = tfms_from_model(archit, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
data = ImageClassifierData.from_paths(path, bs=bs, tfms=tfms, num_workers=4)
learn_da = ConvLearner.pretrained(archit, data, ps=0.5)      #_da :: data augmentation.
```


```python
lrf = learn_da.lr_find()
```


    HBox(children=(IntProgress(value=0, description='Epoch', max=1), HTML(value='')))


     96%|█████████▌| 25/26 [00:04<00:00,  5.05it/s, loss=8.09]



```python
learn_da.sched.plot()
```


![png](/images/naira_classification_files/naira_classification_47_0.png)



```python
learn_tf.sched.plot_lr()
```


![png](/images/naira_classification_files/naira_classification_48_0.png)



```python
lr_da = 0.01
```


```python
learn_da.fit(lr_da, 3)
```


    HBox(children=(IntProgress(value=0, description='Epoch', max=3), HTML(value='')))


    epoch      trn_loss   val_loss   accuracy   
        0      1.705401   0.495659   0.866667  
        1      1.121846   0.186173   0.95      
        2      0.86228    0.162368   0.966667  
    





    [array([0.16237]), 0.9666666636864344]




```python
learn_da.unfreeze()
```


```python
dlr_da = np.array([lr/100 ,lr/10 ,lr])
dlr_da
```




    array([0.0001, 0.001 , 0.01  ])




```python
learn_da.fit(dlr_da, 3)
```


    HBox(children=(IntProgress(value=0, description='Epoch', max=3), HTML(value='')))


    epoch      trn_loss   val_loss   accuracy   
        0      0.870899   0.153029   0.95      
        1      0.65304    0.069872   0.975     
        2      0.482148   0.043935   0.991667  
    





    [array([0.04394]), 0.9916666646798452]




```python
learn_da.fit(dlr_da, 2, cycle_len=1)
```


    HBox(children=(IntProgress(value=0, description='Epoch', max=2), HTML(value='')))


    epoch      trn_loss   val_loss   accuracy   
        0      0.152045   0.017834   0.991667  
        1      0.115954   0.014596   0.991667  
    





    [array([0.0146]), 0.9916666646798452]




```python
learn.save('224_r50_naïra_All_layers')
```


```python
learn.load('224_r50_naïra_All_layers')
```


```python
#logarithmic predictions
log_preds_da, y = learn_da.TTA()
```

    


```python
log_preds_da, y
```




    (array([[[ -0.02911,  -6.71116,  -6.37196, ...,  -3.6755 , -10.83042, -10.21062],
             [ -0.00381, -10.47355, -10.12127, ..., -10.45833,  -9.85412,  -6.74461],
             [ -0.00037,  -8.95466, -12.35964, ...,  -8.38403, -14.62125, -12.04709],
             ...,
             [-10.93964, -10.96277, -14.21617, ..., -12.78286, -11.75792,  -0.00013],
             [-17.27452, -15.14946, -16.30872, ..., -15.23285, -15.62271,  -0.     ],
             [-16.29882, -17.04697, -18.98109, ..., -15.56747, -11.85577,  -0.00001]],
     
            [[ -0.00816,  -5.50622,  -6.64942, ...,  -5.90705, -11.8057 , -11.68874],
             [ -0.0067 ,  -8.59207,  -9.77459, ...,  -9.00766,  -8.80763,  -5.36446],
             [ -0.00091,  -9.26725, -10.74794, ...,  -7.14347, -15.16809, -12.86022],
             ...,
             [ -9.40464,  -9.02375, -11.94917, ..., -11.886  , -10.91466,  -0.00045],
             [-15.70375, -13.17634, -15.69993, ..., -13.7455 , -13.07041,  -0.00001],
             [-17.2672 , -14.60529, -19.92577, ..., -14.15851, -12.44216,  -0.00001]],
     
            [[ -0.13447,  -2.9414 ,  -4.05639, ...,  -2.91093,  -9.55217,  -9.23069],
             [ -0.00087, -10.04172, -11.43896, ..., -10.64378, -11.9205 ,  -8.24317],
             [ -0.00154,  -8.09499, -10.65234, ...,  -6.76529, -11.82937, -11.15357],
             ...,
             [-13.72499, -12.88383, -16.50002, ..., -14.53644, -13.28295,  -0.00001],
             [-15.9492 , -12.7002 , -15.25   , ..., -13.48519, -12.50459,  -0.00001],
             [-15.45253, -17.03182, -19.69193, ..., -15.25108, -10.61565,  -0.00003]],
     
            [[ -0.00645,  -5.20103,  -8.1205 , ...,  -7.46251, -12.79394, -11.33294],
             [ -0.00187,  -9.08785, -11.10998, ..., -10.45017, -10.08036,  -6.93706],
             [ -0.00078,  -7.45882, -11.39866, ...,  -8.65603, -13.41383, -11.21497],
             ...,
             [-14.70812, -15.16663, -16.95833, ..., -17.02563, -14.15458,  -0.00001],
             [-15.67043, -14.88226, -15.32212, ..., -12.45475, -14.60657,  -0.00001],
             [-15.92068, -16.34062, -18.49481, ..., -14.34843, -12.22935,  -0.00001]],
     
            [[ -0.00577,  -5.29242,  -7.81651, ...,  -8.18881, -13.29915, -11.09464],
             [ -0.00029,  -9.76076, -11.10865, ..., -11.33792, -12.24837,  -8.91434],
             [ -0.00024,  -8.64907, -11.64188, ..., -10.06426, -14.3803 , -12.28792],
             ...,
             [-14.93715, -14.03831, -17.2505 , ..., -15.37274, -14.47498,  -0.     ],
             [-14.31691, -11.37785, -14.00285, ..., -11.81978, -12.41713,  -0.00003],
             [-12.99364, -14.04403, -18.27118, ..., -13.8132 , -10.44429,  -0.00006]]], dtype=float32),
     array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4,
            4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
            6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]))




```python
log_preds_da.shape
```




    (5, 120, 8)




```python
len(log_preds_da)
```




    5




```python
len(y)
```




    120




```python
#Exponential predictions : probabilities : Through Test Time Augmentation TTA(), take the average of exponential predictions of original and transformed of images.
probabilities_da = np.mean(np.exp(log_preds_da), 0)
```


```python
probabilities_da
```




    array([[0.96504, 0.01372, 0.0042 , 0.00031, 0.     , 0.01667, 0.00002, 0.00003],
           [0.9973 , 0.00009, 0.00003, 0.00103, 0.00001, 0.00004, 0.00005, 0.00145],
           [0.99923, 0.00026, 0.00001, 0.00001, 0.     , 0.00048, 0.     , 0.00001],
           [0.99972, 0.00004, 0.     , 0.00022, 0.     , 0.00001, 0.     , 0.00001],
           [0.9988 , 0.00049, 0.00016, 0.00007, 0.     , 0.00038, 0.00001, 0.00008],
           [0.99582, 0.00095, 0.00184, 0.00003, 0.     , 0.00132, 0.00002, 0.00001],
           [0.99978, 0.00007, 0.     , 0.00007, 0.00001, 0.00006, 0.     , 0.00001],
           [0.99963, 0.00019, 0.00002, 0.00003, 0.     , 0.00013, 0.     , 0.00001],
           [0.99948, 0.00006, 0.00011, 0.00015, 0.     , 0.00016, 0.00001, 0.00003],
           [0.99117, 0.00697, 0.00076, 0.00016, 0.00001, 0.00079, 0.00001, 0.00013],
           [0.99533, 0.00203, 0.00009, 0.00006, 0.00001, 0.00222, 0.00001, 0.00025],
           [0.99897, 0.00082, 0.00004, 0.00001, 0.     , 0.00014, 0.     , 0.00002],
           [0.99591, 0.00072, 0.00037, 0.00099, 0.00004, 0.00172, 0.00012, 0.00012],
           [0.99923, 0.00037, 0.00001, 0.00001, 0.00002, 0.00032, 0.     , 0.00005],
           [0.98246, 0.01223, 0.00023, 0.00003, 0.00002, 0.00493, 0.00002, 0.00007],
           [0.00013, 0.99864, 0.001  , 0.00001, 0.00001, 0.00008, 0.00003, 0.00009],
           [0.     , 0.99999, 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ],
           [0.     , 0.99992, 0.     , 0.00002, 0.     , 0.     , 0.     , 0.00005],
           [0.     , 0.99999, 0.     , 0.     , 0.     , 0.     , 0.00001, 0.     ],
           [0.     , 0.99999, 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ],
           [0.00001, 0.99978, 0.00002, 0.00001, 0.00004, 0.00001, 0.00011, 0.00003],
           [0.     , 1.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ],
           [0.00001, 0.99988, 0.00001, 0.     , 0.00001, 0.00004, 0.00001, 0.00006],
           [0.00001, 0.99985, 0.00005, 0.00001, 0.00001, 0.00006, 0.00001, 0.00001],
           [0.     , 1.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ],
           [0.     , 1.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ],
           [0.     , 0.99994, 0.00001, 0.     , 0.     , 0.00001, 0.00003, 0.     ],
           [0.     , 0.99998, 0.     , 0.     , 0.     , 0.     , 0.00002, 0.     ],
           [0.     , 0.99998, 0.     , 0.     , 0.     , 0.     , 0.00002, 0.     ],
           [0.     , 1.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ],
           [0.00199, 0.01515, 0.98118, 0.00042, 0.00007, 0.00111, 0.     , 0.00007],
           [0.     , 0.     , 0.99998, 0.     , 0.     , 0.     , 0.     , 0.00001],
           [0.     , 0.00002, 0.99997, 0.     , 0.     , 0.     , 0.     , 0.00001],
           [0.00005, 0.00001, 0.99787, 0.00003, 0.00005, 0.00026, 0.00001, 0.00171],
           [0.00013, 0.00003, 0.99752, 0.00168, 0.     , 0.00021, 0.00001, 0.00042],
           [0.00001, 0.00002, 0.99982, 0.     , 0.00006, 0.     , 0.     , 0.00009],
           [0.00022, 0.00019, 0.99683, 0.00092, 0.00001, 0.001  , 0.00032, 0.00052],
           [0.     , 0.     , 0.99998, 0.     , 0.     , 0.00001, 0.     , 0.     ],
           [0.0002 , 0.     , 0.99942, 0.00035, 0.     , 0.00001, 0.     , 0.00002],
           [0.     , 0.     , 0.99999, 0.     , 0.     , 0.     , 0.     , 0.00001],
           [0.     , 0.     , 0.99999, 0.     , 0.     , 0.     , 0.     , 0.     ],
           [0.     , 0.00001, 0.99919, 0.0007 , 0.     , 0.     , 0.     , 0.00008],
           [0.00006, 0.00031, 0.99947, 0.00001, 0.00001, 0.00007, 0.     , 0.00009],
           [0.0003 , 0.00004, 0.99947, 0.     , 0.00002, 0.00003, 0.     , 0.00014],
           [0.00002, 0.     , 0.9993 , 0.00063, 0.00002, 0.     , 0.     , 0.00003],
           [0.00035, 0.00031, 0.     , 0.9903 , 0.00879, 0.00005, 0.00019, 0.     ],
           [0.00004, 0.     , 0.     , 0.99994, 0.     , 0.00001, 0.     , 0.     ],
           [0.     , 0.     , 0.     , 0.99999, 0.     , 0.     , 0.     , 0.     ],
           [0.00001, 0.     , 0.     , 0.99999, 0.     , 0.     , 0.     , 0.00001],
           [0.     , 0.     , 0.     , 0.99999, 0.     , 0.     , 0.     , 0.     ],
           [0.00001, 0.     , 0.     , 0.99999, 0.     , 0.     , 0.     , 0.     ],
           [0.     , 0.     , 0.     , 1.     , 0.     , 0.     , 0.     , 0.     ],
           [0.00005, 0.     , 0.0002 , 0.99973, 0.     , 0.     , 0.     , 0.00001],
           [0.00002, 0.00001, 0.     , 0.99995, 0.00001, 0.     , 0.     , 0.     ],
           [0.     , 0.     , 0.00001, 0.99998, 0.     , 0.00001, 0.     , 0.     ],
           [0.     , 0.     , 0.     , 1.     , 0.     , 0.     , 0.     , 0.     ],
           [0.00017, 0.00003, 0.00004, 0.99965, 0.00001, 0.00002, 0.00006, 0.00001],
           [0.     , 0.     , 0.     , 1.     , 0.     , 0.     , 0.     , 0.     ],
           [0.00002, 0.     , 0.00001, 0.99992, 0.00001, 0.00003, 0.00001, 0.     ],
           [0.00041, 0.0001 , 0.     , 0.9984 , 0.00099, 0.00002, 0.00008, 0.     ],
           [0.     , 0.     , 0.     , 0.     , 1.     , 0.     , 0.     , 0.     ],
           [0.     , 0.     , 0.     , 0.     , 0.99999, 0.     , 0.     , 0.00001],
           [0.     , 0.     , 0.00001, 0.     , 0.99998, 0.     , 0.00001, 0.     ],
           [0.     , 0.00001, 0.     , 0.     , 0.99998, 0.     , 0.00001, 0.     ],
           [0.     , 0.     , 0.     , 0.     , 1.     , 0.     , 0.     , 0.     ],
           [0.     , 0.     , 0.     , 0.     , 1.     , 0.     , 0.     , 0.     ],
           [0.     , 0.     , 0.     , 0.     , 1.     , 0.     , 0.     , 0.     ],
           [0.     , 0.     , 0.     , 0.     , 1.     , 0.     , 0.     , 0.     ],
           [0.     , 0.     , 0.00001, 0.00001, 0.99974, 0.     , 0.     , 0.00024],
           [0.     , 0.     , 0.     , 0.     , 1.     , 0.     , 0.     , 0.     ],
           [0.     , 0.     , 0.     , 0.     , 0.99999, 0.     , 0.     , 0.     ],
           [0.     , 0.     , 0.     , 0.     , 0.99999, 0.     , 0.00001, 0.     ],
           [0.     , 0.     , 0.     , 0.     , 0.99999, 0.     , 0.     , 0.00001],
           [0.     , 0.     , 0.     , 0.00003, 0.99995, 0.     , 0.00001, 0.00001],
           [0.     , 0.00003, 0.     , 0.     , 0.99996, 0.     , 0.     , 0.     ],
           [0.00003, 0.00145, 0.00121, 0.0001 , 0.00001, 0.99654, 0.00058, 0.00006],
           [0.00003, 0.00001, 0.00084, 0.00004, 0.00001, 0.99896, 0.00003, 0.00007],
           [0.00001, 0.00013, 0.00252, 0.00001, 0.00003, 0.99708, 0.00004, 0.00017],
           [0.     , 0.00001, 0.00001, 0.     , 0.     , 0.99996, 0.00001, 0.     ],
           [0.     , 0.     , 0.02452, 0.00001, 0.     , 0.97546, 0.     , 0.00001],
           [0.     , 0.00001, 0.00022, 0.     , 0.     , 0.99974, 0.00001, 0.00002],
           [0.     , 0.00001, 0.00018, 0.00001, 0.     , 0.99978, 0.00001, 0.00001],
           [0.     , 0.     , 0.00518, 0.00001, 0.     , 0.99481, 0.     , 0.     ],
           [0.00002, 0.00002, 0.03556, 0.00019, 0.00007, 0.96378, 0.0001 , 0.00027],
           [0.     , 0.     , 0.00003, 0.     , 0.     , 0.99997, 0.     , 0.     ],
           [0.00056, 0.23984, 0.00022, 0.02076, 0.00072, 0.73162, 0.00585, 0.00041],
           [0.00004, 0.00002, 0.00385, 0.00011, 0.00003, 0.99453, 0.00005, 0.00137],
           [0.     , 0.00001, 0.00005, 0.     , 0.     , 0.99994, 0.00001, 0.     ],
           [0.00001, 0.00007, 0.00003, 0.     , 0.     , 0.99989, 0.     , 0.     ],
           [0.00024, 0.00033, 0.00021, 0.00005, 0.     , 0.99914, 0.00001, 0.00002],
           [0.     , 0.00001, 0.     , 0.     , 0.00001, 0.     , 0.99997, 0.     ],
           [0.00037, 0.00001, 0.00018, 0.01835, 0.0028 , 0.00052, 0.97715, 0.00062],
           [0.00001, 0.00042, 0.00001, 0.00001, 0.0037 , 0.00001, 0.99575, 0.00008],
           [0.00054, 0.00016, 0.00024, 0.00603, 0.00007, 0.00174, 0.99113, 0.00009],
           [0.     , 0.     , 0.     , 0.     , 0.00004, 0.     , 0.99994, 0.00001],
           [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 1.     , 0.     ],
           [0.     , 0.     , 0.     , 0.     , 0.00001, 0.     , 0.99998, 0.     ],
           [0.00013, 0.00004, 0.0001 , 0.00095, 0.0001 , 0.00005, 0.99862, 0.00002],
           [0.0001 , 0.00001, 0.00008, 0.0002 , 0.00028, 0.00045, 0.99877, 0.0001 ],
           [0.00003, 0.00005, 0.00004, 0.00002, 0.00305, 0.00071, 0.99608, 0.00001],
           [0.00076, 0.00185, 0.00047, 0.00031, 0.00018, 0.00834, 0.98557, 0.00252],
           [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 1.     , 0.     ],
           [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 1.     , 0.     ],
           [0.     , 0.00002, 0.     , 0.     , 0.     , 0.     , 0.99998, 0.     ],
           [0.00002, 0.0002 , 0.     , 0.00056, 0.00066, 0.00061, 0.99795, 0.00001],
           [0.     , 0.     , 0.     , 0.     , 0.00001, 0.     , 0.00001, 0.99998],
           [0.00008, 0.00001, 0.00001, 0.00003, 0.00046, 0.00004, 0.00004, 0.99934],
           [0.00005, 0.00027, 0.00018, 0.00004, 0.00005, 0.00004, 0.0022 , 0.99717],
           [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.00001, 0.99998],
           [0.00002, 0.00003, 0.00009, 0.00003, 0.00002, 0.00002, 0.00096, 0.99883],
           [0.     , 0.     , 0.     , 0.     , 0.00001, 0.00003, 0.00001, 0.99995],
           [0.00003, 0.00016, 0.     , 0.00214, 0.00022, 0.00006, 0.00017, 0.99723],
           [0.01941, 0.11729, 0.00854, 0.00221, 0.00354, 0.46078, 0.0088 , 0.37942],
           [0.00002, 0.00026, 0.     , 0.     , 0.00023, 0.00015, 0.00001, 0.99933],
           [0.     , 0.     , 0.     , 0.     , 0.00065, 0.     , 0.     , 0.99935],
           [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 1.     ],
           [0.     , 0.00001, 0.     , 0.     , 0.     , 0.     , 0.00001, 0.99998],
           [0.00002, 0.00003, 0.     , 0.00005, 0.00001, 0.     , 0.00001, 0.99988],
           [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.99999],
           [0.     , 0.     , 0.     , 0.     , 0.00001, 0.     , 0.00001, 0.99998]], dtype=float32)




```python
len(probabilities_da)
```




    120




```python
#Model accuracy
model_accur = accuracy_np(probabilities_da, y)
model_accur
```




    0.9916666666666667



##### *__A little function to manage floating decimals:__* this helps in shrinking the floating point to present the result as we want.


```python
def floating_decimals(f_val, dec):
    prc = "{:."+str(dec)+"f}"   #first cast decimal as string
    #print(prc) # strformat output is {:.3f}
    return prc.format(f_val)
```


```python
#Shrinking our model accuracy result to 2 floating points.
model_accur = floating_decimals(model_accur*100, 2)
model_accur
```




    '99.17'




```python
 F"Our Naïra bank notes\' classifier can perform with {model_accur}% accuracy."
```




    "Our Naïra bank notes' classifier can perform with 99.17% accuracy."



### **Let us explore the results**


```python
preds_da = np.argmax(probabilities_da, axis=1)
```


```python
probabilities_da = probabilities_da[:,1]
```


```python
probabilities_da
```




    array([0.01372, 0.00009, 0.00026, 0.00004, 0.00049, 0.00095, 0.00007, 0.00019, 0.00006, 0.00697, 0.00203,
           0.00082, 0.00072, 0.00037, 0.01223, 0.99864, 0.99999, 0.99992, 0.99999, 0.99999, 0.99978, 1.     ,
           0.99988, 0.99985, 1.     , 1.     , 0.99994, 0.99998, 0.99998, 1.     , 0.01515, 0.     , 0.00002,
           0.00001, 0.00003, 0.00002, 0.00019, 0.     , 0.     , 0.     , 0.     , 0.00001, 0.00031, 0.00004,
           0.     , 0.00031, 0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.00001, 0.     ,
           0.     , 0.00003, 0.     , 0.     , 0.0001 , 0.     , 0.     , 0.     , 0.00001, 0.     , 0.     ,
           0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.00003, 0.00145, 0.00001,
           0.00013, 0.00001, 0.     , 0.00001, 0.00001, 0.     , 0.00002, 0.     , 0.23984, 0.00002, 0.00001,
           0.00007, 0.00033, 0.00001, 0.00001, 0.00042, 0.00016, 0.     , 0.     , 0.     , 0.00004, 0.00001,
           0.00005, 0.00185, 0.     , 0.     , 0.00002, 0.0002 , 0.     , 0.00001, 0.00027, 0.     , 0.00003,
           0.     , 0.00016, 0.11729, 0.00026, 0.     , 0.     , 0.00001, 0.00003, 0.     , 0.     ],
          dtype=float32)




```python
from sklearn.metrics import confusion_matrix
```


```python
model_conf_matrix = confusion_matrix(y, preds_da)
```


```python
plot_confusion_matrix(model_conf_matrix, data.classes, figsize=(10, 5), title = 'Naïra bank notes classification')
```


```python
def rand_by_mask(mask): return np.random.choice(np.where(mask)[0], 4, replace=False)
def rand_by_correct(is_correct): return rand_by_mask((preds_da == data.val_y)==is_correct)
```


```python
def plot_val_with_title(idxs, title):
    imgs = np.stack([data.val_ds[x][0] for x in idxs])
    title_probabilities_da = [probabilities_da[x] for x in idxs]
    print(title)
    return plots(data.val_ds.denorm(imgs), rows=1, titles=title_probabilities_da)
```


```python
def plots(ims, figsize=(12,6), rows=1, titles=None):
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims)//rows, i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i])
```


```python
def most_by_mask(mask, mult):
    idxs = np.where(mask)[0]
    return idxs[np.argsort(mult * probabilities_da[idxs])[:4]]

def most_by_correct(y, is_correct): 
    mult = -1 if (y==1)==is_correct else 1
    return most_by_mask(((preds_da == data.val_y)==is_correct) & (data.val_y == y), mult)
```


```python
list_Notes_type = data.classes
list_Notes_type
```




    ['N10', 'N100', 'N1000', 'N20', 'N200', 'N5', 'N50', 'N500']




```python
list_Notes_type[0], list_Notes_type[1], list_Notes_type[2], list_Notes_type[3], list_Notes_type[4], list_Notes_type[5], list_Notes_type[6], list_Notes_type[7]
```




    ('N10', 'N100', 'N1000', 'N20', 'N200', 'N5', 'N50', 'N500')




```python
# 1. A few correct labels at random
plot_val_with_title(most_by_correct(0, True), "N10: Most Correctly classified 10 Naïra bank notes")
```

    N10: Correctly classified 10 Naïra bank notes



![png](/images/naira_classification_files/naira_classification_83_1.png)



```python
plot_val_with_title(most_by_correct(1, True), "N100: Most Correctly classified 100 Naïra bank notes")
```

    N100: Correctly classified 100 Naïra bank notes



![png](/images/naira_classification_files/naira_classification_84_1.png)



```python
plot_val_with_title(most_by_correct(5, True), "N5: Most correctly classified 5 Naïra bank notes.")
```

    N5: correctly classified 5 Naïra bank notes.



![png](/images/naira_classification_files/naira_classification_85_1.png)



```python
plot_val_with_title(most_by_correct(7, True), "N500: Most correctly classified 500 Naïra bank notes.")
```

    N500: Most correctly classified 500 Naïra bank notes.



![png](/images/naira_classification_files/naira_classification_86_1.png)



```python
plot_val_with_title(most_by_correct(7, False), "N500: INCORRECTLY classified 500 Naïra bank note. It is actually the only incorrect classified one in the whole validation set as shown in the confusion matrix!!")
```

    N500: INCORRECTLY classified 500 Naïra bank note. It is actually the only incorrect classified one in the whole validation set as shown in the confusion matrix!!



![png](/images/naira_classification_files/naira_classification_87_1.png)

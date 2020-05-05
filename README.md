### **Detect COVID-19 by using chest X-Ray Images**

------

The purpose of this project is to classify the chest X-ray images of a person into 4 classes - bacterial pneumonia, viral pneumonia, covid-19 and normal .

#### Summary

------

For solving this problem ,we have gone through two datasets, one is Mendeley dataset which is collected from two online available datasets ( https://github.com/ieee8023/covid-chestxray-dataset and https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia),the second is the covid19-detection-xray-dataset available on kaggle ( https://www.kaggle.com/darshan1504/covid19-detection-xray-dataset ).

First we went through the Mendeley dataset[1] to build a model (so that we can fine-tune it later to 4 classes) ,here we have done covid-19 Vs non covid-19 binary classification.

With this dataset we have experimented with 4 models:

- Baseline Model (Accuracy 94%),
- Using MobileNet as a pre-trained model with the addition of a prediction layer (Accuracy 100%),
- Using VGG16 as a pre-trained model with the addition of a prediction layer (Accuracy 96%),
- Using MobileNetV2 as a pre-trained model with the addition of a prediction layer (Accuracy 100%).

Secondly,we went through a kaggle dataset[2] ,where we fine-tuned our above MobilenetV2 model (trained on mendeley dataset[1]) on the kaggle dataset[2] containing 4 classes (Covid-19,Normal,Viral pneumonia & Bacterial pneumonia) .The Accuracy achieved was 77.48% .

The performance for Non Covid-19 pneumonia classes (pneumonia-bacterial and pneumonia-viral) are comparatively lower than other two classes and contributes to the lower overall accuracy. If we combine the pneumonia-bacterial and pneumonia-viral into one single class as Pneumonia class, then the overall accuracy increases significantly.

So, as a next step we fine-tuned our MobilenetV2 model trained on the Mendeley dataset[1] on the data obtained by combining bacterial and viral pneumonia classes i.e, Covid-19,Normal & Pneumonia. In this model , we were able to improve the accuracy to 90.42%.



#### Covid-19 vs Non Covid-19 Classification:-

###### Baseline Model on Mendeley Dataset( Model I )

------

This Model has basic architecture which contains 5 Conv2D layers with kernal size (3,3) , 5 MaxPooling2D layer with kernal size (2,2), and one dense layer with activation function 'relu' and no of neurons is 512. The input size provided to the model is (244,244,3).

Hyperparameters:

- Optimizer :Adam

- Learning rate : 0.0001

- Epochs : 10

- No of classes : 2 {'COVID-19': 0, 'Non-COVID-19': 1}

Notebook Reference:
- https://github.com/puja431996/Notebook/blob/master/COVID_19_Basic.ipynb
  

###### Model trained on Mendeley dataset using MobileNet pretrained model (Model II)

------

MobileNet:

- MobileNets are light weight deep neural networks best suited for mobile and embedded vision applications.
- MobileNets are based on a streamlined architecture that uses depth wise separable convolutions.
- MobileNet uses two simple global hyperparameters that efficiently trades off between accuracy and latency. 
- MobileNet could be used in object detection, finegrain classification, face recognition, large-scale geo localization etc.

 Approach:

- Fine-tuned the MobileNet model pretrained on ImageNet for our covid, non covid data.
- Trained it upto 4 epochs and achieved accuracy nearly 99%.

Hyperparameters:

- Optimizer : Adam

- Learning rate : 0.0001

- Epochs : 5

- No of classes : 2 {'COVID-19': 0, 'Non-COVID-19': 1}

 Notebook Reference:
- https://github.com/puja431996/Notebook/blob/master/COVID_19(MobileNet).ipynb 


###### Model trained on Mendeley dataset using VGG16 pretrained model (Model III)

------

VGG16:

VGG is a Convolutional Neural Network architecture, It was proposed by [Karen Simonyan](http://www.robots.ox.ac.uk/~karen/) and [Andrew Zisserman ](https://en.wikipedia.org/wiki/Andrew_Zisserman)of [Oxford Robotics Institute](https://en.wikipedia.org/wiki/Oxford_Robotics_Institute) in the year 2014. It was submitted to Large Scale Visual Recognition Challenge 2014 (ILSVRC2014) and The model achieves 92.7% top-5 test accuracy in ImageNet. [ImageNet](https://en.wikipedia.org/wiki/ImageNet) is one of the largest data-sets available , which is a dataset of over 14 million images belonging to 1000 classes.

Approach:

- Fine-tuned the VGG model pretrained on ImageNet for our covid, non covid data.
- Trained it upto 10 epochs and achieved accuracy nearly 96%.

Hyperparameters:

- Optimizer : Adam

- Learning rate : 0.0001

- Epochs : 10

- No of classes : 2 {'COVID-19': 0, 'Non-COVID-19': 1}

Notebook Reference:
- https://github.com/puja431996/Notebook/blob/master/COVID_19(VGG16).ipynb


###### Model trained on Mendeley dataset using MobileNetV2 pretrained model (Model IV)

------

This model is the advanced version of model MobileNet, called MobileNet Version 2.

Approach:

- Applied Transfer Learning technique with MobileNetV2 for our the mendley dataset.
- Finetuned this model with a slower learning rate.
- After that the model achieved 100% accuracy on training data as well as validation data.

Hyperparameters:

- Optimizer : Adam

- Learning rate : 0.0001

- Epochs : 30 (with Transfer Learning) + 20 (with Fine Tune)

- No of classes : 2 {'COVID-19': 0, 'Non-COVID-19': 1}

Notebook Reference:
- https://github.com/puja431996/Notebook/blob/master/MobileNetV2.ipynb  


#### Covid-19, Normal, Viral Pneumonia & Bacterial Pneumonia Classification :-

------

Model trained on Kaggle dataset[2] with MobileNetV2 pretrained on Mendeley dataset[1].

Approach: 

- Used the kaggle dataset[2] to build a model for 4-class classification,i.e Covid-19,Normal,Viral Pneumonia and Bacterial Pneumonia.
- Fine tuned the previous model (MobileNetV2 which was trained on Mendeley data[1] with binary classification) for our now 4 class problem.
- Achieved accuracy was upto 77.48%.

Hyperparameters:

- Optimizer : Adam

- Learning rate: 0.0001

- Epochs : 15

- Batch size : 32

- No of classes : 4 {'BacterialPneumonia': 0, 'COVID-19': 1, 'Normal': 2, 'ViralPneumonia': 3} 

 Notebook Reference:
- https://github.com/puja431996/Notebook/blob/master/Multi_Class_MobileNet(4_Class).ipynb 


#### Covid-19,Normal & Pneumonia Classification

------

Model trained on Kaggle dataset[2] with MobileNetV2 pretrained on Mendeley dataset[1].

Approach: 

- Used the kaggle dataset[2] to build a model for Three-class classification. 
- Used Transfer Learning on the previous model (MobileNetV2 which was trained on Mendeley data[1] with binary classification) for our now 3 class problem. 
- After fine tuning the above model ,we were able to boost the accuracy to upto 90.42%

Hyperparameters:

- Optimizer : Adam
- Learning rate without fine tune: 0.0001
- Learning rate with fine tune: 0.0001/10
- Epochs without fine tune: 20
- Epochs with fine tune: 20
- Batch size for training generator : 32
- Batch size for validation generator: 1
- No of classes : 3 {'COVID-19': 0, 'Normal': 1, ‘Pneumonia’: 2}


 Notebook Reference:
- https://github.com/puja431996/Notebook/blob/master/Three_Class_MobileNetV2(3_Class).ipynb


#### Referenced Datasets

------

[1]: https://data.mendeley.com/datasets/2fxz4px6d8/4	" Mendeley dataset"<br/>
[2]: https://www.kaggle.com/darshan1504/covid19-detection-xray-dataset	" Kaggle dataset"


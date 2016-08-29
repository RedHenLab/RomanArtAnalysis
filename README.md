#GSoC 2016 with Red Hen Lab
This is my Google Summer of Code 2016 project with [the Distributed Little Red Hen Lab](http://www.redhenlab.org/).

The GSoC [project page](https://summerofcode.withgoogle.com/projects/#5484824705892352) shows the details about this project.


#Introduction

This project is a **relatively independent research project** with Redhen lab. So **I created this repository to contain my research code for experiments and the analysis results instead of contributing to some existing code repository**. As directed by my mentors, this project is focused on analysis of the images of ancient roman statues for their social values. **I wrote most of the code in this repository to train classifiers and analyze data. Lots of analysis results (tables, figures and webpages) are produced which can help the experts in Redhen lab to analyze the meaning of communication of the ancient roman statues.**



This project contains parts as follow:

1. webpage parsers and crawlers for data collecting from art databases 

2. simple annotation tool for roman statue images

3. roman statue face pre-process pipeline

4. face frontalization

5. meta keywords analysis for roman statue data

6. geometry feature extractor for roman statue faces

7.  transfer learning and result analysis for social value inference of roman statue faces

8.  Convolutional Network classifiers for roman statue faces (training and testing code)

9.  Deconvolutional Network for visualization of the learned classifers.


#Method and Code Documentation

##Webpage Parsers and Crawlers for Data Collecting from Art Databases

The HTMLParser and state machine is used for parsing the webpages of the databases and downloading images and meta-data.

###related code:

1.  dataCollecting/downloader4\[database\].py: parsing the webpages and download the images. Because each database has totally different webpage structures, each downloader is designed particularlly for one certain database. The url of the database is already hard-coded into the code. 

2.  dataCollecting/singlePageImageFetcher.py: fetching all the images on a single webpage. This tool can be used with a common webpage to download all the images on it.

###Dependencies:
Python 2.7: The language of these tools. I did not test it with Python 3, so it is not recommended to run them with Python 3.

All the library used are contained in the standard Python package, so there are no extra dependencies.


###Deployment instructions:
1. python downloader4\[database\].py LOCAL-PATH-TO-SAVE-THE-DATA BEGIN-INDEX END-INDEX
2. Set the PAGE\_BASE and PAGE\_LIST variables in the code before run it. And run as: python singlePageImageFetcher.py LOCAL-PATH-TO-SAVE-THE-DATA

###Sample Output:
All the roman art data on pisa is downloaded with this two tools.



##Simple Annotator for Roman Statue Images 
An OpenCV high-gui based simple image annotation tool. This tool is used to annotate a set of images with a variable. The possible value of the variable can be two or more. The annotation results will be stored in a text file.

###related code:

dataCollecting/simpleAnnotationTool.py

###Dependencies
1. Python 2.7
2. OpenCV python bindings (pip install pyopencv or apt-get install python-opencv)

###Deployment instructions:
Store all the images in a single directory

To run: python simpleAnnotationTool.py Path-to-the-directory-containing-images Path-to-the-parameter-file 

The parameter file should look like:

key1 tag1
key2 tag2
...

e.g.

1 male
2 female
3 notSure

The keys can be simple characters on the key board like 0,1,2,a,b,c, but q can not be used.


Press key to annotate (map from keys to tags is in the parameter file)

press space to skip current image

The annotation file will be stored in the same directory with the images

The file name is annotation\_\_numberOfLabel-label0-label1-...-lastLabel\_\_year-month-day\_hh-mm-ss\_\_randomNumber.txt

e.g. annotation\_\_3-male-female-notSure\_\_2016-05-16\_19-40-06\_\_54.txt

The random number from 0 to 99 is to avoid possible conflict

Press q to quit



##Roman Statue Face Pre-process Pipeline

This pre-processing pipeline tool takes a list of images as input. It detects faces and face landmarks, then crop and align the largest face on each image.

It supports 2 alignment mode: affine warp and face frontalization which can be selected in the code with the "MODE" swich. The usage: Give the path to the trained shape predictor model as the first argument and then the directory containing the facial images. The code will detetct faces, align the largest one, crop it and save it with "\_crop.jpg" suffix.

###related code:

ML4RomeArtf/facePrepPipeline.py

This pre-processing pipeline support 2 mode: affine warp and face frontalization which can be selected in the code with the "MODE" swich. 

###Dependencies
1. Python 2.7
2. OpenCV python bindings (pip install pyopencv or apt-get install python-opencv)
3. skimage, numpy(apt-get install python-skimage python-numpy / or install with pip)
4. dlib (pip install dlib)
5. dlib's trained shape predictor: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
6. All the dependencies of the shape frontalization part.

###Deployment Instructions:
1. Download the shape predictor and unzip it.
2. Set the MODE variable in the code.
3. Run the code as: "python facePrepPipeline.py path-to-the-downloaded-shape-predictor path-to-directory-containing-images". The successfully processed image will be saved to origin-image\_crop.jpg or origin-image\_frontal.jpg.

###Sample Output
You can see these two webpages (http://pisa.vrnewsscape.ucla.edu/roman/rome101/show4iccv15.html
http://pisa.vrnewsscape.ucla.edu/roman/rome101/show4us10k.html ) for samples. The images on right side of the page is the pre-processed results.

##Face Frontalization

###related code:

FaceFrontalisation/*

I modified the code and parameters from https://github.com/ChrisYang/facefrontalisation and use it for the pre-processing pipeline of statue faces.

###Dependencies
1. Python 2.7
2. OpenCV python bindings (pip install pyopencv or apt-get install python-opencv)
3. scipy, pyplot (can be installed with pip)
4. dlib (pip install dlib)
5. dlib's trained shape predictor: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
6. Pre-defined 3D model: already contained in the repository.

###Deployment Instructions:
It is recommended that you only use this part in the face pre-process pipeline. When the MODE is set to 1, the pipeline will automatically use this part.


##Meta Keywords Analysis for Roman Statue Data

Regarding the database on laststatues.classics.ox.ac.uk: Using the text in each attributes' field, I analyzed the frequency of each words. I also calculated the co-occurrence and the correlation coefficient for the words occurrence vectors. The results are here (https://github.com/mfs6174/GSoC2016-RedHen/tree/master/ML4RomeArt/keywordResults).

I developed a script for automatic classifier training with the 179 dimensional features for all the keywords which occur > 10 times. Because most keywords lead to unbalanced classification problems, I use informative down-sampling on majority classes. Each ensembled classifier is composed of a bag of SVM classifiers , which are trained with the minority class samples and some majority class samples in the similar size with minority class samples sampled with bootstrapping and will vote to make predictions. The parameters for SVM were choose by grid search cross validation. You can see the results here ( https://github.com/mfs6174/GSoC2016-RedHen/blob/master/ML4RomeArt/keywordResults/keywordClassifiersROC.csv ). The Area Under the Curve (AUC) scores and the cross validation standard deviations for each keyword are shown. I selected the keywords with AUC >= 0.75 and STD <=  0.15 which I believe are reasonably classified. You can see it here ( https://github.com/mfs6174/GSoC2016-RedHen/blob/master/ML4RomeArt/keywordResults/keywordClassifiersROCGood.csv ). I think among those, the word "empress" is interesting. The reason of some of the keywords can be classified may be the keywords only occur in male or female statues, and the gender can be easily classified.


**This part of the code is mainly used with the database on laststatues.classics.ox.ac.uk in this project. But if some other databases with details meta-data (like keywords) are available in the future, the code can be used with them too (need to change the field mask which is hard-coded in the code).**
###related code:

1. ML4RomeArt/keywordAnalysis.py

2. ML4RomeArt/keywordClassifiersFactory.py

###Dependencies
1. Python 2.7
2. python-sklearn
3. pandas (apt-get install python-pandas)

###Deployment Instructions:

####keywordAnalysis
1. Firstly, you need to set the RESULT\_PATH variable in the code to the path you want to save the analysis results and set the FIELD\_MASK list to the meta-data field you want to include in the analysis.
2. Run the analysis code as: "python keywordAnalysis.py path-to-the-jsons-containing-meta-data". The json files are downloaded by the database crawler.  The code will produce wordCount.csv containing the keyword frequency information and cooccurrence.csv/correlationCoef.csv/correlationSort.csv/absoluteCorrelationSort.csv. All the files are correlation analysis about the keywords.
3. Set the "maxDf" and "minDf" variables first (or just use the default value)  and run the analysis code as: "python keywordAnalysis.py path-to-the-jsons-containing-meta-data 1" , the code will procude "TF-reduce.pkl". This file will be used in the automatically generating of keyword classifiers.

####keywordClassifiersFactory
1. Please set the "MODEL\_PATH", "KEYWORD\_PATH" variables in the keywordClassifiersFactory.py to give the input and output directory.
2. Run the code as "python keywordClassifiersFactory.py pickled-Word-Count-file(TF-reduce.pkl) number-of-database \[face-feature-of-one-database path-to-the-dir-with-json-containing-meta-data-of-one-database end-pattern-of-one-database\] ". The face feature is generated with the Geometry Feature Extractor (next part). The end pattern is the pattern between the object id and the suffix. The images with the same object id is the different photos for a same object. For the "ancientrome.ru" database, the pattern is ".jpg". For the "laststatues.classics.ox.ac.uk" database, the pattern is "\_".
3. For each keyword with occurrence > 10 times, the code will train a classifier to predict if one image has the keyword and do cross-validation with it. The code will produce "keywordClassifiersROC.csv" showing the Area Under the Curve (AUC) scores and the cross validation standard deviations.  The keywords with AUC >= 0.75 and STD <=  0.15 are believed as  reasonably classified and put into "keywordClassifiersROCGood.csv". The trained classifiers are saved to "MODEL\_PATH".


###Sample Output
Please see the  ML4RomeArt/keywordResults directory in the repository.

##Geometry Feature Extractor for Roman Statue Faces

At first, I developed a feature extractor to extract structural features from statue faces. I use dlib's facial landmark detector to get 68 landmarks for each detected face.  I designed 43 geometry features based on some other reference papers and my own understanding. Some of them like height and width of faces, noses and eyes are similar with features in Jungseock's ICCV paper. The features also include some ratio, angle and elliptocytosis eccentricity values. The normalized coordinates of the landmarks are also used as features. The number of feature dimensions is 179 in total. I found that  the frontal face detector in dlib can detect lots of "not so frontal" faces. They are weird after the alignment with 2D affine transformation and lead to incorrect feature values. So I decide to use features extracted from frontalized faces  and  I modified the feature extractor to add more feature dimensions like the distance between every landmarks pair and so on. The dimension of the structure feature I am using now is 2200+. The large dimension should allow me to just train linear SVR instead of SVR with rbf kernel to get some fine results. So I can use Liblinear instead of LibSVM to speed up the grid search cross validation. 

###related code:

ML4RomeArt/faceStructureFeatures.py

###Dependencies
Same with Face Pre-process Pipeline part.


###Deployment Instructions:
1. Please set the "NORM\_MODE" and "LOAD\_MODE" variables in the code. The "NORM\_MODE" controls whether to use the face frontalization to generate features or just uese the 2D affine warp. The "LOAD\_MODE = 1" is used to process the US10K dataset with a text file containing the image list. If you are generating features for some normal image collection, set "LOAD\_MODE = 0". 
2. Run the code as: "python faceStructureFeatures.py path-to-dir-containing-images file-to-save-the-features". You can use the saved feature file in the keywordClassifiersFactory (last part) and the transfer learning (next part). The code should be ran with both the statue face images and the mordern photo face images.

###Sample Output:

The output is saved in binary format and used for training classifiers. It is hard to directly examine the output.


##Transfer Learning and Result Analysis for Social Value Inference of Roman Statue Faces
I use the dataset for my mentor's ICCV paper on social dimensions of faces of politicians and the US10K dataset. The datasets contain facial photographs labeled with visual variables and social perceptions. I train regressors for these variables with geometry features on this dataset and use it to predict the variables for the statue faces (transfer learning). 

I trained SVR regressors for each social perception with the Jungseock's dataset. The hyper-parameters were choose by cross validation grid search.  The average MSE is about 0.08.  The social perception with the largest cross validation MSE is "Old". Its MSE is 0.18.  I applied the regressors to the laststatues database to get the predicted social perception values. The correlation coefficients between keywords occurrence and social perception values were calculated. You can see the results here (https://github.com/mfs6174/GSoC2016-RedHen/blob/master/ML4RomeArt/keywordResults/correlationKeywordsSocialEval.csv). I also generated the most correlated pairs by sorting the coefficients with absolute value. You can see the results here ( https://github.com/mfs6174/GSoC2016-RedHen/blob/master/ML4RomeArt/keywordResults/keywordSocialCorrPairsSortedMoreThan10.csv ). Most highly correlated pairs are gender related. Some material, geographic location or time-period related words are also in some pairs with absolute values of correlation coefficients > 0.1. 

 I implemented more evaluation metrics for the social attributes regressors like the pair-wise classification accuracy ( PWCA) in  my mentor's ICCV paper. I adjusted some details on the feature extraction and the trained the regressors again with MAPE or PWCA as the validation metrics. The average accuracy is about 60% over dimensions and 71% for Energetic. I also tried to train XGBoost regressors but they cannot over-perform SVR.

I implemented Shapiro-Wilk test for normality to test the social attributes values. The annotation values and predicted values for photos are normally distributed for most dimensions except Trustworthy and Old. But the predicted values for statues are never normally distributed.

I have got the boxplot figures for annotation values and predicted values for photos and predicted values for statues (this is the order on the figures) . You can compare the photos and the statues with the boxplot.

As for the us10k dataset,  you can see the correlation coefficients analysis like I did on the last dataset here ( https://github.com/mfs6174/GSoC2016-RedHen/blob/master/ML4RomeArt/keywordResults/us10K_keywordSocialCorrPairsSorted.csv ) and here ( https://github.com/mfs6174/GSoC2016-RedHen/blob/master/ML4RomeArt/keywordResults/us10K_correlationKeywordsSocialEval.csv) .

###related code:

1. ML4RomeArt/socialLearningFromStructure.py

2. ML4RomeArt/htmlShow.py

3. ML4RomeArt/dataLoder.py

###Dependencies
1. Python 2.7
2. python-sklearn
3. pyplot
4. scipy
5. xgboost (pip install xgbmagic )
6. outlier\_utils (pip install outlier\_utils)
7. pandas

###Deployment Instructions:
1. Set all the variables on the begining of the code to specify the related input and output path you are using and the suffix for the dataset. If the us10k dataset is being processed, set the "TRAIN\_DATA\_MODE" as 1. The code can be used with iccv15 dataset or us10k dataset to produce analysis results for each dataset.
2. Training the regressors for social attributes  with modern portraits dataset: Run "python socialLearningFromStructure.py train annoatation-data-of-the-dataset saved-face-feature-file-of-the-dataset". For the iccv15 dataset, the "annoatation-data-of-the-dataset" is the ".mat" file. For the us10k dataset, it is the "psychology-attributes.txt". These files are provided by Jungseock and not included in the repository.
3. Validating the trained regressors: Run "python socialLearningFromStructure.py validation annoatation-data-of-the-dataset saved-face-feature-file-of-the-dataset path-to-output-the-most-and-least-face-images". The code will print the validation metrics to the screen and csv file. In additional, the code will select the face images with the highest or the lowest predicted values of each social attributes and save them to "path-to-output-the-most-and-least-face-images" directory.
4. Transfering the trained regressors to Roman statue face images: Run "python socialLearningFromStructure.py transfer saved-statue-face-feature-file file-to-save-the-social-evaluation-result-of-statues  saved-photo-face-feature-file label-file". The last argument is needed only when transfering the us10k dataset.  The regressors will be applied to the statue faces and produce predicted social attribute values. The saved social evaluation results will be used in the next functions.
5. Analyzing the correlations between predicted social attribute values and keywords of the statue: Run "python socialLearningFromStructure.py analysis pickled-Word-Count-file(TF-reduce.pkl produced by the keyword analysis part) file-with-saved-social-evaluation-result".The code will produce "correlationKeywordsSocialEval.csv" containing the correlation coefficients between predicted social attribute values and keywords of the statues.
6. Generating dimension analysis visualization figures: Run the code as "python socialLearningFromStructure.py pca pickled-Word-Count-file(TF-reduce.pkl produced by the keyword analysis part) file-with-saved-social-evaluation-result-of-statues validation-result(produced by function 3)". The code will produce a visualization figure for the dimension analysis with the social attributes values and keywords occurrence. The figure file name is "DATASET\_PREF+\_Attributes_Keywords.png" and you can change the figure's resolution with "FIG\_SIZE" variable in the code.
7. Selecting statue faces with the highest or the lowest predicted values of each attribute for analysis: Run the code as "python socialLearningFromStructure.py argsort file-with-saved-social-evaluation-result-of-statues output-directory path-to-directory-containing-preprocessed-face-images".
8. Comparing the predicted values of social attributes between roman statue faces and modern portraits in the dataset and producing boxplot for visualized analysis: Run the code as "python socialLearningFromStructure.py compare file-with-saved-social-evaluation-result-of-statues saved-face-feature-file-of-the-dataset annoatation-data-of-the-dataset". The code will compare the distribution of the predicted values between roman statue faces and modern portraits, perform some hypothetical test and generate boxplot figures to the figure path.
9. Generating html webpages to show the predictions: Run "python socialLearningFromStructure.py show file-with-saved-social-evaluation-result-of-statues output-path text-file-containing-the-list-of-the-images-to-show" and "python htmlShow.py html-root-dir(the output-path in tje last command) html-file-name". The webpage " http://pisa.vrnewsscape.ucla.edu/roman/rome101/show4iccv15.html " is generated with this function.

###Sample Output:
Please see the  ML4RomeArt/keywordResults directory in the repository and refer to the results section of this documentation.


##Convolutional Network Classifiers for Roman Statue Faces (Training and Testing Code)

I have developed a tool which can automatically build image classification dataset from the keywords. I built two datasets. The first one is gender dataset which is merged from one built from laststatues database with the gender field and another one built from ancientrome database with finding words like "man", "male", "female". It has 1805 images (1417 male, 388 female). The second one is beard dataset which is built from laststatues database with the beard field. It only has 704 images ( 88 long-bearded, 169 short-bearded, 201 stubble-bearded and 246 clean-shaven ).

I have trained convolutional neural network classifiers for the two datasets. The size of the dataset are quite small for image classification problem, so the large networks won't converge. I use a very small network with 4 VGG-style convolutional layers and about 390k parameters with heavy regularization.

The 10 folds for cross validation are split by objects instead of images (one object has more than one image) like the common face related tasks. The different objects in the website are still be possible to be the same statue or statues of the same person, so you need to know the cross validation results are still more optimistic than the real performance. The gender dataset is not balanced so the female images are over-sampled to avoid lazy classifier.

The best results I have got for now is 83.16% average accuracy  for binary gender classification and 55.11% for 4-classes beard classification.

I also used the production year annotations provided by my mentors to build a dataset and train a CNN classifier. The classifier will predict if the statue is produced before 96 AD or not. The accuracy is 67.98%.

###related code:

1. ML4RomeArt/buildDatasetFromKeywords.py

2. ML4RomeArt/makeYearsDataset.py

3. ML4RomeArt/CNN4Portraits_keras.py


##Deconvolutional Network for visualization of the learned classifers.

I have implemented the deconv network described in this post and Zeiler's ECCV paper. The code I found on Github is just not the right implementation. So I had to rewrite it according to the paper. The deconvolution results are shown in result section.

###related code:

1. ML4RomeArt/DeconvTool/*

2. ML4RomeArt/sampleSubsetImages.py






#Results

In this section, some analysis results (most are visualized) are listed.

##The html showing the photos, frontalized faces and the social values predicted by the transfered regressors

The photos are choose manually from rome101.com database. They are realistic and the people are famous. The names of the people are in the image file name.   

http://pisa.vrnewsscape.ucla.edu/roman/rome101/show4iccv15.html
http://pisa.vrnewsscape.ucla.edu/roman/rome101/show4us10k.html
##Deconvolution visualization for the CNN classifiers

For the gender classifier, the "filter 0" is keeping the male activation and the "filter 1" is keeping the female activation. As far as I can see, it seems that there are some differences with the mouths ( woman's lips are rounded) , noses and eyes ( there are more shape shadow with man's noses and eyes).



For the  beard classifier, filters 0/3 are keeping long-bearded/clean-shaven's activations.



For the production year classifier, the "filter 0" is keeping the "early" (before 96 AD) activation and the "filter 1" is keeping the "late" (after 96 AD)  activation. You can see the patterns showing frowning with contracted corrugator muscles, and patterns about the iris and pupil of the eye in the "late" activation. 



1. Beard: https://github.com/mfs6174/GSoC2016-RedHen/blob/master/ML4RomeArt/Figures/beard_1_0.png https://github.com/mfs6174/GSoC2016-RedHen/blob/master/ML4RomeArt/Figures/beard_1_3.png 

2. Gender: https://github.com/mfs6174/GSoC2016-RedHen/blob/master/ML4RomeArt/Figures/gender_1_0.png https://github.com/mfs6174/GSoC2016-RedHen/blob/master/ML4RomeArt/Figures/gender_1_1.png 

3. Production Year: https://github.com/mfs6174/GSoC2016-RedHen/blob/master/ML4RomeArt/Figures/production_year_0.png https://github.com/mfs6174/GSoC2016-RedHen/blob/master/ML4RomeArt/Figures/production_year_1.png 

##Boxplot figures comparing the predicted social variables between roman statues and modern portraits of American people.

1. ICCV15 dataset with ancientrome.ru database https://github.com/mfs6174/GSoC2016-RedHen/blob/master/ML4RomeArt/Figures/iccv15_arru_compare.png 

2. ICCV15 dataset with laststatues.classics.ox.ac.uk database https://github.com/mfs6174/GSoC2016-RedHen/blob/master/ML4RomeArt/Figures/iccv15_lsuk_compare.png 

3. US10K dataset with  ancientrome.ru database https://github.com/mfs6174/GSoC2016-RedHen/blob/master/ML4RomeArt/Figures/us10K_arru_compare.png 

4. US10K dataset with   laststatues.classics.ox.ac.uk database https://github.com/mfs6174/GSoC2016-RedHen/blob/master/ML4RomeArt/Figures/us10K_lsacuk_compare.png 

##Dimension analysis figures between the predicted social variables and the keywords occurance for the roman statues database.

Here the eventual goal is to understand the interrelationships between the keywords in the basis of the exhibited traits. For example, we may see two keywords very close in this space like "emperor" and "crown", both of which might lie on the area of some specific social attributes.


1. https://github.com/mfs6174/GSoC2016-RedHen/blob/master/ML4RomeArt/Figures/iccv15___Attributes_Keywords_2000.png

2. https://github.com/mfs6174/GSoC2016-RedHen/blob/master/ML4RomeArt/Figures/us10k___Attributes_Keywords_2000.png 

#Still To-Do

1. Improve the face frontalization algorithm to get more valid results.

2. Train the production year classifier with more annotated images to get better results.

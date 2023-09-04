# Histopathological Image Classification

This repository comprises my solution to a datachallenge organised at Telecom Paris at the end of a Machine Learning course, before delving deeper into Deep Learning. It contains :
* A [jupyter notebook](./Histopathological_Image_Classification.ipynb) detailing my classification algorithm and the choices taken to develop it
* A [presentation](./Restitution_Datachallenge.pdf) of the approach in PDF format, used to explain my reasoning to the whole promotion at the end of the challenge
* The [images](./data-challenge) used as dataset for the challenge


## Context of the project

The goal of the project was to classify breast cancer histopathological images into 8 different classes, each identified by different letters in the image filename. An overview of the different classes involved is given in the table below :
<table>
  <thead>
    <tr>
      <th align="center">Class ID</th>
      <th align="center">Identifying letters</th>
      <th align="center">Tumor full name</th>
      <th colspan="3" align="center">Images sample</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">1</td>
      <td align="center">F</td>
      <td align="center">Fibroadenoma (benign)</td>
      <td align="center"><img src="./data-challenge/Train/SOB_B_F-14-29960AB-100-002.png" alt="F1" style="max-width: 100%;"></td>
      <td align="center"><img src="./data-challenge/Train/SOB_B_F-14-23060CD-100-011.png" alt="F2" style="max-width: 100%;"></td>
      <td align="center"><img src="./data-challenge/Train/SOB_B_F-14-21998CD-100-025.png" alt="F3" style="max-width: 100%;"></td>
    </tr>
    <tr>
      <td align="center">2</td>
      <td align="center">DC</td>
      <td align="center">Carcinoma (malignant)</td>
      <td align="center"><img src="./data-challenge/Train/SOB_M_DC-14-10926-100-003.png" alt="DC1" style="max-width: 100%;"></td>
      <td align="center"><img src="./data-challenge/Train/SOB_M_DC-14-11031-100-014.png" alt="DC2" style="max-width: 100%;"></td>
      <td align="center"><img src="./data-challenge/Train/SOB_M_DC-14-13412-100-003.png" alt="DC3" style="max-width: 100%;"></td>
    </tr>
    <tr>
      <td align="center">3</td>
      <td align="center">PC</td>
      <td align="center">Papillary Carcinoma (malignant)</td>
      <td align="center"><img src="./data-challenge/Train/SOB_M_PC-14-9146-100-015.png" alt="PC1" style="max-width: 100%;"></td>
      <td align="center"><img src="./data-challenge/Train/SOB_M_PC-14-15687B-100-013.png" alt="PC2" style="max-width: 100%;"></td>
      <td align="center"><img src="./data-challenge/Train/SOB_M_PC-14-19440-100-001.png" alt="PC3" style="max-width: 100%;"></td>
    </tr>
    <tr>
      <td align="center">4</td>
      <td align="center">PT</td>
      <td align="center">Phyllodes Tumor (benign)</td>
      <td align="center"><img src="./data-challenge/Train/SOB_B_PT-14-21998AB-100-012.png" alt="PT1" style="max-width: 100%;"></td>
      <td align="center"><img src="./data-challenge/Train/SOB_B_PT-14-21998AB-100-041.png" alt="PT2" style="max-width: 100%;"></td>
      <td align="center"><img src="./data-challenge/Train/SOB_B_PT-14-29315EF-100-005.png" alt="PT3" style="max-width: 100%;"></td>
    </tr>
    <tr>
      <td align="center">5</td>
      <td align="center">MC</td>
      <td align="center">Mucinous Carcinoma (malignant)</td>
      <td align="center"><img src="./data-challenge/Train/SOB_M_MC-14-10147-100-004.png" alt="MC1" style="max-width: 100%;"></td>
      <td align="center"><img src="./data-challenge/Train/SOB_M_MC-14-12773-100-008.png" alt="MC2" style="max-width: 100%;"></td>
      <td align="center"><img src="./data-challenge/Train/SOB_M_MC-14-12773-100-021.png" alt="MC3" style="max-width: 100%;"></td>
    </tr>
    <tr>
      <td align="center">6</td>
      <td align="center">LC</td>
      <td align="center">Lobular Carcinoma (malignant)</td>
      <td align="center"><img src="./data-challenge/Train/SOB_M_LC-14-16196-100-004.png" alt="LC1" style="max-width: 100%;"></td>
      <td align="center"><img src="./data-challenge/Train/SOB_M_LC-14-12204-100-031.png" alt="LC2" style="max-width: 100%;"></td>
      <td align="center"><img src="./data-challenge/Train/SOB_M_LC-14-16196-100-003.png" alt="LC3" style="max-width: 100%;"></td>
    </tr>
    <tr>
      <td align="center">7</td>
      <td align="center">A</td>
      <td align="center">Adenosis (benign)</td>
      <td align="center"><img src="./data-challenge/Train/SOB_B_A-14-22549CD-100-002.png" alt="A1" style="max-width: 100%;"></td>
      <td align="center"><img src="./data-challenge/Train/SOB_B_A-14-22549AB-100-002.png" alt="A2" style="max-width: 100%;"></td>
      <td align="center"><img src="./data-challenge/Train/SOB_B_A-14-22549G-100-008.png" alt="A3" style="max-width: 100%;"></td>
    </tr>
    <tr>
      <td align="center">8</td>
      <td align="center">TA</td>
      <td align="center">Tubular Adenona (benign)</td>
      <td align="center"><img src="./data-challenge/Train/SOB_B_TA-14-3411F-100-004.png" alt="TA1" style="max-width: 100%;"></td>
      <td align="center"><img src="./data-challenge/Train/SOB_B_TA-14-3411F-100-007.png" alt="TA2" style="max-width: 100%;"></td>
      <td align="center"><img src="./data-challenge/Train/SOB_B_TA-14-3411F-100-012.png" alt="TA31" style="max-width: 100%;"></td>
    </tr>
  </tbody>
</table>

All images are extracted from the **[BreakHis dataset](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)**. It is used as a benchmark in many medical imaging competition, but often only for binary classification (identifying if a tumor is benign or malignant). Histologically benign is a term referring to a lesion that does not match any criteria of malignancy – e.g., marked cellular atypia, mitosis, disruption of basement membranes, metastasize, etc. Normally, benign tumors are relatively “innocents”, presents slow growing and remains localized. Malignant tumor is a synonym for cancer: lesion can invade and destroy adjacent structures (locally invasive) and spread to distant sites (metastasize) to cause death. <br>
The samples present in the dataset were collected by SOB method, also named partial mastectomy or excisional biopsy. This type of procedure, compared to any methods of needle biopsy, removes the larger size of tissue sample and is done in a hospital with general anesthetic.

The annoted dataset (Train folder) used in this challenge consists of **422 images** randomly extracted from BreakHis, the number of images to classify (Test folder) contains **207 images**. <br>
The images are of dimension 700x456 or 700x460 pixels in RGB format. <br>
The **metric** used to rank the submissions in this datachallenge was the **[F1-score](https://en.wikipedia.org/wiki/F-score)**, which gives equal importance to precision and recall. The accuracy of the submitted classifiers was also displayed, but not used for scoring.

## Key difficulties

Three main difficulties had to be addressed during this datachallenge :
* **Small size of the dataset** <br>
The state of the art for histopathological image classification is currently composed of methods based on Deep Learning, which require a consequent number of images to train from scratch. The small number of training images made this kind of approach unreasonable, so I instead opted for more traditionnal image classification techniques based on feature extraction and classical machine learning (e.g. SVM, Random Forest, Boosting, Logistic Regression). The small number of images could alos be leveraged by using more computationally intensive but rigorous cross-validation methods, such as **Leave-One-Out**.
* **Class imbalance** <br>
As can be seen on the following graph, the repartition of images in each class is heavily imbalanced.
<p align="center"><img src="./images/Class_imbalance.png" /></p>

This can bias the model towards more represented classes, and make the learning of general features harder for the least represented classes (LC and TA in this case).

* **Multi-labels images** <br>
From the images alone, finding the dataset from which they had been extracted was not difficult. However, the annotations for each image are made by experts who have access to more than just the histopathological images, and who know that several images actually come from a single slice, from a single patient - information which is not available on the test set. Finally, several types of tumor may be present in a single image. For all these reasons, many images can actually be classified into several classes. There is an especially high number of such cases for classes **DC** (Carcinoma) and **LC** (Lobular Carcinoma), as can be seen in the following examples where the same image was found in the Train folder, in the Test folder and in the BreakHis dataset with a name different from the Train Folder


| Image | Name in the Test folder | Name in the Train folder | Alias name in the BreakHis dataset | Possible classes |
|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
![SLO_01](./data-challenge/Test/SOB_18.png)  |  SOB_18 | SOB_M_LC-14-13412-100-026  |  SOB_M_DC-14-13412-100-026 | LC or DC
![SLO_01](./data-challenge/Test/SOB_28.png)  |  SOB_28 | SOB_M_LC-14-13412-100-025  |  SOB_M_DC-14-13412-100-025 | LC or DC
![SLO_01](./data-challenge/Test/SOB_29.png)  |  SOB_29 | SOB_M_LC-14-13412-100-001  |  SOB_M_DC-14-13412-100-001 | LC or DC

Even with a perfect classifier, getting a perfect score is thus dependent on luck !

## Results
![](./images/screenshot-results.png)


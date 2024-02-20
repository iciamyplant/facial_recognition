# Projet : 
1. Detection du visage
2. coder un algo qui classifie le visage : Nom, genre H/F = classification, le but est d'associer un label à une image + âge = regression pour prédire un nombre
3. coder un algo de détection d'objets dangereux ?
4. utiliser les librairies ou des modèles pré-entrainés pour l'expérience finale

# 1. Detection du visage

**Computer vision** =  discipline informatique permettant de construire des systèmes qui obtiennent des informations à partir d'images. Les images étant un ensemble de nombres avec une certaine structure, une matrice. Dans le cas d'une image en niveaux de gris, chaque nombre (pixel) représente une intensité différente allant de 0 (blanc) à 1 (noir pur) ==> de 0 à 255. Si nous travaillons avec des images colorées, alors la même matrice que ci-dessous, mais en 3 fois (RVB), chaque matrice représentant une intensité de couleur différente ==> 0.0.0 à 255.255.255

<p align="center">
<img width="450" src="https://github.com/iciamyplant/facial_recognition/assets/57531966/cf9ced56-1a3c-4554-8a00-8a64792d9d13">
<p align="center">

**Face detection** = computer vision task in which we detect the presence of human faces and its location (x1, y1, x2, y2) within an image or a video stream. C'est un problème de regression, le but est de prédire les coordonnées (continues) du rectangle entourant le visage, cordonnée en haut à gauche (x1,y1) et coordonnée en bas à droite (x2,y2). Plusieurs obstacles à la détection : occlusion, orientation of face, expression, lighting, accessories. Au fil de temps nombreuses avancées : Viola Jones Algorithm, Histogram of oriented gradients (HOG), FDDB, *advent of deep learning techniques 2012, more robust face detectors have been developed*, Anootated Faces in the Wild, Pascal Face, SSD (slower then HOG), MTCNN (CNNs connected in a cascated manner, not fast for real time applications), UFDD, RetinaFace, MediaPipe (super real time performances), YuNet. 

**Feature** = Pour détecter des visages (ou objets), l'ordinateur s'appuie sur des features = is a piece of information in an image that is relevant to solving a certain problem. It could be something as simple as a single pixel value, or more complex like edges, corners, and shapes. You can combine multiple simple features into a complex feature. Applying certain operations to an image produces information that could be considered features as well. Computer vision and image processing have a large collection of useful features and feature extracting operations.


### Viola-Jones Algorithm

Nous allons implémenter un système de detection faciale utilisant Viola-Jones algorithm, easiest face recognition system you can create, but there are more advanced techniques to do the same project. L'objectif ici est surtout d'avoir un aperçu des bases du fonctionnement des systèmes de détection d'objets. Viola-Jones Algorithm was created mainly to work with frontal faces and operate with grayscale images, donc input image doit etre in grayscale. À partir de cette image, l’algorithme examine de nombreuses sous-régions plus petites et tente de trouver un visage en recherchant des caractéristiques spécifiques dans chaque sous-région. Il doit vérifier de nombreuses positions et échelles différentes, car une image peut contenir de nombreux visages de différentes tailles
[Official Paper](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf)
[Bonne vidéo](https://www.youtube.com/watch?v=p9vq90NYHMs)
[Tutoriel](https://realpython.com/traditional-face-detection-python/)

Etapes :
1. Selecting Haar-like features
2. Creating an integral image
3. Running AdaBoost training
4. Creating classifier cascades

#### 1. Haar feature selection is used to detect the features of a face

All human faces share some similarities : eye region is darker than the bridge of the nose, the cheeks are brighter than the eye region. A simple way to find out which region is lighter or darker is to sum up the pixel values of both regions and comparing them. The sum of pixel values in the darker region will be smaller than the sum of pixels in the lighter region. Plus il y a une différence, plus y a du constraste entre les deux régions. C'est sur ce principe que Alfred Haar created features detector.

Une Haar feature prend une partie rectangulaire d'une image et divise ce rectangle en plusieurs parties. 4 basic types of Haar-like features : 

Alfred Haar created features detector, and these detectors will look for specific places where they can fit. 

Haar features are rectangular regions masked over an image. Within each rectangle, the summation of pixels is calculated and then the difference between the black and white regions is calculated. For exemple, eyes region is darker than cheeks. Then, renvoit une valeur totale qui doit être supérieure à un seuil pour considérer qu'il y a bien un visage.

<p align="center">
<img width="450" src="https://github.com/iciamyplant/facial_recognition/assets/57531966/594b9497-ff00-403a-ad79-86e54dd49637">
<p align="center">

Mais difficile de faire les calculs en testant les features sur toute l'image. In a 24*24 resolution image, there are 180k+ features, not all are usefull. So the goal is to only use the useful features and reduce processing time. Integral image is a solution to fasten the processing time

#### 3. an integral image is calculated to fasten the processing : in an integral image, every pixel is the summation of the pixels above and to the left of it (voir vidéo explication avec exemple).

#### 4. adaboost training to properly locate and train the features and pour améliorer the processing time

All these feature detectors by their own are unable to perform good predictions, so they are considered weak classifiers. But what if we could use a lot of these weak classifiers at the same time in order to create a strong one ?
Not all features are needed. We need to eliminate the undesired features to fasten the process and get accurate results. We need to train the features on the images to only use the right features in the right place. 
- provide lots of facial image data to the algorithm training and non-facial images for differenciation.
- Between the useful features, not all of them are of the same significance. Therefore, the creators of Viola Jones proposed what is called a strong feature combining weak features with their respective weights (= on combine des features faibles avec leurs poids respectifs, ce qui crée une feature forte). Combining features together is what make them strong. How can we find the weights of each feature ?
- We give the system facial images indicating that they are positive examples and non facial images
- Initialize weights for each image
- for each classifier, we normalize the weights. A classifier with one feature is used and trained on all images and the error is computed. If a face is detected on a facial image, the error is 0. Otherwise error is 1. Inversement for non facial images. The error is then multiplied by the significance of the image. The lower error is chosen, and the weights are updated

==> it's mathematical operations but the idea is that we are trying to classify images as faces or non-faces, so essentially what this algorithm does is to penalize more those misclassified images (false positives and negatives) by incrementing its weights (this is their importance), so the algorithm is going to look for features that better adapt to the picture. Algorithm tries to reduce the error, iteration after iteration

==> it's mathematical operations but the idea is that we are trying to classify images as faces or non-faces, donc l'algo va 
- run et dire si telle ou telle image est un visage ou non
- calculer un taux d'erreur basé sur la somme pondérée des instances où la classification a été mauvaise, le but étant de réduire ce taux d'erreur
- mettre à jour les poids (l'importance de chaque feature) selon un processus itératif, en pénalisant davantage les images mal classées (faux positifs et négatifs)
Dans le but donc de rechercher des caractéristiques qui s'adaptent mieux à l'image. 

5. cascading to distinguish whether an image contains a face or not
Après avoir performing the adaboost training, on a la first and second most important features. Feature on the eyes and chins, where the eyes are darker than chins is the most significant. The second one indicates that the bridge of the nose is brighter than its surroundings. A fast way to check if the image contains a facial feature is to cascade the classifiers. If the first feature is approuved then it moves on for the second classifier until all of the features are approved. Then a face is detected. Much faster then trying all of the face detecting features.

![résumé du process](https://github.com/iciamyplant/facial_recognition/assets/57531966/3da197a5-8525-4aeb-82cb-a034d64e84dc)



### Viola-Jones Algorithm implementation

```
pip install scikit-image
pip install scikit-learn
```


### CNN Model implementation

[Tutorial](https://realpython.com/face-recognition-with-python/#prerequisites)

### Face detection using pre-trained model

[Face detection using pre-trained model - Google Collab](https://colab.research.google.com/github/dortmans/ml_notebooks/blob/master/face_detection.ipynb)













# 1. Les réseaux de neurones convolutifs

Tous ces algos se basent sur les mêmes concepts, les réseaux de neurones convolutifs.

Ici on va utiliser du Deep Learning, c’est un pan de l’intelligence artificielle qui se base sur l’apprentissage profond et les réseaux de neurones, et ca marche particulièrement bien sur les données non structurées (information that is not arranged according to a preset data model or schema, pas besoin de la traiter avant utilisation) type image ou texte. La on va pas faire un réseau classique avec plein de couches de neurones successives, on va faire ce qu’on **appelle un CNN (Convolutional Neural Network) qui se base sur des filtres de convolutions** et qui sont particulièrement adpatés aux images.

**Un filtre de convolution** c’est une matrice qui va traverser toute l’image de gauche à droite et de bas en haut en appliquant successivement des opérations de convolution, i.e. des sommes de produits termes à termes. Quand un filtre se propage sur une image, il somme les résultats de ces opérations de convolution sur les différents canaux de couleur. Cette logique est généralisable à toute donnée d’input qui est composée de plusieurs couches de matrices à 2 dimensions.

![filtreconvolution](https://github.com/iciamyplant/facial_recognition/assets/57531966/915a15cc-d503-4b3b-a1a7-40e28b520edc)

On appelle ca un filtre car ca va modifier l’image en gros, par exemple t’as des filtres qui affichent le gradient ou qui détectent les bords, ce genre de trucs.

![applicationfiltre](https://github.com/iciamyplant/facial_recognition/assets/57531966/660e37d0-b299-417f-9818-77c54c1ffb7d)

On va donc appliquer plusieurs couches de convolutions les unes après les autres, ce qui va en fait correspondre à la phase de "feature extraction”. On va ensuite applatir notre dernière image, utiliser le vecteur qui nous est donné dans un réseau dense et confronter la prédicition qui est faite avec la réalité. Encore une fois on a une fonction de perte à minimiser, le but est qu’au fur et à mesure, les prédicitions soient de plus en plus proches de la réalité en optimisant les paramètres des filtres à chaque fois qu’on passe dans le réseau.

L’idée générale est que le réseau va apprendre par lui même ou regarder sur l’image grâce aux différents filtres, en fonction de la classification/régression qu’on a envie de faire.

![process](https://github.com/iciamyplant/facial_recognition/assets/57531966/77f8cab4-b79b-42ba-b47d-2043f02bce48)




# 3. Algo de classification du visage : Nom, H/F, âge

**Facial recognition** = involves identifying the face in the image as belonging to person X and not person Y. It is often used for biometric purposes, like unlocking your smartphone
**Facial analysis** = tries to understand something about people from their facial features, like determining their age, gender, or the emotion they are displaying.
**Facial tracking** = is mostly present in video analysis and tries to follow a face and its features (eyes, nose, and lips) from frame to frame. The most popular applications are various filters available in mobile apps like Snapchat.

- Nom, H/F = Classifiction du visage
- Age = Regression
- Emotion ?

Ici on aura besoin d’une base de donnée de nos visages et d’entrainer sur les labels qu’on veut, en choisissant bien les bonnes fonction de pertes en sortie du réseau (cross entropy pour classification, mean average error (ou MSE, RMSE) pour regression)
[Age Estimation Bdd](https://paperswithcode.com/datasets?task=age-estimation&page=1)
[best datasets for emotion detection](https://paperswithcode.com/datasets?task=age-estimation&page=1)
[YouTube Faces Database](https://www.cs.tau.ac.il/~wolf/ytfaces/)

**Transfert Learning** : Quelque chose qui se fait vachement en IA, l’idée générale c’est que y a des modèles pré-entrainés qui ont des poids de feature extraction quasi optimisés car ils ont été entrainés sur des dizaines de milliers d’images très différentes et sur des problématiques variées.
Le but c’est alors de repartir de la partie extraction de features avec les poids optimisés et de rajouter des couches de classification derrière qui sont adaptés à notre problématique. On y passe nos données pour l’entrainement, et ce qui va se passer c’est que les poids des filtres ont déjà été optimisés et on va optimiser que la deuxième partie du réseau, ce qui fait qu’on a besoin de moins de données et que théoriquement, ca va plus vite.

![trasnfertlearning](https://github.com/iciamyplant/facial_recognition/assets/57531966/58c0e934-c387-4dd2-8626-8746ff3725ee)
[Face Recognition Model Using Transfer Learning - Medium](https://python.plainenglish.io/face-recognition-model-using-transfer-learning-9554340e6c9d)
[Face recognition using Transfer learning and VGG16 - Medium](https://medium.com/analytics-vidhya/face-recognition-using-transfer-learning-and-vgg16-cf4de57b9154)
[Face Recognition using Transfer Learning on MobileNet - Medium](https://medium.com/analytics-vidhya/face-recognition-using-transfer-learning-on-mobilenet-cf632e25353e)

[Real-Time Face Recognition: An End-To-End Project - ](https://towardsdatascience.com/real-time-face-recognition-an-end-to-end-project-b738bb0f7348)

# 4. with Opencv & Face recognition

[pypi tuto](https://pypi.org/project/face-recognition/)
[Face Detection with Python using OpenCV](https://www.datacamp.com/tutorial/face-detection-python-opencv)

Opencv = Open Source Computer Vision Library, Cv2 module is the main module in OpenCV that provides developers with an easy-to-use interface for working with image and video processing functions
[cv2 functions explained](https://konfuzio.com/en/cv2/)

```
sudo rm /usr/lib/python3.11/EXTERNALLY-MANAGED
pip install opencv-python
pip install face_recognition
```
```
## Encoder l'image
img = cv2.imread #function to read an image from a file. It takes the file path as input and returns a numpy array containing the image
cv2.imshow("Img", img) #cv2.imshow is a function used to display an image. It takes the image as input and opens a new window to display the image
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #we need to convert our image from bgr to rgb
```
```
## Is img & img2 the same person ?
result = face_recognition.compare_faces([img_encoding], img_encoding2)
print("Result: ", result) #result is boolean 
```
```
## in real time
#cap = cv2.VideoCapture(2) #allows working with video either by capturing via live webcam or by a video file
#cap = cv2.VideoCapture("file") #capture d'une video, doit etre au format avi
while True:
    ret, frame = cap.read() #videocapture.read = Grabs, decodes and returns the next video frame, renvoie false si aucune image n'a été saisie
```

"the algorithm works great, 99% accuracy"
[video capture function explained](https://www.geeksforgeeks.org/how-to-get-properties-of-python-cv2-videocapture-object/)
[videcapture.read & others explained](https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html#a473055e77dd7faa4d26d686226b292c1)
[Enable Webcam on VM](https://www.youtube.com/watch?v=ec4-1gF-cNU)

#### Tutorials face reco
- https://pypi.org/project/face-recognition/
- https://www.youtube.com/watch?v=5yPeKQzCPdI
- https://www.youtube.com/watch?v=pQvkoaevVMk
- https://www.youtube.com/watch?v=vA-JiuYX--Y
- https://www.youtube.com/watch?v=PdkPI92KSIs
- https://towardsdatascience.com/building-fast-facial-recognition-for-videos-and-images-7b9f3e7c240c

#### Tutorials age & gender
https://www.youtube.com/watch?v=hvNj7Js9RyA
https://www.youtube.com/watch?v=V0qeBb8F8XY

# Projet : 
1. Detection du visage
2. coder un algo qui classifie le visage : Nom, genre H/F = classification, le but est d'associer un label à une image + âge = regression pour prédire un nombre
3. coder un algo de détection d'objets dangereux ?
4. utiliser les librairies ou des modèles pré-entrainés pour l'expérience finale

# 1. Detection du visage

Computer vision =  discipline informatique permettant de construire des systèmes qui obtiennent des informations à partir d'images. Les images étant un ensemble de nombres avec une certaine structure, une matrice. Dans le cas d'une image en niveaux de gris, chaque nombre (pixel) représente une intensité différente allant de 0 (blanc) à 1 (noir pur) ==> de 0 à 255. Si nous travaillons avec des images colorées, alors nous avons trois canaux différents (RVB), nous avons donc la même image précédente mais trois fois, chaque matrice représentant une intensité de couleur différente ==> 0.0.0 à 255.255.255

Face detection = computer vision task in which we detect the presence of human faces and its location (x1, y1, x2, y2) within an image or a video stream. C'est un problème de regression, le but est de prédire les coordonnées (continues) du rectangle entourant le visage, cordonnée en haut à gauche (x1,y1) et coordonnée en bas à droite (x2,y2). Plusieurs obstacles à la detection : occlusion, orientation of face, expression, lighting, accessories. Au fil de temps nombreuses avancées : Viola Jones Algorithm, Histogram of oriented gradients (HOG), FDDB, *advent of deep learning techniques 2012, more robust face detectors have been developed*, Anootated Faces in the Wild, Pascal Face, SSD(slower then HOG), MTCNN (CNNs connected in a cascated manner, not fast for real time applications), UFDD, RetinaFace, MediaPipe (super real time performances), YuNet. 


#### Viola-Jones Algorithm

Nous allons implémenter un système de detection faciale utilisant Viola-Jones algorithm, easiest face recognition system you can create, but there are more advanced techniques to do the same project. L'objectif ici est surtout d'avoir un aperçu des bases du fonctionnement des systèmes de détection d'objets. 
[Official Paper](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf)
[Bonne vidéo](https://www.youtube.com/watch?v=p9vq90NYHMs)
[Tutoriel débutant](https://medium.com/@juanlux7/creating-a-face-recognition-system-from-scratch-83b709cd0560)

Viola-Jones Algorithm was created mainly to work with frontal faces and operate with grayscale images. The way it works is by applying some kind of filter (kernel) throughout the whole image (pretty similar to what a Convolutional Neural network does), starting from the top left corner and checking each pixel, and while doing that is looking for human features like eyebrows, eyes, noses etc. Et pour ça, voilà les étapes :

1. input the desired image to the system in greyscale
2. Haar feature selection is used to detect the features of a face

Many common features on a human face like: two eyes, two eyebrowns, one mouth, one nose ==> we can say that the human face has a pattern ==> we have to detect thoses features. Alfred Haar created features detector, and these detectors will look for specific places where they can fit, checking different sections of the image and look for patterns where each individual Haar feature detector can fit in to. Haar features are rectangular regions masked over an image. Within each rectangle, the summation of pixels is calculated and then the difference between the black and white regions is calculated. For exemple, eyes region is darker than cheeks. Then, renvoit une valeur totale qui doit être supérieure à un seuil pour considérer qu'il y a bien un visage.

![exampleharr](https://github.com/iciamyplant/facial_recognition/assets/57531966/bbc49cb2-fa3d-46eb-bc42-e3eff7b07841)

3. an integral image is calculated to fasten the processing
Mais pas possible de faire les calculs en testant les features sur toute l'image. In a 24*24 resolution image, there are 180k+ features, not all are usefull. So the goal is to only use the useful features and reduce processing time. Integral image is a solution : In an integral image, every pixel is the summation of the pixels above and to the left of it (voir vidéo explication avec exemple).

5. adaboost training to properly locate and train the features and pour améliorer the processing time
6. cascading to distinguish whether an image contains a face or not


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

# 2. Algo de detection du visage from scratch

[Face detection using pre-trained model - Google Collab](https://colab.research.google.com/github/dortmans/ml_notebooks/blob/master/face_detection.ipynb)


# 3. Algo de classification du visage : Nom, H/F, âge
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

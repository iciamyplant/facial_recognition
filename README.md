# Projet : 
1. Detection du visage
   + Viola-Jones Algorithm : understand the basics
   + Face detection model from scratch
   + Face detection using pre-trained model : Python and Tensorflow with VGG16 pre-trained
2. Facial recognition
   + Age recognition from scratch
   + Gender recognition from scratch
   + Face recognition - Opencv & face_recognition
3. Utiliser des modèles pré-entrainés pour l'expérience finale

# 1. Detection du visage

**Computer vision** =  discipline informatique permettant de construire des systèmes qui obtiennent des informations à partir d'images. Les images étant un ensemble de nombres avec une certaine structure, une matrice. Dans le cas d'une image en niveaux de gris, chaque nombre (pixel) représente une intensité différente allant de 0 (blanc) à 1 (noir pur) ==> de 0 à 255. Si nous travaillons avec des images colorées, alors la même matrice que ci-dessous, mais en 3 fois (RVB), chaque matrice représentant une intensité de couleur différente ==> 0.0.0 à 255.255.255

<p align="center">
<img width="450" src="https://github.com/iciamyplant/facial_recognition/assets/57531966/cf9ced56-1a3c-4554-8a00-8a64792d9d13">
<p align="center">

**Face detection** = computer vision task in which we detect the presence of human faces and its location (x1, y1, x2, y2) within an image or a video stream. C'est un problème de regression, le but est de prédire les coordonnées (continues) du rectangle entourant le visage, cordonnée en haut à gauche (x1,y1) et coordonnée en bas à droite (x2,y2). Plusieurs obstacles à la détection : occlusion, orientation of face, expression, lighting, accessories. Au fil de temps nombreuses avancées : Viola Jones Algorithm, Histogram of oriented gradients (HOG), FDDB, *advent of deep learning techniques 2012, more robust face detectors have been developed*, Anootated Faces in the Wild, Pascal Face, SSD (slower then HOG), MTCNN (CNNs connected in a cascated manner, not fast for real time applications), UFDD, RetinaFace, MediaPipe (super real time performances), YuNet. 

**Feature** = Pour détecter des visages (ou objets), l'ordinateur s'appuie sur des features = is a piece of information in an image that is relevant to solving a certain problem. It could be something as simple as a single pixel value, or more complex like edges, corners, and shapes. You can combine multiple simple features into a complex feature. Applying certain operations to an image produces information that could be considered features as well. Computer vision and image processing have a large collection of useful features and feature extracting operations.


### Viola-Jones Algorithm : understand the basics

Nous allons implémenter un système de detection faciale utilisant Viola-Jones algorithm, easiest face recognition system you can create, but there are more advanced techniques to do the same project. L'objectif ici est surtout d'avoir un aperçu des bases du fonctionnement des systèmes de détection d'objets. Viola-Jones Algorithm was created mainly to work with frontal faces and operate with grayscale images, donc input image doit etre in grayscale. À partir de cette image, l’algorithme examine de nombreuses sous-régions plus petites et tente de trouver un visage en recherchant des caractéristiques spécifiques dans chaque sous-région. Il doit vérifier de nombreuses positions et échelles différentes, car une image peut contenir de nombreux visages de différentes tailles
[Official Paper](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf)
[Bonne vidéo](https://www.youtube.com/watch?v=p9vq90NYHMs)
[Tutoriel](https://realpython.com/traditional-face-detection-python/)
[Example of a Viola-Jones Algorithm from scratch](https://github.com/sunsided/viola-jones-adaboost/blob/master/viola-jones.ipynb)

Etapes :
1. Selecting Haar-like features
2. Creating an integral image
3. Running AdaBoost training
4. Creating classifier cascades

#### 1. Haar feature selection

All human faces share some similarities : eye region is darker than the bridge of the nose, the cheeks are brighter than the eye region. A simple way to find out which region is lighter or darker is to sum up the pixel values of both regions and comparing them. The sum of pixel values in the darker region will be smaller than the sum of pixels in the lighter region. Plus il y a une différence, plus y a du constraste entre les deux régions. C'est sur ce principe que Alfred Haar created features detector.

Une Haar feature prend une partie rectangulaire d'une image et divise ce rectangle en plusieurs parties. 4 basic types of Haar-like features : Horizontal feature with two rectangles & Vertical feature with two rectangles (utiles pour détecter les bords), Vertical feature with three rectangles (detects lines), Diagonal feature with four rectangles (deteclte les caractéristiques diagonales) and these features will look for specific places where they can fit. The value of the feature is calculated as a single number: the sum of pixel values in the black area minus the sum of pixel values in the white area. For uniform areas like a wall, this number would be close to zero and won’t give you any meaningful information. To be useful, a Haar-like feature needs to give you a large number, meaning that the areas in the black and white rectangles are very different. There are known features that perform very well to detect human faces. On peut utiliser ce principe pour déterminer quelles zones d'une image donnent une réponse forte (un grand nombre) pour une feature spécifique. On peut combiner many of these features to understand if an image region contains a human face.

<p align="center">
<img width="450" src="https://github.com/iciamyplant/facial_recognition/assets/57531966/3f78e329-af35-4101-a998-fb82a1857a54">
<p align="center">

As mentioned, the Viola-Jones algorithm calculates a lot of these features in many subregions of an image. This quickly becomes computationally expensive: it takes a lot of time using the limited resources of a computer. In a 24*24 resolution image, there are 180k+ features, not all are usefull. To tackle this problem, Viola and Jones used integral images to fasten the processing time

#### 2. Creating an integral image

**integral image** = name of a data structure & an algorithm used to obtain this data structure = moyen rapide et efficace de calculer la somme des valeurs de pixels dans une image ou une partie rectangulaire d’une image. L'image intégrale peut être calculée en un seul passage sur l'image originale. 
Calcul : in an integral image, every pixel is the summation of the pixels above and to the left of it.

<p align="center">
<img width="450" src="https://github.com/iciamyplant/facial_recognition/assets/57531966/ce32c8c1-b8a3-4251-9346-6c6b2160b899">
<p align="center">

But how do you decide which of these features and in what sizes to use for finding faces in images? This is solved by a machine learning algorithm called **boosting**. Specifically, we will learn about AdaBoost (Adaptive Boosting)

#### 3. Running AdaBoost training

Boosting is based on the following question: “Can a set of weak learners create a single strong learner?” 
- **A weak learner (or weak classifier)** = is defined as a classifier that is only slightly better than random guessing. In face detection, this means that a weak learner can classify a subregion of an image as a face or not-face only slightly better than random guessing. In the Viola-Jones algorithm, each Haar-like feature represents a weak learner, by their own the detectors are unable to perform realy good predictions
- **A strong learner** = is substantially better at picking faces from non-faces.
- So the power of boosting comes from combining many (thousands) of weak classifiers into a single strong classifier

![weakstrongclassifier](https://github.com/iciamyplant/facial_recognition/assets/57531966/770920d5-6e09-4be3-84c2-70057d1ded08)

Comment décider du type et de la taille d'une feature qui entre dans le classificateur final ? Pour ça, Adaboost training (Adaboost is a Machine Learning algorithm), on entraîne le modèle avec un grand nombre d'images. 

AdaBoost vérifie les performances de chaque feature qu'on lui fournit. 
Pour calculer les performances d'une feature (weak classificateur), vous l'évaluez sur toutes les sous-régions de toutes les images utilisées pour l'entraînement. Certaines sous-régions produiront une réponse forte. Celles-ci seront classées comme positives, ce qui signifie que le weak classifier pense qu’elle contiennent un visage humain. Les sous-régions qui ne produisent pas de réponse forte ne contiennent pas de visage humain d'après le weak classifier, donc elles sont classées comme négatives.
Les weak classifiers qui ont bien fonctionné se voient attribuer une signifiance ou un poids plus élevé. Le résultat final est un classifier fort, également appelé boosted classifier, qui contient les weak classifiers les plus performants.

1. We give the system facial images (positive examples) and non facial images (negative examples)
2. Initialize weights for each image
3. For each classifier :
-  + We normalize the weights
-  + A classifier restreint à une feature utilisée est entraîné sur toutes les images, et l'erreur est calculée (erreur de 0 s'il s'est pas trompé, 1 s'il s'est trompé).
-  + The error is then multiplied by the significance of the image
   + le lowest error is chosen and then les poids sont mis à jour de cette manière : beta increases with the error hence the lower the error is, the higher alpha is. So a feature with a low error is given a higher importance in the strong classifier
   + the weight of the image is updated for the next iteration by reducing it for the images that were correctly classified
 4. The final strong classifier is one, when the sum of the weighted features is higher than half of the sum of the weights

Résumé : it's mathematical operations but the idea is that we are trying to classify images as faces or non-faces. L'algo va run et dire si telle ou telle image est un visage ou non, calculer un taux d'erreur basé sur la somme pondérée des instances où la classification a été mauvaise, le but étant de réduire ce taux d'erreur, mettre à jour les poids (l'importance de chaque feature) selon un processus itératif, en pénalisant davantage les images mal classées (faux positifs et négatifs), dans le but donc de rechercher des caractéristiques qui s'adaptent le mieux à l'image. 


#### 4. Cascading Classifiers

Viola et Jones ont évalué des centaines de milliers de classificateurs spécialisés dans la recherche de visages dans les images. Mais il serait coûteux en termes de calcul d'exécuter tous ces classificateurs sur chaque région de chaque image, c'est pourquoi ils ont créé une technique qu'on appelle une cascade de classificateurs.

On transforme le classificateur fort (constitué de milliers de classificateurs faibles) en une cascade où chaque classificateur faible représente une étape (permet d'éliminer rapidement les non-visages).
- Lorsqu'une sous-région d'image entre dans la cascade, elle est évaluée par la première étape. Si cette étape évalue la sous-région comme positive (=pense qu'il y a un visage), le résultat de l'étape est *maybe*
- Si une sous-région obtient un *maybe*, elle est envoyée à l'étape suivante de la cascade, et ainsi de suite
- Processus répété jusqu'à ce que l'image traverse toutes les étapes de la cascade. Si tous les classificateurs approuvent l'image, elle est finalement classée comme visage humain
- ==> classificateurs les plus performants doivent être placés au début de la cascade. Dans Viola-Jones les plus performants :  Feature on the eyes and chins, where the eyes are darker than chins is the most significant. The second one indicates that the bridge of the nose is brighter than its surroundings.

![résumé du process](https://github.com/iciamyplant/facial_recognition/assets/57531966/3da197a5-8525-4aeb-82cb-a034d64e84dc)



### CNN : utilisé aujourd'hui

[Explication video debutant](https://www.youtube.com/watch?v=QzY57FaENXg) ;
[Explication un peu plus poussée de Science4all](https://www.youtube.com/watch?v=zG_5OtgxfAg&t=305s) ;
[Explication bien qui fait un pont avec les haar-like feature à 7:19](https://www.youtube.com/watch?v=YRhxdVk_sIs)

Mais depuis l'avènement de techniques de deep learning, de nouvelles méthodes de computer vision plus robustes ont été développées. Le DL fonctionne particulièrement bien sur les données non structurées (information that is not arranged according to a preset data model or schema, pas besoin de la traiter avant utilisation) type image ou texte. Mais pour faire du computer vision, on va pas faire un réseau classique avec plein de couches de neurones successives, on va faire ce qu’on **appelle un CNN (Convolutional Neural Network)** qui est particulièrement adpatés aux images. Un **regular neural network** se compose d'une input layer (accepts input in different forms), hidden layers (perform calculations on these inputs), and an output layer (delivers the outcome of the calculations and extractions). Each of these layers contain neurons that are connected to neurons in the previous layer and each neuron has its own weight. Alors que a **CNN** = is a type of neural network, that is most often applied to image processing problems and NLP. Et voilà comment fonctionne a CNN :

The basis of a CNN is that it contains **convolution layers**. Juste like any layer, a convolutional layer receives input, then transforms the input in a some way, and outpouts the transofrm input to the next layer. In a convolutionnal layer this transformation is a convolution operation. Les convolutionnal layers are able to detect patterns (= une image est composée de plein d'éléments : des objets, des textures, edges, formes... donc ces patterns peuvent etre des bords..) in images, thanks to filters. In each convolutionnal layer we need to specify a number of filters the layer should have. Un filtre peut détecter un pattern, comme les bords de l'image (le trait d'une falaise par exemple), some filters may detect circles, squares etc. A **convolutionnal filter** can technically just be thought of as a relatively small matrix for which we decide the number of rows and the number of columns that this matrix has. And the values within the matrix are initialized with random numbers. Once the convolutionnal layer recives the input, la matrice qui va traverser toute l’image de gauche à droite et de bas en haut en appliquant successivement des opérations de convolution, i.e. des sommes de produits termes à termes. Quand un filtre se propage sur une image, il somme les résultats de ces opérations de convolution sur les différents canaux de couleur (t’as des filtres qui affichent le gradient ou qui détectent les bords, ce genre de trucs). Cette logique est généralisable à toute donnée d’input qui est composée de plusieurs couches de matrices à 2 dimensions. The deeper our network goes, the sophisticated these filters become (detect eyes, dogs...). Les CNN se composent donc d'une superposition de couches de neurones (convolutionnal layers) qui analysent les images. Chaque couche correspond au calcul de representation de plus en plus abstraite du contenu de l'image (par exemple une couche pour la luminosité et couleur, différentes correlations entre les pixels voisins, une couche pour mettre à bout ces correlations pour identifier des lignes directrices,..). Le réseau de neurone calcule ainsi une abstraction croissante des données. Ce qui lui permet de calculer une représentation sémantique de l'image.

- Chaque neurone de chaque couche intermediaire n'est exposé qu'à un champ récépteur particulier (cad une toute petite région de l'image, kernel_size), et l'analyse de ce champ récépteur et la même que l'analyse qu'effectue un autre neurone de la même couche avec son propre champ récepteur.
- Exemple = imaginons une couche qui analyse la luminosité, le premier neurone n'analyse qu'une toute petite région de l'image, et le neurone va multiplier la luminosité de chaque pixel par un pois sinaptique, et calculer la somme, puis calculer la somme des luminosités pondérées de tous les pixels de la région. Puis activation = cad qu'on va retenir que la partie positive du résultat (fonction d'activation relu). Les pois synaptiques forment alors une matrice dite de convolution (filtre). Et l'idée des CNN c'est d'utiliser la même matrice de convolution pour tous les champs recepteurs de l'image. Donc les autres neurones d'une même couche appliquerons les mêmes calculs mais sur des regions différentes de l'image.
- C'est à dire que pour une seule et même région de l'image, elle sera analyser par des neurones mais de différentes couches,  mais chacun de ces neurones utilise une matrice de convolution différente. Donc chaque sous-region sera traduite par un certain nombre de nombres qui résume le contenu de cette région. Et ce sont ces matrices de convolution qu'on va chercher à améliorer.
- On va donc appliquer plusieurs couches de convolutions les unes après les autres, ce qui va en fait correspondre à la phase de "feature extraction”
- Mais ce faisant on ne diminue pas la taille des données à analyser, plus y a de filtres plus on augmente la dimension des données, donc on ajoute généralement une phase de réduction de la dimensionnalité qui revient a résumer l'information de plusieurs neurones voisins en une seule information (=pooling).
- On va ensuite applatir notre dernière image, utiliser le vecteur qui nous est donné dans un réseau dense et confronter la prédicition qui est faite avec la réalité. Encore une fois on a une fonction de perte à minimiser, le but est qu’au fur et à mesure, les prédicitions soient de plus en plus proches de la réalité en optimisant les paramètres des filtres à chaque fois qu’on passe dans le réseau.
- L’idée générale est que le réseau va apprendre par lui même où regarder sur l’image grâce aux différents filtres, en fonction de la classification/régression qu’on a envie de faire.

<p align="center">
<img width="450" src="https://github.com/iciamyplant/facial_recognition/assets/57531966/77f8cab4-b79b-42ba-b47d-2043f02bce48">
<p align="center">











### Face detection using pre-trained model : Python and Tensorflow with VGG16 pre-trained

[Tutoriel, Face Detection Model with Python and Tensorflow from scratch](https://www.youtube.com/watch?v=N_W4EYtsa10)

2 principaux gros outils pour faire du DL = **TensorFlow** = Open source framework developed by Google to build various machine learning and deep learning models publié en 2015. For exemple, can be used to train and execute : NLP, neural network image recognition, digit classification, etc. Tensorflow is focus to reduce the complexity of implementing computations on large numerical datasets. Et **Pytorch** = framework de deep learning basé sur Torch, développé par le groupe de recherche AI de Facebook, publié en open-source en 2017, écrit en python.

- the input will be the individuals pixels
- and the oupout the patterns we are trying to classify








[Vidéo en anglais où le mec résume à son bureau](https://www.youtube.com/watch?v=py5byOOHZM8)


[CNN Tutorial pre trained model](https://realpython.com/face-recognition-with-python/#prerequisites)
[Face detection using pre-trained model - Google Collab](https://colab.research.google.com/github/dortmans/ml_notebooks/blob/master/face_detection.ipynb)

**Transfert Learning** : Quelque chose qui se fait vachement en IA, l’idée générale c’est que y a des modèles pré-entrainés qui ont des poids de feature extraction quasi optimisés car ils ont été entrainés sur des dizaines de milliers d’images très différentes et sur des problématiques variées.
Le but c’est alors de repartir de la partie extraction de features avec les poids optimisés et de rajouter des couches de classification derrière qui sont adaptés à notre problématique. On y passe nos données pour l’entrainement, et ce qui va se passer c’est que les poids des filtres ont déjà été optimisés et on va optimiser que la deuxième partie du réseau, ce qui fait qu’on a besoin de moins de données et que théoriquement, ca va plus vite.

<p align="center">
<img width="450" src="https://github.com/iciamyplant/facial_recognition/assets/57531966/58c0e934-c387-4dd2-8626-8746ff3725ee">
<p align="center">

face_recognition to detect the face in each image and get its encoding. This is an array of numbers describing the features of the face, and it’s used with the main model underlying face_recognition to reduce training time while improving the accuracy of a large model. This is known as transfer learning.








# 2. Facial recognition 

**Facial recognition** = involves identifying the face in the image as belonging to person X and not person Y

**Facial analysis** = tries to understand something about people from their facial features, like determining their age, gender, or the emotion they are displaying

**Facial tracking** = is mostly present in video analysis and tries to follow a face and its features (eyes, nose, and lips) from frame to frame

Objectif : réussir à reconnaître le visage + analyser le genre et la tranche d'âge


### Age recognition from scratch

#### 1. Dataset

Face_recognition/Age_Gender_Recognition/from_scratch/utkcropped [UtkCropped Base de données](https://www.kaggle.com/datasets/abhikjha/utk-face-cropped/data). The dataset consists of over 20,000 face images with annotations of age, gender, and ethnicity. Age from 0 to 116 years old. The images cover large variation in pose, facial expression, illumination, occlusion, resolution, etc. Files : age_genre_ethnicite.numero.jpg. Genre : (0=male, 1=female). Ethnicité : (0=white, 1=Black, 2=Asian, 3=Indian, 4=Hispanic)

#### 2. Training

Pincipe : un modèle de Machine Learning est capable d’apprendre de façon autonome à partir d’un jeu de données, dans l’objectif de prédire des comportements sur un autre jeu de données. Pour cela, il trouve des relations sous-jacentes entre des variables explicatives indépendantes (les pixels cad l'image, X) et une variable cible dans le dataset initial (l'âge, y). Puis il utilise ces patterns pour prédire ou classifier des nouvelles données.

Donc, il faut :
- diviser le jeu de données initial en deux ensembles : une partie pour le training, une partie pour les tests (on a pris que les 2000 premières données pour que ça aille plus vite)
- fit, cad entraîner sur une partie des données : X_train, y_train
- notre modèle va trouver des relations sous-jacentes entre des variables explicatives indépendantes (X = les pixels) et une variable cible (y = l'âge)
- puis on teste les capacités prédictives de notre modèle sur des nouvelles données : X_test, y_test
- plusieurs  métriques peuvent être utilisées pour cette évaluation : dans le cas d’une régression linéaire, **le coefficient de détermination**, **la RMSE** et **la MAE** sont privilégiés. Dans le cas d’une classification, **l’accuracy, la précision, le recall et le F1-score** sont privilégiés. Ces scores sur l’ensemble test permettent donc de déterminer si le modèle est performant et à quel point il doit être amélioré avant de pouvoir prédire sur un nouveau dataset

##### a. Séparation des données 
```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123) #train_test_split permet de faire cette séparation en deux ensembles, en apprentissage supervisé, prend en argument les deux arrays l'une avec les variables explicatives et l'autre la variable cible (= les labels), test_size = soit un nombre décimal compris entre 0 et 1 représentant une proportion du jeu de données, soit un nombre entier représentant un nombre d’exemples du jeu de données, random_state =  nombre qui contrôle la façon dont le générateur pseudo-aléatoire divise les données
```

##### b. Création du modèle & compilation

modèle très classique avec des couches de convolutions successives (avec pooling et dropout). La couche de sortie est un seul neurone sans fonction d'activation pcq on est sur une régression.
Exemple fonctionnement couches de convolutions succesives [vidéo](https://www.youtube.com/watch?v=JboZfxUjLSk)

```
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
input_shape = (200,200,3)
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='valid', input_shape=input_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.2)) #layer to avoid overfitting

model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.2))

model.add(Flatten()) #transforme en une liste de pixels

model.add(Dense(units=64, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(units=1))

model.summary()
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
```

<p align="center">
<img width="450" src="https://github.com/iciamyplant/facial_recognition/assets/57531966/a5d95992-5c02-4c37-bc79-b84d0089c016">
<p align="center">


|Type of Layer|Parameters|Layer Definition|Parameters Definition|
|-----|-----|---|---|
|Conv2d|filters=16, kernel_size=(5,5), padding='valid', input_shape=input_shape, activation='relu'|The first layer of a Convolutional Neural Network is always a Convolutional Layer. Convolutional layers apply a convolution operation to the input, passing the result to the next layer. Le type de convolution le plus couramment utilisé est la couche de convolution 2D, le kernel « glisse » sur les données d'entrée 2D, effectuant une multiplication par éléments |**kernel_size** = le kernel correspond à la convolution matrix et kernel_size est la taille qu'on donne à cette matrice. A kernel describes a filter that we are going to pass over an input image. To make it simple, the kernel will move over the whole image, from left to right, from top to bottom by applying a convolution product. The output of this operation is called a filtered image. **Filters** = ou kernels, chaqe kernel est un filtre différent qu'on applique à l'image, différentes oppérations. **activation** = on va retenir que la partie positive du résultat |
|Max_pooling|pool_size=(2, 2)|||
|Dropout|rate=0.2|||
|Conv2D|pool_size=(2, 2)|||
|MaxPooling2D|filters=32, kernel_size=(3, 3), padding='valid', activation='relu'|||
|Conv2D|filters=64, kernel_size=(3, 3), padding='valid', activation='relu'|||
|MaxPooling2D|pool_size=(2, 2)|||
|Dropout|rate=0.2|||
|Flatten|rate=0.2|||
|Dense|units=64, activation='relu'|||
|Dropout|rate=0.2|||
|Dense|units=1|||


##### c. Entraînement
```
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                              epochs=10,  batch_size=32)
```
**overfitting (ou sur-apprentissage)** = décrit une situation où le modèle construit est trop complexe (avec trop de variables explicatives par exemple), tel qu’il apprend parfaitement les données d’entraînement mais n’arrive pas à se généraliser sur d’autres données
**underfitting (ou sous-apprentissage)** = décrit une situation où le modèle est trop simple ou mal choisi (choix d’une régression linéaire sur des données ne respectant pas ses hypothèses par exemple), tel qu’il apprend mal

##### d. Prédiction

```
y_pred = model.predict(X_test)
y_pred
```

### Gender recognition from scratch


Classification, le but est d'associer un label à une image
Ici on aura besoin d’une base de donnée de nos visages et d’entrainer sur les labels qu’on veut, en choisissant bien les bonnes fonction de pertes en sortie du réseau (cross entropy pour classification, mean average error (ou MSE, RMSE) pour regression)

[Age Estimation Bdd](https://paperswithcode.com/datasets?task=age-estimation&page=1)
[best datasets for emotion detection](https://paperswithcode.com/datasets?task=age-estimation&page=1)
[YouTube Faces Database](https://www.cs.tau.ac.il/~wolf/ytfaces/)


### with pre-trained model - OpenCV et facial_recognition

[Face Recognition Model Using Transfer Learning - Medium](https://python.plainenglish.io/face-recognition-model-using-transfer-learning-9554340e6c9d)
[Face recognition using Transfer learning and VGG16 - Medium](https://medium.com/analytics-vidhya/face-recognition-using-transfer-learning-and-vgg16-cf4de57b9154)
[Face Recognition using Transfer Learning on MobileNet - Medium](https://medium.com/analytics-vidhya/face-recognition-using-transfer-learning-on-mobilenet-cf632e25353e)

[Real-Time Face Recognition: An End-To-End Project - ](https://towardsdatascience.com/real-time-face-recognition-an-end-to-end-project-b738bb0f7348)






















--------------------------------------------------- BROUILLON ---------------------------------------------------






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

https://we.tl/t-Jjq3qU0Kjc
## Prerequisites
```
sudo rm /usr/lib/python3.11/EXTERNALLY-MANAGED
pip install opencv-python
pip install face_recognition
```
Library Opencv = image and video processing. Cv2 module is the main module in OpenCV that provides developers with an easy-to-use interface for working with image and video processing functions
[cv2 functions explained](https://konfuzio.com/en/cv2/)


## Encoder l'image

```
img = cv2.imread #function to read an image from a file. It takes the file path as input and returns a numpy array containing the image
cv2.imshow("Img", img) #cv2.imshow is a function used to display an image. It takes the image as input and opens a new window to display the image
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #we need to convert our image from bgr to rgb
```

## Is img & img2 the same person ?

```
result = face_recognition.compare_faces([img_encoding], img_encoding2)
print("Result: ", result) #result is boolean 
```

"the algorithm works great, 99% accuracy"

## in real time

[video capture function explained](https://www.geeksforgeeks.org/how-to-get-properties-of-python-cv2-videocapture-object/)
[videcapture.read & others explained](https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html#a473055e77dd7faa4d26d686226b292c1)
```
#cap = cv2.VideoCapture(2) #allows working with video either by capturing via live webcam or by a video file
#cap = cv2.VideoCapture("file") #capture d'une video, doit etre au format avi
while True:
    ret, frame = cap.read() #videocapture.read = Grabs, decodes and returns the next video frame, renvoie false si aucune image n'a été saisie
```

[Enable Webcam on VM](https://www.youtube.com/watch?v=ec4-1gF-cNU)



Tutorials
- https://www.youtube.com/watch?v=5yPeKQzCPdI
- https://www.youtube.com/watch?v=pQvkoaevVMk
- https://www.youtube.com/watch?v=vA-JiuYX--Y
- https://www.youtube.com/watch?v=PdkPI92KSIs
- https://towardsdatascience.com/building-fast-facial-recognition-for-videos-and-images-7b9f3e7c240c


# face-landmark-detection-and-alignment
this implements face landmark detection and alignment code

> download the face landmark model from:
> https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2 
> and place the extracted file inside models

run commands:
```
pip install -r requirements.txt
python main.py

```
> Note: checkout **config.py** to finetune some parameters

## OUTPUT:

## face landmark
![face landmark](https://github.com/humandotlearning/face-landmark-detection-and-alignment/blob/master/media/face_landmarks.gif) 

## face aligned
![face aligned](https://github.com/humandotlearning/face-landmark-detection-and-alignment/blob/master/media/face_alligned.gif)

References:
1. [face-alignment-with-opencv-and-python](https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/)
2. [facial-landmarks-dlib-opencv-python](https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/)

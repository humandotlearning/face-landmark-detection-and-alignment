# Refer dlib documentation
# download and extract from: https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2
MODEL_PTH = 'models/shape_predictor_68_face_landmarks.dat'

# output file name
OUT_NAME = "media/face_landmark.avi"

# dimension of video
(W,H) = (640,480)

# save video of aligned face
aligned_face = True
aligned_vid_name = "media/aligned_face.avi"
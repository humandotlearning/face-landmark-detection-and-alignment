import cv2
import numpy as np
import time
import dlib
from imutils.face_utils import FaceAligner

from face_utils import face_utils
from face_landmark import draw_landmark
import config

# Refer dlib documentation
# download and extract from: https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2
model_path = config.MODEL_PTH
# shape predictor for facial landmarks
sp_predictor = dlib.shape_predictor(model_path)

# desiredFaceWidth=256 default
fa = FaceAligner(sp_predictor, desiredFaceWidth=256)

# output file name
out_name = config.OUT_NAME

detector = dlib.get_frontal_face_detector()

# dimension of video
(W,H) = (config.W, config.H)

def showInMovedWindow(winname, img, x, y):
    """
        draws cv2 window at x,y coordinate
    """
    cv2.namedWindow(winname)        # Create a named window
    cv2.moveWindow(winname, x, y)   # Move it to (x,y)
    cv2.imshow(winname,img)

def main():

    # video capture object
    cam = cv2.VideoCapture(0)

    # video write object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_name, fourcc, 20.0, (W,H))

    if config.aligned_face == True:
        vid2 = cv2.VideoWriter(config.aligned_vid_name, fourcc, 20.0, (W,H))

    while True:

        ret, im = cam.read()

        # checks if frames are being received or not 
        if ret == False:
            continue

        # flipping horizontally for better viewing
        im = cv2.flip(im, 1)
        orig_im = im.copy()

        # image(bgr) to grayscale for face detection
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        # face detection
        dets = detector(gray,1)

        # draws detected faces
        im = face_utils.draw_bbox(im, dets)

        # draw face landmarks
        im = draw_landmark(im, dets, sp_predictor)

        # allign face of the largest face in image
        # largest face is done only for ease of use, you can put it in loop to allign all faces
        bbox = face_utils.largest_bbox(dets)
        if bbox:
            alligned_face = fa.align(orig_im ,gray, bbox)
            # shows aligned face image
            # cv2.imshow("alligned_face", alligned_face)

            w,h,_ = alligned_face.shape

            v_width = int((W - w)/2)
            h_width = int((H - h)/2)
            padded_img = cv2.copyMakeBorder(alligned_face,h_width,h_width,v_width,v_width,cv2.BORDER_CONSTANT,value=(0,0,0))
            # draws image window in the top-left corner of screen
            showInMovedWindow("alligned_face", padded_img, 10, 10)

            # writes aligned image in video
            vid2.write(padded_img)

        
        # final image
        final_im = im

        # writes image in video
        out.write(final_im)

        # shows image
        # cv2.imshow("final_im", final_im)
        # draws image window in the top-right corner of screen
        showInMovedWindow("final_im", final_im, 30000, 0)

        

        # exit video when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()








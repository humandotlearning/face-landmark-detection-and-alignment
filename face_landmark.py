import cv2
from face_utils import face_utils

def draw_landmark(image, rects, predictor, convert_to_rgb=False):
    """
        params:
            image
            rect: dlib_shape_predictor.recangle
            predictor: dlib shape predictor object
        
        returns: landmark drawn image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    im_color = image

    if convert_to_rgb:
        im_color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # face landmark code
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(im_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # show the face number
        cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(im_color, (x, y), 1, (0, 0, 255), -1)
            
    return im_color
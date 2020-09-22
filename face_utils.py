import numpy as np
import cv2

class face_utils():
    def __init__(self):
#         self.shape = shape
#         self.rect = rect
        pass
    
    
    def shape_to_np( shape , dtype="int"):
        """
            take the shape landmark predicted object and converts it to numpy
        """
        
        # initialize the list of (x, y)-coordinates
        coords = np.zeros((68, 2), dtype=dtype)
        # loop over the 68 facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        # return the list of (x, y)-coordinates
        return coords

    def rect_to_bb(rect ):
        """
            take a bounding predicted by dlib and convert it
            to the format (x, y, w, h) as we would normally do
            with OpenCV
        
        """
#         rect = self.rect
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
        # return a tuple of (x, y, w, h)
        return (x, y, w, h)

    def draw_bbox(image, dets):
        """
            draws faces detected

            params:
                image
                dets: dlib face detection object

            returns:
                image with bbox drawn on them
        
        """

        for bbox in dets:
            (x1,y1) = (bbox.left(),bbox.top())
            (x2,y2) = (bbox.right(),bbox.bottom())
            
            image = cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2 )
                    
        return image

    def largest_bbox(dets):
        """
            returns largest detected face
        """
        largest = -999
        for bbox in dets:
            x1 = min(bbox.left(),bbox.right())
            x2 = max(bbox.left(),bbox.right())
            y1 = min(bbox.top(),bbox.bottom())
            y2 = max(bbox.top(),bbox.bottom())
            area = (x2-x1) * ( y2 - y1)
            if area > largest:
                dets = bbox
                largest = area
                
        return dets

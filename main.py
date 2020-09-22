import cv2
import numpy as np
import time

def main():
    # output file name
    out_name = "media/tmp.avi"

    # dimension of video
    (w,h) = (640,480)

    # video capture object
    cam = cv2.VideoCapture(0)

    # video write object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_name, fourcc, 20.0, (w,h))

    while True:

        ret, im = cam.read()

        # checks if frames are being received or not 
        if ret == False:
            continue

        # flipping horizontally for better viewing
        im = cv2.flip(im, 1)

        
        # final image
        final_im = im

        # writes image in video
        out.write(final_im)

        # shows image
        cv2.imshow("final_im", final_im)

        # exit video when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()








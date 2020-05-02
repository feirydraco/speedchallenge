import cv2
import sys
import numpy as np

MIN_MATCH_COUNT = 10


class SpeedDetector():
    def __init__(self, video, annotations):
        self.video = cv2.VideoCapture(video)
        if self.video.isOpened() == False:
            print("Error accessing video.")
            sys.exit(0)
        self.truth = list(open(annotations, "r"))

    def cleanup(self, video):
        video.release()
        cv2.destroyAllWindows()

    def eval(self):
        orb = cv2.ORB_create(nfeatures=100)
        frame_idx = 0
        prev_frame = None
        while True:
            ret, frame = self.video.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if ret == True:
                kp = orb.detect(frame, None)
                kp, des = orb.compute(frame, kp)
                if prev_frame is not None:
                    kp_prev = orb.detect(prev_frame, None)
                    kp_prev, des_prev = orb.compute(prev_frame, kp_prev)

                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    matches = bf.match(des_prev, des)
                    matches = sorted(matches, key = lambda x:x.distance)

                    img3 = cv2.drawMatches(prev_frame,kp_prev,frame,kp,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                    cv2.imshow("frame", img3)


                    
                # print(type(kp), type(des))
                # print(kp, des)

                cv2.drawKeypoints(frame, kp, frame, (0, 255, 255), flags=0)

                
                print("GroundTruth:", self.truth[frame_idx])
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break
                frame_idx += 1
                prev_frame = frame.copy()
            else:
                break
        self.cleanup(self.video)


if __name__ == "__main__":
    speed = SpeedDetector("data/train.mp4", "data/train.txt")
    speed.eval()
# lk_params = dict( winSize  = (15, 15),
#                   maxLevel = 2,
#                   criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


# ret, frame = cap.read()
# prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# p0 = orb.detect(prev, None)
# p0, des0 = orb.compute(prev, p0)
# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     if ret == True:
#         img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         p1, st, err = cv2.calcOpticalFlowPyrLK(prev.astype(np.uint8), img.astype(np.uint8), p0, None, **lk_params)

#         img2 = frame.copy()
#         # draw only keypoints location,not size and orientation
#         # cv2.drawKeypoints(img, kp, img2, color=(0, 255, 0), flags=0)
#         # Our operations on the frame come here

#         # Display the resulting frame
#         cv2.imshow('frame', img2)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     prev = img
# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()

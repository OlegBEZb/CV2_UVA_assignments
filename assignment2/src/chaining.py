import numpy as np
import cv2 as cv
from fundamental_matrix import keypoint_matcher


def get_names_image_pair(image_number):
    next_number = image_number + 1
    if image_number < 10:
        name1 = f"../Data/House/frame0000000{image_number}.png"
    else:
        name1 = f"../Data/House/frame000000{image_number}.png"
    if next_number < 10:
        name2 = f"../Data/House/frame0000000{image_number}.png"
    else:
        name2 = f"../Data/House/frame000000{image_number}.png"
    return name1, name2


def chaining():

    # initialize the PVM, -1 is placeholder
    PVM = np.full((100,1),-1)

    # loop through all image pairs and perform keypoint_matcher
    for i in range(0,49): #stop should be 49, but lower is nice for debugging
        name1, name2 = get_names_image_pair(i+1) #the file names start at 1 instead of 0
        image1 = cv.imread(name1)
        image2 = cv.imread(name2)
        matches, _, _, kp1, kp2 = keypoint_matcher(image1, image2)
        
        for m in matches:
            found = False
            query_kp = kp1[m[0].queryIdx]
            train_kp = kp2[m[0].trainIdx]
            x1, y1 = query_kp.pt
            x2, y2 = train_kp.pt

            # check whether the point in query_kp is already included in PVM
            # TODO This does not work yet. Method incorrect or indexing mistake?
            for column_number, x_candid in enumerate(PVM[2*i,:]):
                if x_candid == x1 and PVM[2*i+1,column_number] == y1:
                    if PVM[2*(i+1),column_number] == -1 and PVM[2*(i+1)+1,column_number] == -1:
                        PVM[2*(i+1),column_number] = x2
                        PVM[2*(i+1)+1,column_number] = y2
                        found = True
            
            if not found:
                new_col = np.full((100,1),-1)
                new_col[2*i] = x1
                new_col[2*i+1] = y1
                new_col[2*(i+1)] = x2
                new_col[2*(i+1)+1] = y2
                PVM = np.append(PVM, new_col, axis=1)

    PVM = np.delete(PVM, obj=0, axis=0) #delete placeholder column
    return PVM



if __name__ == '__main__':
    chaining()
import numpy as np
import cv2 as cv
from fundamental_matrix import keypoint_matcher
from tqdm import tqdm


def get_names_image_pair(image_number):
    name1 = get_image_name(image_number)
    name2 = get_image_name(image_number + 1)
    return name1, name2


def get_image_name(image_number):
    if image_number < 10:
        name = f"../Data/House/frame0000000{image_number}.png"
    else:
        name = f"../Data/House/frame000000{image_number}.png"
    return name


def get_row_value_idx(row, value):
    row = row.tolist()
    if value in row:
        return row.index(value)
    else:
        return None


def chaining(**_matching_kwargs):
    # initialize the PVM, -1 is placeholder
    m = 49*2
    sparse_fill_value = -1e-4
    PVM = np.full((m, 1), sparse_fill_value)  # num of images twice

    # loop through all image pairs and perform keypoint_matcher
    for left_img_idx in tqdm(range(1, 49)):  # in case of 360 this should also connect the last with the first
        i = (left_img_idx - 1) * 2
        right_image_idx = left_img_idx + 1  # TODO: change in flexible i, j pairs
        # if right_image_idx == 50: right_image_idx = 1   # in case of 360 this should also connect the last with the first
        j = (right_image_idx - 1) * 2
        name1, name2 = get_image_name(left_img_idx), get_image_name(right_image_idx)
        image1 = cv.imread(name1)
        image2 = cv.imread(name2)
        print(f"comparing table x indices {i, j} and images {name1, name2}")
        _, matched_points1, matched_points2, _, _ = keypoint_matcher(image1, image2, **_matching_kwargs)

        for p1, p2 in zip(matched_points1, matched_points2):
            point_already_introduced = False
            x1, y1 = p1
            x2, y2 = p2

            # comparing rows: (0, 1) vs (2, 3); (2, 3) vs (4, 5)
            pvm_x1_idx = get_row_value_idx(PVM[i, :], x1)
            if pvm_x1_idx:
                if PVM[i + 1, pvm_x1_idx] == y1:
                    # print(f'point ({i, pvm_x1_idx}) exists. updating {j, pvm_x1_idx}')
                    PVM[j, pvm_x1_idx] = x2
                    PVM[j + 1, pvm_x1_idx] = y2
                    point_already_introduced = True

            pvm_x2_idx = get_row_value_idx(PVM[j, :], x2)
            if pvm_x2_idx:
                if PVM[j + 1, pvm_x2_idx] == y2:
                    # print(f'point ({j, pvm_x2_idx}) exists. updating {i, pvm_x2_idx}')
                    PVM[i, pvm_x2_idx] = x2
                    PVM[i + 1, pvm_x2_idx] = y2
                    point_already_introduced = True

            #             # check whether the point in query_kp is already included in PVM
            #             # TODO This does not work yet. Method incorrect or indexing mistake?
            #             for column_number, x_candid in enumerate(PVM[2 * i, :]):
            #                 if x_candid == x1 and PVM[2 * i + 1, column_number] == y1:
            #                     if PVM[2 * (i + 1), column_number] == -1 and PVM[2 * (i + 1) + 1, column_number] == -1:
            #                         PVM[2 * (i + 1), column_number] = x2
            #                         PVM[2 * (i + 1) + 1, column_number] = y2
            #                         point_already_introduced = True

            if not point_already_introduced:
                new_col = np.full((m, 1), sparse_fill_value)
                new_col[i] = x1
                new_col[i + 1] = y1
                new_col[j] = x2
                new_col[j + 1] = y2
                PVM = np.append(PVM, new_col, axis=1)

    PVM = np.delete(PVM, obj=0, axis=1)  # delete placeholder column
    return PVM


if __name__ == '__main__':
    chaining()

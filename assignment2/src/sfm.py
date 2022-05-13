import numpy as np
from scipy.spatial import procrustes
from chaining import chaining


def normalize(column):
    return column - np.mean(column)


def factorize_and_stitch(pvm):
    # 1. Select a dense block and construct D
    D = pvm.copy()

    # 2 Normalize the point coordinates by translating them to mean
    norm_D = np.apply_along_axis(normalize, 1, D)

    # 3 Apply SVD
    U, W, V_T = np.linalg.svd(norm_D)
    W = np.diag(W)

    # 4 Get measurement and shape matrix
    W3 = W[:3, :3]
    U3 = U[:, :3]
    V_T3 = V_T[:3, :]
    M = U3 @ np.sqrt(W3)
    S = np.sqrt(W3) @ V_T3

    # 5 Eliminate affine ambiguity (additional improvement)
    L = np.linalg.pinv(M) @ np.eye(M.shape[0]) @ np.linalg.pinv(M).T
    C = np.linalg.cholesky(L)

    M = M @ C
    S = np.linalg.inv(C) @ S

    return S


def find_nonzero_runs(a, sparse_fill_value):
    # Create an array that is 1 where a is nonzero, and pad each end with an extra 0.
    isnonzero = np.concatenate(([0], (np.asarray(a) != sparse_fill_value).view(np.int8), [0]))
    absdiff = np.abs(np.diff(isnonzero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges


def extract_dense_blocks(PVM, min_number_of_patches=3, min_number_of_images=3, sparse_fill_value=0):
    patches_num = PVM.shape[1]

    dense_blocks_list = []
    for i in range(patches_num - 1):  # the last column may not start a block
        sequences = find_nonzero_runs(PVM[:, i], sparse_fill_value=sparse_fill_value)
        longest_seq = max(sequences, key=lambda x: x[1] - x[0])
        print(f'in column {i} the longest sequence indices: {longest_seq}')

        seq_len = longest_seq[1] - longest_seq[0]
        if seq_len < min_number_of_images * 2:
            print(f'less than {min_number_of_images} images')
            continue
        else:
            for j in range(i + 1, patches_num):
                # by searching for all other patches that are visible in at least the same views
                if any(PVM[longest_seq[0]: longest_seq[1], j] == sparse_fill_value):  # seq end condition
                    if j - i > min_number_of_patches - 1:  # at least 2 points matched
                        dense_blocks_list.append((longest_seq[0], longest_seq[1], i, j))
                    else:
                        print('do not make a dense matrix from one patch')
                    break
                else:
                    print('dense block is rising')
    return dense_blocks_list


def stardardize(mtx):
    trace = (mtx @ mtx.T).trace()
    mtx_stand = mtx / np.sqrt(trace)
    return mtx_stand


def allign_cloud_points_cumulatively(s_list):
    mtx1_stand = stardardize(s_list[0])

    already_alligned_standardized = [mtx1_stand]

    for points_to_allign in s_list[1:]:
        print('raw points_to_allign\n', points_to_allign)

        root = np.vstack(already_alligned_standardized)
        points_to_allign = points_to_allign.T

        pad_num = root.shape[0] - points_to_allign.shape[0]
        print(f'accumulated shape: {root.shape[0]} procrusted with {points_to_allign.shape[0]}')

        if pad_num >= 0:
            shape_before = points_to_allign.shape
            print(f'Adding {pad_num} zero points to the new S')
            points_to_allign = np.pad(points_to_allign, [(0, abs(pad_num)), (0, 0)], constant_values=0)
            print('adjusted points_to_allign\n', points_to_allign)
        else:
            root = np.pad(root, [(0, abs(pad_num)), (0, 0)], constant_values=0)
            print(f'Adding {abs(pad_num)} zero points to the accumulated')

        print(f'Procrusting matrices of shapes {root.shape} and {points_to_allign.shape}')
        try:
            mtx1, mtx2, _ = procrustes(root, points_to_allign)
        except:
            print('was not able to procrust. Probably, the S matrix is of zeros')
            continue

        # NOT SURE ABOUT THIS MOMENT
        if pad_num >= 0:  # returning to the original shape = dropping added zeros
            mtx2 = mtx2[: shape_before[0], :shape_before[1]]

        mtx2_stand = stardardize(mtx2)

        assert np.isclose((mtx2_stand @ mtx2_stand.T).trace(), 1.,
                          rtol=1e-05, atol=1e-08)
        assert np.isclose((mtx1 @ mtx1.T).trace(), 1.,
                          rtol=1e-05, atol=1e-08)

        already_alligned_standardized.append(mtx2_stand)
        print('\n')

    return already_alligned_standardized


if __name__ == '__main__':
    PVM = chaining()
    already_alligned_standardized = factorize_and_stitch(PVM)
    print(already_alligned_standardized)

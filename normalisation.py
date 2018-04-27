import numpy as np
from sklearn import preprocessing

def shape_for_scaler(cqt_segments_array_train_or_valid_or_test):
    num_samples, height, width = cqt_segments_array_train_or_valid_or_test.shape
    array_transposed = np.transpose(cqt_segments_array_train_or_valid_or_test)
    print('transposed:')
    print(array_transposed)
    first_int = height * width
    array_transposed_reshaped = array_transposed.reshape(first_int, num_samples)
    print('array transposed reshaped:')
    print(array_transposed_reshaped)
    transposed_reshaped_transposed = np.transpose(array_transposed_reshaped)
    print('transposed reshaped transposed:')
    print(transposed_reshaped_transposed)
    return transposed_reshaped_transposed, num_samples, height, width

def create_scaler(transposed_reshaped_transposed):
    scaler = preprocessing.StandardScaler()
    scaler.fit(transposed_reshaped_transposed)
    return scaler

def feature_standardize_array(array_shaped_for_scaler, scaler, num_samples, height, width):
    standardized = scaler.transform(array_shaped_for_scaler)
    print('standardized:')
    print(standardized)
    untransposed = np.transpose(standardized)
    print('untransposed:')
    print(untransposed)
    #TODO: check that this reshape is correct (use diff len vectors in diff dimens)
    #replace 2, 2, 2, with input_width, height, depth?
    reshaped = untransposed.reshape(width, height, num_samples)
    print('reshaped:')
    print(reshaped)
    transposed = np.transpose(reshaped)
    print('transposed:')
    print(transposed)
    return transposed

# working mini version:
#
# print('og:')
# #is it valid to normalize overall rather than by bin?
cqt_segments_array = [[[-11, 2], [1, 2]],
                      [[-8, -2], [1, 10]]
                      ]
# cqt_segments_array = np.array(cqt_segments_array)
# print(cqt_segments_array)
#
# print('----------')
#
# cqt_segments_array_transposed = np.transpose(cqt_segments_array)
# print('transposed:')
# print(cqt_segments_array_transposed)
#
# print('----------')
#
# cqt_segments_transposed_reshaped = \
#     np.array(cqt_segments_array_transposed)
# cqt_segments_transposed_reshaped = \
#     cqt_segments_array_transposed.reshape(4, 2)
#
# print('cqt segments transposed reshaped:')
# print(cqt_segments_transposed_reshaped)
#
# print('transposed:')
# cqt_segments_transposed_reshaped_transposed = np.transpose(cqt_segments_transposed_reshaped)
# #so no renaming
# cqt_segments_transposed_reshaped = cqt_segments_transposed_reshaped_transposed
# print(cqt_segments_transposed_reshaped)
#
# scaler = preprocessing.StandardScaler()
# scaler.fit(cqt_segments_transposed_reshaped)
# standardized = scaler.transform(cqt_segments_transposed_reshaped)
# print('standardized:')
# print(standardized)

def main():
    # for testing
    # three_d_array = [[[-11, 2, 1, 1, 1, 1, 1],
    #                   [1, 2, 1, 1, 1, 1, 1],
    #                   [1, 2, 1, 1, 1, 1, 1]],
    #
    #                   [[-8, -2, 1, 1, 1, 1, 1],
    #                    [1, 10, 1, 1, 1, 1, 1],
    #                    [1, 2, 1, 1, 1, 1, 1]]
    #                   ]
    # three_d_array = np.array(three_d_array)

    three_d_array = [[[-11, 2], [1, 2]],
                          [[-8, -2], [1, 10]]
                          ]
    three_d_array = np.array(three_d_array)

    array_shaped_for_scaler, num_samples, height, width = shape_for_scaler(three_d_array)
    scaler = create_scaler(array_shaped_for_scaler)
    standardized = feature_standardize_array(array_shaped_for_scaler, scaler, num_samples, height, width)

if __name__ == '__main__':
    main()
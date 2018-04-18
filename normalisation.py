import numpy as np

#is it valid to normalize overall rather than by bin?
cqt_segments_array = [[[-11, 2], [1, 2]],
                      [[-8, 2], [1, 10]],
                      [[-11, 2], [1, 2]]]
cqt_segments_array = np.array(cqt_segments_array)
shape = cqt_segments_array.shape

#np.random.random((3, 8, 17))

# cqt_min = cqt_segments_array.min(axis=(0, 1, 2), keepdims=True)
# cqt_max = cqt_segments_array.max(axis=(0, 1, 2), keepdims=True)
#better alternative:
cqt_amax = np.amax(cqt_segments_array)
cqt_min = np.amin(cqt_segments_array)

cqt_segments_array_normalized = np.array(cqt_segments_array)

mean = np.mean(cqt_segments_array)
print(mean)

for cqt_segment in cqt_segments_array_normalized:
    for row in cqt_segment:
        for num in row:
            num = (num - mean)/(cqt_max - cqt_min)

print(np.mean(cqt_segments_array_normalized))


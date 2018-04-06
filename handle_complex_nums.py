import numpy as np

a = np.array([1+2j, 3+4j, 5+6j])
# a_real is a VIEW of a. It doesn't take up any more space in memory. It DOES have the base of
# the imaginary numbers.
a_real = a.real
# a_copy_real creates a new object that does NOT have that base.
a_copy_real = np.array(a_real)

print(a_real)

print(type(a[0]))
print(type(a_real[0]))


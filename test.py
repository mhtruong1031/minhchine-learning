import numpy as np

key  = np.array([[2, 16], [10, 31]])
dkey = np.array([[[-31/98, 8/49],[5/49,-1/49]]])

string = "i would have loved to live through the mundane with you"
if len(string) % 2 != 0:
    string += " "

num_str = [ord(x) - ord('a') + 1 if x != " " else 0 for x in string]

if len(string) % 2 != 0:
    string += " "

encoded_num = []
decoded_num = []

# Encoding
for i in range(0, int(len(string)/key.shape[1])):
    token = num_str[2*i:2*i+2]
    encoded_num.append(list(np.dot(key, np.array(token).reshape((2, 1))).reshape((2,))))

# Decoding
for token in encoded_num:
    decoded_num.append(list(np.dot(dkey, np.array(token).reshape((2, 1))).reshape((2,)).tolist()))

print(num_str)
print(encoded_num)
print(decoded_num)
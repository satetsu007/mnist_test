import numpy as np
from PIL import Image
from sklearn.datasets import fetch_mldata 

mnist = fetch_mldata('MNIST original',data_home = ".")
data = mnist["data"]
target = mnist["target"]

print(target)
print(len(data[0]))
print(len(target))

for k in range(70000):
	i = data[k]
	num = []
	for j in range(28):
		index = i[j*28:(j*28)+28]
		num.append(index)

	img = Image.fromarray(np.array(num))
	img.save('./mnist/mnist'+str(k)+'.jpg')

import models
import numpy as np
import matplotlib.pyplot as plt
import utils


batch_size = 1
# model = models.ModularCPPN([500, 500], h_layer_num=5, layer_sizes=[60], scale=0.1)
model = utils.build_basic_repeating_network([500, 500], hidden_size=60, repeat_num=1)

zs = np.random.randn(batch_size, 20)
image = model.forward(zs)[0]
print(image.shape)

plt.imshow(image)
plt.show()

from gp.configs.res_config import ResConfig
from gp.res.data_generator import GenerateData
import matplotlib.pyplot as plt
from tqdm import tqdm
config = ResConfig()
generator = GenerateData(config)
loop = tqdm(generator.next_batch(), total=10)
t=0
for batch_x, batch_y, batch_actions, batch_rewards, new_sequence in loop:
   t+=1
   print(batch_x.shape)
   plt.imsave('./test/'+str(t),batch_x[0,0, :, :, 0], cmap='gray')
   if t>10:
      break

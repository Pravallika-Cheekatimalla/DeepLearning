#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt

fid_scores_dcgan = np.load('./Pravallika/FID/DCGAN.npy')
fid_scores_wgangp = np.load('./Pravallika/FID/WGANGP.npy')
fid_scores_acgan = np.load('./Pravallika/FID/ACGAN.npy')

plt.figure(figsize=(10, 5))
plt.title("FID Score Comparison for DCGAN, WGAN-GP, and ACGAN")
plt.plot(fid_scores_dcgan, label="DCGAN", color='blue')
plt.plot(fid_scores_wgangp, label="WGAN-GP", color='green')
plt.plot(fid_scores_acgan, label="ACGAN", color='red')
plt.xlabel("epochs")
plt.ylabel("FID Score")
plt.legend()

plt.savefig('./Results/FID.jpg', format='jpeg', dpi=100, bbox_inches='tight')
plt.show()

wgangp_min = np.min(fid_scores_wgangp)
wgangp_mean = np.mean(fid_scores_wgangp)
final_wgangp_score = fid_scores_wgangp[-1] 

wgangp_min, wgangp_mean, final_wgangp_score


# In[ ]:





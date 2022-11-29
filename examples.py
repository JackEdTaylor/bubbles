# %% Setup

from PIL import Image
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import bubbles

np.random.seed(152872)

# %% Example 1 - face

face = Image.open(op.join('img', 'face.png'))

face1, mask, mu_x, mu_y, sigma = bubbles.bubbles_mask(im=face, sigma=[20,20,20,20,20], bg=127)

print(mu_x)
print(mu_y)
print(sigma)

face1.save(op.join('examples', 'face1.png'))

fig = plt.figure(figsize=(4.35, 5))
plt.imshow(mask)
plt.colorbar()
fig.savefig(op.join('examples', 'face1_mask.png'), dpi=100, bbox_inches='tight')

# %% Example 2 - face's eyes

face2 = bubbles.bubbles_mask(im=face, mu_x=[85, 186.7], mu_y=[182.5, 182.5], sigma=[20, 10], bg=127)[0]
face2.save(op.join('examples', 'face2.png'))

# %% Example 3 - letter a

a = Image.open(op.join('img', 'a.png'))

a1, mask, mu_x, mu_y, sigma = bubbles.bubbles_mask_nonzero(im=a, sigma=[10,10,10,10], bg=127, max_sigma_from_nonzero=2)

a1.save(op.join('examples', 'a1.png'))

# demonstrate that the space is unused
a2, mask, mu_x, mu_y, sigma = bubbles.bubbles_mask_nonzero(im=a, sigma=np.repeat(3, repeats=150), bg=127, max_sigma_from_nonzero=1)

a2.save(op.join('examples', 'a2.png'))


a_arr = np.asarray(a).copy()
a_arr[a_arr==127] = 0
a_arr[:,:,1] = 0
a_arr[:,:,2] = mask * 255
Image.fromarray(a_arr).save(op.join('examples', 'a2_locs.png'))

# %% Example 4 - cat

cat = Image.open(op.join('img', 'cat.png'))
cat1 = bubbles.bubbles_mask(im=cat, sigma=np.repeat(10, 20), bg=127)[0]
cat2 = bubbles.bubbles_mask(im=cat, sigma=np.repeat(10, 20), bg=0)[0]
cat3 = bubbles.bubbles_mask(im=cat, sigma=np.repeat(10, 20), bg=[127, 0, 127])[0]

cat1.save(op.join('examples', 'cat1.png'))
cat2.save(op.join('examples', 'cat2.png'))
cat3.save(op.join('examples', 'cat3.png'))

# %% Example 5 - masks

# same bubble parameters for all masks
mu_y = [20, 30, 70]
mu_x = [20, 30, 90]
sigma = [5, 10, 7.5]
sh = (100, 100)

masks = [bubbles.build_mask(mu_y, mu_x, sigma, sh, scale=True, max_merge=False),
         bubbles.build_mask(mu_y, mu_x, sigma, sh, scale=True, max_merge=True),
         bubbles.build_mask(mu_y, mu_x, sigma, sh, scale=False, max_merge=False),
         bubbles.build_mask(mu_y, mu_x, sigma, sh, scale=False, max_merge=True)]

for i in range(4):
    fig = plt.figure(figsize=(3, 2.5))
    plt.imshow(masks[i], interpolation=None)
    plt.colorbar()
    fig.savefig(op.join('examples', f'mask{i+1}.png'), dpi=100, bbox_inches='tight')


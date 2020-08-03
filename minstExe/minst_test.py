# coding=utf-8
import time
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models

# x = np.linspace(0, 10, 1000)
# y = np.sin(x)
# plt.figure(figsize=(6, 4))
# plt.plot(x, y, color="red", linewidth=1)
# plt.xlabel("x")  # xlabel、ylabel：分别设置X、Y轴的标题文字。
# plt.ylabel("sin(x)")
# plt.title("正弦曲线图")  # title：设置子图的标题。
# plt.ylim(-1.1, 1.1)  # xlim、ylim：分别设置X、Y轴的显示范围。
# plt.savefig('quxiantu.png', dpi=120, bbox_inches='tight')
# plt.show()
# plt.close()
s_img_url_init = 'init.jpg'
s_img_url_beibian = "beibian.jpg"
s_img_url_beijing = "beijing.jpg"
s_img_url_zimu = 'zimu.jpg'
s_img_url_zitai = 'zitai.jpg'
s_img_url_other = '0.jpg'
vgg = models.vgg16(pretrained=True)
#vgg.features(x)
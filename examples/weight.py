import matplotlib.pyplot as plt
import numpy as np

# 定义权重列表（共用数据）
# weights = [0.4136,
# 0.0174,
# 0.9037,
# 0.1132,
# 0.9496,
# 0.8844,
# 0.9606,
# 0.0463,
# 0.0264,
# 0.0373,
# 0.0173,
# 0.0419,
# 0.9759,
# 0.0326,
# 0.0326,
# 0.0258,
# 0.9759,
# 0.0326,
# 0.9673,
# 0.9673

# ]
weights = [0.49,
0.31,
0.75,
0.08,
0.52,
0.50,
0.34,
0.32,
0.12,
0.49,
0.30,
0.49,
0.87,
0.13
]
weight_array = np.array(weights).reshape(1, -1)

# 统一配置：解决字体警告、优化矢量图渲染、低版本兼容无框
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']  # 优先使用Linux环境自带的DejaVu Sans，解决字体查找警告
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
# 全局默认隐藏所有坐标轴边框（低版本兼容）
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.bottom'] = False
cmap = 'Blues'

# ---------------------- 第一张图：正方形格子权重热力图（低版本兼容无框） ----------------------
fig1, ax1 = plt.subplots(figsize=(50, 2))

# 绘制正方形热力图
im1 = ax1.imshow(weight_array, cmap=cmap, aspect=1, vmin=0, vmax=1)

# 大幅放大、加粗数字标注
for i in range(len(weights)):
    ax1.text(i, 0, f'{weights[i]:.2f}', ha='center', va='center', fontsize=50)

# 彻底关闭坐标轴（低版本兼容，无任何边框/刻度残留）
ax1.set_xticks([])
ax1.set_yticks([])
ax1.axis('off')  # 核心：关闭整个坐标轴，替代frameon，无任何边框残留

# 保存为PDF矢量格式（移除废弃的frameon，靠axis('off')实现无框）
plt.savefig('square_weight_heatmap_no_border.svg', dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.close(fig1)
print("第一张图（完全无框PDF）已保存为 square_weight_heatmap_no_border.svg")

# ---------------------- 第二张图：独立0-1范围颜色图例（低版本兼容无框） ----------------------
dummy_data = np.linspace(0, 1, 100).reshape(1, -1)
fig2, ax2 = plt.subplots(figsize=(2, 6))

# 绘制虚拟渐变图
im2 = ax2.imshow(dummy_data, cmap=cmap, aspect='auto', vmin=0, vmax=1)

# 彻底关闭虚拟图的坐标轴（低版本兼容，无残留）
ax2.set_xticks([])
ax2.set_yticks([])
ax2.axis('off')

# 添加0-1范围颜色条（低版本兼容：移除frameon，后续手动隐藏边框）
cbar2 = plt.colorbar(im2, ax=ax2, fraction=1.0, pad=0.0)
cbar2.set_ticks([0, 0.5, 1])
cbar2.ax.tick_params(labelsize=10, length=0)  # length=0 去除刻度线，更纯净

# 关键：低版本兼容，手动隐藏颜色条边框（替代废弃的frameon=False）
cbar2.outline.set_visible(False)

# 保存为PDF矢量格式（移除废弃的frameon，靠axis('off')实现无框）
plt.savefig('0_1_color_legend_no_border.svg', dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.close(fig2)
print("第二张图（完全无框PDF）已保存为 0_1_color_legend_no_border.svg")
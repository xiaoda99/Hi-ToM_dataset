import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

x_labels = [11, 12, 13, 21, 22, 23, 31, 32, 33]
x_positions = range(len(x_labels))  # 即 [0, 1, 2, 3, 4, 5, 6, 7, 8]

y1=[1.0, 0.8, 0.65, 1.0, 0.75, 0.7, 1.0, 0.75, 0.6]
y2=[1.0, 0.8, 0.9, 1.0, 0.5, 0.55, 1.0, 0.35, 0.35]
y3=[1.0,1.0,0.8182,1.0,0.7273,0.5455,1.0,0.5455,0.4545]

plt.figure(figsize=(10, 6))
plt.plot(x_positions, y1, marker='o', color='blue', label='1')
plt.plot(x_positions, y2, marker='s', color='red', label='2')
plt.plot(x_positions, y3, marker='^', color='green', label='3')
plt.xticks(x_positions, x_labels)
plt.title('不同情况下的折线图', fontsize=16)
plt.xlabel('横坐标', fontsize=14)
plt.ylabel('纵坐标', fontsize=14)
plt.ylim(0, 1.1)
plt.legend()

plt.grid(True, linestyle='--', alpha=0.6)
def add_value_labels(x_pos, y_vals, color, offset_y=0.03):
    for i, (x, y) in enumerate(zip(x_pos, y_vals)):
        plt.text(x, y + offset_y, f'{y:.2f}',
                 color=color, fontsize=9, ha='center', va='bottom')

add_value_labels(x_positions, y1, 'blue')    
add_value_labels(x_positions, y2, 'red')     
add_value_labels(x_positions, y3, 'green')  

plt.tight_layout() 
plt.show()
# 第一行是min_tom，第二行hi-tom去掉选择加大题量，第三行初始
# 4.5
# y1=[1.0, 0.8, 0.65, 1.0, 0.75, 0.7, 1.0, 0.75, 0.6]
# y2=[1.0, 0.8, 0.9, 1.0, 0.5, 0.55, 1.0, 0.35, 0.35]
# y3=[1.0,1.0,0.8182,1.0,0.7273,0.5455,1.0,0.5455,0.4545]
# 4.0
# y1=[1.0, 0.7, 0.55, 1.0, 0.8, 0.75, 1.0, 0.6, 0.4]
# y2=[1.0, 0.95, 0.8, 1.0, 0.8, 0.7, 1.0, 0.8, 0.5]
# y3=[1.0,0.8182,0.6364,1.0,0.7273,0.6364,1.0,0.8182,0.6364]
# opus
# y1=[1.0, 0.9, 0.55, 1.0, 0.85, 0.7, 1.0, 0.8, 0.7]
# y2=[1.0, 1.0, 0.9, 1.0, 0.8, 0.65, 1.0, 0.6, 0.45]
# y3=[1.0,1.0,0.8182,1.0,0.8182,0.6364,1.0,0.6364,0.6364]

import numpy as np
import matplotlib.pyplot as plt
import math
import random
from mpl_toolkits.mplot3d import Axes3D

k = 5.0
l = 1.0

DELTA_T = 0.001
MAX_T = 3.0

def dx(x, y, z, t):
    return -k * x * y
    
def dy(x, y, z, t):
    return k * x * y - l * y
    
def dz(x, y, z, t):
    return l * y
    
def calc_conceved_quantity(x, y, z, t):
    return (l/k)*math.log(x) - x - y
    
def runge_kutta(init_x, init_y, init_z, init_t):
    x, y, z, t = init_x, init_y, init_z, init_t
    history = [[x, y, z, t, calc_conceved_quantity(x, y, z, t)]]
    cnt = 0
    while t < MAX_T:
        cnt += 1
        k1_x = DELTA_T*dx(x, y, z, t)
        k1_y = DELTA_T*dy(x, y, z, t)
        k1_z = DELTA_T*dz(x, y, z, t)
        
        k2_x = DELTA_T*dx(x + k1_x/2, y + k1_y/2, z + k1_z/2, t + DELTA_T/2)
        k2_y = DELTA_T*dy(x + k1_x/2, y + k1_y/2, z + k1_z/2, t + DELTA_T/2)
        k2_z = DELTA_T*dz(x + k1_x/2, y + k1_y/2, z + k1_z/2, t + DELTA_T/2)
        
        k3_x = DELTA_T*dx(x + k2_x/2, y + k2_y/2, z + k2_z/2, t + DELTA_T/2)
        k3_y = DELTA_T*dy(x + k2_x/2, y + k2_y/2, z + k2_z/2, t + DELTA_T/2)
        k3_z = DELTA_T*dz(x + k2_x/2, y + k2_y/2, z + k2_z/2, t + DELTA_T/2)
        
        k4_x = DELTA_T*dx(x + k3_x, y + k3_y, z + k3_z, t + DELTA_T)
        k4_y = DELTA_T*dy(x + k3_x, y + k3_y, z + k3_z, t + DELTA_T)
        k4_z = DELTA_T*dz(x + k3_x, y + k3_y, z + k3_z, t + DELTA_T)
        
        x += (k1_x + 2*k2_x + 2*k3_x + k4_x)/6
        y += (k1_y + 2*k2_y + 2*k3_y + k4_y)/6
        z += (k1_z + 2*k2_z + 2*k3_z + k4_z)/6
        t += DELTA_T
        x = max(0, x)
        y = max(0, y)
        z = max(0, z)
        if cnt % 100 == 0:
            history.append([x, y, z, t, calc_conceved_quantity(x, y, z, t)])
    return history
    
if __name__ == '__main__':

    fig = plt.figure()
    ax = Axes3D(fig)

    for i in range(0, 100):
        init_x = random.random()
        init_y = random.random()
    
        # ルンゲクッタ法で数値計算
        history = np.array(runge_kutta(init_x, init_y, init_z = 0, init_t = 0))
        
        x_min, x_max = min(history[:,0]), max(history[:,0])
        y_min, y_max = min(history[:,1]), max(history[:,1])
        z_min, z_max = min(history[:,2]), max(history[:,2])
        t_min, t_max = 0, MAX_T
        
        '''
        # x vs yの位相図をプロット
        #plt.subplot(4, 1, 1)
        plt.title("x vs y")
        plt.xlim(0, 1)
        plt.ylim(0, 1.4)
        plt.scatter(history[:,0], history[:,1])
        '''
        '''
        # x(健康な人の数)の時間変化をプロット
        plt.subplot(4, 1, 2)
        plt.title("t vs x")
        plt.xlim(t_min, t_max)
        plt.ylim(0, x_max)
        plt.scatter(history[:,3], history[:,0])
        
        # y(病気の人の数)の時間変化をプロット
        plt.subplot(4, 1, 3)
        plt.title("t vs y")
        plt.xlim(t_min, t_max)
        plt.ylim(0, y_max)
        plt.scatter(history[:,3], history[:,1])
        
        # 保存量の時間変化をプロット
        plt.subplot(4, 1, 4)
        plt.title(u"t vs conserved quantity")
        plt.xlim(t_min, t_max)
        plt.scatter(history[:,3], history[:,4])
        '''
        ax.scatter3D(history[:,0], history[:,1], history[:,2])
        
    plt.show()

    
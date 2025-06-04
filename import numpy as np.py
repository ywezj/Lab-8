import numpy as np
import matplotlib.pyplot as plt

# Параметры
L = 3*np.pi/2
x = np.linspace(0, L, 500)
f = lambda x: 1 + np.sin(x)

# Общий ряд Фурье
def S_general(x, N):
    a0 = 2 + 4/(3*np.pi)
    sum = a0/2
    if N >= 1:
        a1 = -2/5
        b1 = 1 + 6/(5*np.pi)
        sum += a1*np.cos(4*x/3) + b1*np.sin(4*x/3)
    return sum

# Ряд по синусам
def S_sin(x, N):
    sum = 0
    for n in range(1, N+1):
        if n == 1:
            bn = (12 + 5*np.pi)/(5*np.pi)
        elif n == 2:
            bn = 9/(2*np.pi)
        elif n == 4:
            bn = -9/(7*np.pi)
        else:
            bn = 0
        sum += bn*np.sin(2*n*x/3)
    return sum

# Ряд по косинусам
def S_cos(x, N):
    a0 = 2 + 4/(3*np.pi)
    sum = a0/2
    for n in range(1, N+1):
        if n == 1:
            an = 12/(5*np.pi) - 1/2
        elif n == 2:
            an = 9/(4*np.pi)
        elif n == 3:
            an = 2/np.pi
        else:
            an = 9/(n**2*np.pi)
        sum += an*np.cos(2*n*x/3)
    return sum

# Построение графиков
plt.figure(figsize=(15, 10))

# Общий ряд
plt.subplot(3, 1, 1)
plt.plot(x, f(x), 'k', label='Исходная')
plt.plot(x, S_general(x, 1), 'b--', label='S1')
plt.plot(x, S_general(x, 5), 'r--', label='S5')
plt.title('Общий ряд Фурье')
plt.legend()

# Ряд по синусам
plt.subplot(3, 1, 2)
plt.plot(x, f(x), 'k')
plt.plot(x, S_sin(x, 5), 'g--', label='S5')
plt.plot(x, S_sin(x, 20), 'm--', label='S20')
plt.title('Ряд по синусам')
plt.legend()

# Ряд по косинусам
plt.subplot(3, 1, 3)
plt.plot(x, f(x), 'k')
plt.plot(x, S_cos(x, 5), 'c--', label='S5')
plt.plot(x, S_cos(x, 20), 'y--', label='S20')
plt.title('Ряд по косинусам')
plt.legend()

plt.tight_layout()
plt.show()
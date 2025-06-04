
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Загрузка изображения с проверкой
input_image = cv2.imread('variant-7.jpg')
if input_image is None:
    print("Ошибка: не удалось загрузить изображение 'variant-7.jpg'")
    print("Убедитесь, что файл существует в текущей директории")
    exit()

# Конвертация цветов
norm = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

# Отражения
flipped_code_0 = cv2.flip(norm, 1)  # Вертикальное отражение
flipped_code_1 = cv2.flip(flipped_code_0, 0)  # Горизонтальное отражение

# Отображение результата
plt.imshow(flipped_code_1)
plt.title('Результат преобразования')
plt.axis('off')  # Скрыть оси
plt.show()
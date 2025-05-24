import time
import numpy as np
import cv2

def load_fly_image(path, size=(64, 64)):
    """Загружаем изображение мухи с прозрачным фоном (если есть альфа-канал)"""
    fly_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if fly_img is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение по пути: {path}")
    
    # Изменяем размер изображения мухи
    fly_img = cv2.resize(fly_img, size)
    
    # Если нет альфа-канала, создаем его (предполагая, что черный цвет - это фон)
    if fly_img.shape[2] == 3:
        gray = cv2.cvtColor(fly_img, cv2.COLOR_BGR2GRAY)
        _, alpha = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        fly_img = cv2.merge([fly_img[:,:,0], fly_img[:,:,1], fly_img[:,:,2], alpha])
    
    return fly_img

def overlay_image(background, overlay, position):
    """Наложение изображения с прозрачностью на фон"""
    x, y = position
    h, w = overlay.shape[:2]
    
    # Область, куда будем накладывать изображение
    roi = background[y:y+h, x:x+w]
    
    # Маска из альфа-канала и инвертированная маска
    overlay_img = overlay[:,:,:3]
    mask = overlay[:,:,3] / 255.0
    inv_mask = 1.0 - mask
    
    # Наложение с учетом прозрачности
    for c in range(0, 3):
        roi[:,:,c] = (overlay_img[:,:,c] * mask + roi[:,:,c] * inv_mask).astype(np.uint8)
    
    return background

def video_processing():
    # Загружаем изображение мухи
    try:
        fly_img = load_fly_image('fly64.png')
    except Exception as e:
        print(f"Ошибка загрузки изображения мухи: {e}")
        return
    
    fly_size = fly_img.shape[0]  # Предполагаем квадратное изображение
    fly_half = fly_size // 2
    
    cap = cv2.VideoCapture(0)
    down_points = (640, 480)
    i = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, down_points, interpolation=cv2.INTER_LINEAR)
        frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        ret, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

        contours, hierarchy = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            
            # Рисуем прямоугольник вокруг объекта
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Вычисляем центр объекта
            obj_center = (x + w // 2, y + h // 2)
            
            # Рисуем точку в центре объекта
            cv2.circle(frame, obj_center, 5, (0, 0, 255), -1)
            
            # Вычисляем расстояние от центра кадра до центра объекта
            distance = np.sqrt((frame_center[0] - obj_center[0]) ** 2 + (frame_center[1] - obj_center[1]) ** 2)
            
            # Выводим расстояние
            cv2.putText(frame, f"Distance: {distance:.1f} px", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Наложение изображения мухи (центрируем по объекту)
            fly_x = obj_center[0] - fly_half
            fly_y = obj_center[1] - fly_half
            
            # Проверяем, чтобы координаты не выходили за границы кадра
            if fly_x >= 0 and fly_y >= 0 and (fly_x + fly_size) < frame.shape[1] and (fly_y + fly_size) < frame.shape[0]:
                frame = overlay_image(frame, fly_img, (fly_x, fly_y))
            
            if i % 5 == 0:
                print("Object center:", obj_center, "| Distance:", distance)

        cv2.imshow('Tracking with Fly', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.1)
        i += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_processing()
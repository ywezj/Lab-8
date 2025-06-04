import time
import numpy as np
import cv2

def video_processing():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка: не удалось открыть камеру!")
        return
    
    down_points = (640, 480)
    i = 0
    
    # Создаём окно с возможностью изменения размера
    cv2.namedWindow('Object Tracking', cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка: не удалось получить кадр!")
            break

        # Изменение размера кадра
        frame = cv2.resize(frame, down_points, interpolation=cv2.INTER_LINEAR)
        
        # Определяем центр кадра
        frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)
        
        # Преобразование в градации серого и размытие
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Бинаризация с инверсией (чтобы метка была белой на чёрном фоне)
        _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

        # Поиск контуров
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) > 0:
            # Находим контур с максимальной площадью
            c = max(contours, key=cv2.contourArea)
            
            # Получаем прямоугольник, описывающий контур
            x, y, w, h = cv2.boundingRect(c)
            
            # Фильтр по размеру (чтобы игнорировать мелкие объекты)
            if w > 50 and h > 50:
                # Рисуем прямоугольник вокруг метки
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Вычисляем центр метки
                obj_center = (x + w // 2, y + h // 2)
                
                # Рисуем центр метки
                cv2.circle(frame, obj_center, 5, (0, 0, 255), -1)
                
                # Рисуем центр кадра
                cv2.circle(frame, frame_center, 5, (255, 0, 0), -1)
                
                # Рисуем линию между центрами
                cv2.line(frame, frame_center, obj_center, (255, 0, 255), 2)
                
                # Вычисляем расстояние между центрами
                distance = np.sqrt((frame_center[0] - obj_center[0]) ** 2 + 
                              (frame_center[1] - obj_center[1]) ** 2)
                
                # Выводим информацию на кадр
                cv2.putText(frame, f"Distance: {distance:.1f} px", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"Object at: {obj_center}", (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                # Выводим информацию в консоль
                if i % 5 == 0:
                    print(f"Object center: {obj_center} | Distance: {distance:.1f} px")

        # Отображаем кадр
        cv2.imshow('Object Tracking', frame)
        
        # Выход по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.1)
        i += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_processing()

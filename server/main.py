import cv2
import numpy as np
import time
import random
import string
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# Конфигурация
CONFIG = {
    "font_path": "C:/Windows/Fonts/arial.ttf",
    "checkpoint_interval": 0.5,
    "min_size": 500,
    "max_objects": 8,
    "parked_threshold": 4,
    "tolerance": 1.5,
    "prediction_strength": 0.7,
    "path_thickness": 1,
    "center_dot_radius": 2,
    "object_lifetime": 2
}

# Генерация случайного тега
def generate_random_tag(length=5):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

# Функция для вычисления расстояния между двумя точками
def calculate_distance(p1, p2):
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

# Функция для вычисления скорости (расстояние / время)
def calculate_speed(distance, time_diff):
    return distance / time_diff if time_diff > 0 else 0

# Функция для рисования перевернутого треугольника на изображении
def draw_inverted_triangle(img, points, color=(255, 0, 0), thickness=2):
    p1, p2, p3 = points
    cv2.line(img, tuple(p1), tuple(p2), color, thickness)
    cv2.line(img, tuple(p2), tuple(p3), color, thickness)
    cv2.line(img, tuple(p3), tuple(p1), color, thickness)

# Функция для определения точек перевернутого треугольника (пирамида)
def get_inverted_triangle_points(x1, y1, x2, y2):
    top_point = ((x1 + x2) // 2, y2)
    return [(x1, y1), (x2, y1), top_point]

# Функция для определения направления движения
def determine_direction_from_path(path, tolerance=5):
    if len(path) < 2:
        return 'none'

    start_point = path[0]
    end_point = path[-1]

    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]

    if abs(dx) > tolerance and abs(dy) <= tolerance:
        return 'right' if dx > 0 else 'left'
    elif abs(dy) > tolerance and abs(dx) <= tolerance:
        return 'down' if dy > 0 else 'up'
    else:
        return 'diagonal'

# Переходная матрица для предсказаний
def calculate_next_flow(flow_values):
    weights = np.array([0.5, 0.3, 0.2])  # Веса для предсказателей
    predictors = [
        lambda x: x[-1],  # Последнее значение
        lambda x: x[-1] * 0.9 + x[-2] * 0.1 if len(x) > 1 else x[-1],  # Сглаживание
        lambda x: np.mean(x[-3:]) if len(x) > 3 else x[-1],  # Среднее
    ]
    predictions = np.array([predictor(flow_values) for predictor in predictors])
    next_flow = np.dot(weights, predictions)
    return next_flow

# Функция для наложения полупрозрачных путей
def overlay_line_with_gradient(frame, start_point, end_point, start_color, end_color, alpha=0.25, thickness=2):
    steps = int(calculate_distance(start_point, end_point))
    for i in range(steps):
        interpolated_color = tuple(
            int(start_color[j] + (end_color[j] - start_color[j]) * i / steps) for j in range(3)
        )
        t = i / steps
        intermediate_point = (int(start_point[0] * (1 - t) + end_point[0] * t),
                              int(start_point[1] * (1 - t) + end_point[1] * t))
        if i > 0:
            cv2.line(frame, prev_point, intermediate_point, interpolated_color, thickness)
        prev_point = intermediate_point

# Функция для отображения путей и чекпоинтов
def draw_paths_and_centers(frame, tracked_objects, direction_colors):
    for vehicle_tag, data in tracked_objects.items():
        points = data["points"]

        if len(points) < 2:
            continue

        direction = determine_direction_from_path(points)
        color = direction_colors.get(direction, (255, 255, 0))

        for i in range(1, len(points)):
            prev_point = points[i - 1]
            current_point = points[i]
            overlay_line_with_gradient(frame, prev_point, current_point, color, color, alpha=0.25, thickness=CONFIG["path_thickness"])

        # Центр объекта
        last_position = data["last_position"]
        cv2.circle(frame, last_position, CONFIG["center_dot_radius"], (0, 255, 0), -1)

        # Отображение чекпоинтов
        for idx, point in enumerate(points):
            checkpoint_color = color  # Используем тот же цвет для чекпоинтов
            cv2.circle(frame, point, 3, checkpoint_color, -1)

            # Отображение метки с токеном и индексом чекпоинта
            label = f"{vehicle_tag}.{idx + 1}"
            cv2.putText(frame, label, (point[0] + 5, point[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

# Таймеры для обновления значений
class UpdateTimer:
    def __init__(self, interval):
        self.interval = interval
        self.last_update = None

    def should_update(self):
        current_time = time.time()
        if self.last_update is None or (current_time - self.last_update >= self.interval):
            self.last_update = current_time
            return True
        return False

# Таймеры
flow_timer = UpdateTimer(CONFIG["checkpoint_interval"])
state_timer = UpdateTimer(2)

# Функция для сглаживания координат
def smooth_coordinates(current_position, last_position, alpha=0.5):
    smoothed_x = int(alpha * current_position[0] + (1 - alpha) * last_position[0])
    smoothed_y = int(alpha * current_position[1] + (1 - alpha) * last_position[1])
    return smoothed_x, smoothed_y

# Функция для обнаружения пробок
def detect_traffic_jam(tracked_objects, parked_threshold=CONFIG["parked_threshold"]):
    total_count = len(tracked_objects)
    if total_count == 0:
        return False  # Если нет объектов, то нет пробки

    parked_count = sum(1 for data in tracked_objects.values() 
                       if len(data["points"]) > parked_threshold and np.allclose(data["points"][-1], data["points"][-parked_threshold], atol=5))
    
    return (parked_count / total_count) > 0.8  # Если более 80% машин двигаются медленно

# Функция для вычисления точности предсказания
def calculate_accuracy(predicted_flow, real_flow):
    if real_flow == 0:
        return 100 if predicted_flow < 1 else 0
    error = abs(predicted_flow - real_flow) / real_flow
    accuracy = max(0, (1 - error) * 100)
    return accuracy

# Обновленная функция для обработки кадра
def frame_process(frame, model, tracked_objects, flow_history):
    current_time = time.time()

    # Удаление устаревших объектов
    to_remove = []
    for tag, data in tracked_objects.items():
        if current_time - data["last_checkpoint"] > CONFIG["object_lifetime"]:
            to_remove.append(tag)
    for tag in to_remove:
        del tracked_objects[tag]

    # Обработка кадра
    results = model(frame, stream=True)

    # Обработка объектов
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls = int(box.cls[0])

            if cls not in [2, 7]:  # Предполагается, что это только автомобили
                continue

            width_obj = x2 - x1
            height_obj = y2 - y1
            if width_obj * height_obj < CONFIG["min_size"]:
                continue

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            object_tag = None
            min_distance = float('inf')

            # Определение ближайшего отслеживаемого объекта
            for obj, data in tracked_objects.items():
                prev_x, prev_y = data["last_position"]
                distance = calculate_distance((center_x, center_y), (prev_x, prev_y))

                if distance < min_distance and distance < 50:
                    min_distance = distance
                    object_tag = data["tag"]

            if not object_tag:
                object_tag = generate_random_tag()

            if object_tag not in tracked_objects:
                tracked_objects[object_tag] = {
                    "tag": object_tag,
                    "last_position": (center_x, center_y),
                    "points": [(center_x, center_y)],
                    "last_checkpoint": current_time,
                }

            smoothed_position = smooth_coordinates((center_x, center_y), tracked_objects[object_tag]["last_position"])
            tracked_objects[object_tag]["last_position"] = smoothed_position

            # Обновление чекпоинтов
            if current_time - tracked_objects[object_tag]["last_checkpoint"] > CONFIG["checkpoint_interval"]:
                tracked_objects[object_tag]["points"].append(smoothed_position)
                if len(tracked_objects[object_tag]["points"]) > 15:
                    tracked_objects[object_tag]["points"].pop(0)
                tracked_objects[object_tag]["last_checkpoint"] = current_time

    # Общая скорость для предсказания потока
    total_vehicles = len(tracked_objects)
    total_speed = sum(calculate_vehicle_speed(data, current_time) 
                      for data in tracked_objects.values())
    avg_speed = total_speed / total_vehicles if total_vehicles > 0 else 0.1

    # Прогнозирование потока на основе средней скорости
    video_flow = total_vehicles * np.exp(-CONFIG["prediction_strength"] / (avg_speed + 1))
    flow_history.append(video_flow)
    flow_history = flow_history[-10:]

    # Предсказание потока
    predicted_flow = calculate_next_flow(flow_history)

    # Статистика точности
    real_flow = total_vehicles * np.exp(-CONFIG["prediction_strength"] / (avg_speed + 1))  # Реальный поток
    accuracy = calculate_accuracy(predicted_flow, real_flow)

    global last_flow_value, last_traffic_state
    if flow_timer.should_update():
        last_flow_value = predicted_flow

    if state_timer.should_update():
        if detect_traffic_jam(tracked_objects):  # Проверка на пробку
            last_traffic_state = "Пробки"
        elif last_flow_value < 15.0:
            last_traffic_state = "Свободная дорога"
        elif 15.0 < last_flow_value < 30.0:
            last_traffic_state = "Среднее движение"
        elif 30.0 < last_flow_value < 50.0:
            last_traffic_state = "Тяжелое движение"
        else:
            last_traffic_state = "Высокая плотность"

# Функция для отображения путей и чекпоинтов
def draw_paths_and_centers(frame, tracked_objects, direction_colors):
    for vehicle_tag, data in tracked_objects.items():
        points = data["points"]

        if len(points) < 2:
            continue

        direction = determine_direction_from_path(points)
        color = direction_colors.get(direction, (255, 255, 0))

        for i in range(1, len(points)):
            prev_point = points[i - 1]
            current_point = points[i]
            overlay_line_with_gradient(frame, prev_point, current_point, color, color, alpha=0.25, thickness=CONFIG["path_thickness"])

        # Центр объекта
        last_position = data["last_position"]
        cv2.circle(frame, last_position, CONFIG["center_dot_radius"], (0, 255, 0), -1)

        # Отображение чекпоинтов
        for idx, point in enumerate(points):
            checkpoint_color = color  # Используем тот же цвет для чекпоинтов
            cv2.circle(frame, point, 3, checkpoint_color, -1)

            # Отображение метки с токеном и индексом чекпоинта
            label = f"{vehicle_tag}.{idx + 1}"
            cv2.putText(frame, label, (point[0] + 5, point[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

# Таймеры для обновления значений
class UpdateTimer:
    def __init__(self, interval):
        self.interval = interval
        self.last_update = None

    def should_update(self):
        current_time = time.time()
        if self.last_update is None or (current_time - self.last_update >= self.interval):
            self.last_update = current_time
            return True
        return False

# Таймеры
flow_timer = UpdateTimer(CONFIG["checkpoint_interval"])
state_timer = UpdateTimer(2)

# Функция для сглаживания координат
def smooth_coordinates(current_position, last_position, alpha=0.5):
    smoothed_x = int(alpha * current_position[0] + (1 - alpha) * last_position[0])
    smoothed_y = int(alpha * current_position[1] + (1 - alpha) * last_position[1])
    return smoothed_x, smoothed_y

# Функция для обнаружения пробок
# Функция для вычисления скорости движения автомобиля на основе координат
def calculate_vehicle_speed(vehicle_data, current_time):
    if len(vehicle_data["points"]) < 2:
        return 0
    last_position = vehicle_data["points"][-1]
    prev_position = vehicle_data["points"][-2]
    time_diff = current_time - vehicle_data["last_checkpoint"]
    distance = calculate_distance(last_position, prev_position)
    speed = calculate_speed(distance, time_diff)
    return speed

# Функция для определения пробки (на основе скорости машин)
def detect_traffic_jam(tracked_objects, parked_threshold=CONFIG["parked_threshold"]):
    total_count = len(tracked_objects)
    if total_count == 0:
        return False  # Если нет объектов, то нет пробки

    parked_count = sum(1 for data in tracked_objects.values() 
                       if len(data["points"]) > parked_threshold and np.allclose(data["points"][-1], data["points"][-parked_threshold], atol=5))
    
    return (parked_count / total_count) > 0.8  # Если более 80% машин двигаются медленно


# Функция для вычисления точности предсказания
def calculate_accuracy(predicted_flow, real_flow):
    if real_flow == 0:
        return 100 if predicted_flow < 1 else 0
    error = abs(predicted_flow - real_flow) / real_flow
    accuracy = max(0, (1 - error) * 100)
    return accuracy

# Обновленная функция для обработки кадра
def frame_process(frame, model, tracked_objects, flow_history):
    current_time = time.time()

    # Удаление устаревших объектов
    to_remove = []
    for tag, data in tracked_objects.items():
        if current_time - data["last_checkpoint"] > CONFIG["object_lifetime"]:
            to_remove.append(tag)
    for tag in to_remove:
        del tracked_objects[tag]

    # Обработка кадра
    results = model(frame, stream=True)

    # Обработка объектов
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls = int(box.cls[0])

            if cls not in [2, 7]:  # Предполагается, что это только автомобили
                continue

            width_obj = x2 - x1
            height_obj = y2 - y1
            if width_obj * height_obj < CONFIG["min_size"]:
                continue

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            object_tag = None
            min_distance = float('inf')

            # Определение ближайшего отслеживаемого объекта
            for obj, data in tracked_objects.items():
                prev_x, prev_y = data["last_position"]
                distance = calculate_distance((center_x, center_y), (prev_x, prev_y))

                if distance < min_distance and distance < 50:
                    min_distance = distance
                    object_tag = data["tag"]

            if not object_tag:
                object_tag = generate_random_tag()

            if object_tag not in tracked_objects:
                tracked_objects[object_tag] = {
                    "tag": object_tag,
                    "last_position": (center_x, center_y),
                    "points": [(center_x, center_y)],
                    "last_checkpoint": current_time,
                }

            smoothed_position = smooth_coordinates((center_x, center_y), tracked_objects[object_tag]["last_position"])
            tracked_objects[object_tag]["last_position"] = smoothed_position

            # Обновление чекпоинтов
            if current_time - tracked_objects[object_tag]["last_checkpoint"] > CONFIG["checkpoint_interval"]:
                tracked_objects[object_tag]["points"].append(smoothed_position)
                if len(tracked_objects[object_tag]["points"]) > 15:
                    tracked_objects[object_tag]["points"].pop(0)
                tracked_objects[object_tag]["last_checkpoint"] = current_time

    # Общая скорость для предсказания потока
    total_vehicles = len(tracked_objects)
    total_speed = sum(calculate_vehicle_speed(data, current_time) 
                      for data in tracked_objects.values())
    avg_speed = total_speed / total_vehicles if total_vehicles > 0 else 0.1

    # Прогнозирование потока на основе средней скорости
    video_flow = total_vehicles * np.exp(-CONFIG["prediction_strength"] / (avg_speed + 1))
    flow_history.append(video_flow)
    flow_history = flow_history[-10:]

    # Предсказание потока
    predicted_flow = calculate_next_flow(flow_history)

    # Статистика точности
    real_flow = total_vehicles * np.exp(-CONFIG["prediction_strength"] / (avg_speed + 1))  # Реальный поток
    accuracy = calculate_accuracy(predicted_flow, real_flow)

    global last_flow_value, last_traffic_state
    if flow_timer.should_update():
        last_flow_value = predicted_flow

    if state_timer.should_update():
        if detect_traffic_jam(tracked_objects):  # Проверка на пробку
            last_traffic_state = "Пробки"
        elif last_flow_value < 15.0:
            last_traffic_state = "Свободная дорога"
        elif 15.0 < last_flow_value < 30.0:
            last_traffic_state = "Среднее движение"
        elif 30.0 < last_flow_value < 40.0:
            last_traffic_state = "Загруженная дорога"
        else:
            last_traffic_state = "Пробки"

    # Обновление текста о состоянии трафика
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype(CONFIG["font_path"], 20)

    text = f"Текущий поток: {last_flow_value:.2f} | Состояние трафика: {last_traffic_state} | Точность: {accuracy:.2f}%"
    text_bbox = draw.textbbox((15, 25), text, font=font)
    draw.rectangle([(15, 25), (text_bbox[2], text_bbox[3])], fill=(0, 0, 0, 128))
    draw.text((15, 25), text, fill=(255, 255, 255), font=font)

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR), tracked_objects, flow_history


def run_video_analysis(video_path, output_path):
    # Загрузка моделей
    model_traffic = YOLO('models/yolo11x.pt')  # Модель для анализа движения
    model_crash = YOLO('models/bestbest.pt')  # Модель для обнаружения аварий

    cap = cv2.VideoCapture(video_path)

    global last_flow_value, last_traffic_state
    last_flow_value = 0
    last_traffic_state = "Нет данных"
    
    if not cap.isOpened():
        print("Ошибка при открытии видео файла!")
        return

    # Инициализация истории движения и отслеживаемых объектов
    flow_history = []
    tracked_objects = {}

    # Определение цветов для направлений движения
    direction_colors = {
        'up': (0, 0, 255),  # Красный для движения вверх
        'down': (0, 255, 0),  # Зеленый для движения вниз
        'left': (255, 0, 0),  # Синий для движения влево
        'right': (255, 255, 0),  # Желтый для движения вправо
        'diagonal': (255, 0, 255)  # Фиолетовый для диагонального движения
    }

    # Настроим VideoWriter для сохранения результата
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Кодек mp4v для записи в формате MP4
    out = cv2.VideoWriter(output_path, fourcc, 30, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Обработка каждого кадра
        frame, tracked_objects, flow_history = frame_process(
            frame,
            model=model_traffic,
            tracked_objects=tracked_objects,
            flow_history=flow_history
        )

        # Анализ аварий
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        crash_results = model_crash.predict(source=rgb_frame, conf=0.5)  # Ожидаем предсказание от модели

        # Обработка обнаруженных аварий
        crash_count = 0
        for result in crash_results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Координаты прямоугольника
                confidence = box.conf[0]

                if confidence > 0.5 and int(box.cls[0]) == 0:  # Проверка на класс "авария"
                    crash_count += 1
                    triangle_points = get_inverted_triangle_points(x1, y1, x2, y2)
                    draw_inverted_triangle(frame, triangle_points, color=(0, 0, 255), thickness=3)

        # Отображение путей и направлений движения
        draw_paths_and_centers(frame, tracked_objects, direction_colors)

        # Сохранение обработанного кадра в выходное видео
        out.write(frame)

     

    # Освобождаем ресурсы
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Видео обработано и сохранено в:", output_path)

# Пример использования:
video_path = 'cars.mp4'  # Входной видеофайл
output_path = 'output_video.mp4'  # Выходной видеофайл с расширением .mp4
run_video_analysis(video_path, output_path)
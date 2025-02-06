import os
from pathlib import Path
from typing import Any

import keras
import numpy as np
from flask import Flask, redirect, render_template, request
from PIL import Image

app_dir = Path(os.getcwd())

# Параметры изображения
IMG_SIZE = (224, 224)

# Загружаем обученную модель
models_dir = app_dir.joinpath("models")
models = sorted(models_dir.iterdir())
model_name = models_dir.joinpath(models[-1])
print(f"Model: {model_name}")
model: Any = keras.models.load_model(model_name)  # latest by default

# Словарь классов (нужно заменить на реальные классы, которые использовались при обучении)
class_names = ["Урбанистика", "Фантастика", "Пейзаж"]

# Создаем Flask-приложение
app = Flask(__name__, template_folder=app_dir.joinpath("gui_eval", "templates"))


# Главная страница с формой загрузки
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Проверяем, загружен ли файл
        file = request.files.get("file")
        if not file or file.filename == "":
            return redirect(request.url)

        # Открываем изображение напрямую из потока
        img = Image.open(file.stream)

        # Классифицируем изображение
        result = classify_image(img)

        return render_template("index.html", result=result)

    return render_template("index.html", result=None)


# Функция для предсказания класса изображения
def classify_image(img):
    img = img.resize(IMG_SIZE)
    img = img.convert("RGB")
    img_array = np.array(img) / 255.0  # Нормализуем изображение
    img_array = np.expand_dims(img_array, axis=0)  # Добавляем размерность для батча
    predictions = model.predict(img_array)[0]  # Предсказания модели
    result_text = "Результаты классификации:\n"
    for i, class_name in enumerate(class_names):
        confidence = predictions[i] * 100  # Уверенность по каждому классу
        result_text += f"\n{class_name}: {confidence:.2f}%\n"
    return result_text


# Запускаем сервер Flask
if __name__ == "__main__":
    app.run("127.0.0.1", port=5380, debug=False)

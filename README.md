# Neural Style Transfer

Это проект для переноса стиля между изображениями с использованием нейронных сетей и фреймворка Streamlit.

## Описание

Данный проект позволяет загружать контент-изображение и стиль-изображение, а затем выполнять перенос стиля, создавая новое изображение с содержимым контент-изображения и стилем стиль-изображения.

## Установка и запуск

### Использование Docker

1. Создайте Docker образ:
   docker build -t neural-style-transfer .
2. Запустите Docker контейнер:
   docker run -p 8501:8501 neural-style-transfer
3. Откройте приложение в браузере по адресу:
   http://localhost:8501

### Локальная установка

1. Клонируйте репозиторий:
   git clone https://github.com/yourusername/NeuralStyleTransfer.git
   cd NeuralStyleTransfer
2. Создайте и активируйте виртуальное окружение:
   python -m venv .venv
   source .venv/bin/activate  # Для Windows: .venv\Scripts\activate
3. Установите зависимости:
   pip install -r requirements.txt
4. Запустите приложение:
   streamlit run app.py

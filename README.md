# Neural Style Transfer

Это проект для переноса стиля между изображениями с использованием нейронных сетей и фреймворка Streamlit.

## Описание

Данный проект позволяет загружать контент-изображение и стиль-изображение, а затем выполнять перенос стиля, создавая новое изображение с содержимым контент-изображения и стилем стиль-изображения.

## Установка и запуск

### Использование Docker

1. ```docker build -t neural-style-transfer .```
2. ```docker run -p 8501:8501 neural-style-transfer```
3. ```http://localhost:8501```

### Локальная установка

1. ```git clone https://github.com/yourusername/NeuralStyleTransfer.git```
2. ```cd NeuralStyleTransfer```
3. ```python -m venv .venv```
   ```source .venv/bin/activate  # Для Windows: .venv\Scripts\activate```
4. ```pip install -r requirements.txt```
5. ```streamlit run app.py```

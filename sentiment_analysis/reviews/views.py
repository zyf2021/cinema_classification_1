from django.shortcuts import render
from .forms import ReviewForm
import pickle
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from .utils import preprocess



# Загрузите вашу модель и векторизатор
with open(r'C:\Users\Asus\PycharmProjects\cinema_classification\sentiment_analysis\reviews\model.pkl', 'rb') as f:
    model = pickle.load(f)

with open(r'C:\Users\Asus\PycharmProjects\cinema_classification\sentiment_analysis\reviews\vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)



def home(request):
    if request.method == 'POST':
        form = ReviewForm(request.POST)
        if form.is_valid():
            rating = form.cleaned_data['rating']
            comment = form.cleaned_data['comment']
            # Предобработка комментария
            # processed_comment = preprocess(comment)  # Используйте вашу функцию предобработки
            # Векторизация
            # tfidf_comment = vectorizer.transform([processed_comment])

            # Применяем предобработку
            preprocessed_comment = preprocess(comment)
            # Преобразуем новый комментарий в TF-IDF вектор
            new_comment_tfidf = vectorizer.transform([preprocessed_comment])
            # Создание массива для рейтинга
            new_rating_array = np.array([[rating]])
            # Преобразуем рейтинг в разреженный формат
            new_rating_sparse = csr_matrix(new_rating_array)
            # Объединение признаков
            new_input = hstack([new_comment_tfidf, new_rating_sparse])
            # Прогнозирование
            prediction = model.predict(new_input)
            # Интерпретация результата
            sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
            return render(request, 'reviews/result.html',
                          {'rating': rating, 'comment': comment, 'sentiment': sentiment})
    else:
        form = ReviewForm()
    return render(request, 'reviews/home.html', {'form': form})

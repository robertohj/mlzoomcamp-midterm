FROM python:3.11-slim
WORKDIR /app

RUN pip install pandas
RUN pip install scikit-learn
RUN pip install Flask
RUN pip install waitress
RUN pip install requests

COPY ["predict_model_employee_review.py", "model.bin","./"]

EXPOSE 9696

ENTRYPOINT ["waitress-serve", "--host=127.0.0.1", "--port=9696", "predict_model_employee_review:app"]
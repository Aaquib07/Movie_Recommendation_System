# Movie_Recommendation_System
## How to run the application?
* Fork this repository
* Create a virtual environment on your local machine
* * In Linux
```bash
python3 -m venv (YOUR_VIRTUAL_ENV_NAME)
```
* * In Windows
```bash
python -m venv (YOUR_VIRTUAL_ENV_NAME)
```
* Activate your virtual environment
* * In Linux
```bash
source (YOUR_VIRTUAL_ENV_NAME)/bin/activate
```
* * In Windows
```bash
(YOUR_VIRTUAL_ENV_NAME)\Scripts\activate
```
* Install all the required libraries from requirements.txt file
```bash
pip install -r requirements.txt
```
* Run the recommender.py file
* * In Linux
```bash
python3 recommender.py
```
* * In Windows
```bash
python recommender.py
```
* Run the app.py file
```bash
streamlit run app.py
```

## Description
- A Streamlit-based application to recommend movies to the user.
- Content-based filtering has been used.
- Cosine similarity has been used to calculate the similarity between movies.

## Dataset
TMDB 5000 Movie Dataset has been used as our project dataset. This dataset contains information about movies such as cast, crews, genres, etc. The link for this dataset is [here](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

## Contact
* Email ID: [aaquibasrar4@gmail.com](mailto:aaquibasrar4@gmail.com)
* LinkedIn: [https://www.linkedin.com/in/aaquib-asrar/](https://www.linkedin.com/in/aaquib-asrar/)

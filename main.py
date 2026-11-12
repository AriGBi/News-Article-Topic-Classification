import numpy as np
import pandas as pd
import html
import re
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC



def clean_html(text):

    if not isinstance(text, str):
        return ""
    text=text.lower()
    text = html.unescape(text)
    text = re.sub(r'[<>/="]',' ',text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\b[a-z]\b', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def preprocess_data(df, is_eval=False):
    
    df_copy = df.copy()
    if not is_eval and len(df_copy) > 53004:
        x_copy = np.vstack((df_copy.iloc[:53004,:], df_copy.iloc[53005:,:]))
        df_copy = pd.DataFrame(x_copy, columns=df.columns)


    if not is_eval:
        df_copy = df_copy[df_copy['article'].astype(str).str.strip() != ''].copy()
        df_copy = df_copy.drop_duplicates(subset=['title', 'article', 'label'], keep='first')
        df_copy = df_copy.drop_duplicates(subset=['title', 'article'], keep=False)

    df_copy["all_text"] = (df_copy["source"].astype(str) + "." + df_copy["title"].astype(str) + "." +  df_copy["article"].astype(str))
    df_copy["all_text"] = df_copy["all_text"].apply(clean_html)
    df_copy['all_text'] = df_copy['all_text'].replace(r'[\n\r\t]+', ' ', regex=True)
    df_copy['all_text'] = df_copy['all_text'].replace(r'\s+', ' ', regex=True).str.strip()


    if not is_eval:
        df_copy = df_copy.drop_duplicates(subset=['all_text', 'label'], keep='first')
        df_copy = df_copy.drop_duplicates(subset=['all_text'], keep=False)
        df_copy = df_copy[df_copy['all_text'].str.len() > 20]


    df_copy['num_word_log'] = df_copy['all_text'].apply(lambda x: np.log1p(len(str(x).split())))

    df_copy['date_missing'] = df_copy['timestamp'].str.startswith('0000').astype(int)
    df_copy['timestamp_dt'] = pd.to_datetime(df_copy['timestamp'], errors='coerce')
    df_copy['year'] = df_copy['timestamp_dt'].dt.year
    df_copy['day_of_week'] = df_copy['timestamp_dt'].dt.dayofweek

    df_clean = df_copy.drop(columns=["article", "title", "timestamp", "timestamp_dt"], errors='ignore')
    
    return df_clean


def to_list(x):
    return x.iloc[:, 0].tolist()


if __name__ == "__main__":


    df=pd.read_csv('development.csv')
    ev = pd.read_csv("evaluation.csv")
        
    df_train = preprocess_data(df, is_eval=False)
    df_eval = preprocess_data(ev, is_eval=True)

    num_features = ["page_rank", "year", "day_of_week", "date_missing", "num_word_log"]
    cat_features = ["source"]
    text_col = ["all_text"]
    
    X = df_train[num_features + cat_features + text_col]
    y = df_train["label"].astype(int).values
    
    preproc = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy='mean'), StandardScaler()), num_features),
        ("text", make_pipeline(FunctionTransformer(to_list, validate=False), 
                               TfidfVectorizer(min_df=2, max_df=0.3, ngram_range=(1,3), sublinear_tf=True, stop_words="english", strip_accents='unicode')), text_col),
        ("cat", make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown="ignore")), cat_features)
    ])
    
    X_eval = df_eval[num_features + cat_features + text_col]

    Logistic_Regression_model = Pipeline([
        ('preproc', preproc),
        ('clf', LogisticRegression(C=4, class_weight='balanced', max_iter=1000, random_state=42, solver='saga'))
    ])
    Logistic_Regression_model.fit(X, y)
    y_pred_LR = Logistic_Regression_model.predict(X_eval)
    submission = pd.DataFrame({
        'Id': ev['Id'], 
        'Predicted': y_pred_LR
    })
    submission.to_csv("submission_Logistic_Regression.csv", index=False)


    Linear_SVC_model = Pipeline([
        ('preproc', preproc),
        ('clf', LinearSVC(C=0.5, class_weight='balanced', random_state=42))
    ])
    Linear_SVC_model.fit(X, y)
    y_pred_L_SVC = Linear_SVC_model.predict(X_eval)
    submission = pd.DataFrame({
        'Id': ev['Id'], 
        'Predicted': y_pred_L_SVC
    })
    submission.to_csv("submission_Linear_SVC.csv", index=False)

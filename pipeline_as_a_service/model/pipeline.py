import pandas as pd
import dill
import datetime

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.model_selection import cross_val_score, StratifiedKFold


def pipeline():
    import pandas as pd
    df = get_final_merged_data_set()

    categorical_columns = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent', 'utm_keyword',
                           'device_category', 'device_brand', 'geo_country', 'geo_city']

    column_transformer = ColumnTransformer(transformers=[
                                                        ('Target encoding', TargetEncoder(), categorical_columns)
                                                        ], remainder='passthrough')

    preprocessor = Pipeline(steps=[
        ('filter', FunctionTransformer(filter_data)),
        ('imp_const_other', FunctionTransformer(get_simple_imputing)),
        ('replace_wrong_values', FunctionTransformer(get_replace_wrong_values)),
        ('features_creation', FunctionTransformer(get_data_from_datetime_column)),
        ('mmscaler', FunctionTransformer(get_mm_scaling)),
        ('transform_columns_by_frequency', FunctionTransformer(get_transform_columns_by_frequency)),
        ('column_transformer', column_transformer)
    ])

    X = df.drop(['event_action'], axis=1)
    y = df['event_action']

    models = (
        LogisticRegression(C=5, solver='saga', class_weight='balanced', penalty='elasticnet', random_state=42,
                           l1_ratio=0.56, max_iter=200, n_jobs=-2),
        RandomForestClassifier(n_estimators=200, class_weight='balanced_subsample', criterion='gini', max_depth=10,
                               max_features='sqrt', min_samples_split=100, n_jobs=-2, random_state=42)
        )

    best_score = .0
    best_pipe = None
    for model in models:
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        score = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc', n_jobs=-2)
        print(f'model: {type(model).__name__}, roc_auc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')

        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, ROC-AUC: {best_score:.4f}')

    best_pipe.fit(X, y)
    with open('model.pkl', 'wb') as file:
        dill.dump({
            'model': best_pipe,
            'metadata': {
                'name': 'ML model for predicting event action',
                'author': 'Alexandr Ezerskiy',
                'version': 1.0,
                'date': datetime.datetime.now(),
                'type': type(best_pipe.named_steps["classifier"]).__name__,
                'ROC-AUC': best_score
            }
        }, file)


def get_final_merged_data_set() -> pd.DataFrame:
    import pandas as pd
    df1 = pd.read_csv('./../../data/ga_hits.csv')
    df1 = df1.drop_duplicates()

    target_events = ['sub_car_claim_click', 'sub_car_claim_submit_click', 'sub_open_dialog_click',
                     'sub_custom_question_submit_click', 'sub_call_number_click', 'sub_callback_submit_click',
                     'sub_submit_success', 'sub_car_request_submit_click']

    df1_1 = df1.copy()
    df1_1['event_action'] = df1_1['event_action'].apply(lambda x: 1 if x in target_events else 0)
    df1_short = df1_1.groupby(['session_id'])[['event_action']].sum()
    df1_short['event_action'] = df1_short['event_action'].apply(lambda x: 1 if x >= 1 else 0)

    df2 = pd.read_csv('./../../data/ga_sessions.csv')
    df2 = df2.drop_duplicates()

    df = pd.merge(left=df2, right=df1_short, on='session_id', how='inner')

    return df


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    df2 = df.copy()
    columns_to_drop = ['session_id', 'client_id', 'device_os', 'visit_time', 'device_model',
                       'device_screen_resolution', 'device_browser']

    return df2.drop(columns_to_drop, axis=1)


def get_simple_imputing(df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    from sklearn.impute import SimpleImputer
    df2 = df.copy()
    columns_const = ['utm_source', 'utm_campaign', 'utm_adcontent', 'device_brand', 'utm_keyword']

    imp_const_other = SimpleImputer(strategy='constant', fill_value='other')
    df2[columns_const] = imp_const_other.fit_transform(df2[columns_const])

    return df2


def get_replace_wrong_values(df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    df2 = df.copy()
    df2['utm_medium'] = df2['utm_medium'].apply(
        lambda x: x.replace('(not set)', 'other').replace('(none)', 'other'))

    mode_country = df2['geo_country'].mode()[0]
    df2['geo_country'] = df2['geo_country'].apply(lambda x: x.replace('(not set)', mode_country))

    mode_city = df2['geo_city'].mode()[0]
    df2['geo_city'] = df2['geo_city'].apply(lambda x: x.replace('(not set)', mode_city))

    return df2


def get_data_from_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    df2 = df.copy()
    df2['visit_date'] = pd.to_datetime(df['visit_date'])
    df2['month'] = df2['visit_date'].dt.month
    df2['dayofweek'] = df2['visit_date'].dt.dayofweek
    df2 = df2.drop(columns=['visit_date'])

    return df2


def get_mm_scaling(df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    df2 = df.copy()
    mmscaler = MinMaxScaler()
    mmscaled_train = mmscaler.fit_transform(df2[['visit_number', 'month', 'dayofweek']])
    df2[['visit_number', 'month', 'dayofweek']] = mmscaled_train

    df2 = df2.drop(columns=['visit_number', 'month', 'dayofweek'])

    return df2


def get_transform_columns_by_frequency(df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    df2 = df.copy()
    min_count = int(df2.shape[0] * 0.001)
    list_categorical_columns = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent', 'utm_keyword',
                                'device_category', 'device_brand', 'geo_country', 'geo_city']

    for column in list_categorical_columns:
        df_value_counts = df2[column].value_counts().to_frame().reset_index()
        df_value_counts.columns = ['unique_value', 'count']
        unique_values_with_condition = df_value_counts[df_value_counts['count'] >= min_count]['unique_value'].tolist()

        df2[column] = df2[column].apply(lambda x: x if x in unique_values_with_condition else 'rare')

    return df2




if __name__ == '__main__':
    pipeline()

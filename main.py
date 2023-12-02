import numpy as np
from sklearn import datasets as loadDataset
from sklearn.decomposition import PCA, FastICA, FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import FeatureAgglomeration
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.svm import SVC
import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2, SelectKBest, SequentialFeatureSelector
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold


def heatmap(arr, dataset_name):
    """
        Tworzy i zapisuje heatmapę macierzy korelacji oraz wykres wartości własnych.

        Parameters:
            arr (numpy.ndarray): Macierz danych.
            dataset_name (str): Nazwa zbioru danych.

        Returns:
            None

        Examples:
            >>> heatmap(X, 'breast_cancer')
        """
    dataframe = pd.DataFrame(arr)

    inf_values = np.isinf(dataframe)
    nan_values = np.isnan(dataframe)

    # Sprawdź, czy są jakieś nieskończoności
    if np.any(inf_values):
        print("Znaleziono nieskończoności...")
        print(np.where(inf_values))

    # Sprawdź, czy są jakiekolwiek NaN-y
    if np.any(nan_values):
        print("Znaleziono NaN-y...")
        print(np.where(nan_values))

    corr = dataframe.corr()

    ax = sns.heatmap(corr, xticklabels=corr.columns.values + 1, yticklabels=corr.columns.values + 1, cmap="Greens",
                     annot=True, fmt=".2f", annot_kws={"size": 5})
    ax.set_title(f'Macierz korelacji dla zbioru danych {dataset_name}')
    ax.set_xlabel('Zmienna')
    ax.set_ylabel('Zmienna')
    plt.savefig('matrices/corr_matrix_' + str(dataset_name) + '.svg')
    plt.clf()
    eig_vals, eig_vecs = np.linalg.eig(corr)
    plt.bar(range(len(eig_vals)), eig_vals)
    plt.title(f'Wartości własne dla zbioru {dataset_name}')
    plt.xlabel('Numer głównej składowej')
    plt.ylabel('Wartość własna')
    plt.xticks(range(len(eig_vals)), range(1, len(eig_vals) + 1))
    plt.savefig('eigenvals/eigenvalues_' + str(dataset_name) + '.svg')
    plt.clf()

def load_data(dataset_name):
    """
        Wczytuje dane z wybranego zbioru danych.

        Parameters:
            dataset_name (str): Nazwa zbioru danych.

        Returns:
            tuple: Krotka zawierająca macierz danych (X) i wektor etykiet (y).

        Examples:
            >>> X, y = load_data('breast_cancer')
        """
    if dataset_name == 'breast_cancer':
        data = loadDataset.load_breast_cancer()  # https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset
        X = data.data
        y = data.target
    elif dataset_name == 'sonar':  # https://www.kaggle.com/datasets/rupakroy/sonarcsv
        data = np.loadtxt('datasets/sonar.all-data.txt', delimiter=',', dtype=str)
        X = data[:, :-1].astype(float)
        y = (data[:, -1] == 'R').astype(int)
    elif dataset_name == 'pistachio':
        data = pd.read_csv(
            'datasets/pistachio.csv')  # https://www.kaggle.com/datasets/amirhosseinmirzaie/pistachio-types-detection
        X = data.drop(columns=['Class']).values
        y = data['Class'].map({'Kirmizi_Pistachio': 0, 'Siit_Pistachio': 1}).values
    elif dataset_name == 'weather':
        # Wczytaj dane
        data = pd.read_csv('datasets/weather.csv', skiprows=range(1, 2526), nrows=366)  # rok 2020
        data = data.drop(columns=['Date'])

        # Zamień kierunki wiatru na liczby
        wind_direction_mapping = {'W': 0, 'WNW': 1, 'NW': 2, 'NNW': 3, 'N': 4, 'NNE': 5, 'NE': 6, 'ENE': 7,
                                  'E': 8, 'ESE': 9, 'SE': 10, 'SSE': 11, 'S': 12, 'SSW': 13, 'SW': 14, 'WSW': 15}

        data['WindGustDir'] = data['WindGustDir'].map(wind_direction_mapping)
        data['WindDir9am'] = data['WindDir9am'].map(wind_direction_mapping)
        data['WindDir3pm'] = data['WindDir3pm'].map(wind_direction_mapping)

        X = data.drop(columns=['RainToday']).values
        # Przypisz do zmiennej y kolumnę "RainToday"
        y = data['RainToday'].map({'No': 0, 'Yes': 1}).values
    elif dataset_name == 'nba':
        data = pd.read_csv('datasets/nba.csv')
        data = data.drop(columns=['name'])
        data = data.dropna(subset=['3p'])

        X = data.drop(columns=['target_5yrs']).values
        y = data['target_5yrs'].values.astype(int)

    heatmap(X, dataset_name)
    return X, y


def rescale_mcc(mcc):
    """
        Przeskalowuje wartość współczynnika korelacji Matthews'a (MCC) do zakresu [0, 1].

        Parameters:
            mcc (float): Wartość współczynnika korelacji Matthews'a.

        Returns:
            float: Przeskalowana wartość MCC w zakresie [0, 1].

        Examples:
            >>> rescaled_value = rescale_mcc(0.5)
        """
    return (mcc + 1) / 2


def choose_classifier(choice):
    """
       Wybiera i zwraca klasyfikator oraz odpowiadający mu słownik parametrów na podstawie wyboru użytkownika.

       Parameters:
           choice (int): Numer wybranego klasyfikatora.

       Returns:
           tuple: Krotka zawierająca obiekt klasyfikatora oraz słownik parametrów dla tego klasyfikatora.

       Raises:
           ValueError: Jeśli podany numer klasyfikatora nie jest obsługiwany.

       Examples:
           >>> clf, param_grid = choose_classifier(0)
       """
    if choice == 0:
        clf = SVC(probability=False)
        param_grid = {'kernel': ['rbf', 'poly', 'sigmoid']}
    elif choice == 1:
        clf = LogisticRegression(max_iter=10000)
        param_grid = {'C': [0.1, 1, 10, 100]}
    elif choice == 2:
        clf = KNeighborsClassifier()
        param_grid = {'n_neighbors': range(3, 10), 'algorithm': ['ball_tree', 'kd_tree', 'brute']}
    elif choice == 3:
        clf = GaussianNB()
        param_grid = {}
    elif choice == 4:
        clf = DecisionTreeClassifier()
        param_grid = {'criterion': ['gini', 'entropy', 'log_loss']}
    elif choice == 5:
        clf = xgb.XGBClassifier()  # https://xgboost.readthedocs.io/en/stable/parameter.html
        param_grid = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]}
    elif choice == 6:
        clf = MLPClassifier(max_iter=10000)
        param_grid = {'hidden_layer_sizes': [(10,), (20,), (50,), (100,)], 'learning_rate_init': [0.001, 0.01]}
    elif choice == 7:
        clf = RandomForestClassifier()
        param_grid = {'n_estimators': [80, 100, 120]}

    return clf, param_grid


if __name__ == '__main__':
    datasets = ['breast_cancer', 'pistachio', 'sonar', 'weather', 'nba']

    print("Dostępne zbiory danych:")
    for i, dataset in enumerate(datasets, 1):
        print(f"{i}. {dataset}")

    choice = int(input("Wybierz zbiór danych: "))
    if 1 <= choice <= len(datasets):
        dataset_name = datasets[choice - 1]
    else:
        print("Nieprawidłowy wybór. Wprowadź liczbę od 1 do", len(datasets))
        exit()

    X, y = load_data(dataset_name)

    methods = [
        ("PCA", PCA(), {'n_components': [0.9, 0.95, 0.99], 'svd_solver': ['auto', 'full'],
                        'whiten': [True, False], 'tol': [0, 0.01, 0.05]}),
        ("LDA", LinearDiscriminantAnalysis(), {}),
        # ("LDA", LinearDiscriminantAnalysis(), {'solver': ['svd', 'eigen']}),
        ("CA", FeatureAgglomeration(), {'n_clusters': range(2, 10, 1)}),
        ("Chi2", SelectKBest(chi2), {'k': range(1, X.shape[1] + 1)}),
        ("RFE", RFE(estimator=LogisticRegression(max_iter=10000)), {'n_features_to_select': range(1, X.shape[1] + 1)}),
        ("VT", VarianceThreshold(), {}),
        # ("SFS", SequentialFeatureSelector(KNeighborsClassifier(n_neighbors=3), cv=3, n_jobs=-1, direction='backward'),
        # {'n_features_to_select': np.arange(0.1, 1, 0.1)}),

    ]

    num_experiments = int(input("Podaj liczbę powtórzeń eksperymentu: "))

    all_results = {method: [] for method, _, _ in methods}
    all_results['Brak'] = []

    for clf_type in tqdm(range(0, 8), desc='Trwa wykonywanie eksperymentu'):
        clf, param_grid = choose_classifier(clf_type)

        mcc_only = {method: [] for method, _, _ in methods}
        mcc_only['Brak'] = []

        for i in tqdm(range(num_experiments), desc='Iteracja'):

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            for name, method, method_grid in methods:
                steps = [(name.lower(), method), (clf.__class__.__name__.lower(), clf)]

                pipe = Pipeline(steps)

                temp_param_grid = {f'{name.lower()}__{key}': value for key, value in method_grid.items()}
                temp_param_grid.update(
                    {f'{clf.__class__.__name__.lower()}__{key}': value for key, value in param_grid.items()})

                grid_search = GridSearchCV(pipe, temp_param_grid, cv=3, n_jobs=-1)

                grid_search.fit(X_train, y_train)

                best_params = grid_search.best_params_

                # print('Best parameters for', name, ':', best_params)

                y_pred = grid_search.predict(X_test)
                report = classification_report(y_test, y_pred, output_dict=True)

                matthews = rescale_mcc(matthews_corrcoef(y_test, y_pred))

                precision_avg = (report['0']['precision'] + report['1']['precision']) / 2
                recall_avg = (report['0']['recall'] + report['1']['recall']) / 2
                f1_avg = (report['0']['f1-score'] + report['1']['f1-score']) / 2

                all_results[name].append([precision_avg, recall_avg, f1_avg, report['accuracy'], matthews])
                mcc_only[name].append(matthews)

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)

            matthews = rescale_mcc(matthews_corrcoef(y_test, y_pred))
            precision_avg = (report['0']['precision'] + report['1']['precision']) / 2
            recall_avg = (report['0']['recall'] + report['1']['recall']) / 2
            f1_avg = (report['0']['f1-score'] + report['1']['f1-score']) / 2

            all_results['Brak'].append([precision_avg, recall_avg, f1_avg, report['accuracy'], matthews])
            mcc_only['Brak'].append(matthews)

        avg_results = {method: np.mean(results, axis=0) for method, results in all_results.items()}
        avg_results_list = [[method] + result.tolist() for method, result in avg_results.items()]

        headers = ["Metoda", "Precyzja", "Czułość", "Miara F1", "Dokładność", "Matthews"]
        tabela = pd.DataFrame(avg_results_list, columns=headers)

        clf_info = f"Klasyfikator: {clf.__class__.__name__}"

        plt.figure(figsize=(10, 6))
        sns.set(font_scale=1)
        ax = sns.heatmap(tabela.set_index('Metoda').iloc[:, :5], annot=True, fmt=".3f", cmap="YlGnBu", cbar=True)
        iters_text = "eksperymentów" if num_experiments > 1 else "eksperymentu"
        plt.title(
            f"Metryki klasyfikacji dla {dataset_name} [Średnia z {num_experiments} {iters_text}] \n{clf_info}")

        plt.savefig('metrics/klasyfikator_' + str(clf.__class__.__name__) + '.svg')
        plt.clf()

        matthews_df = pd.DataFrame(mcc_only)
        matthews_df.to_csv('matthews/matthews_' + str(clf.__class__.__name__) + '.csv', header=True, index=False)

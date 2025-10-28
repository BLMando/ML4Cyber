# 🛡️ CyberForecast — Analisi, Classificazione e Previsione delle Minacce Informatiche Globali (2015–2024)

**Progetto di Machine Learning per l'analisi predittiva delle minacce informatiche globali utilizzando tecniche di classificazione, clustering e forecasting.**

---

## 📋 **Panoramica del Progetto**

### **Obiettivi**
Applicare tecniche avanzate di Machine Learning per:
- **📊 EDA**: Analisi esplorativa approfondita dei dati di cybersecurity
- **🎯 Classificazione**: Predizione del tipo di attacco usando LazyPredict + tuning
- **🔍 Clustering**: Identificazione di pattern nascosti con K-Means, DBSCAN, PCA
- **📈 Forecasting**: Previsione temporale degli attacchi con Prophet/ARIMA

### **Dataset**
`Global_Cybersecurity_Threats_2015-2024.csv` - 10 anni di dati su minacce informatiche globali con:
- **Dimensione temporale**: 2015-2024
- **Variabili categoriche**: Attack Type, Country, Target Industry, Attack Source
- **Variabili numeriche**: Financial Loss, Affected Users, Resolution Time
- **Copertura geografica**: Dati multinazionali

---

## 🚀 **Quick Start**

### **1. Setup Ambiente**
```bash
# Clone repository
git clone <repository-url>
cd ML4Cyber

# Setup ambiente virtuale
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Installa dipendenze
pip install -r requirements.txt

```

### **2. Configurazione**
```bash
# Copia e modifica configurazione
cp config/config.yaml.example config/config.yaml
```

### **3. Esecuzione**
```bash
# Esegui notebooks in ordine
jupyter notebook notebooks/01_EDA.ipynb
jupyter notebook notebooks/02_Classification.ipynb
jupyter notebook notebooks/03_Clustering.ipynb
jupyter notebook notebooks/04_Forecasting.ipynb
```

---

## 📁 **Struttura del Progetto**

```
ML4Cyber/
├── data/
│   ├── raw/                           # Dati originali (mai modificare)
│   ├── processed/                     # Dati preprocessati
│   └── interim/                       # Dati intermedi
├── notebooks/
│   ├── 01_EDA.ipynb                  # Analisi esplorativa
│   ├── 02_Classification.ipynb       # Modelli di classificazione
│   ├── 03_Clustering.ipynb           # Analisi cluster
│   └── 04_Forecasting.ipynb          # Previsioni temporali
├── src/
│   ├── data/
│   │   ├── loader.py                 # Caricamento dati
│   │   └── preprocessor.py           # Pipeline preprocessing
│   ├── features/
│   │   └── engineering.py            # Feature engineering
│   ├── models/
│   │   ├── classifier.py             # Modelli classificazione
│   │   ├── clusterer.py              # Modelli clustering
│   │   └── forecaster.py             # Modelli forecasting
│   ├── visualization/
│   │   └── plots.py                  # Grafici standardizzati
│   └── utils/
│       ├── config.py                 # Configurazioni
│       └── helpers.py                # Utility
├── tests/                            # Unit tests
├── results/
│   ├── figures/                      # Grafici salvati
│   ├── models/                       # Modelli serializzati
│   └── reports/                      # Report generati
├── config/
│   └── config.yaml                   # Parametri configurabili
├── requirements.txt                  # Dipendenze Python
├── .gitignore                        # File da ignorare
└── README.md                         # Questo file
```

---

## 🔧 **Best Practices di Preprocessing**

### **📋 Checklist Obbligatoria**

#### **1. Gestione Missing Values**
```python
# SEMPRE controllare missing values
df.isnull().sum()

# Strategie per tipo:
# - Numeriche: mean/median/mode
# - Categoriche: mode/new_category  
# - Time series: forward/backward fill
```

#### **2. Train/Test Split PRIMA del Preprocessing**
```python
# ✅ CORRETTO
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler.fit(X_train)  # Fit solo su train

# ❌ SBAGLIATO - Data Leakage
scaler.fit(X)  # Fit su tutto il dataset
```

#### **3. Encoding e Scaling**
```python
# Categoriche
pd.get_dummies(df['category'])  # OneHot per nominali
LabelEncoder()                  # Label per ordinali

# Numeriche (SEMPRE per algoritmi distance-based)
StandardScaler()    # Media=0, std=1
MinMaxScaler()      # Range [0,1]
RobustScaler()      # Robusto agli outliers
```

### **⚡ Pipeline Modulari Avanzate**

#### **1. Pipeline per Diversi Task**
```python
# Pipeline Classificazione
def create_classification_pipeline(model):
    return Pipeline([
        ('preprocessor', CyberPreprocessor().create_preprocessing_pipeline()),
        ('classifier', model)
    ])

# Pipeline Clustering (senza target)
def create_clustering_pipeline():
    return Pipeline([
        ('preprocessor', CyberPreprocessor().create_preprocessing_pipeline()),
        ('pca', PCA(n_components=0.95)),
        ('clusterer', KMeans(random_state=42))
    ])
```

#### **2. Feature Engineering Avanzato**
```python
# Feature temporali
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['is_weekend'] = df['date'].dt.dayofweek >= 5

# Feature aggregate
df['loss_log'] = np.log1p(df['financial_loss'])
df['loss_per_user'] = df['financial_loss'] / df['affected_users']
```

#### **3. Gestione Outliers**
```python
# IQR Method
Q1, Q3 = df['col'].quantile([0.25, 0.75])
IQR = Q3 - Q1
lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
df_clean = df[(df['col'] >= lower) & (df['col'] <= upper)]

# Z-Score Method (|z| > 3)
from scipy import stats
df_clean = df[np.abs(stats.zscore(df['col'])) < 3]
```

---

## 🔧 **Pipeline di Preprocessing Avanzate**

### **Pipeline Completa per CyberForecast**
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

class CyberPreprocessor:
    def __init__(self):
        self.numerical_cols = ['Financial Loss (in Million $)', 
                              'Number of Affected Users',
                              'Incident Resolution Time (in Hours)']
        
        self.categorical_cols = ['Country', 'Target Industry', 
                               'Attack Source', 'Security Vulnerability Type']
        
    def create_preprocessing_pipeline(self):
        # Pipeline numeriche
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Pipeline categoriche
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
        ])
        
        # Combina pipeline
        return ColumnTransformer([
            ('num', numeric_pipeline, self.numerical_cols),
            ('cat', categorical_pipeline, self.categorical_cols)
        ])
    
    def engineer_features(self, df):
        df = df.copy()
        
        # Feature temporali
        df['Attack_Frequency'] = df.groupby(['Country', 'Year'])['Attack Type'].transform('count')
        
        # Feature finanziarie
        df['Loss_Per_User'] = df['Financial Loss (in Million $)'] / (df['Number of Affected Users'] + 1)
        df['Loss_Log'] = np.log1p(df['Financial Loss (in Million $)'])
        
        # Feature di gravità
        df['Severity_Score'] = (df['Financial Loss (in Million $)'] * 
                               df['Number of Affected Users'] / 
                               df['Incident Resolution Time (in Hours)'])
        
        return df
```

---

## 🤖 **Approccio Modeling Dettagliato**

### **1. Classificazione (Attack Type Prediction)**

#### **Workflow Completo**
```python
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

class AttackTypeClassifier:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.preprocessor = CyberPreprocessor()
        
    def lazy_predict_benchmark(self, X_train, X_test, y_train, y_test):
        # Preprocessing
        pipeline = self.preprocessor.create_preprocessing_pipeline()
        X_train_processed = pipeline.fit_transform(X_train)
        X_test_processed = pipeline.transform(X_test)
        
        # LazyPredict
        clf = LazyClassifier(verbose=0, ignore_warnings=True)
        models, predictions = clf.fit(X_train_processed, X_test_processed, y_train, y_test)
        
        return models.head(5)
    
    def optimize_top_models(self, X_train, y_train, top_models):
        param_grids = {
            'RandomForestClassifier': {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [10, 20, None],
                'classifier__min_samples_split': [2, 5, 10]
            },
            'XGBClassifier': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [3, 6, 9],
                'classifier__learning_rate': [0.01, 0.1, 0.2]
            }
        }
        
        results = {}
        for model_name in top_models.index[:3]:
            if model_name in param_grids:
                model = self._get_model_instance(model_name)
                pipeline = Pipeline([
                    ('preprocessor', self.preprocessor.create_preprocessing_pipeline()),
                    ('classifier', model)
                ])
                
                grid_search = GridSearchCV(
                    pipeline, param_grids[model_name],
                    cv=5, scoring='f1_weighted', n_jobs=-1
                )
                
                grid_search.fit(X_train, y_train)
                results[model_name] = {
                    'best_model': grid_search.best_estimator_,
                    'best_score': grid_search.best_score_,
                    'best_params': grid_search.best_params_
                }
        
        return results
```

#### **Domande di Ricerca Specifiche**
- **Predittività**: Quale combinazione di features (settore + paese + vulnerabilità) predice meglio il tipo di attacco?
- **Feature Importance**: Le perdite finanziarie sono più predittive del settore target?
- **Generalizzazione**: I modelli addestrati su dati 2015-2020 performano bene su 2021-2024?
- **Bias Geografico**: Esistono bias nei modelli verso paesi con più dati?

### **2. Clustering (Pattern Discovery)**

#### **Approccio Multi-Algoritmo**
```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA

class CyberThreatClusterer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.preprocessor = CyberPreprocessor()
        
    def find_optimal_clusters(self, X, max_clusters=10):
        # Preprocessing + PCA
        pipeline = Pipeline([
            ('preprocessor', self.preprocessor.create_preprocessing_pipeline()),
            ('pca', PCA(n_components=0.95))
        ])
        X_processed = pipeline.fit_transform(X)
        
        results = {'k': [], 'silhouette': [], 'inertia': []}
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state)
            labels = kmeans.fit_predict(X_processed)
            
            results['k'].append(k)
            results['silhouette'].append(silhouette_score(X_processed, labels))
            results['inertia'].append(kmeans.inertia_)
        
        return pd.DataFrame(results)
    
    def apply_clustering_algorithms(self, X, optimal_k=5):
        # Preprocessing + PCA
        pipeline = Pipeline([
            ('preprocessor', self.preprocessor.create_preprocessing_pipeline()),
            ('pca', PCA(n_components=0.95))
        ])
        X_processed = pipeline.fit_transform(X)
        
        algorithms = {
            'KMeans': KMeans(n_clusters=optimal_k, random_state=self.random_state),
            'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
            'Agglomerative': AgglomerativeClustering(n_clusters=optimal_k)
        }
        
        results = {}
        for name, algorithm in algorithms.items():
            labels = algorithm.fit_predict(X_processed)
            
            if len(set(labels)) > 1 and -1 not in labels:
                silhouette = silhouette_score(X_processed, labels)
            else:
                silhouette = None
            
            results[name] = {
                'labels': labels,
                'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
                'silhouette_score': silhouette
            }
        
        return results, X_processed
```

#### **Domande di Ricerca Specifiche**
- **Clustering Geografico**: Paesi con economia simile subiscono attacchi simili?
- **Clustering Temporale**: Esistono "ondate" di attacchi coordinati nel tempo?
- **Clustering Settoriale**: Settori con infrastrutture simili mostrano vulnerabilità correlate?
- **Clustering Multi-dimensionale**: Come si raggruppano gli attacchi considerando tipo+paese+settore insieme?

### **3. Forecasting (Time Series Prediction)**

#### **Approccio Multi-Modello**
```python
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

class CyberThreatForecaster:
    def __init__(self):
        self.models = {}
        
    def prepare_time_series(self, df, target_col, freq='M'):
        if target_col == 'attack_count':
            ts = df.groupby(['Year', 'Month']).size()
        else:
            ts = df.groupby(['Year', 'Month'])[target_col].sum()
        
        # Crea indice temporale
        ts.index = pd.to_datetime([f"{year}-{month:02d}-01" 
                                  for year, month in ts.index])
        
        return ts.resample(freq).sum()
    
    def prophet_forecast(self, ts, periods=12, seasonality=True):
        # Prepara dati per Prophet
        df_prophet = pd.DataFrame({
            'ds': ts.index,
            'y': ts.values
        })
        
        # Configura modello
        model = Prophet(
            yearly_seasonality=seasonality,
            weekly_seasonality=False,
            daily_seasonality=False
        )
        
        model.fit(df_prophet)
        
        # Forecast
        future = model.make_future_dataframe(periods=periods, freq='M')
        forecast = model.predict(future)
        
        return model, forecast
    
    def arima_forecast(self, ts, order=(1,1,1), periods=12):
        model = ARIMA(ts, order=order)
        fitted_model = model.fit()
        
        forecast = fitted_model.forecast(steps=periods)
        conf_int = fitted_model.get_forecast(steps=periods).conf_int()
        
        return fitted_model, forecast, conf_int
    
    def compare_models(self, ts, test_size=12):
        # Split train/test
        train = ts[:-test_size]
        test = ts[-test_size:]
        
        results = {}
        
        # Prophet
        prophet_model, prophet_forecast = self.prophet_forecast(train, periods=test_size)
        prophet_pred = pd.Series(
            prophet_forecast['yhat'][-test_size:].values,
            index=test.index
        )
        results['Prophet'] = {
            'MAE': mean_absolute_error(test, prophet_pred),
            'RMSE': np.sqrt(mean_squared_error(test, prophet_pred))
        }
        
        # ARIMA
        arima_model, arima_forecast, _ = self.arima_forecast(train, periods=test_size)
        arima_pred = pd.Series(arima_forecast, index=test.index)
        results['ARIMA'] = {
            'MAE': mean_absolute_error(test, arima_pred),
            'RMSE': np.sqrt(mean_squared_error(test, arima_pred))
        }
        
        return results
```

#### **Domande di Ricerca Specifiche**
- **Trend Analysis**: Il numero di attacchi ransomware cresce più velocemente di altri tipi?
- **Stagionalità**: Esistono picchi di attacchi durante festività o periodi specifici?
- **Previsioni Economiche**: Le perdite finanziarie seguono trend economici globali?
- **Efficacia Difese**: L'adozione di AI-based detection riduce la frequenza degli attacchi nel tempo?

### **Workflow di Integrazione**
```python
def run_complete_analysis(df):
    # 1. Preprocessing
    preprocessor = CyberPreprocessor()
    df_engineered = preprocessor.engineer_features(df)
    
    # 2. Classificazione
    classifier = AttackTypeClassifier()
    X = df_engineered.drop('Attack Type', axis=1)
    y = df_engineered['Attack Type']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    top_models = classifier.lazy_predict_benchmark(X_train, X_test, y_train, y_test)
    optimized_models = classifier.optimize_top_models(X_train, y_train, top_models)
    
    # 3. Clustering
    clusterer = CyberThreatClusterer()
    optimal_k_df = clusterer.find_optimal_clusters(X)
    cluster_results, X_processed = clusterer.apply_clustering_algorithms(X)
    
    # 4. Forecasting
    forecaster = CyberThreatForecaster()
    ts_attacks = forecaster.prepare_time_series(df, 'attack_count')
    forecast_comparison = forecaster.compare_models(ts_attacks)
    
    return {
        'classification': optimized_models,
        'clustering': cluster_results,
        'forecasting': forecast_comparison
    }
```

---

## 📊 **Visualizzazioni e Reporting**

### **Template Grafici Standardizzati**
```python
# Configurazione globale
plt.style.use('seaborn-v0_8-darkgrid')
PLOT_CONFIG = {
    'figsize': (12, 6),
    'dpi': 300,
    'title_fontsize': 16
}

# Grafici interattivi con Plotly
import plotly.express as px
fig = px.scatter(df, x='financial_loss', y='affected_users', 
                color='attack_type', title='Attack Impact Analysis')
```

### **Dashboard EDA**
- Distribuzione attacchi per tipo/paese/settore
- Evoluzione temporale delle minacce
- Correlazioni tra variabili
- Analisi perdite finanziarie

---
---

### **Riproducibilità**
- **Random Seeds**: Tutti i processi random usano `seed=42`
- **Environment**: `requirements.txt` con versioni pinned
- **Config**: Parametri centralizzati in `config.yaml`
- **Documentation**: Codice completamente documentato

---

## 📚 **Dipendenze e Requisiti**

### **Core Libraries**
```txt
# Data Processing
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0

# Visualization  
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0

# Machine Learning
lazypredict>=0.2.12
xgboost>=1.6.0
lightgbm>=3.3.0

# Time Series
prophet>=1.1.0
statsmodels>=0.13.0

# Experiment Tracking
mlflow>=2.0.0

# Testing
pytest>=7.0.0
```

### **Requisiti Sistema**
- Python 3.8+
- RAM: 8GB+ raccomandati
- Storage: 2GB per dati e modelli

---

## 🎯 **Domande di Ricerca Principali**

### **EDA**
1. Quali settori subiscono le perdite finanziarie maggiori?
2. Come varia la frequenza degli attacchi nel tempo?
3. Esistono correlazioni tra tipo di attacco e vulnerabilità?

### **Classificazione**
1. È possibile predire il tipo di attacco dalle caratteristiche del target?
2. Quali feature sono più discriminanti per ogni classe?
3. Come varia l'accuratezza per diversi paesi/settori?

### **Clustering**
1. Esistono gruppi di paesi con pattern di attacco simili?
2. Si possono identificare cluster temporali negli attacchi?
3. Quali settori mostrano vulnerabilità correlate?

### **Forecasting**
1. Il numero di attacchi mostra un trend di crescita?
2. Esistono pattern stagionali nelle minacce informatiche?
3. Come evolveranno le perdite finanziarie nei prossimi anni?

---

## 📋 **Checklist Progetto**

### **Setup Iniziale**
- [ ] Ambiente virtuale configurato
- [ ] Dipendenze installate
- [ ] Struttura cartelle creata
- [ ] Git repository inizializzato

### **Sviluppo**
- [ ] EDA completata con visualizzazioni
- [ ] Preprocessing pipeline implementata
- [ ] LazyPredict benchmark eseguito
- [ ] Modelli ottimizzati con GridSearch
- [ ] Clustering analysis completata
- [ ] Forecasting implementato

### **Finalizzazione**
- [ ] Risultati documentati
- [ ] Grafici salvati in alta risoluzione
- [ ] Modelli serializzati
- [ ] Report finale generato
- [ ] Codice pulito e commentato
- [ ] README aggiornato

---

---

## 🎓 **Note Accademiche**

Questo progetto è stato sviluppato come tesina universitaria seguendo rigorose best practices di Data Science:
- **Metodologia scientifica** con ipotesi e validazione
- **Riproducibilità** garantita da seed fissi e documentazione
- **Qualità del codice** con testing e linting automatici
- **Visualizzazioni professionali** per comunicazione efficace

**Obiettivo**: Dimostrare competenze complete in Machine Learning applicato alla cybersecurity, dalla data exploration al deployment di modelli predittivi.

---

*Ultimo aggiornamento: Ottobre 2024*
#Libraries for sql
import pyodbc 
from dotenv import dotenv_values #import the dotenv_values function from the dotenv package
import warnings 
warnings.filterwarnings('ignore')

#libraries for handling data
import pandas as pd
import numpy as np
pd.set_option('display.max_rows', None)

#libraries for visulation
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.offline as offline
from plotly.offline import init_notebook_mode, iplot
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
offline.init_notebook_mode(connected=True) # Configure Plotly to run 
import statsmodels.api as sm
from scipy.stats import chi2_contingency

#Feature processing libraries
from sklearn.impute import SimpleImputer
#import phix
from phik import phik_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

#algorithm libraries 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import GridSearchCV

#pipelines and transformers
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer 
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

#model evaluation
from sklearn.metrics import classification_report,fbeta_score,make_scorer
from sklearn.metrics import confusion_matrix

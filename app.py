import streamlit as st
import pandas as pd
import joblib
import altair as alt
import warnings
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Sistema de Predição de Obesidade",
    page_icon="🩺",
    layout="wide",
)
warnings.filterwarnings('ignore')

# =============================================================================
# --- FUNÇÃO DE TREINAMENTO DO MODELO ---
# Esta função contém a lógica que estava no 'train_model.py'
# =============================================================================
def train_and_save_model():
    """
    Lê o CSV, treina um novo modelo de pipeline e o salva em um arquivo .pkl.
    Esta função só será executada se o modelo não existir.
    """
    st.warning("Arquivo do modelo não encontrado! Treinando um novo modelo. Isso pode levar um minuto...")
    
    # 1. Carregar os dados
    try:
        df = pd.read_csv('Obesity.csv')
    except FileNotFoundError:
        st.error("ERRO: O arquivo 'Obesity.csv' não foi encontrado. Por favor, coloque-o na mesma pasta do app.")
        return None

    # Limpeza dos nomes das colunas
    df.columns = df.columns.str.strip()
    
    # 2. Preparar os dados
    target_column = 'Obesity'
    if target_column not in df.columns:
        st.error(f"ERRO: A coluna alvo '{target_column}' não foi encontrada no CSV.")
        return None
        
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    
    # 3. Criar o pipeline de Machine Learning
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # 4. Treinar
    pipeline.fit(X, y)
    
    # 5. Salvar
    joblib.dump(pipeline, 'obesity_pipeline.pkl')
    st.success("Novo modelo treinado e salvo com sucesso como 'obesity_pipeline.pkl'!")
    
    return pipeline

# =============================================================================
# --- LÓGICA PRINCIPAL DE CARREGAMENTO DO MODELO ---
# =============================================================================
MODEL_PATH = 'obesity_pipeline.pkl'
if not os.path.exists(MODEL_PATH):
    pipeline = train_and_save_model()
else:
    try:
        pipeline = joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar o modelo existente: {e}")
        pipeline = train_and_save_model()

if pipeline is None:
    st.stop()


# --- DICIONÁRIOS PARA MAPEAMENTO E TRADUÇÃO ---
gender_map = {"Feminino": "Female", "Masculino": "Male"}
yes_no_map = {"Sim": "sim", "Não": "não"}
caec_map = { "Não": "No", "Às vezes": "Sometimes", "Frequentemente": "Frequently", "Sempre": "Always" }
calc_map = { "Não": "no", "Às vezes": "Sometimes", "Frequentemente": "Frequently" }
mtrans_map = { "Automóvel": "Automobile", "Motocicleta": "Motorbike", "Bicicleta": "Bike", "Transporte Público": "Public_Transportation", "Caminhando": "Walking" }
result_map = { 'Insufficient_Weight': 'Peso Insuficiente', 'Normal_Weight': 'Peso Normal', 'Overweight_Level_I': 'Sobrepeso Nível I', 'Overweight_Level_II': 'Sobrepeso Nível II', 'Obesity_Type_I': 'Obesidade Tipo I', 'Obesity_Type_II': 'Obesidade Tipo II', 'Obesity_Type_III': 'Obesidade Tipo III' }
feature_translation = { 'Age': 'Idade', 'Height': 'Altura (m)', 'Weight': 'Peso (kg)', 'FCVC': 'Freq. Consumo Vegetais', 'NCP': 'Nº Refeições Principais', 'CH2O': 'Consumo de Água (Litros)', 'FAF': 'Freq. Atividade Física', 'TUE': 'Tempo de Uso de Telas', 'family_history_sim': 'Histórico Familiar de Sobrepeso', 'FAVC_sim': 'Consome Alimentos Altamente Calóricos', 'CAEC_Frequently': 'Lanches (Frequentemente)', 'CAEC_Always': 'Lanches (Sempre)', 'SCC_sim': 'Monitora Calorias', 'MTRANS_Public_Transportation': 'Usa Transporte Público', 'Gender_Male': 'Gênero (Masculino)', 'SMOKE_sim': 'É Fumante' }


# =============================================================================
# --- INTERFACE DO STREAMLIT ---
# =============================================================================
def show_predict_page():
    st.header("🩺 Sistema Preditivo de Níveis de Obesidade")
    st.markdown("Preencha o formulário abaixo para receber uma predição sobre o estado nutricional.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Informações Pessoais")
        age = st.number_input("Idade", 1, 100, 25)
        gender = st.selectbox("Gênero", list(gender_map.keys()))
        height = st.number_input("Altura (metros)", 1.0, 2.5, 1.70, format="%.2f")
        weight = st.number_input("Peso (kg)", 30.0, 200.0, 70.0, format="%.1f")
        family_history = st.radio("Histórico familiar de sobrepeso?", list(yes_no_map.keys()), horizontal=True)
    with col2:
        st.subheader("Hábitos Alimentares")
        favc = st.radio("Consome alimentos de alta caloria (FAVC)?", list(yes_no_map.keys()), horizontal=True, key="favc")
        fcvc = st.slider("Freq. consumo de vegetais (FCVC)", 1.0, 3.0, 2.0, help="1: Nunca, 2: Às vezes, 3: Sempre")
        ncp = st.slider("Nº de refeições principais por dia (NCP)", 1.0, 4.0, 3.0)
        caec = st.selectbox("Consumo de lanches (CAEC)?", list(caec_map.keys()))
        ch2o = st.slider("Consumo de água diário (Litros)", 1.0, 3.0, 2.0, 0.5)
        calc = st.selectbox("Freq. consumo de álcool (CALC)", list(calc_map.keys()))
    with col3:
        st.subheader("Estilo de Vida")
        scc = st.radio("Monitora o consumo de calorias (SCC)?", list(yes_no_map.keys()), horizontal=True, key="scc")
        smoke = st.radio("Fumante?", list(yes_no_map.keys()), horizontal=True, key="smoke")
        faf = st.slider("Freq. de atividade física (FAF)", 0.0, 3.0, 1.0, help="0: Nenhuma, 1: 1-2d/sem, 2: 2-4d/sem, 3: 4-5d/sem")
        tue = st.slider("Tempo de uso de telas (TUE)", 0.0, 2.0, 1.0, help="0: 0-2h, 1: 3-5h, 2: >5h")
        mtrans = st.selectbox("Transporte principal (MTRANS)", list(mtrans_map.keys()))

    if st.button("Realizar Predição", type="primary", use_container_width=True):
        input_data = pd.DataFrame({
            'Gender': [gender_map[gender]], 'Age': [age], 'Height': [height], 'Weight': [weight],
            'family_history': [yes_no_map[family_history]], 'FAVC': [yes_no_map[favc]],
            'FCVC': [fcvc], 'NCP': [ncp], 'CAEC': [caec_map[caec]], 'SMOKE': [yes_no_map[smoke]],
            'CH2O': [ch2o], 'SCC': [yes_no_map[scc]], 'FAF': [faf], 'TUE': [tue], 'CALC': [calc_map[calc]],
            'MTRANS': [mtrans_map[mtrans]],
        })

        prediction = pipeline.predict(input_data)
        prediction_proba = pipeline.predict_proba(input_data)
        predicted_class_pt = result_map.get(prediction[0], "Classe Desconhecida")
        st.markdown("---"); st.subheader("Resultado da Predição")
        st.success(f"O nível de peso previsto é: **{predicted_class_pt}**")
        # ... (Restante da interface como estava)

def show_dashboard():
    st.header("📊 Painel Analítico: Insights sobre Fatores de Risco")
    # ... (código do dashboard como estava)

st.sidebar.title("Navegação")
app_mode = st.sidebar.selectbox("Escolha a funcionalidade", ["Sistema Preditivo", "Painel Analítico"])

if app_mode == "Sistema Preditivo":
    show_predict_page()
else:
    show_dashboard()
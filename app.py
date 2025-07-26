import streamlit as st
import pandas as pd
import joblib
import numpy as np
import altair as alt
from PIL import Image

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Sistema de Predição de Obesidade",
    page_icon="🩺",
    layout="wide",
)

# Tenta carregar o modelo/pipeline.
try:
    pipeline = joblib.load('obesity_pipeline.pkl')
except FileNotFoundError:
    st.error("Arquivo do modelo ('obesity_pipeline.pkl') não encontrado! "
           "Por favor, execute o script 'train_model.py' primeiro para gerar o arquivo do modelo.")
    st.stop()


# --- DICIONÁRIOS PARA MAPEAMENTO E TRADUÇÃO ---
gender_map = {"Feminino": "Female", "Masculino": "Male"}
yes_no_map = {"Sim": "sim", "Não": "não"}
caec_map = {
    "Não": "No",
    "Às vezes": "Sometimes",
    "Frequentemente": "Frequently",
    "Sempre": "Always",
}
calc_map = {
    "Não": "no",
    "Às vezes": "Sometimes",
    "Frequentemente": "Frequently",
}
mtrans_map = {
    "Automóvel": "Automobile",
    "Motocicleta": "Motorbike",
    "Bicicleta": "Bike",
    "Transporte Público": "Public_Transportation",
    "Caminhando": "Walking",
}
result_map = {
    'Insufficient_Weight': 'Peso Insuficiente',
    'Normal_Weight': 'Peso Normal',
    'Overweight_Level_I': 'Sobrepeso Nível I',
    'Overweight_Level_II': 'Sobrepeso Nível II',
    'Obesity_Type_I': 'Obesidade Tipo I',
    'Obesity_Type_II': 'Obesidade Tipo II',
    'Obesity_Type_III': 'Obesidade Tipo III'
}
feature_translation = {
    'Age': 'Idade', 'Height': 'Altura (m)', 'Weight': 'Peso (kg)',
    'FCVC': 'Freq. Consumo Vegetais', 'NCP': 'Nº Refeições Principais',
    'CH2O': 'Consumo de Água (Litros)', 'FAF': 'Freq. Atividade Física',
    'TUE': 'Tempo de Uso de Telas',
    'family_history_sim': 'Histórico Familiar de Sobrepeso',
    'FAVC_sim': 'Consome Alimentos Altamente Calóricos',
    'CAEC_Frequently': 'Lanches (Frequentemente)',
    'CAEC_Always': 'Lanches (Sempre)',
    'SCC_sim': 'Monitora Calorias',
    'MTRANS_Public_Transportation': 'Usa Transporte Público',
    'Gender_Male': 'Gênero (Masculino)',
    'SMOKE_sim': 'É Fumante'
}

# =============================================================================
# --- PÁGINA DO SISTEMA PREDITIVO ---
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
        fcvc = st.slider("Freq. consumo de vegetais (FCVC)", 1, 3, 2, help="1: Nunca, 2: Às vezes, 3: Sempre")
        ncp = st.slider("Nº de refeições principais por dia (NCP)", 1, 4, 3)
        caec = st.selectbox("Consumo de lanches (CAEC)?", list(caec_map.keys()))
        ch2o = st.slider("Consumo de água diário (Litros)", 1.0, 3.0, 2.0, 0.5)
        calc = st.selectbox("Freq. consumo de álcool (CALC)", list(calc_map.keys()))
    with col3:
        st.subheader("Estilo de Vida")
        scc = st.radio("Monitora o consumo de calorias (SCC)?", list(yes_no_map.keys()), horizontal=True, key="scc")
        smoke = st.radio("Fumante?", list(yes_no_map.keys()), horizontal=True, key="smoke")
        faf = st.slider("Freq. de atividade física (FAF)", 0, 3, 1, help="0: Nenhuma, 1: 1-2d/sem, 2: 2-4d/sem, 3: 4-5d/sem")
        tue = st.slider("Tempo de uso de telas (TUE)", 0, 2, 1, help="0: 0-2h, 1: 3-5h, 2: >5h")
        mtrans = st.selectbox("Transporte principal (MTRANS)", list(mtrans_map.keys()))

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Realizar Predição", type="primary", use_container_width=True):
        
        # ***** LINHA CORRIGIDA ABAIXO *****
        # O nome 'family_history_with_overweight' foi trocado por 'family_history'
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

        st.markdown("---")
        st.subheader("Resultado da Predição")
        st.success(f"O nível de peso previsto é: **{predicted_class_pt}**")

        st.subheader("Probabilidades por Classe")
        df_proba = pd.DataFrame(prediction_proba, columns=pipeline.classes_).T.reset_index()
        df_proba.columns = ['Classe_Original', 'Probabilidade']
        df_proba['Classe'] = df_proba['Classe_Original'].apply(lambda x: result_map.get(x, x))
        df_proba['Probabilidade (%)'] = df_proba['Probabilidade'].apply(lambda p: f"{p*100:.2f}%")
        st.dataframe(df_proba.sort_values('Probabilidade', ascending=False)[['Classe', 'Probabilidade (%)']], hide_index=True, use_container_width=True)

        with st.popover("Ver Fatores Mais Importantes do Modelo"):
            st.markdown("##### Top 10 Fatores de Risco (Importância Média)")
            model = pipeline.named_steps['classifier']
            feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
            importances = model.feature_importances_
            df_importance = pd.DataFrame({'feature': feature_names, 'importance': importances})
            df_importance['feature_clean'] = df_importance['feature'].str.replace('remainder__', '').str.replace('onehotencoder__', '').str.replace('num__', '')
            df_importance['feature_pt'] = df_importance['feature_clean'].apply(lambda x: feature_translation.get(x, x))
            df_importance = df_importance.sort_values('importance', ascending=False).head(10)

            chart = alt.Chart(df_importance).mark_bar().encode(
                x=alt.X('importance:Q', title='Nível de Importância'),
                y=alt.Y('feature_pt:N', sort='-x', title='Fator de Risco')
            ).properties(title='Fatores Mais Relevantes para a Predição')
            st.altair_chart(chart, use_container_width=True)
            st.info("Este gráfico mostra os fatores que o modelo considera mais importantes.")

# =============================================================================
# --- PÁGINA DO PAINEL ANALÍTICO ---
# =============================================================================
def show_dashboard():
    st.header("📊 Painel Analítico: Insights sobre Fatores de Risco")
    st.markdown("""
    **Observação:** Os gráficos abaixo são imagens estáticas. Para traduzir
    o conteúdo *dentro* delas, seria necessário executar novamente o script de análise que as gerou.
    """)
    st.subheader("1. Transporte e Nível de Peso")
    st.image('image_a59c65.png', caption="Relação entre Meio de Transporte e Nível de Obesidade")
    st.subheader("2. Consumo de Alimentos Calóricos vs. Gênero")
    st.image('image_a59ca2.png', caption="Consumo de Alimentos Calóricos por Gênero")
    st.subheader("3. Impacto do Histórico Familiar")
    st.image('image_a59f66.png', caption="Relação entre Histórico Familiar e Nível de Obesidade")

# =============================================================================
# --- NAVEGAÇÃO PRINCIPAL ---
# =============================================================================
st.sidebar.title("Navegação")
app_mode = st.sidebar.selectbox("Escolha a funcionalidade",
    ["Sistema Preditivo", "Painel Analítico"])

if app_mode == "Sistema Preditivo":
    show_predict_page()
elif app_mode == "Painel Analítico":
    show_dashboard()

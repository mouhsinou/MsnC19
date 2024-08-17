import streamlit as st
import pandas as pd
import io
import plotly.express as px
import requests

# Configuration de la page
st.set_page_config(page_title="COVID-19 Breath Analysis", page_icon="🫁", layout="wide")

# Styles CSS personnalisés
st.markdown("""
<style>
    .reportview-container {
        background: #f0f8ff;
    }
    .sidebar .sidebar-content {
        background: #e6f2ff;
    }
    .Widget>label {
        color: #0066cc;
        font-weight: bold;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #007bff;
        border-radius: 5px;
    }
    h1 {
        color: #007bff;
    }
    h2 {
        color: #17a2b8;
    }
</style>
""", unsafe_allow_html=True)

# Fonctions utilitaires
def load_patient_data(file):
    content = file.read().decode("utf-8")
    lines = content.splitlines()

    patient_id = lines[0].split(":")[1].strip()
    
    data_lines = lines[3:]
    data = []
    for line in data_lines:
        parts = line.split()
        if len(parts) > 1 and parts[0] != 'Min:Sec':
            try:
                min_sec = parts[0]
                measurements = list(map(float, parts[1:]))
                data.append([min_sec] + measurements)
            except ValueError:
                continue
    
    columns = ['Min_Sec'] + [f'D{i}' for i in range(1, 65)]
    df = pd.DataFrame(data, columns=columns)
    df['Patient_ID'] = patient_id
    
    return df

def combine_all_patients(uploaded_files):
    all_data = []
    for file in uploaded_files:
        df = load_patient_data(file)
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def convert_to_seconds(time_str):
    try:
        min, sec = map(float, time_str.split(':'))
        return min * 60 + sec
    except ValueError:
        return None

# Pages de l'application
def main_page():
    st.title('🫁 COVID-19 Diagnosis from Breath Analysis')

    # Choix du type de diagnostic
    diagnosis_type = st.selectbox("Choisissez le type de diagnostic", ["Diagnostic pour un seul patient", "Diagnostic pour plusieurs patients"])

    if diagnosis_type == "Diagnostic pour un seul patient":
        st.header('📊 Diagnostic pour un seul patient')

        col1, col2 = st.columns(2)
        
        with col1:
            min_sec = st.number_input('Min_Sec', min_value=0.0, step=0.1)
        
        input_data = {'Min_Sec': min_sec}
        
        for i in range(1, 65):
            if i <= 32:
                with col1:
                    input_data[f'D{i}'] = st.number_input(f'D{i}', key=f'D{i}')
            else:
                with col2:
                    input_data[f'D{i}'] = st.number_input(f'D{i}', key=f'D{i}')

        if st.button('🔬 Prédire'):
            with st.spinner('Analyse en cours...'):
                response = requests.post('https://msnoc19.onrender.com/predict/', json=input_data)
                if response.status_code == 200:
                    result = response.json()["prediction"]
                    if result == "POSITIVE":
                        st.error(f'Résultat du test : {result}')
                    else:
                        st.success(f'Résultat du test : {result}')
                    st.balloons()
                else:
                    st.warning('Erreur dans la prédiction. Veuillez vérifier les valeurs saisies.')

    elif diagnosis_type == "Diagnostic pour plusieurs patients":
        st.header('📁 Prédiction en lot')

        uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")

        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            original_data = data.copy()
            
            if 'Patient_ID' in data.columns:
                data = data.drop(columns=['Patient_ID'])
            
            if st.button('🔬 Prédire en lot'):
                with st.spinner('Traitement des prédictions en lot...'):
                    response = requests.post('https://msnoc19.onrender.com/predict_batch/', json=data.to_dict(orient='records'))
                    if response.status_code == 200:
                        predictions = response.json()['predictions']
                        st.success('Prédictions terminées!')
                        
                        original_data['Prédiction'] = predictions
                        
                        st.subheader('Résultats des Prédictions')
                        st.dataframe(original_data)
                        
                        # Visualisation
                        fig = px.pie(original_data, names='Prédiction', title='Distribution des Prédictions')
                        st.plotly_chart(fig)
                        
                        csv_buffer = io.StringIO()
                        original_data.to_csv(csv_buffer, index=False)
                        csv_str = csv_buffer.getvalue()
                        
                        st.download_button(
                            label="📥 Télécharger les résultats au format CSV",
                            data=csv_str,
                            file_name="resultats_predictions.csv",
                            mime="text/csv",
                        )
                    else:
                        st.error('Erreur dans la prédiction. Veuillez vérifier le fichier d’entrée.')

def prepare_data_page():
    st.title('🔬 Préparation des données')

    uploaded_files = st.file_uploader("Téléchargez plusieurs fichiers texte", type="txt", accept_multiple_files=True)

    if uploaded_files:
        with st.spinner('Combinaison et préparation des données...'):
            combined_df = combine_all_patients(uploaded_files)
            
            combined_df['Min_Sec'] = combined_df['Min_Sec'].apply(convert_to_seconds)
            
            d_columns = [f'D{i}' for i in range(1, 65)]
            
            df_grouped = combined_df.groupby('Patient_ID')[d_columns].mean()
            df_grouped1 = combined_df.groupby('Patient_ID')['Min_Sec'].mean().reset_index()
            
            df_final = df_grouped1.set_index('Patient_ID').join(df_grouped)
            
            columns_order = ['Min_Sec'] + [col for col in df_final.columns if col != 'Min_Sec']
            df_final = df_final[columns_order]
            
            st.success('Données combinées et préparées avec succès!')
            st.subheader('Aperçu des données finales:')
            st.dataframe(df_final.head())
            
            csv = df_final.to_csv(index=True)
            st.download_button(
                label="📥 Télécharger les données préparées au format CSV",
                data=csv,
                file_name="donnees_patients_preparees.csv",
                mime="text/csv",
            )

# Navigation principale
def main():
    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Aller à', ['🏠 Page principale', '🔬 Préparer les données'])

    if page == '🏠 Page principale':
        main_page()
    elif page == '🔬 Préparer les données':
        prepare_data_page()
    
    st.sidebar.write("**Mes Coordonnées :**")
    st.sidebar.write("**Nom:** MAMA Moussinou")
    st.sidebar.write("**Email:** mamamouhsinou@gmail.com")
    st.sidebar.write("**Téléphone:** +229 95231680")
    st.sidebar.write("**LinkedIn:** moussinou-mama-8b6270284")

if __name__ == "__main__":
    main()

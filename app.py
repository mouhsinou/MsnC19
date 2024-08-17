import streamlit as st
import requests
import pandas as pd
import os
import io
import plotly.express as px

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
def load_patient_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
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

def combine_all_patients(directory):
    all_data = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            df = load_patient_data(file_path)
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

    st.header('📊 Individual Patient Analysis')
    
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

    if st.button('🔬 Predict'):
        with st.spinner('Analyzing...'):
            response = requests.post('https://msnoc19.onrender.com/predict/', json=input_data)
            if response.status_code == 200:
                result = response.json()["prediction"]
                if result == "POSITIVE":
                    st.error(f'Test Result: {result}')
                else:
                    st.success(f'Test Result: {result}')
                st.balloons()
            else:
                st.warning('Error in prediction. Please check the input values.')

    st.header('📁 Batch Prediction')

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        original_data = data.copy()
        
        if 'Patient_ID' in data.columns:
            data = data.drop(columns=['Patient_ID'])
        
        if st.button('🔬 Predict Batch'):
            with st.spinner('Processing batch prediction...'):
                response = requests.post('https://msnoc19.onrender.com/predict_batch/', json=data.to_dict(orient='records'))
                if response.status_code == 200:
                    predictions = response.json()['predictions']
                    st.success('Predictions completed!')
                    
                    original_data['Prediction'] = predictions
                    
                    st.subheader('Prediction Results')
                    st.dataframe(original_data)
                    
                    # Visualisation
                    fig = px.pie(original_data, names='Prediction', title='Distribution of Predictions')
                    st.plotly_chart(fig)
                    
                    csv_buffer = io.StringIO()
                    original_data.to_csv(csv_buffer, index=False)
                    csv_str = csv_buffer.getvalue()
                    
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv_str,
                        file_name="prediction_results.csv",
                        mime="text/csv",
                    )
                else:
                    st.error('Error in prediction. Please check the input file.')

def prepare_data_page():
    st.title('🔬 Data Preparation')

    directory = st.text_input('📁 Enter the directory path containing the text files:')

    if st.button('🔄 Combine and Prepare Data'):
        if directory and os.path.isdir(directory):
            with st.spinner('Combining and preparing data...'):
                combined_df = combine_all_patients(directory)
                
                combined_df['Min_Sec'] = combined_df['Min_Sec'].apply(convert_to_seconds)
                
                d_columns = [f'D{i}' for i in range(1, 65)]
                
                df_grouped = combined_df.groupby('Patient_ID')[d_columns].mean()
                df_grouped1 = combined_df.groupby('Patient_ID')['Min_Sec'].mean().reset_index()
                
                df_final = df_grouped1.set_index('Patient_ID').join(df_grouped)
                
                columns_order = ['Min_Sec'] + [col for col in df_final.columns if col != 'Min_Sec']
                df_final = df_final[columns_order]
                
                st.success('Data combined and prepared successfully!')
            
            st.subheader('Preview of Final Data:')
            st.dataframe(df_final.head())
            
            csv = df_final.to_csv(index=True)
            st.download_button(
                label="📥 Download Prepared Data CSV",
                data=csv,
                file_name="prepared_patient_data.csv",
                mime="text/csv",
            )
        else:
            st.error('Please enter a valid directory path.')

# Navigation principale
def main():
    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Go to', ['🏠 Main Page', '🔬 Prepare Data'])

    if page == '🏠 Main Page':
        main_page()
    elif page == '🔬 Prepare Data':
        prepare_data_page()
    
    st.sidebar.write("**Mes Coordonnées :**")
    st.sidebar.write("**Nom:** MAMA Moussinou")
    st.sidebar.write("**Email:** mamamouhsinou@gmail.com")
    st.sidebar.write("**Téléphone:** +229 95231680")
    st.sidebar.write("**LinkedIn:** moussinou-mama-8b6270284")

if __name__ == "__main__":
    main()
import os
import joblib
import pandas as pd

def load_model(model_path: str = "models/lightgbm_credit_model.joblib"):
    """Carrega o artefato do modelo treinado."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado no caminho: {model_path}")
    
    model = joblib.load(model_path)
    print(f"✅ Modelo carregado com sucesso a partir de '{model_path}'!")
    return model

def make_predictions(data_path: str = "data/train_pipeline_woe.parquet", model_path: str = "models/lightgbm_credit_model.joblib"):
    """Realiza a inferência em novos dados utilizando o modelo LightGBM treinado."""
    model = load_model(model_path)
    
    print(f"⏳ Carregando dados para inferência de '{data_path}'...")
    df = pd.read_parquet(data_path)
    
    # Separa as features (removendo colunas de identificação e target, caso existam)
    colunas_ignorar = ["id_cliente", "target", "sk_id_curr"]
    features = [c for c in df.columns if c.lower() not in colunas_ignorar]
    
    X = df[features]
    
    print("🚀 Gerando previsões de risco de crédito...")
    # Prediz a probabilidade de inadimplência (classe 1)
    prob_inadimplencia = model.predict_proba(X)[:, 1]
    
    # Cria um DataFrame com os resultados
    df_resultado = pd.DataFrame({
        "id_cliente": df["id_cliente"] if "id_cliente" in df.columns else range(len(df)),
        "probabilidade_inadimplencia": prob_inadimplencia
    })
    
    print("\n--- Amostra das Primeiras Previsões ---")
    print(df_resultado.head(10))
    
    return df_resultado

if __name__ == "__main__":
    # Executa a inferência de teste utilizando a base processada
    make_predictions()
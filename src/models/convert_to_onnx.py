"""Script to convert the Scikit-Learn baseline pipeline to ONNX format."""

import joblib
from skl2onnx import to_onnx
from skl2onnx.common.data_types import StringTensorType

# Mantivemos o nome do arquivo para não quebrar a sua API
JOBLIB_PATH = "models/linear_svm_baseline.joblib" 
ONNX_PATH = "models/model_optimized.onnx"

def main():
    print(f"Carregando modelo base: {JOBLIB_PATH}")
    pipeline = joblib.load(JOBLIB_PATH)
    
    initial_type = [('medical_abstract', StringTensorType([None, 1]))]
    
    print("Convertendo pipeline para formato ONNX...")
    # Sem hacks! O LogisticRegression é 100% suportado nativamente.
    onx = to_onnx(pipeline, initial_types=initial_type, target_opset=12)
    
    with open(ONNX_PATH, "wb") as f:
        f.write(onx.SerializeToString())
        
    print(f"Modelo ONNX salvo com sucesso em: {ONNX_PATH}")

if __name__ == "__main__":
    main()
# main.py
"""
Iris Flower Classification - Complete Pipeline
"""

import joblib
from src.data_preprocessing import load_iris_data, preprocess_data
from src.model_training import train_models
from src.evaluation import evaluate_model

def main():
    print("🌸 Starting Iris Flower Classification Project...\n")
    
    # Load data
    df, class_names = load_iris_data()
    print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]-1} features")
    print(f"Classes: {class_names}\n")
    
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # Train models
    models = train_models(X_train, y_train)
    
    # Evaluate all models and find the best one
    print("="*60)
    best_model = None
    best_acc = 0
    best_name = ""
    
    for name, model in models.items():
        print(f"\n🔍 Evaluating {name}...")
        acc = evaluate_model(model, X_test, y_test, class_names)
        
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name
    
    print("="*60)
    print(f"\n🏆 Best Model: {best_name} with Accuracy: {best_acc:.4f}")
    
    # Save the best model and scaler
    joblib.dump(best_model, f'models/best_{best_name}.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print(f"✅ Best model and scaler saved successfully!")
    
    print("\n🎉 Project completed successfully!")

if __name__ == "__main__":
    main()

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve

class AudioFeatureExtractor:
    def __init__(self, sr=22050, n_mels=128, fmax=8000, n_bins=84, bins_per_octave=12, n_mfcc=13):
        self.sr = sr
        self.n_mels = n_mels
        self.fmax = fmax
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.n_mfcc = n_mfcc
    
    def extract_mel_spectrogram(self, audio_path):
        y, sr = librosa.load(audio_path, sr=self.sr)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels, fmax=self.fmax)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    def extract_cqt_spectrogram(self, audio_path):
        y, sr = librosa.load(audio_path, sr=self.sr)
        cqt = librosa.cqt(y, sr=sr, n_bins=self.n_bins, bins_per_octave=self.bins_per_octave)
        cqt_db = librosa.amplitude_to_db(np.abs(cqt))
        return cqt_db
    
    def extract_mfcc_delta(self, audio_path):
        y, sr = librosa.load(audio_path, sr=self.sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        mfcc_features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
        return mfcc_features
    
    def extract_all_features(self, audio_path):
        mel_spec = self.extract_mel_spectrogram(audio_path)
        cqt_spec = self.extract_cqt_spectrogram(audio_path)
        mfcc_features = self.extract_mfcc_delta(audio_path)
        
        features = {
            'mel_spectrogram': mel_spec,
            'cqt_spectrogram': cqt_spec,
            'mfcc_delta': mfcc_features
        }
        
        return features
    
    def plot_features(self, features, audio_path):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        librosa.display.specshow(features['mel_spectrogram'], x_axis='time', y_axis='mel', 
                                fmax=8000, ax=axes[0, 0])
        axes[0, 0].set_title('Mel Spectrogram')
        axes[0, 0].set_colorbar(label='dB')
        
        librosa.display.specshow(features['cqt_spectrogram'], x_axis='time', y_axis='cqt_hz', 
                                ax=axes[0, 1])
        axes[0, 1].set_title('CQT Spectrogram')
        axes[0, 1].set_colorbar(label='dB')
        
        librosa.display.specshow(features['mfcc_delta'][:13], x_axis='time', ax=axes[1, 0])
        axes[1, 0].set_title('MFCC Coefficients')
        axes[1, 0].set_colorbar(label='Value')
        
        librosa.display.specshow(features['mfcc_delta'][13:26], x_axis='time', ax=axes[1, 1])
        axes[1, 1].set_title('MFCC Delta Coefficients')
        axes[1, 1].set_colorbar(label='Value')
        
        plt.suptitle(f'Audio Features: {audio_path}')
        plt.tight_layout()
        plt.show()
    
    def save_features(self, features, base_filename):
        np.save(f'{base_filename}_mel.npy', features['mel_spectrogram'])
        np.save(f'{base_filename}_cqt.npy', features['cqt_spectrogram'])
        np.save(f'{base_filename}_mfcc.npy', features['mfcc_delta'])
        print(f"Features saved as {base_filename}_*.npy")
    
    def load_features(self, base_filename):
        features = {
            'mel_spectrogram': np.load(f'{base_filename}_mel.npy'),
            'cqt_spectrogram': np.load(f'{base_filename}_cqt.npy'),
            'mfcc_delta': np.load(f'{base_filename}_mfcc.npy')
        }
        return features

class ModelEnsemble:
    def __init__(self, cnn_model1_path, cnn_model2_path, rf_model_path):
        self.cnn_model1 = load_model(cnn_model1_path)
        self.cnn_model2 = load_model(cnn_model2_path)
        
        with open(rf_model_path, 'rb') as file:
            self.rf_model = pickle.load(file)
        
        self.ensemble_model = None
    
    def get_cnn_predictions(self, model, X):
        predictions = model.predict(X)
        return predictions.flatten()
    
    def create_ensemble_features(self, X_cnn1, X_cnn2, X_rf):
        cnn1_preds = self.get_cnn_predictions(self.cnn_model1, X_cnn1)
        cnn2_preds = self.get_cnn_predictions(self.cnn_model2, X_cnn2)
        rf_preds = self.rf_model.predict_proba(X_rf)[:, 1]
        
        ensemble_features = np.column_stack((cnn1_preds, cnn2_preds, rf_preds))
        return ensemble_features
    
    def train_ensemble(self, X_cnn1, X_cnn2, X_rf, y_true):
        ensemble_features = self.create_ensemble_features(X_cnn1, X_cnn2, X_rf)
        
        self.ensemble_model = LogisticRegression(random_state=42)
        self.ensemble_model.fit(ensemble_features, y_true)
        
        return self.ensemble_model
    
    def ensemble_predict(self, X_cnn1, X_cnn2, X_rf):
        ensemble_features = self.create_ensemble_features(X_cnn1, X_cnn2, X_rf)
        
        final_predictions = self.ensemble_model.predict(ensemble_features)
        final_probabilities = self.ensemble_model.predict_proba(ensemble_features)[:, 1]
        
        return final_predictions, final_probabilities
    
    def evaluate_ensemble(self, X_cnn1, X_cnn2, X_rf, y_true):
        predictions, probabilities = self.ensemble_predict(X_cnn1, X_cnn2, X_rf)
        
        accuracy = accuracy_score(y_true, predictions)
        auc = roc_auc_score(y_true, probabilities)
        
        print(f"Ensemble Model Accuracy: {accuracy:.4f}")
        print(f"Ensemble Model AUC: {auc:.4f}")
        print("\nEnsemble Classification Report:")
        print(classification_report(y_true, predictions))
        
        self.plot_roc_curve(y_true, probabilities, auc)
        self.plot_model_weights()
        
        return accuracy, auc
    
    def plot_roc_curve(self, y_true, probabilities, auc_score):
        fpr, tpr, _ = roc_curve(y_true, probabilities)
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, label=f'Ensemble Model (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Ensemble Model')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_model_weights(self):
        model_weights = self.ensemble_model.coef_[0]
        models = ['CNN Model 1', 'CNN Model 2', 'Random Forest']
        
        plt.figure(figsize=(8, 6))
        plt.bar(models, model_weights)
        plt.title('Model Weights in Logistic Ensemble')
        plt.ylabel('Weight Coefficient')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def save_ensemble_model(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self.ensemble_model, file)
        print(f"Ensemble model saved as {file_path}")

def main():
    feature_extractor = AudioFeatureExtractor()
    
    audio_file_path = 'audio.wav'
    features = feature_extractor.extract_all_features(audio_file_path)
    
    print("Feature Shapes:")
    print(f"Mel Spectrogram: {features['mel_spectrogram'].shape}")
    print(f"CQT Spectrogram: {features['cqt_spectrogram'].shape}")
    print(f"MFCC+Delta Features: {features['mfcc_delta'].shape}")
    
    feature_extractor.plot_features(features, audio_file_path)
    feature_extractor.save_features(features, 'audio_features')
    
    ensemble = ModelEnsemble('cnn_model1.h5', 'cnn_model2.h5', 'random_forest_model.pkl')
    
    ensemble.train_ensemble(X_test_cnn1, X_test_cnn2, X_test_rf, y_test)
    
    ensemble.evaluate_ensemble(X_test_cnn1, X_test_cnn2, X_test_rf, y_test)
    
    ensemble.save_ensemble_model('logistic_ensemble_model.pkl')
    
    test_predictions, test_probabilities = ensemble.ensemble_predict(X_test_cnn1, X_test_cnn2, X_test_rf)
    
    print("\nSample Predictions:")
    for i in range(5):
        print(f"Sample {i+1}: Prediction={test_predictions[i]}, Probability={test_probabilities[i]:.4f}")

if __name__ == "__main__":
    main()
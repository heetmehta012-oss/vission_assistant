#!/usr/bin/env python3
"""
Simple training script for personal object learning.
No complex ML required - just basic computer vision.
"""

import cv2
import os
import numpy as np
import pickle
import time
from collections import defaultdict

class SimpleObjectTrainer:
    def __init__(self, data_dir="training_data"):
        self.data_dir = data_dir
        self.object_data = defaultdict(list)
        os.makedirs(data_dir, exist_ok=True)
    
    def collect_object_samples(self, object_name, num_samples=20):
        """Collect training samples for an object"""
        print(f"📸 Collecting {num_samples} samples for '{object_name}'")
        print("Position the object in different angles and lighting")
        print("Press SPACE to capture, Q to finish early")
        
        cap = cv2.VideoCapture(0)
        collected = 0
        
        while collected < num_samples:
            ret, frame = cap.read()
            if not ret:
                continue
            
            cv2.imshow(f"Training: {object_name} ({collected}/{num_samples})", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space to capture
                # Extract features from the object
                features = self.extract_features(frame)
                self.object_data[object_name].append(features)
                collected += 1
                print(f"✅ Captured sample {collected}/{num_samples}")
                
                # Save the image for reference
                cv2.imwrite(f"{self.data_dir}/{object_name}_{collected:03d}.jpg", frame)
                
            elif key == ord('q'):  # Quit early
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"🎉 Collected {collected} samples for '{object_name}'")
        return collected
    
    def extract_features(self, image):
        """Extract simple visual features from image"""
        features = []
        
        # 1. Color histogram (dominant colors)
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        features.extend(hist.flatten())
        
        # 2. Shape features (contours)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Area and perimeter
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Bounding box aspect ratio
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = w / h if h > 0 else 1
            
            # Circularity
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            
            features.extend([area, perimeter, aspect_ratio, circularity])
        else:
            features.extend([0, 0, 1, 0])  # Default values
        
        # 3. Texture features (simplified)
        # Compute standard deviation of pixel values
        std_dev = np.std(gray)
        features.append(std_dev)
        
        return np.array(features)
    
    def train_classifier(self):
        """Train a simple classifier on collected data"""
        if not self.object_data:
            print("❌ No training data available!")
            return None
        
        print("🧠 Training classifier...")
        
        # Prepare training data
        X = []
        y = []
        
        for object_name, feature_list in self.object_data.items():
            for features in feature_list:
                X.append(features)
                y.append(object_name)
        
        X = np.array(X)
        y = np.array(y)
        
        # Use a simple k-NN classifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.preprocessing import StandardScaler
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train classifier
        classifier = KNeighborsClassifier(n_neighbors=3)
        classifier.fit(X_scaled, y)
        
        print(f"✅ Trained on {len(X)} samples across {len(self.object_data)} objects")
        
        return classifier, scaler
    
    def save_model(self, classifier, scaler, filename="custom_object_model.pkl"):
        """Save the trained model"""
        model_data = {
            'classifier': classifier,
            'scaler': scaler,
            'object_names': list(self.object_data.keys())
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Model saved to {filename}")
    
    def load_model(self, filename="custom_object_model.pkl"):
        """Load a trained model"""
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            print(f"📂 Model loaded from {filename}")
            return model_data['classifier'], model_data['scaler'], model_data['object_names']
        except FileNotFoundError:
            print("❌ No saved model found!")
            return None, None, []
    
    def recognize_object(self, image, classifier, scaler):
        """Recognize an object using trained model"""
        if classifier is None:
            return None
        
        features = self.extract_features(image)
        features_scaled = scaler.transform([features])
        
        # Predict
        prediction = classifier.predict(features_scaled)[0]
        confidence = classifier.predict_proba(features_scaled)[0].max()
        
        return prediction, confidence

def main():
    trainer = SimpleObjectTrainer()
    
    print("🎯 Vision Assistant Training")
    print("=" * 40)
    
    while True:
        print("\nOptions:")
        print("1. Train new object")
        print("2. Test recognition")
        print("3. Save model")
        print("4. Load model")
        print("5. Quit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            object_name = input("Enter object name: ").strip()
            if object_name:
                trainer.collect_object_samples(object_name)
        
        elif choice == '2':
            classifier, scaler, _ = trainer.load_model()
            if classifier is None:
                print("❌ No model loaded. Train or load a model first.")
                continue
            
            print("📷 Testing recognition. Point camera at objects...")
            cap = cv2.VideoCapture(0)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Try to recognize
                prediction, confidence = trainer.recognize_object(frame, classifier, scaler)
                
                # Display result
                if prediction and confidence > 0.5:
                    text = f"{prediction} ({confidence:.2f})"
                    color = (0, 255, 0)
                else:
                    text = "Unknown"
                    color = (0, 0, 255)
                
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.imshow("Object Recognition", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
        
        elif choice == '3':
            classifier, scaler, _ = trainer.load_model()
            if classifier is None:
                classifier, scaler = trainer.train_classifier()
            
            if classifier is not None:
                trainer.save_model(classifier, scaler)
        
        elif choice == '4':
            trainer.load_model()
        
        elif choice == '5':
            break
        
        else:
            print("❌ Invalid choice!")

if __name__ == "__main__":
    main()

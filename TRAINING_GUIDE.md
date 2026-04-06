# Training Guide for Vision Assistant

## 1. Custom YOLO Object Training

### Step 1: Collect Training Data
```bash
# Create dataset folder
mkdir custom_dataset
mkdir custom_dataset/images
mkdir custom_dataset/labels
```

### Step 2: Capture Your Objects
- Take 50-200 photos of each object you want to detect
- Include different angles, lighting, distances
- Examples: your specific mug, laptop model, keys, etc.

### Step 3: Label Images
Use LabelImg or similar tool:
```bash
pip install labelimg
labelimg custom_dataset/images
```

- Draw boxes around your objects
- Save as YOLO format (.txt files)
- Use consistent class names

### Step 4: Create Dataset Config
```yaml
# custom_dataset.yaml
path: custom_dataset
train: images
val: images
nc: 5  # number of classes
names: ['my_mug', 'my_laptop', 'my_keys', 'my_phone', 'my_wallet']
```

### Step 5: Train Model
```python
from ultralytics import YOLO

# Load base model
model = YOLO('yolov8s.pt')

# Train on your data
results = model.train(
    data='custom_dataset.yaml',
    epochs=50,
    imgsz=640,
    device='cpu'  # or 'cuda' if you have GPU
)

# Export your custom model
model.export(format='pt')
```

### Step 6: Use Custom Model
```python
# In detector.py
self.model = YOLO('custom_model.pt')  # Your trained model
```

## 2. Environment-Specific AI Training

### Step 1: Collect Room Data
```python
# Auto-collect room images
import cv2
import os

def collect_room_data():
    cap = cv2.VideoCapture(0)
    count = 0
    
    while count < 200:  # Collect 200 images
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f'room_data/img_{count:04d}.jpg', frame)
            count += 1
        cv2.waitKey(1000)  # 1 second between shots
    
    cap.release()
```

### Step 2: Create Environment Classifier
```python
# train_environment.py
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def extract_features(image):
    # Extract color histograms
    hist = cv2.calcHist([image], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
    
    # Extract edge density
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Extract texture features
    # ... add more features
    
    return np.concatenate([hist.flatten(), [edge_density]])

def train_environment_classifier():
    # Load labeled room images
    kitchen_features = [extract_features(img) for img in load_images('kitchen')]
    office_features = [extract_features(img) for img in load_images('office')]
    
    X = np.array(kitchen_features + office_features)
    y = np.array(['kitchen']*len(kitchen_features) + ['office']*len(office_features))
    
    # Train classifier
    clf = RandomForestClassifier()
    clf.fit(X, y)
    
    return clf
```

## 3. Personal Object Learning

### Step 1: Create Learning System
```python
# learning_system.py
class PersonalObjectLearner:
    def __init__(self):
        self.known_objects = {}
        self.object_features = {}
    
    def learn_object(self, image, object_name):
        """Learn a new personal object"""
        features = self.extract_features(image)
        
        if object_name not in self.object_features:
            self.object_features[object_name] = []
        
        self.object_features[object_name].append(features)
        
        # Keep only last 20 examples per object
        if len(self.object_features[object_name]) > 20:
            self.object_features[object_name] = self.object_features[object_name][-20:]
    
    def recognize_object(self, image):
        """Try to recognize learned objects"""
        features = self.extract_features(image)
        
        best_match = None
        best_score = 0
        
        for object_name, feature_list in self.object_features.items():
            # Compare with stored examples
            for stored_features in feature_list:
                similarity = self.calculate_similarity(features, stored_features)
                if similarity > best_score and similarity > 0.7:  # Threshold
                    best_score = similarity
                    best_match = object_name
        
        return best_match
    
    def save_learning(self, filename):
        """Save learned objects"""
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self.object_features, f)
    
    def load_learning(self, filename):
        """Load learned objects"""
        import pickle
        try:
            with open(filename, 'rb') as f:
                self.object_features = pickle.load(f)
        except FileNotFoundError:
            self.object_features = {}
```

### Step 4: Voice Command Integration
```python
# Add to command_handler.py
def _learn_object(self, object_name):
    """Learn a new personal object"""
    self.speaker.speak(f"Show me the {object_name} so I can learn it.")
    
    # Wait 3 seconds for user to position object
    time.sleep(3)
    
    with self.frame_lock:
        frame = self.state.get("latest_frame")
        if frame is not None:
            self.learning_system.learn_object(frame, object_name)
            self.speaker.speak(f"I've learned what {object_name} looks like.")
        else:
            self.speaker.speak("No camera frame available.")

# Add command handling
elif any(kw in text for kw in ["learn", "remember object", "teach object"]):
    object_name = self._extract_name(text, ["learn", "remember", "teach"])
    if object_name:
        self._learn_object(object_name)
```

## 4. Quick Start Training

### Option A: Face Recognition (Easiest)
```bash
# Just use voice commands
"computer" → "remember John"  # Look at camera
"computer" → "remember Mary"  # Look at camera
```

### Option B: Custom Objects (Medium)
```bash
# Collect 50 photos per object
python collect_object_data.py
# Label them
labelimg object_images/
# Train
python train_custom_yolo.py
```

### Option C: Environment Learning (Advanced)
```bash
# Collect room data
python collect_room_data.py
# Train environment classifier
python train_environment.py
```

## Training Tips

1. **Start Small** - Train on 3-5 important objects first
2. **Good Lighting** - Consistent lighting improves accuracy
3. **Multiple Angles** - Capture objects from different perspectives
4. **Clean Background** - Remove clutter when training
5. **Regular Updates** - Retrain monthly for best results

## Hardware Requirements

- **CPU Training**: Works but slow (2-6 hours)
- **GPU Training**: Much faster (30 minutes - 1 hour)
- **Storage**: 1-5GB for training data
- **RAM**: 8GB+ recommended for training

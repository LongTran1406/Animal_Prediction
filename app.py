from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import cv2

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model('model/vgg16_model.h5')

# Define the image size that the model expects
target_size = (224, 224)

def normalization(pic_test):
    mn = 0
    mx = 1
    norm_type = cv2.NORM_MINMAX
    b, g, r = cv2.split(pic_test)
    b_normalized = cv2.normalize(b.astype(np.float32), None, mn, mx, norm_type)
    g_normalized = cv2.normalize(g.astype(np.float32), None, mn, mx, norm_type)
    r_normalized = cv2.normalize(r.astype(np.float32), None, mn, mx, norm_type)
    
    normalized_image = cv2.merge((b_normalized, g_normalized, r_normalized))
    return normalized_image

@app.route('/')
def home():
    return render_template('index.html')  # Upload page

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    if file:
        # Save uploaded file
        img_path = os.path.join('static', file.filename)
        file.save(img_path)
        
        # Read the image with OpenCV
        pic_test = cv2.imread(img_path)
        
        # Normalize the image
        normalized_image = normalization(pic_test)
        
        # Resize image to fit the model input
        normalized_image = cv2.resize(normalized_image, target_size)
        
        # Convert to RGB (since OpenCV reads in BGR)
        normalized_image = cv2.cvtColor(normalized_image, cv2.COLOR_BGR2RGB)
        
        # Convert image to array and expand dimensions for prediction
        img_array = image.img_to_array(normalized_image)
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction)  # Get the predicted class index
        
        # Define your class labels in English
        class_labels = [
            'Dog', 'Horse', 'Elephant', 'Butterfly', 'Chicken', 
            'Cat', 'Cow', 'Sheep', 'Spider', 'Squirrel'
        ]

        # Map the predicted index to a class label
        predicted_label = class_labels[predicted_class_index]
        predicted_prob = prediction[0][predicted_class_index] * 100
        
        # Debugging output
        predicted_prob = round(predicted_prob)
        print(f"Prediction: {prediction}")
        print(f"Predicted Class Index: {predicted_class_index}")
        print(f"Predicted Label: {predicted_label}")
        print(f"Prediction Probability: {predicted_prob:.2f}%")
        
        # Pass both predicted_label and predicted_prob to the template
        img_filename = os.path.basename(img_path)  # Extract filename only
        return render_template('result.html', predicted_label=predicted_label, predicted_prob=predicted_prob, img_filename=img_filename)




if __name__ == '__main__':
    app.run(debug=True)

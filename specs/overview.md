Cerdas Isyarat is an interactive web application designed to help children learn BISINDO (Bahasa Isyarat Indonesia) in a fun and engaging way. One of its key features is an AI-powered computer vision tool that allows users to practice by translating words into sign language gestures and receiving instant feedback, making the learning experience more interactive and effective.

**this project uses Python with the following key libraries:**

- scikit-learn for machine learning algorithms and metrics
- numpy for numerical operations
- pandas for data manipulation
- imgaug for image augmentation
- mediapipe for hand landmarks detection
- ONNX for browser inferrence

**Here's the workflow of the project:**

- Load the dataset and preprocess the images
- Augment the images using imgaug with 50 images for each Alphabet
- Extract hand landmarks from the images using mediapipe
- Extract features from the hand landmarks
- Train and evaluate machine learning models
- Compare the models and choose the best one
- Use ONNX on web browser to capture real-time video and classify the BISINDO alphabet
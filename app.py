import numpy as np
import streamlit as st
import cv2
from tensorflow.keras.models import load_model
import random

# تحميل النموذج المدرب
@st.cache_resource
def load_model_cached(model_path):
    return load_model(model_path)

# تحميل البيانات
@st.cache_data
def load_data(x_path, y_path):
    X_data = np.load(x_path)
    Y_data = np.load(y_path)
    return X_data, Y_data

# تحميل النموذج المدرب
model_path = '/Users/mk/Downloads/vgg19_the_last_model.keras'  # استبدل بالمسار الصحيح
model = load_model_cached(model_path)

# تحميل البيانات
X_data, Y_data = load_data('/Users/mk/Downloads/X_data.npy', '/Users/mk/Downloads/Y_data.npy')

# تقسيم البيانات
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.2)

# واجهة Streamlit
st.title("Cataract Diagnosis")
st.write("Here are some images used for training, along with their actual labels and the model's predictions.")

# عرض 3 صور مع التصنيف الفعلي والتنبؤات
y_pred = model.predict(x_test)  # توقعات النموذج على بيانات الاختبار

# عرض 3 صور فقط في كل مرة
st.write("Model predictions for random samples from test data:")
for i in range(3):  # عرض 3 صور فقط في كل مرة
    sample = random.choice(range(len(x_test)))  # اختيار عينة عشوائية
    image = x_test[sample]
    category = y_test[sample]
    pred_category = y_pred[sample]
    
    # تصنيف الصورة الفعلي
    label = "Normal" if category == 0 else "Cataract"
    
    # تصنيف التنبؤ (اعتمادًا على الاحتمالية، إذا كانت > 0.5 نعتبرها "Cataract")
    pred_label = "Cataract" if pred_category[0] > 0.5 else "Normal"
    
    # عرض الصورة في Streamlit بحجم أصغر
    st.image(image, caption=f"Actual: {label}\nPrediction: {pred_label}", width=300)  # عرض الصورة بحجم أصغر

# الآن، إضافة خيار لتحميل صورة جديدة من قبل المستخدم
st.write("Upload an image of the eye for cataract detection and diagnosis.")

# رفع صورة جديدة
uploaded_file = st.file_uploader("Please upload an eye image to detect cataracts (Accepted formats: JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # قراءة الصورة المرفوعة
    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    
    # تغيير الحجم ليطابق المدخلات للنموذج
    img_resized = cv2.resize(img, (224, 224))  # التأكد من تصغير الصورة إلى الحجم المطلوب
    img_resized = np.array(img_resized) / 255.0  # تطبيع الصورة
    
    # عرض الصورة المرفوعة بحجم مناسب
    st.image(img_resized, caption="Uploaded Image", width=300)  # عرض الصورة بحجم أصغر

    # إضافة بعد للدفعة
    img_resized = np.expand_dims(img_resized, axis=0)
    
    # التنبؤ
    prediction = model.predict(img_resized)

    # عرض النتيجة
    pred_label = 'Cataract' if prediction[0] > 0.5 else 'Normal'
    
    st.write(f"Prediction: {pred_label}")
    st.write(f"Prediction probability: {prediction[0]}")

    # إذا كانت الاحتمالية منخفضة جدًا (أقل من 0.5) يمكن أن تعتبر التنبؤ غير موثوق
    if prediction[0] < 0.5:
        st.write("The prediction is not confident. Please upload a clearer image.")

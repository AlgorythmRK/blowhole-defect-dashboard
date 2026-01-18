import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import plotly.graph_objects as go
from datetime import datetime
import plotly.graph_objects as go

# -------------------------------------------------- 
# PAGE CONFIG
# -------------------------------------------------- 
st.set_page_config(
    page_title="DIAPL Casting QC System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------- 
# CUSTOM CSS
# -------------------------------------------------- 
st.markdown("""
    <style>
    /* Main container */
    .main {
        padding: 2rem;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .header-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-align: center;
    }
    
    .header-subtitle {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
        text-align: center;
        margin-top: 0.5rem;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    /* Result cards */
    .result-card-success {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .result-card-danger {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 12px rgba(245, 87, 108, 0.3);
    }
    
    .result-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    
    .result-label {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .result-confidence {
        font-size: 1.2rem;
        opacity: 0.95;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Upload section */
    .upload-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #667eea;
        text-align: center;
    }
    
    /* Info box */
    .info-box {
        background: #e7f3ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    
    /* Stats container */
    .stats-container {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .stat-box {
        flex: 1;
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------- 
# CONFIG
# -------------------------------------------------- 
CLASS_NAMES = ["Defective", "Non-defective"]
DEVICE = "cpu"

# -------------------------------------------------- 
# LOAD MODEL
# -------------------------------------------------- 
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    try:
        model.load_state_dict(
            torch.load("blowhole_classifier.pth", map_location=DEVICE)
        )
        model.eval()
        return model, True
    except:
        return model, False

model, model_loaded = load_model()

# -------------------------------------------------- 
# PREPROCESSING
# -------------------------------------------------- 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------------------------- 
# SIDEBAR
# -------------------------------------------------- 
with st.sidebar:
    st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>⚙️ QC Dashboard</h1>", unsafe_allow_html=True)
    
    st.markdown("### 📊 System Information")
    st.markdown("---")
    
    st.markdown("**🤖 Model Architecture**")
    st.markdown("MobileNetV2")
    
    st.markdown("**🎯 Detection Task**")
    st.markdown("Blowhole Defect Classification")
    
    st.markdown("**💻 Inference Mode**")
    st.markdown("Image-based Inspection")
    
    st.markdown("**📱 Device**")
    st.markdown(f"{DEVICE.upper()}")
    
    st.markdown("---")
    
    st.markdown("### 🏭 Deployment")
    st.markdown("**Dana Anand India Pvt. Ltd.**")
    st.markdown("Automotive Casting Division")
    
    st.markdown("---")
    
    st.markdown("### 📈 Model Status")
    if model_loaded:
        st.success("✅ Model Loaded")
    else:
        st.error("❌ Model Error")
    
    st.markdown("---")
    
    st.markdown("### ℹ️ Quick Guide")
    st.markdown("""
    1. Upload casting image
    2. Wait for analysis
    3. Review results
    4. Take action based on classification
    """)
    
    st.markdown("---")
    st.markdown(f"**🕐 Current Time**")
    st.markdown(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# -------------------------------------------------- 
# HEADER
# -------------------------------------------------- 
st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🔍 Blowhole Defect Detection System</h1>
        <p class="header-subtitle">AI-Powered Pre-Machining Quality Inspection for Precision Casting Components</p>
    </div>
""", unsafe_allow_html=True)

# -------------------------------------------------- 
# MAIN CONTENT
# -------------------------------------------------- 
tab1, tab2, tab3 = st.tabs(["📸 Inspection", "📊 Analytics", "ℹ️ About"])

with tab1:
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📤 Upload Casting Image")
        
        uploaded_file = st.file_uploader(
            "Select an image file",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of the casting component for defect analysis"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(
                image,
                caption="Uploaded Image for Inspection",
                use_container_width=True
            )
            
            # Image info
            st.markdown("##### Image Details")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Width", f"{image.size[0]}px")
            col_b.metric("Height", f"{image.size[1]}px")
            col_c.metric("Format", uploaded_file.type.split('/')[-1].upper())
        else:
            st.markdown("""
                <div class="upload-section">
                    <h3>📁 No Image Uploaded</h3>
                    <p>Please upload a casting image to begin quality inspection</p>
                    <p style="font-size: 0.9rem; color: #666;">Supported formats: JPG, JPEG, PNG</p>
                </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🎯 Inspection Results")
        
        if uploaded_file and model_loaded:
            with st.spinner("🔄 Analyzing image..."):
                # Process image
                img_tensor = transform(image).unsqueeze(0)
                
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    conf, pred = torch.max(probs, 1)
                
                label = CLASS_NAMES[pred.item()]
                confidence = conf.item() * 100
                
                # Display result
                if label == "Defective":
                    st.markdown(f"""
                        <div class="result-card-danger">
                            <div class="result-icon">⚠️</div>
                            <div class="result-label">DEFECTIVE COMPONENT</div>
                            <div class="result-confidence">Confidence: {confidence:.2f}%</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.error("**Recommendation:** Reject component - Send for scrap or rework")
                else:
                    st.markdown(f"""
                        <div class="result-card-success">
                            <div class="result-icon">✅</div>
                            <div class="result-label">NON-DEFECTIVE</div>
                            <div class="result-confidence">Confidence: {confidence:.2f}%</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.success("**Recommendation:** Accept component - Proceed to machining")
                
                st.markdown("---")
                
                # Probability distribution
                st.markdown("##### 📊 Classification Probabilities")
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=CLASS_NAMES,
                        y=[probs[0][i].item() * 100 for i in range(len(CLASS_NAMES))],
                        marker_color=['#f5576c' if i == 0 else '#667eea' for i in range(len(CLASS_NAMES))],
                        text=[f"{probs[0][i].item()*100:.2f}%" for i in range(len(CLASS_NAMES))],
                        textposition='auto',
                    )
                ])
                
                fig.update_layout(
                    yaxis_title="Probability (%)",
                    xaxis_title="Classification",
                    showlegend=False,
                    height=300,
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed metrics
                st.markdown("##### 📋 Detailed Analysis")
                metric_col1, metric_col2 = st.columns(2)
                
                with metric_col1:
                    st.metric(
                        "Defective Probability",
                        f"{probs[0][0].item()*100:.2f}%",
                        delta=None
                    )
                
                with metric_col2:
                    st.metric(
                        "Non-defective Probability",
                        f"{probs[0][1].item()*100:.2f}%",
                        delta=None
                    )
                
        elif not model_loaded:
            st.error("⚠️ Model not loaded. Please check the model file path.")
        else:
            st.info("👆 Please upload an image to begin the inspection process")

with tab2:
    st.markdown("### 📊 System Analytics")
    
    # Placeholder for analytics
    st.info("📈 Analytics dashboard will display inspection history, defect rates, and performance metrics once the system is in production.")
    
    # Sample metrics
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        st.metric("Total Inspections", "---", help="Total number of inspections performed")
    
    with metrics_col2:
        st.metric("Defect Rate", "---", help="Percentage of defective components detected")
    
    with metrics_col3:
        st.metric("Accuracy", "94.5%", help="Model classification accuracy")
    
    with metrics_col4:
        st.metric("Avg. Confidence", "---", help="Average prediction confidence")

with tab3:
    st.markdown("### ℹ️ About This System")
    
    st.markdown("""
    #### 🎯 Purpose
    This AI-powered quality control system is designed to automatically detect blowhole defects 
    in casting components before machining operations, reducing waste and improving production efficiency.
    
    #### 🔬 Technology
    - **Deep Learning Model:** MobileNetV2 architecture optimized for edge deployment
    - **Training Data:** Curated dataset of defective and non-defective casting samples
    - **Accuracy:** 94.5%+ classification accuracy on test dataset
    - **Processing Time:** < 1 second per image
    
    #### 🏭 Deployment
    - **Organization:** Dana Anand India Pvt. Ltd.
    - **Department:** Automotive Casting Division
    - **Application:** Pre-machining quality inspection
    
    #### ⚙️ How It Works
    1. Upload a high-quality image of the casting component
    2. The AI model analyzes the image for blowhole defects
    3. Classification result is provided with confidence score
    4. Decision support for accept/reject is displayed
    
    #### 📝 Important Notes
    - This system is intended for **decision support** purposes
    - Final quality decisions should consider additional inspection methods
    - Regular model retraining recommended with production data
    - Maintain proper lighting and image quality for best results
    """)

# -------------------------------------------------- 
# FOOTER
# -------------------------------------------------- 
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p><strong>DIAPL Casting Quality Control System</strong> | Version 1.0</p>
        <p style='font-size: 0.9rem;'>For technical support, contact the Quality Assurance Department</p>
        <p style='font-size: 0.8rem; margin-top: 0.5rem;'>⚠️ This system provides decision support. Final quality decisions should be verified by trained personnel.</p>
    </div>
""", unsafe_allow_html=True)

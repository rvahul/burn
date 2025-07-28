import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
import matplotlib.pyplot as plt
from rembg import remove
import io
import base64
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Medical Burn Analysis System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .severity-mild {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .severity-moderate {
        background-color: #fdeaa7;
        border-left: 4px solid #fd7e14;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .severity-severe {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .fluid-calc {
        background: #e3f2fd;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #2196f3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class BurnAnalysisApp:
    """Complete Burn Analysis Application"""
    
    def __init__(self):
        self.colors = {
            'burn_mild': (255, 255, 0),      # Yellow
            'burn_moderate': (255, 165, 0),   # Orange  
            'burn_severe': (255, 0, 0),       # Red
            'healthy_skin': (0, 255, 0),      # Green
            'mask_overlay': (255, 0, 255)     # Magenta
        }
        
        # Rule of Nines percentages for different body parts
        self.rule_of_nines = {
            'head_neck': 9,
            'each_arm': 9,
            'chest': 9,
            'abdomen': 9,
            'upper_back': 9,
            'lower_back': 9,
            'each_leg_front': 9,
            'each_leg_back': 9,
            'genitals': 1
        }
    
    def remove_background(self, image):
        """Remove background from the image"""
        try:
            # Convert PIL image to bytes
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='PNG')
            img_bytes = img_bytes.getvalue()
            
            # Remove background
            output_data = remove(img_bytes)
            bg_removed_img = Image.open(io.BytesIO(output_data)).convert('RGB')
            
            return bg_removed_img
        except Exception as e:
            st.warning(f"Background removal failed: {e}. Using original image.")
            return image
    
    def create_burn_mask_color_based(self, image):
        """Create burn mask based on color analysis"""
        img_array = np.array(image)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Define color ranges for different burn conditions
        lower_pink = np.array([0, 30, 50])
        upper_pink = np.array([20, 255, 255])
        pink_mask = cv2.inRange(hsv, lower_pink, upper_pink)
        
        # Red regions
        lower_red1 = np.array([160, 50, 50])
        upper_red1 = np.array([180, 255, 255])
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        
        lower_red2 = np.array([0, 50, 50])
        upper_red2 = np.array([10, 255, 255])
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # White/pale regions
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Combine masks
        burn_mask = cv2.bitwise_or(pink_mask, red_mask)
        burn_mask = cv2.bitwise_or(burn_mask, white_mask)
        
        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        burn_mask = cv2.morphologyEx(burn_mask, cv2.MORPH_CLOSE, kernel)
        burn_mask = cv2.morphologyEx(burn_mask, cv2.MORPH_OPEN, kernel)
        
        # Remove small components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            burn_mask, connectivity=8
        )
        
        min_area = 100
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                burn_mask[labels == i] = 0
        
        return burn_mask, pink_mask, red_mask, white_mask
    
    def create_severity_mask(self, image, burn_mask):
        """Create severity-based classification"""
        img_array = np.array(image)
        severity_mask = np.zeros(burn_mask.shape, dtype=np.uint8)
        
        burn_regions = burn_mask > 0
        
        if np.any(burn_regions):
            burn_pixels = img_array[burn_regions]
            
            red_intensity = burn_pixels[:, 0].astype(float)
            green_intensity = burn_pixels[:, 1].astype(float)
            blue_intensity = burn_pixels[:, 2].astype(float)
            
            redness_ratio = red_intensity / (green_intensity + blue_intensity + 1)
            
            severe_threshold = 1.5
            moderate_threshold = 1.1
            
            severe_pixels = redness_ratio > severe_threshold
            moderate_pixels = (redness_ratio > moderate_threshold) & (redness_ratio <= severe_threshold)
            mild_pixels = redness_ratio <= moderate_threshold
            
            burn_coords = np.where(burn_regions)
            severity_mask[burn_coords[0][severe_pixels], burn_coords[1][severe_pixels]] = 3
            severity_mask[burn_coords[0][moderate_pixels], burn_coords[1][moderate_pixels]] = 2
            severity_mask[burn_coords[0][mild_pixels], burn_coords[1][mild_pixels]] = 1
        
        return severity_mask
    
    def calculate_burn_percentage(self, burn_mask, severity_mask, patient_age, patient_weight):
        """Calculate burn percentage and statistics"""
        total_pixels = burn_mask.shape[0] * burn_mask.shape[1]
        burn_pixels = np.sum(burn_mask > 0)
        
        mild_pixels = np.sum(severity_mask == 1)
        moderate_pixels = np.sum(severity_mask == 2)
        severe_pixels = np.sum(severity_mask == 3)
        
        # Calculate percentages
        total_burn_percentage = (burn_pixels / total_pixels) * 100
        
        # Adjust for patient age (Lund-Browder chart approximation)
        if patient_age < 1:
            head_adjustment = 1.5  # Infants have larger head proportion
        elif patient_age < 5:
            head_adjustment = 1.2
        else:
            head_adjustment = 1.0
        
        # Apply age adjustment (simplified)
        adjusted_burn_percentage = total_burn_percentage * head_adjustment
        
        stats = {
            'total_burn_pixels': int(burn_pixels),
            'total_burn_percentage': float(adjusted_burn_percentage),
            'mild_pixels': int(mild_pixels),
            'moderate_pixels': int(moderate_pixels),
            'severe_pixels': int(severe_pixels),
            'mild_percentage': float((mild_pixels / burn_pixels * 100) if burn_pixels > 0 else 0),
            'moderate_percentage': float((moderate_pixels / burn_pixels * 100) if burn_pixels > 0 else 0),
            'severe_percentage': float((severe_pixels / burn_pixels * 100) if burn_pixels > 0 else 0)
        }
        
        return stats
    
    def calculate_fluid_requirements(self, burn_percentage, patient_age, patient_weight):
        """Calculate fluid requirements using Parkland Formula and other guidelines"""
        
        # Parkland Formula: 4 ml × weight (kg) × % burn
        # First 8 hours: 50% of total
        # Next 16 hours: 50% of total
        
        parkland_total = 4 * patient_weight * burn_percentage
        parkland_first_8h = parkland_total / 2
        parkland_next_16h = parkland_total / 2
        
        # Modified Brook Formula (for comparison)
        brook_total = 2 * patient_weight * burn_percentage
        
        # Maintenance fluid requirements
        if patient_weight <= 10:
            maintenance = 100 * patient_weight
        elif patient_weight <= 20:
            maintenance = 1000 + 50 * (patient_weight - 10)
        else:
            maintenance = 1500 + 20 * (patient_weight - 20)
        
        # Age-specific adjustments
        if patient_age < 2:
            maintenance_multiplier = 1.2  # Higher metabolic rate
        elif patient_age < 12:
            maintenance_multiplier = 1.1
        else:
            maintenance_multiplier = 1.0
        
        maintenance = maintenance * maintenance_multiplier
        
        # Total fluid requirements
        total_first_24h = parkland_total + maintenance
        
        fluid_calc = {
            'parkland_total_24h': float(parkland_total),
            'parkland_first_8h': float(parkland_first_8h),
            'parkland_next_16h': float(parkland_next_16h),
            'brook_total_24h': float(brook_total),
            'maintenance_24h': float(maintenance),
            'total_24h': float(total_first_24h),
            'hourly_rate_first_8h': float(parkland_first_8h / 8),
            'hourly_rate_next_16h': float(parkland_next_16h / 16)
        }
        
        return fluid_calc
    
    def create_visualizations(self, original_image, bg_removed_image, burn_mask, severity_mask):
        """Create comprehensive visualizations"""
        
        # Convert to numpy arrays
        bg_removed_array = np.array(bg_removed_image)
        
        # Create overlays
        overlays = {}
        
        # Basic mask overlay
        basic_overlay = bg_removed_array.copy()
        burn_pixels = burn_mask > 0
        basic_overlay[burn_pixels] = [255, 0, 255]  # Magenta
        overlays['basic_mask'] = basic_overlay
        
        # Severity overlay
        severity_overlay = bg_removed_array.copy()
        mild_pixels = severity_mask == 1
        moderate_pixels = severity_mask == 2
        severe_pixels = severity_mask == 3
        
        severity_overlay[mild_pixels] = self.colors['burn_mild']
        severity_overlay[moderate_pixels] = self.colors['burn_moderate']
        severity_overlay[severe_pixels] = self.colors['burn_severe']
        overlays['severity_overlay'] = severity_overlay
        
        # Blended overlay
        blended_overlay = bg_removed_array.copy().astype(float)
        alpha = 0.6
        mask_color = np.array([255, 0, 255], dtype=float)
        
        for i in range(3):
            blended_overlay[burn_pixels, i] = (
                alpha * mask_color[i] + 
                (1 - alpha) * blended_overlay[burn_pixels, i]
            )
        overlays['blended_overlay'] = blended_overlay.astype(np.uint8)
        
        # Contour overlay
        contour_overlay = bg_removed_array.copy()
        contours, _ = cv2.findContours(burn_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_overlay, contours, -1, (0, 255, 0), 3)
        overlays['contour_overlay'] = contour_overlay
        
        return overlays

def main():
    """Main Streamlit application"""
    
    # Initialize the app
    app = BurnAnalysisApp()
    
    # Header
    st.markdown('<h1 class="main-header">🔥 Medical Burn Analysis System</h1>', unsafe_allow_html=True)
    
    # Sidebar for patient information
    st.sidebar.header("📋 Patient Information")
    
    patient_age = st.sidebar.number_input(
        "Patient Age (years)", 
        min_value=0.1, 
        max_value=100.0, 
        value=5.0, 
        step=0.1,
        help="Age affects burn percentage calculation (Lund-Browder chart)"
    )
    
    patient_weight = st.sidebar.number_input(
        "Patient Weight (kg)", 
        min_value=1.0, 
        max_value=200.0, 
        value=20.0, 
        step=0.1,
        help="Weight is crucial for fluid requirement calculations"
    )
    
    patient_gender = st.sidebar.selectbox(
        "Patient Gender",
        ["Male", "Female", "Other"],
        help="May affect body surface area calculations"
    )
    
    # Analysis settings
    st.sidebar.header("⚙️ Analysis Settings")
    
    remove_bg = st.sidebar.checkbox(
        "Remove Background", 
        value=True,
        help="Automatically remove background for better analysis"
    )
    
    mask_threshold = st.sidebar.slider(
        "Mask Sensitivity", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.5, 
        step=0.1,
        help="Adjust sensitivity for burn detection"
    )
    
    # File upload
    st.header("📤 Upload Medical Images")
    
    uploaded_files = st.file_uploader(
        "Choose medical images...",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Upload medical photographs showing burn injuries"
    )
    
    if uploaded_files:
        for idx, uploaded_file in enumerate(uploaded_files):
            st.markdown(f"## 📊 Analysis Results for: {uploaded_file.name}")
            
            # Load image
            original_image = Image.open(uploaded_file).convert('RGB')
            
            # Create columns for layout
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("🖼️ Original Image")
                st.image(original_image, caption="Original Medical Image", use_column_width=True)
            
            # Process image
            with st.spinner("🔄 Processing image..."):
                
                # Remove background if selected
                if remove_bg:
                    bg_removed_image = app.remove_background(original_image)
                else:
                    bg_removed_image = original_image
                
                # Create burn masks
                burn_mask, pink_mask, red_mask, white_mask = app.create_burn_mask_color_based(bg_removed_image)
                severity_mask = app.create_severity_mask(bg_removed_image, burn_mask)
                
                # Calculate statistics
                burn_stats = app.calculate_burn_percentage(
                    burn_mask, severity_mask, patient_age, patient_weight
                )
                
                # Calculate fluid requirements
                fluid_calc = app.calculate_fluid_requirements(
                    burn_stats['total_burn_percentage'], patient_age, patient_weight
                )
                
                # Create visualizations
                overlays = app.create_visualizations(
                    original_image, bg_removed_image, burn_mask, severity_mask
                )
            
            with col2:
                st.subheader("🎯 Processed Image")
                if remove_bg:
                    st.image(bg_removed_image, caption="Background Removed", use_column_width=True)
                else:
                    st.image(overlays['basic_mask'], caption="Burn Mask Overlay", use_column_width=True)
            
            # Analysis Results
            st.subheader("📈 Burn Analysis Results")
            
            # Key metrics
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<h3>{burn_stats["total_burn_percentage"]:.1f}%</h3>'
                    f'<p>Total Burn Area</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            
            with metric_col2:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<h3>{fluid_calc["total_24h"]:.0f} ml</h3>'
                    f'<p>24h Fluid Requirement</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            
            with metric_col3:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<h3>{fluid_calc["hourly_rate_first_8h"]:.0f} ml/h</h3>'
                    f'<p>First 8h Rate</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            
            with metric_col4:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<h3>{fluid_calc["hourly_rate_next_16h"]:.0f} ml/h</h3>'
                    f'<p>Next 16h Rate</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            
            # Detailed visualizations
            st.subheader("🔍 Detailed Analysis")
            
            vis_tab1, vis_tab2, vis_tab3, vis_tab4 = st.tabs([
                "🎨 Segmentation Masks", 
                "📊 Severity Analysis", 
                "💧 Fluid Calculations", 
                "📋 Clinical Report"
            ])
            
            with vis_tab1:
                mask_col1, mask_col2, mask_col3 = st.columns(3)
                
                with mask_col1:
                    st.image(burn_mask, caption="Binary Burn Mask", use_column_width=True, clamp=True)
                
                with mask_col2:
                    st.image(overlays['severity_overlay'], caption="Severity Classification", use_column_width=True)
                    st.markdown("""
                    **Color Legend:**
                    - 🟡 Yellow: Mild burns
                    - 🟠 Orange: Moderate burns  
                    - 🔴 Red: Severe burns
                    """)
                
                with mask_col3:
                    st.image(overlays['contour_overlay'], caption="Burn Contours", use_column_width=True)
            
            with vis_tab2:
                # Severity breakdown
                severity_data = {
                    'Severity': ['Mild', 'Moderate', 'Severe'],
                    'Percentage': [
                        burn_stats['mild_percentage'],
                        burn_stats['moderate_percentage'],
                        burn_stats['severe_percentage']
                    ],
                    'Area (pixels)': [
                        burn_stats['mild_pixels'],
                        burn_stats['moderate_pixels'],
                        burn_stats['severe_pixels']
                    ]
                }
                
                # Pie chart
                fig = px.pie(
                    values=severity_data['Percentage'],
                    names=severity_data['Severity'],
                    title="Burn Severity Distribution",
                    color_discrete_map={
                        'Mild': '#ffc107',
                        'Moderate': '#fd7e14', 
                        'Severe': '#dc3545'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Severity cards
                if burn_stats['mild_percentage'] > 0:
                    st.markdown(
                        f'<div class="severity-mild">'
                        f'<strong>🟡 Mild Burns:</strong> {burn_stats["mild_percentage"]:.1f}% '
                        f'({burn_stats["mild_pixels"]:,} pixels)<br>'
                        f'Characteristics: Superficial burns, minimal tissue damage'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                if burn_stats['moderate_percentage'] > 0:
                    st.markdown(
                        f'<div class="severity-moderate">'
                        f'<strong>🟠 Moderate Burns:</strong> {burn_stats["moderate_percentage"]:.1f}% '
                        f'({burn_stats["moderate_pixels"]:,} pixels)<br>'
                        f'Characteristics: Partial thickness burns, require medical attention'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                if burn_stats['severe_percentage'] > 0:
                    st.markdown(
                        f'<div class="severity-severe">'
                        f'<strong>🔴 Severe Burns:</strong> {burn_stats["severe_percentage"]:.1f}% '
                        f'({burn_stats["severe_pixels"]:,} pixels)<br>'
                        f'Characteristics: Full thickness burns, critical medical intervention required'
                        f'</div>',
                        unsafe_allow_html=True
                    )
            
            with vis_tab3:
                st.markdown(
                    f'<div class="fluid-calc">'
                    f'<h3>💧 Fluid Resuscitation Protocol</h3>'
                    f'<p><strong>Patient:</strong> {patient_age} years old, {patient_weight} kg {patient_gender.lower()}</p>'
                    f'<p><strong>Burn Area:</strong> {burn_stats["total_burn_percentage"]:.1f}% TBSA</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                # Fluid calculation table
                fluid_df = pd.DataFrame({
                    'Protocol': [
                        'Parkland Formula (Total 24h)',
                        'First 8 hours (50%)',
                        'Next 16 hours (50%)',
                        'Brook Formula (Alternative)',
                        'Maintenance Fluids',
                        'Total 24h Requirement'
                    ],
                    'Volume (ml)': [
                        f"{fluid_calc['parkland_total_24h']:.0f}",
                        f"{fluid_calc['parkland_first_8h']:.0f}",
                        f"{fluid_calc['parkland_next_16h']:.0f}",
                        f"{fluid_calc['brook_total_24h']:.0f}",
                        f"{fluid_calc['maintenance_24h']:.0f}",
                        f"{fluid_calc['total_24h']:.0f}"
                    ],
                    'Rate (ml/h)': [
                        f"{fluid_calc['parkland_total_24h']/24:.0f}",
                        f"{fluid_calc['hourly_rate_first_8h']:.0f}",
                        f"{fluid_calc['hourly_rate_next_16h']:.0f}",
                        f"{fluid_calc['brook_total_24h']/24:.0f}",
                        f"{fluid_calc['maintenance_24h']/24:.0f}",
                        f"{fluid_calc['total_24h']/24:.0f}"
                    ]
                })
                
                st.dataframe(fluid_df, use_container_width=True)
                
                # Fluid timeline chart
                hours = list(range(0, 25))
                cumulative_fluid = []
                
                for hour in hours:
                    if hour <= 8:
                        fluid = hour * fluid_calc['hourly_rate_first_8h']
                    else:
                        fluid = (fluid_calc['parkland_first_8h'] + 
                                (hour - 8) * fluid_calc['hourly_rate_next_16h'])
                    cumulative_fluid.append(fluid)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=hours,
                    y=cumulative_fluid,
                    mode='lines+markers',
                    name='Cumulative Fluid',
                    line=dict(color='blue', width=3)
                ))
                
                fig.update_layout(
                    title="24-Hour Fluid Resuscitation Timeline",
                    xaxis_title="Hours from Burn Injury",
                    yaxis_title="Cumulative Fluid (ml)",
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Critical alerts
                if burn_stats['total_burn_percentage'] > 20:
                    st.error("🚨 **CRITICAL BURN**: >20% TBSA - Immediate intensive care required!")
                elif burn_stats['total_burn_percentage'] > 10:
                    st.warning("⚠️ **MAJOR BURN**: >10% TBSA - Hospital admission recommended")
                else:
                    st.info("ℹ️ **MINOR BURN**: <10% TBSA - Outpatient management may be appropriate")
            
            with vis_tab4:
                # Clinical report
                st.subheader("📋 Clinical Assessment Report")
                
                report_data = {
                    "Assessment Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Patient Information": f"{patient_age} years old, {patient_weight} kg {patient_gender.lower()}",
                    "Image File": uploaded_file.name,
                    "Total Burn Area (TBSA)": f"{burn_stats['total_burn_percentage']:.1f}%",
                    "Burn Classification": "Major" if burn_stats['total_burn_percentage'] > 20 else "Moderate" if burn_stats['total_burn_percentage'] > 10 else "Minor",
                    "Severity Distribution": {
                        "Mild": f"{burn_stats['mild_percentage']:.1f}%",
                        "Moderate": f"{burn_stats['moderate_percentage']:.1f}%", 
                        "Severe": f"{burn_stats['severe_percentage']:.1f}%"
                    },
                    "Fluid Requirements (24h)": {
                        "Parkland Formula": f"{fluid_calc['parkland_total_24h']:.0f} ml",
                        "First 8 hours": f"{fluid_calc['parkland_first_8h']:.0f} ml ({fluid_calc['hourly_rate_first_8h']:.0f} ml/h)",
                        "Next 16 hours": f"{fluid_calc['parkland_next_16h']:.0f} ml ({fluid_calc['hourly_rate_next_16h']:.0f} ml/h)",
                        "Maintenance": f"{fluid_calc['maintenance_24h']:.0f} ml",
                        "Total": f"{fluid_calc['total_24h']:.0f} ml"
                    },
                    "Clinical Recommendations": [
                        "Monitor urine output (0.5-1.0 ml/kg/h adults, 1.0 ml/kg/h children)",
                        "Assess for inhalation injury",
                        "Pain management protocol",
                        "Tetanus prophylaxis if indicated",
                        "Nutritional support planning",
                        "Wound care and dressing protocol"
                    ]
                }
                
                # Display formatted report
                st.json(report_data)
                
                # Download report button
                report_json = str(report_data)
                st.download_button(
                    label="📥 Download Clinical Report",
                    data=report_json,
                    file_name=f"burn_analysis_report_{uploaded_file.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            st.markdown("---")
    
    else:
        # Demo section with example results
        st.header("🎯 Demo: Upload Your Medical Images")
        st.info("👆 Please upload medical images to begin burn analysis")
        
        # Feature showcase
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🔍 **Advanced Segmentation**
            - AI-powered background removal
            - Color-based burn detection  
            - Multi-level severity classification
            - Morphological mask cleaning
            """)
        
        with col2:
            st.markdown("""
            ### 📊 **Clinical Calculations**
            - TBSA percentage (Rule of Nines)
            - Age-adjusted calculations
            - Lund-Browder chart approximation
            - Statistical burn analysis
            """)
        
        with col3:
            st.markdown("""
            ### 💧 **Fluid Management**
            - Parkland Formula calculations
            - Brook Formula comparison
            - Maintenance fluid requirements
            - 24-hour resuscitation timeline
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
        <p>🏥 Medical Burn Analysis System | For Educational and Clinical Reference Use</p>
        <p>⚠️ This tool provides estimates for clinical guidance. Always consult with medical professionals for patient care decisions.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

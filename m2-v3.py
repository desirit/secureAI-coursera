"""
Module 2 Video 3: Adversarial Testing and Robustness Evaluation
Demonstrates adversarial example generation and model robustness testing
Uses Adversarial Robustness Toolbox (ART)
"""

import streamlit as st
import numpy as np
from PIL import Image
import io
import base64

# Page config
st.set_page_config(
    page_title="Adversarial Testing Demo",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Adversarial Testing & Robustness Evaluation")
st.caption("Module 2 - Video 3: Testing AI Model Resilience Against Adversarial Attacks")

# Sidebar
with st.sidebar:
    st.header("⚙️ Attack Configuration")
    
    attack_method = st.selectbox(
        "Attack Method",
        ["Fast Gradient Sign Method (FGSM)",
         "Projected Gradient Descent (PGD)",
         "Carlini & Wagner (C&W)",
         "DeepFool"]
    )
    
    epsilon = st.slider(
        "Perturbation Strength (ε)",
        min_value=0.01,
        max_value=0.5,
        value=0.1,
        step=0.01,
        help="Higher values = more visible perturbations"
    )
    
    st.divider()
    
    st.header("🎯 Test Settings")
    show_perturbation = st.checkbox("Show Perturbation", value=True)
    show_confidence = st.checkbox("Show Confidence Scores", value=True)
    run_robustness_test = st.checkbox("Run Robustness Suite", value=False)

    st.divider()

    st.header("📊 Acceptance Criteria")
    robustness_threshold = st.slider(
        "Minimum Robust Accuracy (%)",
        min_value=50,
        max_value=95,
        value=70,
        step=5,
        help="Minimum accuracy required under adversarial attack to pass deployment"
    )
    st.caption(f"Model must maintain ≥{robustness_threshold}% accuracy under attack")

# Simulated model for demonstration
class DemoImageClassifier:
    """Simulated image classifier for demonstration"""
    
    def __init__(self):
        self.classes = ["Cat", "Dog", "Bird", "Car", "Airplane"]
        self.vulnerability_level = "high"  # high, medium, low
    
    def predict(self, image_array):
        """Simulate model prediction"""
        # For demo: Create deterministic but realistic predictions
        # based on image properties

        if image_array is None:
            return None

        # Check if this is the generated cat image (by checking for tan/brown pixels)
        has_cat_colors = np.any((image_array[:,:,0] == 180) &
                                 (image_array[:,:,1] == 140) &
                                 (image_array[:,:,2] == 100))

        if has_cat_colors:
            # This is our cat image - predict Cat with high confidence
            confidences = np.array([0.94, 0.03, 0.01, 0.01, 0.01])  # Cat, Dog, Bird, Car, Airplane
            predicted_class = "Cat"
        else:
            # For other images, use original logic
            img_sum = np.sum(image_array)
            img_mean = np.mean(image_array)

            # Generate "confidence scores"
            base_confidences = np.random.RandomState(int(img_sum) % 1000).random(5)
            confidences = base_confidences / base_confidences.sum()

            # Make one class dominant
            max_idx = int(img_mean * 5) % 5
            confidences[max_idx] = 0.7 + (confidences[max_idx] * 0.25)
            confidences = confidences / confidences.sum()

            predicted_class = self.classes[np.argmax(confidences)]

        return {
            "class": predicted_class,
            "confidence": float(np.max(confidences)),
            "all_confidences": {self.classes[i]: float(confidences[i]) for i in range(5)}
        }
    
    def predict_adversarial(self, image_array, epsilon):
        """Simulate prediction on adversarial example"""
        if image_array is None:
            return None

        # Check if this was originally the cat image
        # (Note: adversarial image will have slightly different colors due to perturbation)
        # So we check for colors in the range
        has_cat_colors = np.any((image_array[:,:,0] >= 170) & (image_array[:,:,0] <= 190) &
                                 (image_array[:,:,1] >= 130) & (image_array[:,:,1] <= 150) &
                                 (image_array[:,:,2] >= 90) & (image_array[:,:,2] <= 110))

        if has_cat_colors:
            # This is our adversarial cat image
            if epsilon >= 0.1:  # Epsilon 0.1 or higher
                # Attack succeeds - model predicts Dog instead of Cat with 97% confidence
                confidences = np.array([0.01, 0.97, 0.01, 0.005, 0.005])  # Cat->Dog with 97% confidence
                predicted_class = "Dog"
            elif epsilon >= 0.05:
                # Moderate perturbation - might switch to Dog with lower confidence
                confidences = np.array([0.30, 0.55, 0.08, 0.04, 0.03])
                predicted_class = "Dog"
            else:
                # Small perturbation - still Cat but lower confidence
                confidences = np.array([0.75, 0.15, 0.05, 0.03, 0.02])
                predicted_class = "Cat"
        else:
            # For other images, use original logic
            img_sum = np.sum(image_array)
            img_mean = np.mean(image_array)

            # Generate different confidences due to perturbation
            perturbed_seed = int((img_sum + epsilon * 1000)) % 1000
            base_confidences = np.random.RandomState(perturbed_seed).random(5)
            confidences = base_confidences / base_confidences.sum()

            # Adversarial examples often fool the model
            if epsilon > 0.05:  # Significant perturbation
                # Different class becomes dominant (attack succeeds)
                original_max = int(img_mean * 5) % 5
                new_max = (original_max + 1) % 5  # Different class
                confidences[new_max] = 0.6
            else:
                # Small perturbation - might still get right class but lower confidence
                max_idx = int(img_mean * 5) % 5
                confidences[max_idx] = 0.5

            confidences = confidences / confidences.sum()
            predicted_class = self.classes[np.argmax(confidences)]

        return {
            "class": predicted_class,
            "confidence": float(np.max(confidences)),
            "all_confidences": {self.classes[i]: float(confidences[i]) for i in range(5)}
        }

def generate_sample_image():
    """Generate a sample cat-like image for testing"""
    # Create a simple image that resembles a cat silhouette
    img_array = np.ones((224, 224, 3), dtype=np.uint8) * 240  # Light background

    # Draw a simple cat-like shape (circle for head, triangles for ears, oval for body)
    center_y, center_x = 112, 112

    # Body (lower oval)
    for i in range(224):
        for j in range(224):
            # Body
            if ((i - 140) ** 2) / 50**2 + ((j - 112) ** 2) / 40**2 < 1:
                img_array[i, j] = [180, 140, 100]  # Brown/tan color

            # Head (upper circle)
            if (i - 80) ** 2 + (j - 112) ** 2 < 30**2:
                img_array[i, j] = [180, 140, 100]

            # Left ear (triangle)
            if i > 45 and i < 75 and abs(j - 85) < (i - 45) * 0.6:
                img_array[i, j] = [180, 140, 100]

            # Right ear (triangle)
            if i > 45 and i < 75 and abs(j - 139) < (i - 45) * 0.6:
                img_array[i, j] = [180, 140, 100]

            # Eyes
            if (i - 75) ** 2 + (j - 100) ** 2 < 4**2:
                img_array[i, j] = [0, 0, 0]  # Black
            if (i - 75) ** 2 + (j - 124) ** 2 < 4**2:
                img_array[i, j] = [0, 0, 0]

            # Nose (small triangle)
            if i > 85 and i < 92 and abs(j - 112) < (92 - i) * 0.5:
                img_array[i, j] = [255, 150, 150]  # Pink

    return img_array

def add_adversarial_perturbation(image_array, epsilon, method="FGSM"):
    """Simulate adding adversarial perturbation"""
    if image_array is None:
        return None
    
    # Create perturbation based on method
    np.random.seed(42)  # For consistency in demo
    
    if method == "Fast Gradient Sign Method (FGSM)":
        # FGSM adds sign of gradients
        perturbation = np.random.randn(*image_array.shape) * 50
        perturbation = np.sign(perturbation) * epsilon * 255
    
    elif method == "Projected Gradient Descent (PGD)":
        # PGD is iterative FGSM
        perturbation = np.random.randn(*image_array.shape) * 30
        perturbation = np.clip(perturbation, -epsilon * 255, epsilon * 255)
    
    elif method == "Carlini & Wagner (C&W)":
        # C&W uses more sophisticated optimization
        perturbation = np.random.randn(*image_array.shape) * 20
        perturbation = np.tanh(perturbation) * epsilon * 255
    
    else:  # DeepFool
        # DeepFool finds minimal perturbation
        perturbation = np.random.randn(*image_array.shape) * 15
        perturbation = perturbation * epsilon * 255
    
    # Add perturbation
    adversarial = image_array.astype(np.float32) + perturbation
    adversarial = np.clip(adversarial, 0, 255).astype(np.uint8)
    
    return adversarial, perturbation

# Initialize model
if 'model' not in st.session_state:
    st.session_state.model = DemoImageClassifier()

model = st.session_state.model

# Main content
tab1, tab2, tab3, tab4 = st.tabs([
    "🎨 Single Attack Demo",
    "📊 Robustness Testing",
    "📈 Attack Success Rate",
    "📚 Learn More"
])

with tab1:
    st.header("Adversarial Example Generation")
    
    col1, col2, col3 = st.columns(3)
    
    # Generate or upload image
    if 'demo_image' not in st.session_state:
        st.session_state.demo_image = generate_sample_image()
    
    demo_image = st.session_state.demo_image
    
    with col1:
        st.subheader("🖼️ Original Image")
        
        # Display original
        img_pil = Image.fromarray(demo_image)
        st.image(img_pil, use_container_width=True)
        
        # Predict on original
        original_pred = model.predict(demo_image)
        
        if original_pred and show_confidence:
            st.success(f"**Prediction:** {original_pred['class']}")
            st.metric("Confidence", f"{original_pred['confidence']:.1%}")
            
            with st.expander("All Class Scores"):
                for cls, conf in original_pred['all_confidences'].items():
                    st.write(f"{cls}: {conf:.3f}")
        
        if st.button("🔄 Generate New Image"):
            st.session_state.demo_image = generate_sample_image()
            st.rerun()
    
    with col2:
        st.subheader("⚡ Adversarial Perturbation")
        
        if show_perturbation:
            # Generate adversarial example
            adversarial_img, perturbation = add_adversarial_perturbation(
                demo_image, epsilon, attack_method
            )
            
            # Visualize perturbation
            perturbation_vis = np.abs(perturbation).astype(np.uint8)
            pert_pil = Image.fromarray(perturbation_vis)
            st.image(pert_pil, use_container_width=True, caption="Perturbation (amplified for visibility)")
            
            st.info(f"**Method:** {attack_method}")
            st.metric("Epsilon (ε)", f"{epsilon:.2f}")
            st.caption("Higher ε = stronger attack, more visible changes")
        else:
            st.info("Enable 'Show Perturbation' in sidebar")
    
    with col3:
        st.subheader("🎭 Adversarial Image")
        
        if 'adversarial_img' in locals():
            # Display adversarial
            adv_pil = Image.fromarray(adversarial_img)
            st.image(adv_pil, use_container_width=True)
            
            # Predict on adversarial
            adv_pred = model.predict_adversarial(adversarial_img, epsilon)
            
            if adv_pred and show_confidence:
                if adv_pred['class'] != original_pred['class']:
                    st.error(f"**Prediction:** {adv_pred['class']} ❌")
                    st.caption("Attack succeeded! Wrong class predicted.")
                else:
                    st.warning(f"**Prediction:** {adv_pred['class']}")
                    st.caption("Same class, but lower confidence")
                
                st.metric("Confidence", f"{adv_pred['confidence']:.1%}")
                
                with st.expander("All Class Scores"):
                    for cls, conf in adv_pred['all_confidences'].items():
                        st.write(f"{cls}: {conf:.3f}")
    
    # Analysis
    st.divider()
    st.subheader("🔍 Attack Analysis")
    
    if 'original_pred' in locals() and 'adv_pred' in locals():
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            attack_success = original_pred['class'] != adv_pred['class']
            st.metric("Attack Success", "Yes ✓" if attack_success else "No ✗")
        
        with col_b:
            conf_drop = original_pred['confidence'] - adv_pred['confidence']
            st.metric("Confidence Drop", f"{conf_drop:.1%}")
        
        with col_c:
            # Calculate L2 norm of perturbation
            l2_norm = np.linalg.norm(perturbation) / np.linalg.norm(demo_image)
            st.metric("Perturbation Size", f"{l2_norm:.4f}")
        
        if attack_success:
            st.error(f"""
            **🚨 Adversarial Attack Successful!**
            
            - Original prediction: **{original_pred['class']}** ({original_pred['confidence']:.1%})
            - Adversarial prediction: **{adv_pred['class']}** ({adv_pred['confidence']:.1%})
            - The model was fooled by a perturbation of ε={epsilon}
            - To humans, the images look nearly identical
            """)
        else:
            st.success(f"""
            **✅ Model Resisted Attack**
            
            - Both images classified as: **{original_pred['class']}**
            - Confidence dropped from {original_pred['confidence']:.1%} to {adv_pred['confidence']:.1%}
            - Model maintained correct classification despite perturbation
            """)

with tab2:
    st.header("Robustness Test Suite")
    
    st.write("""
    This tests the model's robustness across multiple epsilon values,
    measuring how performance degrades as perturbations increase.
    """)
    
    if st.button("▶️ Run Robustness Test Suite") or run_robustness_test:
        st.subheader("Testing Model Robustness...")
        
        # Test across different epsilon values
        epsilon_values = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3]
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, eps in enumerate(epsilon_values):
            status_text.text(f"Testing ε = {eps:.2f}...")

            # Simulate testing on multiple images with realistic robustness degradation
            # Starting from ~94% clean accuracy, degrading to ~23% at epsilon=0.3
            if eps <= 0.01:
                # Very small perturbation - model mostly robust
                accuracy = 0.94  # Clean accuracy
            elif eps <= 0.03:
                # Small perturbation - slight degradation
                accuracy = 0.89
            elif eps <= 0.05:
                # Moderate perturbation - noticeable degradation
                accuracy = 0.76
            elif eps <= 0.1:
                # Significant perturbation - major degradation
                accuracy = 0.58
            elif eps <= 0.15:
                # Strong perturbation - severe degradation
                accuracy = 0.43
            elif eps <= 0.2:
                # Very strong perturbation - critical degradation
                accuracy = 0.31
            else:  # eps >= 0.3
                # Maximum perturbation - near complete failure
                accuracy = 0.23

            results.append({
                "Epsilon": eps,
                "Accuracy": accuracy,
                "Attack Success Rate": 1 - accuracy
            })

            progress_bar.progress((i + 1) / len(epsilon_values))
        
        status_text.text("Test complete!")
        
        # Display results
        st.divider()
        st.subheader("📊 Test Results")
        
        import pandas as pd
        df = pd.DataFrame(results)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            clean_acc = results[0]['Accuracy']
            st.metric("Clean Accuracy", f"{clean_acc:.1%}")
        
        with col2:
            robust_acc = results[-1]['Accuracy']
            st.metric(f"Robust Accuracy (ε={epsilon_values[-1]})", f"{robust_acc:.1%}")
        
        with col3:
            degradation = clean_acc - robust_acc
            st.metric("Performance Degradation", f"{degradation:.1%}")
        
        # Table
        st.dataframe(df.style.format({
            "Epsilon": "{:.2f}",
            "Accuracy": "{:.1%}",
            "Attack Success Rate": "{:.1%}"
        }), use_container_width=True)
        
        # Verdict
        st.divider()
        st.subheader("🎯 Robustness Assessment")

        threshold_decimal = robustness_threshold / 100
        meets_threshold = robust_acc >= threshold_decimal

        if meets_threshold:
            st.success(f"""
            ✅ **Model PASSES robustness requirements**

            - Clean accuracy: {clean_acc:.1%}
            - Robust accuracy: {robust_acc:.1%} under attack (ε={epsilon_values[-1]})
            - Performance degradation: {degradation:.1%}
            - **Verdict:** Meets acceptance threshold of {robustness_threshold}%
            - Model is approved for deployment
            """)
        else:
            st.error(f"""
            🚨 **Model FAILS robustness requirements**

            - Clean accuracy: {clean_acc:.1%}
            - Robust accuracy: {robust_acc:.1%} under attack (ε={epsilon_values[-1]})
            - Performance degradation: {degradation:.1%}
            - **Required threshold:** {robustness_threshold}%
            - **Gap:** {(threshold_decimal - robust_acc):.1%} below requirement
            - **Verdict:** BLOCKED - Do not deploy
            - **Action Required:** Implement adversarial training before deployment
            """)

with tab3:
    st.header("Attack Success Rates by Method")
    
    st.write("""
    Compare effectiveness of different adversarial attack methods.
    """)
    
    methods = [
        "Fast Gradient Sign Method (FGSM)",
        "Projected Gradient Descent (PGD)",
        "Carlini & Wagner (C&W)",
        "DeepFool"
    ]
    
    # Simulated success rates
    success_rates = {
        "FGSM": 0.65,
        "PGD": 0.82,
        "C&W": 0.91,
        "DeepFool": 0.73
    }
    
    st.subheader("📊 Attack Effectiveness Comparison")
    
    col1, col2, col3, col4 = st.columns(4)
    
    cols = [col1, col2, col3, col4]
    for i, method in enumerate(methods):
        short_name = method.split("(")[0].strip()
        with cols[i]:
            st.metric(
                short_name,
                f"{success_rates[list(success_rates.keys())[i]]:.0%}",
                help=f"Success rate for {method}"
            )
    
    st.divider()
    
    st.subheader("💡 Method Characteristics")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.write("**FGSM (Fast Gradient Sign Method)**")
        st.write("- ✅ Fastest attack")
        st.write("- ⚡ Single-step method")
        st.write("- 📊 ~65% success rate")
        st.write("- 💰 Good for quick testing")
        
        st.write("\n**PGD (Projected Gradient Descent)**")
        st.write("- 🔄 Iterative FGSM")
        st.write("- 💪 More powerful")
        st.write("- 📊 ~82% success rate")
        st.write("- ⏱️ Slower but more effective")
    
    with col_b:
        st.write("**C&W (Carlini & Wagner)**")
        st.write("- 🎯 Most sophisticated")
        st.write("- 🔬 Optimization-based")
        st.write("- 📊 ~91% success rate")
        st.write("- ⚠️ Computationally expensive")
        
        st.write("\n**DeepFool**")
        st.write("- 📏 Finds minimal perturbation")
        st.write("- 🎨 Smaller, less visible changes")
        st.write("- 📊 ~73% success rate")
        st.write("- 🔍 Good for understanding vulnerabilities")

with tab4:
    st.header("Understanding Adversarial Testing")
    
    with st.expander("What are Adversarial Examples?"):
        st.write("""
        Adversarial examples are inputs specifically designed to fool machine learning models.
        They look normal to humans but cause models to make incorrect predictions.
        
        **Key Characteristics:**
        - Small, often imperceptible perturbations
        - Cause confident but incorrect predictions
        - Transferable across models
        - Work in physical world (not just digital)
        
        **Real-World Example:**
        Researchers added small stickers to a stop sign. To humans, it still looked like a stop sign.
        To Tesla's Autopilot AI, it looked like a speed limit sign. That's an adversarial example.
        """)
    
    with st.expander("Why Traditional Testing Fails"):
        st.write("""
        Traditional software testing checks:
        - Does the code work correctly?
        - Does it handle edge cases?
        - Does it perform well?
        
        But for AI systems, we also need to ask:
        - **Can someone craft inputs to fool it?**
        - **How much can they degrade performance?**
        - **What's the minimum perturbation needed to break it?**
        
        A model with 99% accuracy on clean data might have only 20% accuracy on adversarial examples.
        Traditional testing would miss this vulnerability entirely.
        """)
    
    with st.expander("Attack Methods Explained"):
        st.write("""
        **FGSM (Fast Gradient Sign Method):**
        - Takes one step in direction that maximizes loss
        - Fast but less powerful
        - Good for basic robustness testing
        
        **PGD (Projected Gradient Descent):**
        - Iterative version of FGSM
        - Multiple small steps toward fooling the model
        - Stronger attacks, better for thorough testing
        
        **C&W (Carlini & Wagner):**
        - Sophisticated optimization-based attack
        - Finds minimal perturbation to fool model
        - Most powerful but computationally expensive
        
        **DeepFool:**
        - Finds smallest perturbation to cross decision boundary
        - Useful for understanding model vulnerabilities
        - Less about maximum damage, more about minimal attack
        """)
    
    with st.expander("Defense Strategies"):
        st.write("""
        **1. Adversarial Training:**
        - Train model on adversarial examples
        - Most effective defense
        - Increases robustness significantly
        
        **2. Input Preprocessing:**
        - Remove high-frequency noise
        - JPEG compression
        - Random transformations
        
        **3. Defensive Distillation:**
        - Train model to output probabilities, not hard predictions
        - Makes gradients less useful for attacks
        
        **4. Ensemble Methods:**
        - Use multiple models with different architectures
        - Harder to fool all models simultaneously
        
        **5. Detection:**
        - Train separate classifier to detect adversarial examples
        - Reject suspicious inputs before processing
        
        **6. Certified Defenses:**
        - Mathematical guarantees of robustness
        - Randomized smoothing
        - Provable bounds on perturbations
        """)
    
    with st.expander("Testing Best Practices"):
        st.write("""
        **Comprehensive Robustness Testing Should Include:**
        
        1. **Multiple Attack Methods:**
           - Test with FGSM, PGD, C&W, DeepFool
           - Each reveals different vulnerabilities
        
        2. **Range of Perturbation Strengths:**
           - Test ε from 0.01 to 0.5
           - Understand degradation curve
        
        3. **Different Norms:**
           - L∞, L2, L1 perturbations
           - Different threat models
        
        4. **Physical-World Attacks:**
           - Printed images
           - Environmental variations
           - 3D objects
        
        5. **Targeted vs Untargeted:**
           - Untargeted: Any wrong class
           - Targeted: Specific wrong class
        
        6. **Set Acceptance Criteria:**
           - E.g., "Must maintain 70% accuracy under ε=0.1 PGD attack"
           - Document in security requirements
        
        7. **Regular Re-testing:**
           - New attack methods emerge
           - Model updates need re-validation
           - Part of CI/CD pipeline
        """)

# Footer
st.divider()
st.caption("🎯 **Adversarial Robustness Testing** - Essential for AI security validation")
st.caption("⚠️ This demo simulates adversarial attacks for educational purposes")

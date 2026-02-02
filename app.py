import streamlit as st
import pandas as pd
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# --- Page Config ---
st.set_page_config(page_title="Shopper Spectrum", layout="wide")

# --- Title ---
st.title("🛒 Shopper Spectrum: Retail Analytics")
st.markdown("### Customer Segmentation & Product Recommendation System")

# --- Sidebar ---
st.sidebar.header("Navigation")
options = st.sidebar.radio("Go to:", ["Project Overview", "Customer Segmentation", "Product Recommender"])

# --- Load Files ---
@st.cache_data
def load_data():
    # Files load karna
    rfm = pd.read_csv('rfm_analysis.csv')
    product_names = pd.read_csv('product_list.csv')
    return rfm, product_names

@st.cache_resource
def load_models():
    # Models load karna
    with open('kmeans_model.pkl', 'rb') as f:
        kmeans = pickle.load(f)
    with open('scaler_model.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('item_similarity.pkl', 'rb') as f:
        similarity = pickle.load(f)
    return kmeans, scaler, similarity

try:
    rfm_df, products_df = load_data()
    kmeans, scaler, similarity = load_models()
    st.sidebar.success("System Ready: Data Loaded Successfully")
except FileNotFoundError:
    st.error("Error: Files nahi mili! Please check ki saari CSV aur PKL files same folder mein hain.")
    st.stop()

# --- Page 1: Overview ---
if options == "Project Overview":
    st.header("📊 Data Analysis Dashboard")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", rfm_df['CustomerID'].nunique())
    col2.metric("Average Spending", f"£{rfm_df['Monetary'].mean():.2f}")
    col3.metric("Average Frequency", f"{rfm_df['Frequency'].mean():.0f} Orders")

    st.subheader("Customer Segments (Clusters)")
    # Cluster Visualization
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=rfm_df, x='Recency', y='Monetary', hue='Cluster', palette='viridis', ax=ax)
    plt.title("Customer Segments: Recency vs Monetary")
    st.pyplot(fig)
    
    st.write("**Interpretation:**")
    st.write("- **Cluster 0, 1, 2** alag-alag tarah ke customers ko darshate hain (e.g., Loyal, At-Risk, Big Spenders).")

# --- Page 2: Segmentation Prediction ---
elif options == "Customer Segmentation":
    st.header("🔍 Predict Customer Category")
    st.write("Naye customer ka R, F, M daalo aur check karo wo kis group mein aata hai.")

    c1, c2, c3 = st.columns(3)
    recency = c1.number_input("Recency (Days since last visit)", min_value=0, value=10)
    frequency = c2.number_input("Frequency (Number of orders)", min_value=1, value=5)
    monetary = c3.number_input("Monetary (Total Spent £)", min_value=0.0, value=500.0)

    if st.button("Predict Segment"):
        # Log transform (jo humne training mein kiya tha)
        # +1 isliye taaki log(0) error na aaye
        r_log = np.log(recency + 1)
        f_log = np.log(frequency + 1)
        m_log = np.log(monetary + 1)
        
        # Scale input
        input_data = np.array([[r_log, f_log, m_log]])
        input_scaled = scaler.transform(input_data)
        
        # Predict
        cluster = kmeans.predict(input_scaled)[0]
        
        st.success(f"Yeh customer **Cluster {cluster}** mein belong karta hai!")
        
        if cluster == 0: 
            st.info("Tip: Yeh group shayad 'Budget' ya 'Risk' wala hai (Apne graph se verify karein).")
        elif cluster == 1:
            st.info("Tip: Yeh 'Regular' customers ho sakte hain.")
        else:
            st.info("Tip: Yeh 'High Value' customers ho sakte hain.")

# --- Page 3: Recommendation ---
elif options == "Product Recommender":
    st.header("🛍️ Product Recommendation Engine")
    st.write("Ek product select karo, hum usse milte-julte products batayenge.")

    # Dropdown menu
    # Hum sirf pehle 200 products dikha rahe hain taaki list hang na ho
    product_list = products_df['Description'].unique()[:200]
    selected_product = st.selectbox("Select a Product:", product_list)

    if st.button("Show Recommendations"):
        # Selected product ka ID (StockCode) dhundo
        product_id = products_df[products_df['Description'] == selected_product]['StockCode'].values[0]
        
        # Check karo agar ID similarity matrix mein hai
        if product_id in similarity.index:
            # Similarity scores nikalo
            distances = similarity.loc[product_id]
            # Sort karke top 5 nikalo (apne aap ko chhod kar)
            similar_products = distances.sort_values(ascending=False)[1:6]
            
            st.write(f"**Products similar to '{selected_product}':**")
            
            for stock_code, score in similar_products.items():
                # StockCode se wapis naam dhundo
                prod_name = products_df[products_df['StockCode'] == stock_code]['Description'].values
                if len(prod_name) > 0:
                    st.success(f"🛒 {prod_name[0]} (Similarity: {score:.2f})")
        else:
            st.warning("Sorry, is product ka data hamare recommendation engine mein nahi hai.")
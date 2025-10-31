# 🛒 SmartCart: Product Recommendations from Retail Data
# -------------------------------------------------------
# Streamlit Dashboard integrating Apriori Algorithm, Clustering, Outlier Detection, and Visualization

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import ast
import traceback

# --------------------------------------
# Streamlit Page Setup
# --------------------------------------
st.set_page_config(page_title="🛒 SmartCart: Retail Product Insights", layout="wide")
st.title("🛒 SmartCart: Product Recommendations from Retail Data")

# --------------------------------------
# File Upload Section
# --------------------------------------
uploaded_file = st.file_uploader("📂 Upload your Retail Transactions CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    # Dataset overview
    st.subheader("📊 Dataset Overview")
    st.write(data.head())
    st.write("**Shape:**", data.shape)
    st.write("**Columns:**", list(data.columns))
    st.write("**Missing Values:**", data.isnull().sum().sum())

    # -------------------- OUTLIER DETECTION --------------------
    st.subheader("🚨 Outlier Detection using Local Outlier Factor")
    numeric_data = data.select_dtypes(include=['number'])

    if not numeric_data.empty:
        lof = LocalOutlierFactor()
        outlier_labels = lof.fit_predict(numeric_data)
        data['Outlier'] = outlier_labels
        st.write("Outlier Label Counts:")
        st.bar_chart(pd.Series(outlier_labels).value_counts())
    else:
        st.warning("No numeric columns found for outlier detection.")

    # -------------------- K-MEANS CLUSTERING --------------------
    st.subheader("🎯 K-Means Clustering on Numeric Features")
    if not numeric_data.empty:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)

        k = st.slider("Select number of clusters (K)", 2, 10, 3)
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(scaled_data)
        data['Cluster'] = labels
        st.write("Clustered Data Sample:")
        st.dataframe(data.head())

        # Cluster Visualization
        fig, ax = plt.subplots()
        ax.scatter(scaled_data[:, 0], scaled_data[:, 1], c=labels, cmap='viridis')
        plt.title("K-Means Cluster Visualization")
        st.pyplot(fig)
    else:
        st.warning("No numeric features available for clustering.")

    # -------------------- MARKET BASKET ANALYSIS --------------------
    st.subheader("🧺 Market Basket Analysis (Apriori Algorithm)")
    st.info("Each row represents a transaction; products are grouped by Transaction_ID.")

    try:
        if 'Transaction_ID' in data.columns and 'Product' in data.columns:

            # Parse and clean product column
            def safe_parse_products(x):
                try:
                    if isinstance(x, list):
                        items = x
                    elif isinstance(x, str):
                        items = ast.literal_eval(x) if x.strip().startswith('[') else [x]
                    else:
                        items = [str(x)]
                    items = [str(i).lower().strip() for i in items if str(i).strip() != ""]
                    return items
                except Exception:
                    try:
                        parts = str(x).split(',')
                        return [p.lower().strip() for p in parts if p.strip() != ""]
                    except:
                        return []

            data['Product_parsed'] = data['Product'].apply(safe_parse_products)
            st.write("✅ Sample parsed transactions (first 5):")
            st.write(data[['Transaction_ID', 'Product_parsed']].head())

            # Group by transaction
            transactions = data.groupby('Transaction_ID')['Product_parsed'].apply(list).apply(lambda x: sum(x, []))

            # Encode transactions
            te = TransactionEncoder()
            te_ary = te.fit(transactions).transform(transactions)
            basket_df = pd.DataFrame(te_ary, columns=te.columns_, index=transactions.index).astype(int)

            st.write("🧾 Basket matrix created successfully!")
            st.write("Shape:", basket_df.shape)
            st.write("Top products:", list(basket_df.columns[:10]))

            # Apply Apriori
            support_threshold = 0.01
            frequent_items = apriori(basket_df, min_support=support_threshold, use_colnames=True)

            while frequent_items.empty and support_threshold > 0.0005:
                support_threshold /= 2
                frequent_items = apriori(basket_df, min_support=support_threshold, use_colnames=True)

            if frequent_items.empty:
                st.error("⚠️ No frequent itemsets found. Try checking your dataset or lower support threshold.")
            else:
                # Generate rules
                rules = association_rules(frequent_items, metric="lift", min_threshold=1.0)
                rules = rules[(rules['confidence'] >= 0.3) & (rules['lift'] > 1.0)].copy()

                # Convert frozenset → list (for comparisons)
                rules['antecedents'] = rules['antecedents'].apply(lambda x: [str(i).lower() for i in list(x)])
                rules['consequents'] = rules['consequents'].apply(lambda x: [str(i).lower() for i in list(x)])

                st.write(f"📈 Generated {len(rules)} Association Rules (min_support ≈ {support_threshold:.4f})")
                st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(15))

                # Product Recommendation System
                st.subheader("💡 Product Recommendation System")
                product = st.text_input("🔍 Enter a product name (e.g., bread, milk, tea):").strip().lower()
                reverse_lookup = st.checkbox("Also consider rules where the product appears as a consequent (reverse lookup)", value=False)

                if product:
                    if product not in basket_df.columns:
                        st.warning(f"'{product}' not found in products list. Try other variations.")

                    recs_from_ante = []
                    for _, row in rules.iterrows():
                        if product in row['antecedents']:
                            recs_from_ante.extend(row['consequents'])
                    recs_from_ante = list(dict.fromkeys(recs_from_ante))

                    recs_from_cons = []
                    if reverse_lookup:
                        for _, row in rules.iterrows():
                            if product in row['consequents']:
                                recs_from_cons.extend(row['antecedents'])
                        recs_from_cons = list(dict.fromkeys(recs_from_cons))

                    combined = recs_from_ante + [r for r in recs_from_cons if r not in recs_from_ante]

                    if combined:
                        st.success(f"Products frequently bought with **'{product.title()}'**:")
                        for rec in combined:
                            st.markdown(f"- 🛒 **{rec.title()}**")
                    else:
                        st.info(f"No strong associations found for '{product}'. Try another item or lower thresholds.")

                # Top 10 Products Visualization
                st.subheader("🏆 Top 10 Most Frequent Products")
                product_freq = basket_df.sum().sort_values(ascending=False).head(10)
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(x=product_freq.values, y=product_freq.index, ax=ax)
                ax.set_title("Top 10 Most Frequent Products")
                ax.set_xlabel("Frequency")
                st.pyplot(fig)

        else:
            st.warning("Please ensure your dataset has 'Transaction_ID' and 'Product' columns for Apriori.")

    except Exception as e:
        st.error(f"Error in Apriori section: {e}")
        st.text(traceback.format_exc())

    # -------------------- CUSTOM VISUALIZATION SECTION --------------------
    st.subheader("📊 Custom Visualization Tools")

    plot_type = st.selectbox("Choose Plot Type", ["Histogram", "Countplot", "Barplot"])
    column = st.selectbox("Select Column", data.columns)

    if plot_type == "Histogram":
        fig, ax = plt.subplots()
        sns.histplot(data[column], kde=True, ax=ax)
        ax.set_title(f"Histogram of {column}")
        st.pyplot(fig)

    elif plot_type == "Countplot":
        fig, ax = plt.subplots()
        sns.countplot(x=data[column], ax=ax)
        ax.set_title(f"Countplot of {column}")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    elif plot_type == "Barplot":
        x_col = st.selectbox("Select X-axis column", data.columns)
        y_col = st.selectbox("Select Y-axis column (numeric)", numeric_data.columns if not numeric_data.empty else [])
        if y_col:
            fig, ax = plt.subplots()
            sns.barplot(x=data[x_col], y=data[y_col], ax=ax)
            ax.set_title(f"Barplot of {y_col} vs {x_col}")
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.warning("Select a numeric column for the Y-axis.")

    # -------------------- FOOTER --------------------
    st.markdown("---")
    st.caption("© 2025 SmartCart | Shravani Ranjeet Kalapure")

else:
    st.info("👆 Please upload a CSV file to get started.")

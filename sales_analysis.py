import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load raw dataset
df = pd.read_csv("raw_sales_data.csv")

# -----------------------------
# Data Cleaning
# -----------------------------
df["Order_Date"] = pd.to_datetime(df["Order_Date"])
df["Region"] = df["Region"].astype("string").str.strip().str.title()
df["City"] = df["City"].astype("string").str.strip()
df["Product"] = df["Product"].astype("string").str.strip().str.title()
df["Quantity"] = df["Quantity"].fillna(1)
df["Discount"] = df["Discount"].fillna(0)

# Calculated columns
df["Gross_Sales"] = df["Quantity"] * df["Unit_Price"]
df["Discount_Amount"] = df["Gross_Sales"] * df["Discount"]
df["Net_Sales"] = df["Gross_Sales"] - df["Discount_Amount"]

# -----------------------------
# KPI Analysis
# -----------------------------
total_revenue = df["Net_Sales"].sum()
total_orders = df["Order_ID"].nunique()
unique_customers = df["Customer_ID"].nunique()
units_sold = df["Quantity"].sum()
average_order_value = total_revenue / total_orders

print("=== SALES KPI REPORT ===")
print(f"Total Revenue: ${total_revenue:,.2f}")
print(f"Total Orders: {total_orders:,}")
print(f"Unique Customers: {unique_customers:,}")
print(f"Units Sold: {units_sold:,.0f}")
print(f"Average Order Value: ${average_order_value:,.2f}")

# -----------------------------
# EDA
# -----------------------------
region_report = (
    df.groupby("Region")
      .agg(Revenue=("Net_Sales", "sum"),
           Orders=("Order_ID", "nunique"),
           Units_Sold=("Quantity", "sum"))
      .sort_values("Revenue", ascending=False)
)

product_report = (
    df.groupby(["Product", "Category"])
      .agg(Revenue=("Net_Sales", "sum"),
           Units_Sold=("Quantity", "sum"))
      .sort_values("Revenue", ascending=False)
)

df["Month"] = df["Order_Date"].dt.strftime("%b")
df["Month_Number"] = df["Order_Date"].dt.month

monthly_report = (
    df.groupby(["Month_Number", "Month"])
      .agg(Revenue=("Net_Sales", "sum"))
      .sort_values("Month_Number")
)

print("\n=== REVENUE BY REGION ===")
print(region_report)

print("\n=== TOP PRODUCTS ===")
print(product_report.head())

print("\n=== MONTHLY REVENUE ===")
print(monthly_report)

# -----------------------------
# Visualizations
# -----------------------------
plt.figure(figsize=(9, 5))
plt.bar(region_report.index, region_report["Revenue"])
plt.title("Revenue by Region")
plt.xlabel("Region")
plt.ylabel("Revenue")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.bar(product_report.reset_index()["Product"], product_report["Revenue"])
plt.title("Revenue by Product")
plt.xlabel("Product")
plt.ylabel("Revenue")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(monthly_report["Month"], monthly_report["Revenue"], marker="o")
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.show()

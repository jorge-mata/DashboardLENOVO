"""CPU Inventory Management Dashboard for Server Manufacturing.

Features:
- Track CPU purchase orders and inventory
- Monitor Intel/AMD distribution (80/20 target)
- KPIs: Fill rate, on-time delivery, service levels
- Charts: CPU brand distribution, lead time analysis
- Export filtered data for planning
"""

import io
from datetime import datetime, timedelta
import random
import json
import os

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st


def generate_sample_orders(n=50, seed=42):
	random.seed(seed)
	np.random.seed(seed)
	start = datetime.today() - timedelta(days=90)
	statuses = ["Pending", "Processing", "Shipped", "Delivered", "Cancelled"]
	customers = [
		"Acme Co", "Beta LLC", "Gamma Inc", "Delta Ltd", "Epsilon GmbH",
		"Zeta Corp", "Eta Partners", "Theta Traders",
	]

	rows = []
	for i in range(1, n + 1):
		order_date = start + timedelta(days=int(np.random.rand() * 90))
		status = random.choices(statuses, weights=[0.15, 0.2, 0.25, 0.35, 0.05])[0]
		customer = random.choice(customers)
		items = random.randint(1, 10)
		total = round(random.uniform(20, 1500) * items / 3, 2)
		rows.append(
			{
				"order_id": f"ORD-{1000 + i}",
				"date": order_date.date().isoformat(),
				"customer": customer,
				"status": status,
				"items": items,
				"total": total,
			}
		)

	df = pd.DataFrame(rows)
	df["date"] = pd.to_datetime(df["date"])
	return df


def load_csv(uploaded_file):
	try:
		return pd.read_csv(uploaded_file, parse_dates=["date"])
	except Exception:
		# try without parsing
		df = pd.read_csv(uploaded_file)
		if "date" in df.columns:
			df["date"] = pd.to_datetime(df["date"])
		return df


def load_cpu_csv(uploaded_file):
	"""Load and process CPU inventory CSV from Microsoft Forms or other sources."""
	try:
		df = pd.read_csv(uploaded_file)
		
		# Common Microsoft Forms column mappings
		column_mappings = {
			'Timestamp': 'Form Submission Date',
			'Start time': 'Form Submission Date',
			'Completion time': 'Form Completion Date',
			'Email': 'Submitter Email',
			'Name': 'Submitter Name',
			'Id': 'Response ID',
			'Order Number': 'Purchase Order Number',
			'Part Number': 'CPU Part Number',
			'Brand': 'CPU Brand',
			'What is the CPU part number?': 'CPU Part Number',
			'CPU Part Number': 'CPU Part Number',
			'What is the CPU brand?': 'CPU Brand',
			'CPU Brand': 'CPU Brand',
			'How many CPUs are being ordered?': 'Requested Quantity',
			'Requested Quantity': 'Requested Quantity',
			'Confirmed Quantity': 'Confirmed Quantity',
			'What is the supplier name?': 'Supplier',
			'Supplier': 'Supplier',
			'When is the delivery needed?': 'Requested Delivery Date',
			'Requested Delivery Date': 'Requested Delivery Date',
			'Confirmed Delivery Date': 'Confirmed Delivery Date',
			'Confirmed Ship Date': 'Confirmed Ship Date',
			'Material in Warehouse': 'Material in Warehouse',
			'Actual Shipment Date': 'Actual Shipment Date',
			'Delivery Note Number': 'Delivery Note Number',
			'Purchase Order Number': 'Purchase Order Number',
			'PO Number': 'Purchase Order Number',
			'Origin Country': 'Origin',
			'Origin': 'Origin'
		}
		
		# Apply column mappings
		df = df.rename(columns=column_mappings)
		
		# Ensure required columns exist, create defaults if missing
		required_columns = {
			'Purchase Order Number': lambda: range(300000, 300000 + len(df)),
			'CPU Part Number': 'UNKNOWN-CPU',
			'CPU Brand': lambda row: 'Intel' if 'INTEL' in str(row.get('CPU Part Number', '')).upper() else 'AMD' if 'AMD' in str(row.get('CPU Part Number', '')).upper() else 'Intel',  # Default to Intel if unknown
			'Requested Quantity': 1,
			'Confirmed Quantity': lambda row: row.get('Requested Quantity', 1),
			'Requested Delivery Date': datetime.now() + timedelta(days=30),
			'Confirmed Delivery Date': lambda row: pd.to_datetime(row.get('Requested Delivery Date', datetime.now())) + timedelta(days=7),
			'Confirmed Ship Date': lambda row: pd.to_datetime(row.get('Confirmed Delivery Date', datetime.now())) - timedelta(days=5),
			'Material in Warehouse': lambda row: pd.to_datetime(row.get('Requested Delivery Date', datetime.now())) + timedelta(days=random.randint(-10, 20)),
			'Actual Shipment Date': lambda row: pd.to_datetime(row.get('Confirmed Ship Date', datetime.now())) + timedelta(days=random.randint(0, 3)),
			'Delivery Note Number': lambda: range(700000, 700000 + len(df)),
			'Supplier': 'Unknown Supplier',
			'Origin': 'Unknown'
		}
		
		for col, default_value in required_columns.items():
			if col not in df.columns:
				if callable(default_value):
					if col in ['CPU Brand']:
						df[col] = df.apply(default_value, axis=1)
					else:
						df[col] = list(default_value())
				else:
					df[col] = default_value
		
		# Convert date columns with flexible parsing
		date_columns = ['Requested Delivery Date', 'Confirmed Delivery Date', 'Confirmed Ship Date', 'Material in Warehouse', 'Actual Shipment Date']
		for col in date_columns:
			if col in df.columns:
				# Handle both MM/DD/YYYY and YYYY-MM-DD formats
				df[col] = pd.to_datetime(df[col], errors='coerce')

		# Convert numeric columns with better error handling
		numeric_columns = ['Requested Quantity', 'Confirmed Quantity', 'Purchase Order Number', 'Delivery Note Number']
		for col in numeric_columns:
			if col in df.columns:
				df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
		
		# Clean up CPU Brand values - handle case variations
		if 'CPU Brand' in df.columns:
			df['CPU Brand'] = df['CPU Brand'].str.strip().str.title()  # Clean and title case
			# Standardize common variations
			df['CPU Brand'] = df['CPU Brand'].replace({
				'Intel': 'Intel',
				'Amd': 'AMD', 
				'AMD': 'AMD',
				'INTEL': 'Intel',
				'intel': 'Intel',
				'amd': 'AMD'
			})
		
		return df
		
	except Exception as e:
		st.error(f"Error loading CPU CSV: {str(e)}")
		return None


def sanitize_df_for_streamlit(df):
	"""
	Coerce dataframe columns to predictable dtypes so pyarrow/streamlit can serialize them.
	Handles both order data and purchase order data structures.
	"""
	df = df.copy()
	
	# Handle datetime columns
	datetime_cols = ["date", "Requested Delivery Date", "Confirmed Delivery Date", 
					"Confirmed Ship Date", "Material in Warehouse", "Actual Shipment Date"]
	for col in datetime_cols:
		if col in df.columns:
			df[col] = pd.to_datetime(df[col], errors="coerce")

	# Handle string columns
	string_cols = ["customer", "status", "order_id", "Customer Part Number", "Origin"]
	for col in string_cols:
		if col in df.columns:
			df[col] = df[col].astype(str)

	# Handle integer columns
	int_cols = ["items", "Purchase Order Number", "Requested Quantity", 
				"Confirmed Quantity", "Delivery Note Number"]
	for col in int_cols:
		if col in df.columns:
			df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

	# Handle float columns
	float_cols = ["total"]
	for col in float_cols:
		if col in df.columns:
			df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)

	# catch any columns containing container objects and stringify them
	for col in df.columns:
		try:
			if df[col].apply(lambda x: isinstance(x, (dict, list, set, tuple))).any():
				df[col] = df[col].astype(str)
		except Exception:
			# if apply fails, stringify as a fallback
			df[col] = df[col].astype(str)

	return df


def generate_cpu_inventory_dummy(n=200, seed=42):
	"""Generate dummy CPU purchase order data for server manufacturing."""
	random.seed(seed)
	np.random.seed(seed)
	
	start_date = datetime.today() - timedelta(days=180)
	
	# Intel CPUs (80% of orders)
	intel_cpus = [
		"INTEL-E5-2680V4", "INTEL-E5-2690V4", "INTEL-E5-2695V4", "INTEL-E5-2698V4",
		"INTEL-XEON-4210", "INTEL-XEON-4214", "INTEL-XEON-4216", "INTEL-XEON-4220",
		"INTEL-XEON-6230", "INTEL-XEON-6238", "INTEL-XEON-6240", "INTEL-XEON-6248",
		"INTEL-XEON-8260", "INTEL-XEON-8268", "INTEL-XEON-8270", "INTEL-XEON-8280"
	]
	
	# AMD CPUs (20% of orders)
	amd_cpus = [
		"AMD-EPYC-7232P", "AMD-EPYC-7302P", "AMD-EPYC-7402P", "AMD-EPYC-7502P",
		"AMD-EPYC-7252", "AMD-EPYC-7352", "AMD-EPYC-7452", "AMD-EPYC-7552",
		"AMD-EPYC-7262", "AMD-EPYC-7362", "AMD-EPYC-7462", "AMD-EPYC-7562"
	]
	
	# Suppliers for CPUs
	suppliers = ["Intel Direct", "AMD Direct", "Arrow Electronics", "Avnet", "Digi-Key", "Mouser"]
	origins = ["USA", "Malaysia", "Ireland", "China", "Taiwan", "Costa Rica"]
	
	rows = []
	for i in range(n):
		# Purchase Order Number
		po_number = 200000 + i
		
		# CPU Part Number with 80/20 distribution
		if random.random() < 0.8:  # 80% Intel
			cpu_part = random.choice(intel_cpus)
			cpu_brand = "Intel"
		else:  # 20% AMD
			cpu_part = random.choice(amd_cpus)
			cpu_brand = "AMD"
		
		# Requested quantities (typical server CPU orders)
		requested_qty = random.choices([10, 25, 50, 100, 250, 500], weights=[10, 20, 30, 25, 10, 5])[0]
		
		# Confirmed quantity (sometimes less due to supply constraints)
		fill_rate_factor = random.choices([1.0, 0.95, 0.9, 0.8, 0.6, 0.5], weights=[60, 20, 10, 5, 3, 2])[0]
		confirmed_qty = int(requested_qty * fill_rate_factor)
		
		# Requested Delivery Date
		requested_delivery = start_date + timedelta(days=random.randint(0, 180))
		
		# Confirmed Delivery Date (CPU supply chains often have delays)
		delay_days = random.choices([0, 1, 3, 7, 14, 30, 60], weights=[40, 15, 20, 15, 7, 2, 1])[0]
		confirmed_delivery = requested_delivery + timedelta(days=delay_days)
		
		# Confirmed Ship Date (few days before delivery)
		ship_delay = random.randint(2, 8)
		confirmed_ship = confirmed_delivery - timedelta(days=ship_delay)
		
		# Material in Warehouse (when CPU became available)
		warehouse_lead = random.randint(-20, 30)  # Can be before or after requested date
		material_warehouse = requested_delivery + timedelta(days=warehouse_lead)
		
		# Actual Shipment Date (sometimes delayed from confirmed ship date)
		actual_delay = random.choices([0, 1, 2, 5, 10, 15], weights=[65, 15, 10, 7, 2, 1])[0]
		actual_shipment = confirmed_ship + timedelta(days=actual_delay)
		
		# Delivery Note Number
		delivery_note = 600000 + i
		
		# Supplier and Origin
		supplier = random.choice(suppliers)
		origin = random.choice(origins)
		
		rows.append({
			"Purchase Order Number": po_number,
			"CPU Part Number": cpu_part,
			"CPU Brand": cpu_brand,
			"Supplier": supplier,
			"Requested Quantity": requested_qty,
			"Confirmed Quantity": confirmed_qty,
			"Requested Delivery Date": requested_delivery,
			"Confirmed Delivery Date": confirmed_delivery,
			"Confirmed Ship Date": confirmed_ship,
			"Delivery Note Number": delivery_note,
			"Material in Warehouse": material_warehouse,
			"Actual Shipment Date": actual_shipment.strftime('%Y-%m-%d'),
			"Origin": origin
		})
	
	df = pd.DataFrame(rows)
	
	# Convert date columns to proper datetime
	date_columns = ["Requested Delivery Date", "Confirmed Delivery Date", 
					"Confirmed Ship Date", "Material in Warehouse"]
	for col in date_columns:
		df[col] = pd.to_datetime(df[col])
	
	# Convert Actual Shipment Date to datetime
	df["Actual Shipment Date"] = pd.to_datetime(df["Actual Shipment Date"])
	
	return df


def compute_cpu_inventory_kpis(df):
	"""Compute KPIs from CPU inventory data."""
	# Clean data - remove any NaN values for calculations
	clean_df = df.dropna(subset=['CPU Brand'], how='any') if 'CPU Brand' in df.columns else df.dropna()
	
	if len(clean_df) == 0:
		return {
			"fill_rate": 0.0,
			"otd_rate": 0.0,
			"service_level": 0.0,
			"delivery_promise_performance": 0.0,
			"quantity_promise_performance": 0.0,
			"lead_time_variability": 0.0,
			"avg_lead_time": 0.0,
			"material_availability_performance": 0.0,
			"intel_ratio": 0.0,
			"amd_ratio": 0.0,
			"brand_distribution": {},
			"lead_times": [],
			"material_lead_times": [],
			"total_cpus": 0
		}
	
	# Ensure required columns exist with defaults
	if 'Requested Quantity' not in clean_df.columns:
		clean_df['Requested Quantity'] = 1
	if 'Confirmed Quantity' not in clean_df.columns:
		clean_df['Confirmed Quantity'] = clean_df['Requested Quantity']
	if 'CPU Brand' not in clean_df.columns:
		clean_df['CPU Brand'] = 'Unknown'
	
	# 1. Fill Rate: (Sum of Confirmed Quantity / Sum of Requested Quantity) * 100
	total_requested = clean_df["Requested Quantity"].sum()
	total_confirmed = clean_df["Confirmed Quantity"].sum()
	fill_rate = (total_confirmed / total_requested * 100.0) if total_requested > 0 else 0.0
	
	# 2. On-Time Delivery: Orders shipped on or before confirmed delivery date
	if 'Actual Shipment Date' in clean_df.columns and 'Confirmed Delivery Date' in clean_df.columns:
		# Convert to datetime if they aren't already
		clean_df['Actual Shipment Date'] = pd.to_datetime(clean_df['Actual Shipment Date'], errors='coerce')
		clean_df['Confirmed Delivery Date'] = pd.to_datetime(clean_df['Confirmed Delivery Date'], errors='coerce')
		
		on_time_deliveries = (clean_df["Actual Shipment Date"] <= clean_df["Confirmed Delivery Date"]).sum()
		otd_rate = (on_time_deliveries / len(clean_df) * 100.0)
	else:
		otd_rate = 0.0
	
	# 3. Service Level: Orders with material available before/on requested delivery date
	if 'Material in Warehouse' in clean_df.columns and 'Requested Delivery Date' in clean_df.columns:
		clean_df['Material in Warehouse'] = pd.to_datetime(clean_df['Material in Warehouse'], errors='coerce')
		clean_df['Requested Delivery Date'] = pd.to_datetime(clean_df['Requested Delivery Date'], errors='coerce')
		
		material_available_on_time = (clean_df["Material in Warehouse"] <= clean_df["Requested Delivery Date"]).sum()
		service_level = (material_available_on_time / len(clean_df) * 100.0)
	else:
		service_level = 0.0
	
	# 4. Delivery Promise Performance: Confirmed delivery <= Requested delivery
	if 'Confirmed Delivery Date' in clean_df.columns and 'Requested Delivery Date' in clean_df.columns:
		delivery_promises_kept = (clean_df["Confirmed Delivery Date"] <= clean_df["Requested Delivery Date"]).sum()
		delivery_promise_performance = (delivery_promises_kept / len(clean_df) * 100.0)
	else:
		delivery_promise_performance = 0.0
	
	# 5. Quantity Promise Performance: Confirmed >= Requested
	quantity_promises_kept = (clean_df["Confirmed Quantity"] >= clean_df["Requested Quantity"]).sum()
	quantity_promise_performance = (quantity_promises_kept / len(clean_df) * 100.0)
	
	# 6. Lead Time Analysis: Actual shipment - Confirmed ship date
	if 'Actual Shipment Date' in clean_df.columns and 'Confirmed Ship Date' in clean_df.columns:
		clean_df['Confirmed Ship Date'] = pd.to_datetime(clean_df['Confirmed Ship Date'], errors='coerce')
		lead_times = (clean_df["Actual Shipment Date"] - clean_df["Confirmed Ship Date"]).dt.days
		lead_time_variability = float(lead_times.std()) if len(lead_times) > 0 else 0.0
		avg_lead_time = float(lead_times.mean()) if len(lead_times) > 0 else 0.0
		lead_times_values = lead_times.values
	else:
		lead_time_variability = 0.0
		avg_lead_time = 0.0
		lead_times_values = []
	
	# 7. Material Availability Performance: Material available before requested date
	if 'Material in Warehouse' in clean_df.columns and 'Requested Delivery Date' in clean_df.columns:
		material_lead_times = (clean_df["Material in Warehouse"] - clean_df["Requested Delivery Date"]).dt.days
		material_early = (material_lead_times <= 0).sum()  # Negative means early
		material_availability_performance = (material_early / len(clean_df) * 100.0)
		material_lead_times_values = material_lead_times.values
	else:
		material_availability_performance = 0.0
		material_lead_times_values = []
	
	# 8. CPU Brand Distribution Analysis
	brand_counts = clean_df["CPU Brand"].value_counts()
	intel_count = brand_counts.get("Intel", 0)
	amd_count = brand_counts.get("AMD", 0)
	total_cpus = intel_count + amd_count
	
	# If no Intel/AMD found, check for variations in casing
	if total_cpus == 0:
		# Check for case variations
		brand_series = clean_df["CPU Brand"].str.upper()
		intel_count = brand_series.str.contains("INTEL", na=False).sum()
		amd_count = brand_series.str.contains("AMD", na=False).sum()
		total_cpus = intel_count + amd_count
	
	intel_ratio = (intel_count / total_cpus * 100) if total_cpus > 0 else 0
	amd_ratio = (amd_count / total_cpus * 100) if total_cpus > 0 else 0
	
	return {
		"fill_rate": fill_rate,
		"otd_rate": otd_rate,
		"service_level": service_level,
		"delivery_promise_performance": delivery_promise_performance,
		"quantity_promise_performance": quantity_promise_performance,
		"lead_time_variability": lead_time_variability,
		"avg_lead_time": avg_lead_time,
		"material_availability_performance": material_availability_performance,
		"lead_times": lead_times_values,
		"material_lead_times": material_lead_times_values,
		"intel_ratio": intel_ratio,
		"amd_ratio": amd_ratio,
		"brand_distribution": brand_counts.to_dict(),
		"total_cpus": total_cpus
	}


def filter_orders(df, start_date=None, end_date=None, statuses=None, search=None):
	out = df.copy()
	if start_date:
		out = out[out["date"] >= pd.to_datetime(start_date)]
	if end_date:
		out = out[out["date"] <= pd.to_datetime(end_date)]
	if statuses:
		out = out[out["status"].isin(statuses)]
	if search:
		q = search.lower()
		out = out[
			out["customer"].str.lower().str.contains(q)
			| out["order_id"].str.lower().str.contains(q)
		]
	return out


def main():
	st.set_page_config(
		page_title="Supply Chain Dashboard", 
		layout="wide",
		initial_sidebar_state="expanded",
		page_icon="📊"
	)
	
	# Header with styling
	st.markdown("""
	<div style='text-align: center; padding: 1rem 0;'>
		<h1 style='color: #2E86AB; margin-bottom: 0.5rem;'>🖥️ CPU Inventory Management Dashboard</h1>
		<p style='color: #666; font-size: 1.2rem;'>Server Manufacturing CPU Supply Chain</p>
	</div>
	""", unsafe_allow_html=True)
	
	st.markdown("---")

	# Sidebar - data input and filters
	st.sidebar.markdown("## 🔧 Configuration")
	st.sidebar.markdown("---")
	
	with st.sidebar.expander("📁 Data Source", expanded=False):
		data_source = st.radio(
			"Choose data source:",
			["Sample Data", "Upload Orders CSV", "Upload CPU Inventory CSV"]
		)
		
		# Initialize variables to avoid UnboundLocalError
		uploaded = None
		cpu_uploaded = None
		
		if data_source == "Upload Orders CSV":
			uploaded = st.file_uploader("Upload orders CSV", type=["csv"], key="orders")
			if uploaded is None:
				st.info("Using sample data (120 orders)")
		elif data_source == "Upload CPU Inventory CSV":
			cpu_uploaded = st.file_uploader("Upload CPU inventory CSV (from Microsoft Forms)", type=["csv"], key="cpu")
			if cpu_uploaded is None:
				st.info("Using dummy CPU data")
		else:
			st.info("Using sample data")

	if uploaded is not None:
		df = load_csv(uploaded)
	else:
		df = generate_sample_orders(120)

	# sanitize dtypes to avoid pyarrow/streamlit serialization errors
	df = sanitize_df_for_streamlit(df)

	# Ensure expected columns exist
	expected = {"order_id", "date", "customer", "status", "items", "total"}
	missing = expected - set(df.columns)
	if missing:
		st.sidebar.error(f"❌ Missing columns: {', '.join(sorted(missing))}")

	st.sidebar.markdown("---")
	st.sidebar.markdown("## 🎛️ Filters")
	
	# date filter
	min_date = df["date"].min().date()
	max_date = df["date"].max().date()
	start_date, end_date = st.sidebar.date_input(
		"📅 Date Range", 
		[min_date, max_date],
		min_value=min_date, 
		max_value=max_date
	)

	# status filter
	statuses = df["status"].unique().tolist()
	selected_status = st.sidebar.multiselect(
		"📋 Status Filter", 
		options=sorted(statuses), 
		default=sorted(statuses)
	)

	# search
	q = st.sidebar.text_input("🔍 Search (customer or order ID)")
	
	# Supply chain config
	st.sidebar.markdown("---")
	st.sidebar.markdown("## ⚙️ Purchase Order Settings")
	po_count = st.sidebar.slider("� Number of Purchase Orders", min_value=50, max_value=500, value=200)

	# Supply chain config
	st.sidebar.markdown("---")
	st.sidebar.markdown("## ⚙️ CPU Inventory Settings")
	cpu_order_count = st.sidebar.slider("🖥️ Number of CPU Orders", min_value=50, max_value=500, value=200)

	# apply filters
	filtered = filter_orders(df, start_date=start_date, end_date=end_date, statuses=selected_status, search=q)

	# Create tabs for better organization
	tab1, tab2, tab3 = st.tabs(["📊 Orders Overview", "🖥️ CPU Inventory KPIs", "📋 Data Table"])
	
	with tab1:
		st.markdown("### 📈 Key Order Metrics")
		
		# Top KPIs with better styling
		col1, col2, col3 = st.columns(3)
		total_orders = len(filtered)
		total_revenue = filtered["total"].sum() if not filtered.empty else 0.0
		avg_order = filtered["total"].mean() if not filtered.empty else 0.0

		with col1:
			st.metric(
				"🛒 Total Orders", 
				f"{total_orders:,}",
				delta=f"+{total_orders - len(df)//2}" if total_orders > len(df)//2 else f"{total_orders - len(df)//2}"
			)
		with col2:
			st.metric(
				"💰 Total Revenue", 
				f"${total_revenue:,.2f}",
				delta=f"+${total_revenue * 0.1:,.0f}" if total_revenue > 0 else None
			)
		with col3:
			st.metric(
				"📊 Avg Order Value", 
				f"${avg_order:,.2f}",
				delta=f"+${avg_order * 0.05:,.0f}" if avg_order > 0 else None
			)

		st.markdown("---")
		
		# Charts in columns
		col_chart1, col_chart2 = st.columns([2, 1])
		
		with col_chart1:
			st.markdown("#### 📅 Orders Timeline")
			if not filtered.empty:
				orders_time = (
					filtered.groupby(filtered["date"].dt.date)["order_id"].count().reset_index()
				)
				orders_time.columns = ["date", "orders"]
				chart = (
					alt.Chart(orders_time)
					.mark_area(opacity=0.7, color="#2E86AB")
					.encode(
						x=alt.X("date:T", title="Date"),
						y=alt.Y("orders:Q", title="Number of Orders"),
						tooltip=["date:T", "orders:Q"]
					)
					.properties(height=300)
				)
				st.altair_chart(chart, use_container_width=True)
			else:
				st.info("📭 No orders match the current filters.")

		with col_chart2:
			st.markdown("#### 🎯 Status Distribution")
			status_counts = filtered["status"].value_counts().reset_index()
			status_counts.columns = ["status", "count"]
			if not status_counts.empty:
				# Define colors for different statuses
				color_scale = alt.Scale(
					domain=["Delivered", "Shipped", "Processing", "Pending", "Cancelled"],
					range=["#2E8B57", "#4169E1", "#FF8C00", "#FFD700", "#DC143C"]
				)
				pie = (
					alt.Chart(status_counts)
					.mark_arc(innerRadius=40, outerRadius=100)
					.encode(
						theta=alt.Theta("count:Q"),
						color=alt.Color("status:N", scale=color_scale),
						tooltip=["status:N", "count:Q"]
					)
					.properties(height=300)
				)
				st.altair_chart(pie, use_container_width=True)
			else:
				st.info("📊 No status data available.")

	with tab2:
		st.markdown("### 🖥️ CPU Inventory Performance Indicators")
		
		# Generate or load CPU data based on selection
		if data_source == "Upload CPU Inventory CSV" and cpu_uploaded is not None:
			po_data = load_cpu_csv(cpu_uploaded)
			if po_data is not None:
				st.success(f"📁 Loaded {len(po_data)} CPU records from uploaded file")
				
				# Show data preview
				with st.expander("🔍 Uploaded Data Preview", expanded=False):
					st.write("**Column mapping applied:**")
					col_info = {}
					for col in po_data.columns:
						col_info[col] = f"{po_data[col].dtype} ({po_data[col].notna().sum()}/{len(po_data)} non-null)"
					st.json(col_info)
					
					st.write("**Sample data (first 3 rows):**")
					st.dataframe(po_data.head(3))
					
					if 'CPU Brand' in po_data.columns:
						brand_counts = po_data['CPU Brand'].value_counts()
						st.write(f"**CPU Brands found:** {dict(brand_counts)}")
			else:
				st.error("❌ Failed to load uploaded CSV file")
				po_data = generate_cpu_inventory_dummy(n=cpu_order_count)
				st.info("🎲 Using generated dummy CPU data as fallback")
		else:
			po_data = generate_cpu_inventory_dummy(n=cpu_order_count)
			st.info("🎲 Using generated dummy CPU data")
			
		kpis = compute_cpu_inventory_kpis(po_data)

		# CPU Brand Distribution Analysis
		col_brand1, col_brand2 = st.columns([1, 1])
		with col_brand1:
			st.metric(
				"🔵 Intel Distribution", 
				f"{kpis['intel_ratio']:.1f}%",
				help="Current Intel CPU order percentage"
			)
		with col_brand2:
			st.metric(
				"🔴 AMD Distribution", 
				f"{kpis['amd_ratio']:.1f}%",
				help="Current AMD CPU order percentage"
			)
		
		# Brand distribution chart
		if kpis['total_cpus'] > 0:
			brand_data = pd.DataFrame(list(kpis['brand_distribution'].items()), columns=['Brand', 'Count'])
			brand_chart = (
				alt.Chart(brand_data)
				.mark_arc(innerRadius=50, outerRadius=100)
				.encode(
					theta=alt.Theta('Count:Q'),
					color=alt.Color('Brand:N', scale=alt.Scale(range=['#CC0000', '#0066CC'])),
					tooltip=['Brand:N', 'Count:Q']
				)
				.properties(height=200, title="CPU Brand Distribution")
			)
			st.altair_chart(brand_chart, use_container_width=True)

		st.markdown("---")

		# Main KPIs in a nice grid layout
		col_a, col_b, col_c = st.columns(3)
		with col_a:
			st.metric(
				"📦 Fill Rate", 
				f"{kpis['fill_rate']:.1f}%",
				delta=f"{kpis['fill_rate'] - 85:.1f}%" if kpis['fill_rate'] > 85 else f"{kpis['fill_rate'] - 85:.1f}%",
				help="Percentage of requested quantity that was confirmed"
			)
		with col_b:
			st.metric(
				"🚚 On-Time Delivery", 
				f"{kpis['otd_rate']:.1f}%",
				delta=f"{kpis['otd_rate'] - 90:.1f}%" if kpis['otd_rate'] > 90 else f"{kpis['otd_rate'] - 90:.1f}%",
				help="Orders shipped on or before confirmed delivery date"
			)
		with col_c:
			st.metric(
				"🎯 Service Level", 
				f"{kpis['service_level']:.1f}%",
				delta=f"{kpis['service_level'] - 80:.1f}%" if kpis['service_level'] > 80 else f"{kpis['service_level'] - 80:.1f}%",
				help="Orders with material available by requested delivery date"
			)

		st.markdown("---")
		
		# Additional performance metrics
		col_d, col_e, col_f = st.columns(3)
		with col_d:
			st.metric(
				"🏭 Delivery Promise Performance", 
				f"{kpis['delivery_promise_performance']:.1f}%",
				help="Confirmed delivery date <= Requested delivery date"
			)
		with col_e:
			st.metric(
				"🏭 Quantity Promise Performance", 
				f"{kpis['quantity_promise_performance']:.1f}%",
				help="Confirmed quantity >= Requested quantity"
			)
		with col_f:
			st.metric(
				"🏭 Material Availability Performance", 
				f"{kpis['material_availability_performance']:.1f}%",
				help="Material available before requested delivery date"
			)

		st.markdown("---")
		st.markdown("#### ⏱️ Lead Time Analysis")
		
		col_lt1, col_lt2 = st.columns([1, 2])
		with col_lt1:
			st.metric(
				"📈 Lead Time Variability", 
				f"{kpis['lead_time_variability']:.2f} days",
				help="Standard deviation of shipment delays"
			)
			st.write(f"**Average Lead Time:** {kpis['avg_lead_time']:.1f} days")
			if len(kpis['lead_times']) > 0:
				st.write(f"**Median:** {np.median(kpis['lead_times']):.1f} days")
				st.write(f"**Min/Max:** {np.min(kpis['lead_times']):.1f} - {np.max(kpis['lead_times']):.1f} days")
		
		with col_lt2:
			if len(kpis['lead_times']) > 0:
				lt_df = pd.DataFrame({"lead_time": kpis['lead_times']})
				hist = (
					alt.Chart(lt_df)
					.mark_bar(color="#FF6B6B", opacity=0.7)
					.encode(
						x=alt.X("lead_time:Q", bin=alt.Bin(maxbins=25), title="Shipment Delay (days)"),
						y=alt.Y("count()", title="Frequency"),
						tooltip=["count()"]
					)
					.properties(height=300, title="Lead Time Distribution")
				)
				st.altair_chart(hist, use_container_width=True)
			else:
				st.info("📊 No lead time data to display.")
		
		# Material Lead Time Analysis
		st.markdown("#### 📦 Material Availability Analysis")
		if len(kpis['material_lead_times']) > 0:
			mat_lt_df = pd.DataFrame({"material_lead_time": kpis['material_lead_times']})
			mat_hist = (
				alt.Chart(mat_lt_df)
				.mark_bar(color="#4CAF50", opacity=0.7)
				.encode(
					x=alt.X("material_lead_time:Q", bin=alt.Bin(maxbins=25), title="Material Lead Time (days, negative = early)"),
					y=alt.Y("count()", title="Frequency"),
					tooltip=["count()"]
				)
				.properties(height=250, title="Material Availability Lead Time Distribution")
			)
			st.altair_chart(mat_hist, use_container_width=True)
			
			st.info("💡 Negative values indicate material was available before the requested delivery date (good!)")
		
		# Show sample of purchase order data
		with st.expander("📋 Sample Purchase Order Data", expanded=False):
			st.dataframe(po_data.head(10), use_container_width=True)

	with tab3:
		st.markdown("### 📋 Orders Data Table")
		
		col_export, col_info = st.columns([3, 1])
		with col_export:
			csv = filtered.to_csv(index=False)
			st.download_button(
				"📥 Download Filtered CSV", 
				data=csv, 
				file_name=f"filtered_orders_{datetime.now().strftime('%Y%m%d')}.csv", 
				mime="text/csv"
			)
		with col_info:
			st.info(f"📊 Showing {len(filtered)} of {len(df)} orders")
		
		# Enhanced data table
		if not filtered.empty:
			# Add row styling based on status
			styled_df = filtered.sort_values(by="date", ascending=False).reset_index(drop=True)
			st.dataframe(
				styled_df,
				use_container_width=True,
				height=400
			)
		else:
			st.warning("⚠️ No data to display with current filters")

		# Helpful debug / info
		with st.expander("🔧 Technical Info", expanded=False):
			col_tech1, col_tech2 = st.columns(2)
			with col_tech1:
				st.write("**Dataset Info:**")
				st.write(f"• Total rows: {len(df)}")
				st.write(f"• Filtered rows: {len(filtered)}")
				st.write(f"• Date range: {min_date} to {max_date}")
			with col_tech2:
				st.write("**Data Types:**")
				for col, dtype in df.dtypes.items():
					st.write(f"• {col}: {dtype}")
			
			st.write("**Sample Data:**")
			st.dataframe(df.head(3))


if __name__ == "__main__":
	main()

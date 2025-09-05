Orders Dashboard (prototype)

This is a minimal Streamlit prototype for tracking orders.

Files:
- `dashboard.py`: Streamlit app. Run with `streamlit run dashboard.py`.
- `requirements.txt`: Python dependencies.

Quick start

1. (Optional) Create a virtualenv and activate it.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run dashboard.py
```

Usage

- Upload your orders CSV in the sidebar or use the generated sample dataset.
- Expected CSV columns: `order_id`, `date`, `customer`, `status`, `items`, `total`.
- Filter by date, status, or search by customer/order id. Download filtered CSV from the UI.

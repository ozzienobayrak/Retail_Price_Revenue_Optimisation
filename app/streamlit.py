import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
import io
import qrcode

st.set_page_config(page_title="Retail Price and Revenue Optimizer", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
    }

    div[data-testid="stMetric"] {
        background-color: #f9fbfd;
        border: 1px solid #dde7f0;
        padding: 12px;
        border-radius: 10px;
    }

    h1, h2, h3 {
        color: #1f4e79;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
DATA_DIR = BASE_DIR / "data1"
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "lightgbm_model_base.joblib"
SIMULATOR_PATH = DATA_DIR / "simulator_df.csv"
INV_WEEK_PATH = DATA_DIR / "inventory_base_week.csv"
INV_MONTH_PATH = DATA_DIR / "inventory_base_month.csv"

CAT_COLS = [
    "item_id", "dept_id", "cat_id", "store_id", "state_id",
    "weekday", "event_name_1", "event_type_1", "event_name_2", "event_type_2"
]

#Renaming column names-better this way instead of changing names everywhere

DISPLAY_RENAME = {
    "item_id": "Product",
    "store_id": "Store",
    "horizon": "Period",
    "scenario": "Inventory",
    "current_price": "Current Price",
    "candidate_price": "Candidate Price",
    "available_stock": "Available Stock",
    "predicted_sales": "Predicted Sales",
    "feasible_sales": "Feasible Sales",
    "predicted_revenue": "Predicted Revenue",
    "price_change_pct": "Price Change %",
}


def prettify_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=DISPLAY_RENAME)


# -----------------------------
# Load model and data
# -----------------------------
@st.cache_resource
def load_model(model_path: Path):
    return joblib.load(model_path)

@st.cache_data
def load_data(simulator_path, inv_week_path, inv_month_path):
    simulator_df = pd.read_csv(simulator_path, low_memory=False)
    inventory_base_week = pd.read_csv(inv_week_path)
    inventory_base_month = pd.read_csv(inv_month_path)

    if "month_key" not in simulator_df.columns:
        simulator_df["month_key"] = (
            simulator_df["year"].astype(str)
            + "-"
            + simulator_df["month"].astype(str).str.zfill(2)
        )

    for col in CAT_COLS:
        if col in simulator_df.columns:
            simulator_df[col] = simulator_df[col].astype("category") #categories disappear when saved in csv and load. 

    return simulator_df, inventory_base_week, inventory_base_month

try:
    model = load_model(MODEL_PATH)
    simulator_df, inventory_base_week, inventory_base_month = load_data(
        SIMULATOR_PATH, INV_WEEK_PATH, INV_MONTH_PATH
    )
except Exception as e:
    st.error(f"Failed to load app assets: {e}")
    st.stop()

    #fake product names to make app more real!
food_names = [
    "Pasta", "Tomato Sauce", "Olive Oil", "Rice", "Brown Bread",
    "Cheddar Cheese", "Milk", "Yogurt", "Butter", "Chicken Breast",
    "Salmon", "Frozen Pizza", "Ice Cream", "Chocolate Bar", "Granola",
    "Cereal", "Orange Juice", "Coffee", "Green Tea", "Apple Juice",
    "Strawberry Jam", "Peanut Butter", "Honey", "Almonds", "Walnuts"
]

unique_items = sorted(simulator_df["item_id"].astype(str).unique())

name_map = {
    item: f"{food_names[i % len(food_names)]} {i+1}"
    for i, item in enumerate(unique_items)
}

simulator_df["product_name"] = simulator_df["item_id"].astype(str).map(name_map)
# -----------------------------
# Helper apps
# -----------------------------
def get_available_stock(
    item_id,
    store_id,
    horizon,
    scenario,
    inventory_base_week,
    inventory_base_month,
):
    multipliers = {"Low": 0.8, "Medium": 1.0, "High": 1.2}

    if scenario not in multipliers:
        raise ValueError("scenario must be 'Low', 'Medium', or 'High'")

    if horizon == "week":
        row = inventory_base_week[
            (inventory_base_week["item_id"] == item_id)
            & (inventory_base_week["store_id"] == store_id)
        ]
        if row.empty:
            raise ValueError("No weekly inventory baseline found for this item-store")
        baseline = row["recent_avg_weekly_sales"].iloc[0]

    elif horizon == "month":
        row = inventory_base_month[
            (inventory_base_month["item_id"] == item_id)
            & (inventory_base_month["store_id"] == store_id)
        ]
        if row.empty:
            raise ValueError("No monthly inventory baseline found for this item-store")
        baseline = row["recent_avg_monthly_sales"].iloc[0]

    else:
        raise ValueError("horizon must be 'week' or 'month'")

    return round(multipliers[scenario] * baseline)


def infer_feature_cols(df: pd.DataFrame) -> list[str]:
    # Exclude target/helper columns and keep model-ready predictors.
    exclude = {
        "sales",
        "sales_pred",
        "revenue_actual",
        "revenue_pred",
        "pred_sales",
        "pred_revenue",
        "date",
        "month_key",
    }
    feature_cols = [c for c in df.columns if c not in exclude]
    return feature_cols

# summary table for the chosen product
def simulate_price_bundle(
    model,
    simulator_df,
    item_id,
    store_id,
    feature_cols,
    inventory_base_week,
    inventory_base_month,
    horizon="week",
    scenario="Medium",
    grid_low=0.8,
    grid_high=1.2,
    n_prices=9,
    price_col="sell_price",
):
    subset = simulator_df[
        (simulator_df["item_id"] == item_id)
        & (simulator_df["store_id"] == store_id)
    ].copy()

    if subset.empty:
        return pd.DataFrame(), pd.DataFrame()

    available_stock = get_available_stock(
        item_id=item_id,
        store_id=store_id,
        horizon=horizon,
        scenario=scenario,
        inventory_base_week=inventory_base_week,
        inventory_base_month=inventory_base_month,
    )

    current_price = subset[price_col].median()
    price_grid = np.round(np.linspace(current_price * grid_low, current_price * grid_high, n_prices), 2)

    results = []

    for p in price_grid:
        x_sim = subset.copy()
        x_sim[price_col] = p

        if "price_change_1" in x_sim.columns:
            if "price_lag_1" in x_sim.columns:
                x_sim["price_change_1"] = x_sim[price_col] - x_sim["price_lag_1"]
            else:
                x_sim["price_change_1"] = 0.0

        if "price_pct_change_1" in x_sim.columns:
            if "price_lag_1" in x_sim.columns:
                denom = x_sim["price_lag_1"].replace(0, np.nan)
                x_sim["price_pct_change_1"] = (x_sim[price_col] - x_sim["price_lag_1"]) / denom
                x_sim["price_pct_change_1"] = x_sim["price_pct_change_1"].fillna(0.0)
            else:
                x_sim["price_pct_change_1"] = 0.0

        if "price_rel" in x_sim.columns and "price_roll_mean_28" in x_sim.columns:
            denom = x_sim["price_roll_mean_28"].replace(0, np.nan)
            x_sim["price_rel"] = x_sim[price_col] / denom
            x_sim["price_rel"] = x_sim["price_rel"].fillna(1.0)

        pred_sales = np.asarray(model.predict(x_sim[feature_cols]), dtype=float)
        pred_sales = np.clip(pred_sales, 0, None)

        x_sim["pred_sales"] = pred_sales
        x_sim["pred_revenue"] = x_sim["pred_sales"] * x_sim[price_col]

        if horizon == "week":
            grouped = (
                x_sim.groupby(["item_id", "store_id", "wm_yr_wk"], observed=True)
                .agg(pred_sales=("pred_sales", "sum"), pred_revenue=("pred_revenue", "sum"))
                .reset_index()
            )
        elif horizon == "month":
            grouped = (
                x_sim.groupby(["item_id", "store_id", "month_key"], observed=True)
                .agg(pred_sales=("pred_sales", "sum"), pred_revenue=("pred_revenue", "sum"))
                .reset_index()
            )
        else:
            raise ValueError("horizon must be 'week' or 'month'")

        total_pred_sales = grouped["pred_sales"].sum()
        feasible_sales = min(total_pred_sales, available_stock)
        predicted_revenue = p * feasible_sales

        results.append(
            {
                "item_id": item_id,
                "store_id": store_id,
                "horizon": horizon,
                "scenario": scenario,
                "current_price": current_price,
                "candidate_price": p,
                "available_stock": available_stock,
                "predicted_sales": total_pred_sales,
                "feasible_sales": feasible_sales,
                "predicted_revenue": predicted_revenue,
            }
        )

    sim_table = pd.DataFrame(results)
    best_bundle = sim_table.loc[[sim_table["predicted_revenue"].idxmax()]].copy()
    best_bundle["price_change_pct"] = (
        (best_bundle["candidate_price"] - best_bundle["current_price"])
        / best_bundle["current_price"]
    )

    return sim_table, best_bundle

#ploting simulation prices against predicted sales&revenue
def plot_price_simulation(sim_table: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        sim_table["candidate_price"],
        sim_table["predicted_revenue"],
        marker="o",
        color="#2e7d32",
        label="Revenue",
    )

    ax.plot(
        sim_table["candidate_price"],
        sim_table["feasible_sales"],
        marker="o",
        color="#1f77b4",
        label="Sales",
    )

    best_idx = sim_table["predicted_revenue"].idxmax()
    best_row = sim_table.loc[best_idx]
    ax.scatter(
        best_row["candidate_price"],
        best_row["predicted_revenue"],
        color="black",
        s=80,
        zorder=3,
        label="Best revenue",
    )

    ax.set_xlabel("Candidate Price")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend()
    fig.tight_layout()
    return fig

#summary table for chosen product for different price levels
def summarize_store_or_state(
    model,
    simulator_df,
    feature_cols,
    inventory_base_week,
    inventory_base_month,
    group_value,
    group_level="store",
    horizon="week",
    scenario="Medium",
):
    if group_level == "store":
        item_store_pairs = (
            simulator_df[simulator_df["store_id"] == group_value][["item_id", "store_id"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
    elif group_level == "state":
        item_store_pairs = (
            simulator_df[simulator_df["state_id"] == group_value][["item_id", "store_id"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
    else:
        raise ValueError("group_level must be 'store' or 'state'")

    best_rows = []
    for item_id, store_id in item_store_pairs:
        try:
            _, best = simulate_price_bundle(
                model=model,
                simulator_df=simulator_df,
                item_id=item_id,
                store_id=store_id,
                feature_cols=feature_cols,
                inventory_base_week=inventory_base_week,
                inventory_base_month=inventory_base_month,
                horizon=horizon,
                scenario=scenario,
            )
            if not best.empty:
                best_rows.append(best)
        except Exception:
            continue

    if not best_rows:
        return pd.DataFrame()

    return pd.concat(best_rows, ignore_index=True)


# -----------------------------
# Load runtime objects
# -----------------------------
try:
    model = load_model(MODEL_PATH)
    simulator_df, inventory_base_week, inventory_base_month = load_data(
        SIMULATOR_PATH, INV_WEEK_PATH, INV_MONTH_PATH
    )
except Exception as e:
    st.error(f"Failed to load app assets: {e}")
    st.stop()

feature_cols = infer_feature_cols(simulator_df)


# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Simulation Controls")

products = pd.DataFrame({
    "item_id": unique_items,
    "product_name": [name_map[item] for item in unique_items]
})

product_choice = st.sidebar.selectbox(
    "Product",
    products["product_name"].tolist()
)

item_id = products.loc[
    products["product_name"] == product_choice,
    "item_id"
].iloc[0]

item_store_df = simulator_df[simulator_df["item_id"] == item_id]
available_stores = sorted(item_store_df["store_id"].dropna().unique().tolist())

store_id = st.sidebar.selectbox("Store", available_stores)

horizon = st.sidebar.selectbox("Period", ["week", "month"])
scenario = st.sidebar.selectbox("Inventory Scenario", ["Low", "Medium", "High"], index=1)

price_range = st.sidebar.slider(
    "Price range around current price",
    0.6, 1.4, (0.8, 1.2), 0.05
)

n_prices = st.sidebar.slider(
    "Number of candidate prices",
    5, 21, 9, 2
)

st.sidebar.markdown("---")
summary_level = st.sidebar.selectbox("Optional total summary", ["None", "Store", "State"])

# -----------------------------
# Main app
# -----------------------------
left, right = st.columns([8,1])
with left:
    st.title("Retail Price Optimization Simulator")
    st.caption("Revenue-maximizing price simulation using a LightGBM demand model.")

with right:
    st.image(generate_qr_code(APP_URL), width=110)
    
try:
    sim_table, best_bundle = simulate_price_bundle(
        model=model,
        simulator_df=simulator_df,
        item_id=item_id,
        store_id=store_id,
        feature_cols=feature_cols,
        inventory_base_week=inventory_base_week,
        inventory_base_month=inventory_base_month,
        horizon=horizon,
        scenario=scenario,
        grid_low=price_range[0],
        grid_high=price_range[1],
        n_prices=n_prices,
    )
except Exception as e:
    st.error(f"Simulation failed: {e}")
    st.stop()

if sim_table.empty or best_bundle.empty:
    st.warning("No simulation results were produced for this selection.")
    st.stop()

best = best_bundle.iloc[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Recommended Price", f"{best['candidate_price']:.2f}")
col2.metric("Predicted Sales", f"{best['predicted_sales']:.0f}")
col3.metric("Feasible Sales", f"{best['feasible_sales']:.0f}")
col4.metric("Predicted Revenue", f"{best['predicted_revenue']:.0f}")

col5, col6, col7 = st.columns(3)
col5.metric("Current Price", f"{best['current_price']:.2f}")
col6.metric("Price Change", f"{best['price_change_pct'] * 100:.1f}%")
col7.metric("Available Stock", f"{best['available_stock']:.0f}")

st.subheader("Best Price Bundle")

best_display = best_bundle.copy()
best_display["item_id"] = best_display["item_id"].map(name_map)

# round columns
best_display["predicted_sales"] = best_display["predicted_sales"].round(0)
best_display["feasible_sales"] = best_display["feasible_sales"].round(0)
best_display["predicted_revenue"] = best_display["predicted_revenue"].round(0)

st.dataframe(prettify_table(best_display), use_container_width=True)

st.subheader("Revenue and Sales Simulation")
product_name = name_map[item_id]


fig = plot_price_simulation(
    sim_table,
    title=f"{horizon.title()} Simulation for {product_name} in {store_id}"
)

st.pyplot(fig, clear_figure=True)

st.subheader("Candidate Price Table")

sim_display = sim_table.copy()
sim_display["item_id"] = sim_display["item_id"].map(name_map)

sim_display["predicted_sales"] = sim_display["predicted_sales"].round(0)
sim_display["feasible_sales"] = sim_display["feasible_sales"].round(0)
sim_display["predicted_revenue"] = sim_display["predicted_revenue"].round(0)

st.dataframe(
    prettify_table(sim_display.sort_values("predicted_revenue", ascending=False)),
    use_container_width=True
)

if summary_level != "None":
    st.subheader(f"Total Optimized Revenue by {summary_level}")

    if summary_level == "Store":
        summary_df = summarize_store_or_state(
            model=model,
            simulator_df=simulator_df,
            feature_cols=feature_cols,
            inventory_base_week=inventory_base_week,
            inventory_base_month=inventory_base_month,
            group_value=store_id,
            group_level="store",
            horizon=horizon,
            scenario=scenario,
        )
        label = store_id
    else:
        state_id = simulator_df.loc[
            (simulator_df["item_id"] == item_id) & (simulator_df["store_id"] == store_id),
            "state_id",
        ].iloc[0]
        summary_df = summarize_store_or_state(
            model=model,
            simulator_df=simulator_df,
            feature_cols=feature_cols,
            inventory_base_week=inventory_base_week,
            inventory_base_month=inventory_base_month,
            group_value=state_id,
            group_level="state",
            horizon=horizon,
            scenario=scenario,
        )
        label = state_id

    if summary_df.empty:
        st.info("No summary results available for this selection.")
    else:
        total_rev = summary_df["predicted_revenue"].sum()
        st.metric(f"Total optimized revenue for {label}", f"{total_rev:.2f}")
        st.dataframe(
            prettify_table(
                summary_df[[
                "item_id",
                "store_id",
                "candidate_price",
                "predicted_sales",
                "feasible_sales",
                "predicted_revenue",
            ]].sort_values("predicted_revenue", ascending=False)
            ),
            use_container_width=True,
        )
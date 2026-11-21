# Solar Process Heat Tool

Preliminary feasibility tool for a concentrated solar **tower with thermal storage** that supplies
industrial process heat (Egypt and MENA). Interactive Streamlit app.

## What it does
- Site and hourly solar data (NASA POWER), process heat load (MW or steam conditions)
- Tower, receiver, heliostat field and storage sizing (`core/design.py`)
- Mirror soiling from dust (`core/soiling.py`, `geometry.py`, `soiling_field.py`)
- Hourly solar heat, storage and backup simulation (`core/dispatch.py`)
- Investment cost, cost of heat, payback, CO2 (`core/economics.py`)
- A source register for every number (`data/sources.csv`, shown in the app)

## Run on your laptop (Anaconda Prompt)
    cd csp_tool
    pip install -r requirements.txt
    streamlit run app.py

## Run the tests
    python tests/test_slice1.py
    python tests/test_slice2.py
    python tests/test_slice3.py
    python tests/test_slice4.py
    python tests/test_slice5.py

## Update the deployed app
Replace the files in your GitHub repo with this folder's content (keep `app.py` at the top level),
commit, and Streamlit Cloud redeploys automatically.

## Layout
- `app.py` - interface only, no physics
- `core/` - calculations
- `data/sources.csv` - every number, its source and how far to trust it
- `tests/` - checks against hand calculations and the published reference case

## Limits
Water-based design (50 to 150 C), efficiencies from a Tucson design, illustrative dust parameters,
US small-plant costs. See the "About & limits" tab.

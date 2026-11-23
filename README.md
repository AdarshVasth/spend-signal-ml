# SpendSignal

SpendSignal is a small, local-first transaction outlier explorer. Upload a CSV to see rows that differ from patterns in that same file. It uses scikit-learn's Isolation Forest and a Streamlit interface; no paid API or model service is needed.

> A model flag is a prompt to review a transaction, not a fraud verdict. This is an educational portfolio project, not financial advice or a bank-grade fraud system.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The app opens with a deterministic synthetic dataset. Upload your own CSV in the sidebar or download the example. Your CSV is processed in the running app session and is not stored in a database.

## CSV columns

| Column | Example | Notes |
| --- | --- | --- |
| `timestamp` | `2026-11-02 14:30` | ISO date/time; parsed as UTC |
| `amount` | `349.50` | Positive numeric amount |
| `merchant` | `Cafe` | Text label |
| `category` | `Food` | Text label |
| `channel` | `Card` | Text label such as Card or Online |

At least 20 rows are required. Category labels are case-sensitive when the model learns patterns.

## Model and design

The pipeline validates the input, derives log amount and time features, one-hot encodes merchant/category/channel, then fits an `IsolationForest`. It returns the model's decision score, a review flag, and a downloadable result CSV. Lower scores are more unusual. The slider sets the model's expected outlier fraction; it is **not** an estimated probability that any row is fraud.

```text
CSV upload -> validation -> features -> Isolation Forest -> review table + CSV export
```

## Test

```bash
python -m unittest discover -s tests -v
```

## Learning guide

1. Trace `validate_transactions()` to see how data-quality checks prevent malformed values from silently entering a model.
2. Read `_features()` and explain why log amount can reduce the influence of a few very large values.
3. In `analyze_transactions()`, compare `fit_predict()` labels with `decision_function()` scores.
4. Change `contamination` and observe that the share flagged changes by design.
5. Add a feature, such as time since the previous transaction, and write a test for its edge cases.

## Limitations

- It learns from the uploaded file only. Small, biased, or already-compromised histories can produce misleading patterns.
- Merchant names and other categorical values are one-hot encoded. A new merchant can appear unusual simply because it is new.
- Anomaly detection is not supervised fraud classification. No fraud labels, calibrated probabilities, real user data, or performance benchmark are included.
- CSV upload type checks are convenience only. Do not upload sensitive information to a deployment you do not control.

## Project structure

```text
app.py                     Streamlit interface
spend_signal.py            validation, feature engineering, model
data/sample_transactions.csv  redistributable synthetic demo data
tests/                     focused unit tests
```

## References

- scikit-learn: [IsolationForest API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- Streamlit: [file uploader](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader)
- Streamlit: [dataframe](https://docs.streamlit.io/develop/api-reference/data/st.dataframe)
- Streamlit: [chart elements](https://docs.streamlit.io/develop/api-reference/charts)

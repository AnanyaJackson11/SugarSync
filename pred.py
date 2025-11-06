# ppg_realtime_predict_full_fixed.py
import joblib
import time
import json
import re
from pathlib import Path
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy import stats

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor, XGBClassifier

# ---------------- Config ----------------
COM_PORT = "COM5"        # update for your system
BAUDRATE = 115200
TIMEOUT = 1

DATA_PATH = "fin_fr.xlsx"    # used if models missing and training needed
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
REG_MODEL_FILE = MODEL_DIR / "xgb_regressor.joblib"
CLF_MODEL_FILE = MODEL_DIR / "xgb_classifier.joblib"
SCALER_FILE    = MODEL_DIR / "scaler.joblib"
ENCODERS_FILE  = MODEL_DIR / "encoders.joblib"

EXCEL_SAVE_PATH = "ppg_glucose_data_predictions.xlsx"

# sampling & window config
SAMPLING_DELAY = 10   # ms in Arduino
fs = 1000.0 / SAMPLING_DELAY
WINDOW_SIZE = 500
FEATURE_WINDOW = 300
SEGMENT_LENGTH = 600
SEGMENTS_TO_COLLECT = 10

# ppg features
ppg_feature_names = [
    "Mean", "Std", "Variance", "Skewness", "Kurtosis", "Median", "Q25", "Q75",
    "IQR", "Range", "ACDC_Ratio", "ACPP", "DC_Component", "AC_Component", "HeartRate",
    "HRV_RMSSD", "PPI_Mean", "PPI_SD", "Systolic_Peak", "Diastolic_Valley",
    "Pulse_Width", "Peak_Amplitude", "Valley_Depth", "Signal_Quality_Index"
]

# excel columns (+ predictions)
columns = [
    "DateTime", "Name", "Age", "Height", "Weight", "Gender",
    "SleepDuration", "SleepDeviation", "MealFood", "Protein", "Carbs", "Fibre", "Fat", "MealTime",
    "Diabetic", "FamilyHistory", "TimeSinceMeal",
] + ppg_feature_names + ["RawWaveform_Segment", "Timestamps_Segment", "Predicted_Glucose", "Predicted_Category"]

GLUCOSE_BINS = [0, 90, 140, np.inf]
GLUCOSE_LABELS = ["low", "med", "high"]

categorical_cols = ["Name", "Gender", "MealFood", "MealTime", "SleepDuration"]

# ---------- Helpers ----------
BINARY_MAP = {
    '1': 1, '0': 0, 'yes': 1, 'no': 0, 'y': 1, 'n': 0,
    'true': 1, 'false': 0, 't': 1, 'f': 0, 'nan_missing': 0, 'none': 0
}

def to_binary_scalar(x):
    """Convert scalar to 0/1 robustly."""
    if pd.isna(x):
        return 0
    s = str(x).strip().lower()
    if s in BINARY_MAP:
        return BINARY_MAP[s]
    try:
        num = float(s)
        return int(num) if not np.isnan(num) else 0
    except:
        pass
    m = re.search(r'(\d+)', s)
    if m:
        try:
            return int(m.group(1))
        except:
            pass
    return 0

def parse_time_to_minutes(val):
    """Parse many time formats to minutes (float)."""
    if pd.isna(val):
        return np.nan
    try:
        f = float(val)
        return f
    except Exception:
        pass
    s = str(val).strip()
    try:
        parts = [p for p in s.split(':') if p != ""]
        if len(parts) == 1:
            return float(parts[0])
        if len(parts) == 2:
            h = float(parts[0]); m = float(parts[1])
            return h*60.0 + m
        if len(parts) == 3:
            h = float(parts[0]); m = float(parts[1]); sec = float(parts[2])
            return h*60.0 + m + sec/60.0
    except Exception:
        pass
    try:
        td = pd.to_timedelta(s)
        return td.total_seconds() / 60.0
    except Exception:
        return np.nan

def ensure_numeric_X(X):
    """Convert any non-numeric columns to numeric via LabelEncoder (safety net)."""
    non_numeric = X.select_dtypes(exclude=[np.number, 'category', 'bool']).columns.tolist()
    if non_numeric:
        print("Converting non-numeric columns to numeric via LabelEncoder:", non_numeric)
        for c in non_numeric:
            X[c] = X[c].astype(str).fillna("nan_missing")
            X[c] = LabelEncoder().fit_transform(X[c])
    return X

# ---------- Feature extraction ----------
def safe_calculate(func, default=0):
    try:
        result = func()
        return result if not (np.isnan(result) or np.isinf(result)) else default
    except:
        return default

def preprocess_signal(signal):
    if len(signal) < 10:
        return np.array(signal)
    signal = np.array(signal)
    signal = signal - np.mean(signal)
    window_size = max(3, min(len(signal)//10, int(fs * 0.05)))
    if len(signal) >= window_size and window_size > 0:
        kernel = np.ones(window_size) / window_size
        filtered = np.convolve(signal, kernel, mode='same')
        return filtered
    return signal

def create_default_features():
    return {name: 0 for name in ppg_feature_names}

def extract_comprehensive_features(signal, timestamps=None):
    if len(signal) < 20:
        return create_default_features()
    try:
        signal = np.array(signal)
        processed = preprocess_signal(signal)
        f = {}
        f['Mean'] = safe_calculate(lambda: np.mean(processed))
        f['Std'] = safe_calculate(lambda: np.std(processed))
        f['Variance'] = safe_calculate(lambda: np.var(processed))
        f['Skewness'] = safe_calculate(lambda: stats.skew(processed))
        f['Kurtosis'] = safe_calculate(lambda: stats.kurtosis(processed))
        f['Median'] = safe_calculate(lambda: np.median(processed))
        f['Q25'] = safe_calculate(lambda: np.percentile(processed, 25))
        f['Q75'] = safe_calculate(lambda: np.percentile(processed, 75))
        f['IQR'] = f['Q75'] - f['Q25']
        f['Range'] = safe_calculate(lambda: np.max(processed) - np.min(processed))
        dc_component = safe_calculate(lambda: np.mean(processed))
        ac_component = safe_calculate(lambda: np.max(processed) - np.min(processed))
        f['DC_Component'] = dc_component
        f['AC_Component'] = ac_component
        f['ACPP'] = ac_component
        f['ACDC_Ratio'] = safe_calculate(lambda: ac_component / dc_component if abs(dc_component) > 1e-6 else 0)
        # peaks & HR
        try:
            min_distance = max(1, int(fs * 0.4))
            prominence = max(0.1, np.std(processed)*0.3)
            peaks, props = find_peaks(processed, distance=min_distance, prominence=prominence)
            if len(peaks) > 1:
                peak_intervals = np.diff(peaks) / fs
                peak_intervals = peak_intervals[peak_intervals > 0]
                if len(peak_intervals) > 0:
                    heart_rate = safe_calculate(lambda: 60.0 / np.mean(peak_intervals))
                    f['HeartRate'] = min(200, max(40, heart_rate))
                    f['PPI_Mean'] = safe_calculate(lambda: np.mean(peak_intervals) * 1000)
                    f['PPI_SD'] = safe_calculate(lambda: np.std(peak_intervals) * 1000)
                    f['HRV_RMSSD'] = safe_calculate(lambda: np.sqrt(np.mean(np.diff(peak_intervals)**2)) * 1000)
                    peak_ampls = processed[peaks]
                    f['Peak_Amplitude'] = safe_calculate(lambda: np.mean(peak_ampls))
                    f['Systolic_Peak'] = safe_calculate(lambda: np.max(peak_ampls))
                    f['Pulse_Width'] = safe_calculate(lambda: np.mean(np.diff(peaks)) / fs * 1000)
                else:
                    f.update({k:0 for k in ['HeartRate','PPI_Mean','PPI_SD','HRV_RMSSD','Peak_Amplitude','Systolic_Peak','Pulse_Width']})
            else:
                f.update({k:0 for k in ['HeartRate','PPI_Mean','PPI_SD','HRV_RMSSD','Peak_Amplitude','Systolic_Peak','Pulse_Width']})
        except Exception:
            f.update({k:0 for k in ['HeartRate','PPI_Mean','PPI_SD','HRV_RMSSD','Peak_Amplitude','Systolic_Peak','Pulse_Width']})
        # valleys
        try:
            valleys, _ = find_peaks(-processed, distance=max(1, int(fs * 0.4)))
            if len(valleys) > 0:
                valley_depths = processed[valleys]
                f['Valley_Depth'] = safe_calculate(lambda: np.mean(valley_depths))
                f['Diastolic_Valley'] = safe_calculate(lambda: np.min(valley_depths))
            else:
                f['Valley_Depth'] = f['Diastolic_Valley'] = safe_calculate(lambda: np.min(processed))
        except:
            f['Valley_Depth'] = f['Diastolic_Valley'] = 0
        # signal quality
        try:
            peaks_len = len(peaks) if 'peaks' in locals() else 0
            if peaks_len > 2:
                peak_intervals = np.diff(peaks)
                if len(peak_intervals) > 0 and np.mean(peak_intervals) > 0:
                    regularity = 1 - min(1, np.std(peak_intervals) / np.mean(peak_intervals))
                    f['Signal_Quality_Index'] = max(0, min(1, regularity))
                else:
                    f['Signal_Quality_Index'] = 0
            else:
                f['Signal_Quality_Index'] = 0
        except:
            f['Signal_Quality_Index'] = 0
        # ensure all features present
        for feat in ppg_feature_names:
            if feat not in f:
                f[feat] = 0
        return f
    except Exception as e:
        print("Feature extraction error:", e)
        return create_default_features()

# ---------- Model utils ----------
def train_models_from_data(data_path):
    print("Training models from", data_path)
    if not Path(data_path).exists():
        raise FileNotFoundError("Training data not found; provide trained models in 'models/' or create training data at DATA_PATH.")

    df = pd.read_excel(data_path)

    if "GlucoseLevel" not in df.columns:
        raise ValueError("Training data must contain 'GlucoseLevel' column.")

    # ensure required cols exist
    required_cols = ppg_feature_names + ["Name","Age","Height","Weight","Gender","SleepDuration","MealTime","Protein","Carbs","Fibre","Fat","TimeSinceMeal","Diabetic","FamilyHistory","MealFood","SleepDeviation"]
    for c in required_cols:
        if c not in df.columns:
            df[c] = 0

    # parse times -> minutes
    df['MealTime_minutes'] = df['MealTime'].apply(parse_time_to_minutes) if 'MealTime' in df.columns else 0
    df['SleepDuration_minutes'] = df['SleepDuration'].apply(parse_time_to_minutes) if 'SleepDuration' in df.columns else 0
    df['TimeSinceMeal'] = pd.to_numeric(df['TimeSinceMeal'], errors='coerce')

    # coerce ppg feature cols numeric
    for c in ppg_feature_names:
        df[c] = pd.to_numeric(df.get(c, 0), errors='coerce').fillna(0)

    # coerce numeric user input cols
    for c in ["Age","Height","Weight","Protein","Carbs","Fibre","Fat"]:
        df[c] = pd.to_numeric(df.get(c, 0), errors='coerce')
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    # convert Diabetic / FamilyHistory to numeric 0/1
    for bin_col in ['Diabetic', 'FamilyHistory']:
        if bin_col in df.columns:
            df[bin_col] = df[bin_col].apply(to_binary_scalar).astype(int)
        else:
            df[bin_col] = 0

    # label-encode categorical columns and store encoders
    encoders = {}
    for c in categorical_cols:
        df[c] = df[c].astype(str).fillna("nan_missing")
        le = LabelEncoder()
        try:
            df[c] = le.fit_transform(df[c])
        except Exception:
            df[c] = df[c].astype('category').cat.codes
            le = None
        encoders[c] = le

    # prepare numeric_for_scaling and fit scaler
    numeric_for_scaling = [c for c in (ppg_feature_names + ["Age","Height","Weight","Protein","Carbs","Fibre","Fat","TimeSinceMeal","MealTime_minutes","SleepDuration_minutes"]) if c in df.columns]
    scaler = None
    if numeric_for_scaling:
        for c in numeric_for_scaling:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            if df[c].isna().any():
                df[c] = df[c].fillna(df[c].median())
        scaler = StandardScaler()
        scaler.fit(df[numeric_for_scaling])
        df[numeric_for_scaling] = scaler.transform(df[numeric_for_scaling])

    # build X & y
    X_cols = ["Name","Age","Height","Weight","Gender","SleepDuration_minutes","SleepDeviation","MealFood","Protein","Carbs","Fibre","Fat","MealTime_minutes","Diabetic","FamilyHistory","TimeSinceMeal"] + ppg_feature_names
    for c in X_cols:
        if c not in df.columns:
            df[c] = 0
    X = df[X_cols].copy()

    # safety: convert any remaining non-numeric columns
    X = ensure_numeric_X(X)

    y_reg = pd.to_numeric(df["GlucoseLevel"], errors='coerce').fillna(0).astype(float)
    y_clf = pd.cut(y_reg, bins=GLUCOSE_BINS, labels=GLUCOSE_LABELS).astype(str)

    # regression training
    Xr_train, Xr_val, yr_train, yr_val = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    reg = XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
    reg.fit(Xr_train, yr_train, eval_set=[(Xr_val, yr_val)], verbose=False)
    print("Regressor val MAE:", mean_absolute_error(yr_val, reg.predict(Xr_val)))

    # classifier training
    le_target = LabelEncoder()
    y_train_c = le_target.fit_transform(y_clf.fillna("nan_missing"))
    Xc_train, Xc_val, yc_train, yc_val = train_test_split(X, y_train_c, test_size=0.2, random_state=42, stratify=y_train_c if len(np.unique(y_train_c))>1 else None)
    clf = XGBClassifier(objective='multi:softprob', num_class=len(le_target.classes_), n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='mlogloss')
    clf.fit(Xc_train, yc_train, eval_set=[(Xc_val, yc_val)], verbose=False)

    # save artifacts
    joblib.dump(reg, REG_MODEL_FILE)
    joblib.dump(clf, CLF_MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    encoders['target_encoder'] = le_target
    joblib.dump(encoders, ENCODERS_FILE)

    print("Training complete. Models saved to", MODEL_DIR)
    return reg, clf, scaler, encoders

def load_models_if_exist():
    if REG_MODEL_FILE.exists() and CLF_MODEL_FILE.exists() and ENCODERS_FILE.exists():
        reg = joblib.load(REG_MODEL_FILE)
        clf = joblib.load(CLF_MODEL_FILE)
        scaler = joblib.load(SCALER_FILE) if SCALER_FILE.exists() else None
        encoders = joblib.load(ENCODERS_FILE)
        return reg, clf, scaler, encoders
    return None, None, None, None

def prepare_input_row_for_model(row_df, encoders, scaler):
    df = row_df.copy()

    # parse times -> minutes
    df['MealTime_minutes'] = df['MealTime'].apply(parse_time_to_minutes) if 'MealTime' in df.columns else 0
    df['SleepDuration_minutes'] = df['SleepDuration'].apply(parse_time_to_minutes) if 'SleepDuration' in df.columns else 0
    df['TimeSinceMeal'] = pd.to_numeric(df['TimeSinceMeal'], errors='coerce').fillna(0)

    # normalize categorical strings
    for c in categorical_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("nan_missing")
        else:
            df[c] = "nan_missing"

    # convert Diabetic / FamilyHistory to numeric 0/1
    for bin_col in ['Diabetic', 'FamilyHistory']:
        if bin_col in df.columns:
            df[bin_col] = df[bin_col].apply(to_binary_scalar).astype(int)
        else:
            df[bin_col] = 0

    # apply encoders (if available)
    if encoders is not None:
        for c, le in encoders.items():
            if c == 'target_encoder':
                continue
            if c in df.columns:
                vals = df[c].astype(str).tolist()
                if le is not None:
                    known = set(le.classes_.tolist())
                    mapped = [v if v in known else ("nan_missing" if "nan_missing" in known else le.classes_[0]) for v in vals]
                    df[c] = le.transform(mapped)
                else:
                    df[c] = pd.Categorical(df[c]).codes
            else:
                df[c] = 0
    else:
        for c in categorical_cols:
            if c in df.columns:
                df[c] = df[c].astype(str).fillna("nan_missing")
                df[c] = LabelEncoder().fit_transform(df[c])
            else:
                df[c] = 0

    # scale numeric fields if scaler exists
    numeric_for_scaling = [c for c in (ppg_feature_names + ["Age","Height","Weight","Protein","Carbs","Fibre","Fat","TimeSinceMeal","MealTime_minutes","SleepDuration_minutes"]) if c in df.columns]
    if scaler is not None and numeric_for_scaling:
        for c in numeric_for_scaling:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        df[numeric_for_scaling] = scaler.transform(df[numeric_for_scaling])
    else:
        for c in numeric_for_scaling:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    ordered_cols = ["Name","Age","Height","Weight","Gender","SleepDuration_minutes","SleepDeviation","MealFood","Protein","Carbs","Fibre","Fat","MealTime_minutes","Diabetic","FamilyHistory","TimeSinceMeal"] + ppg_feature_names
    for c in ordered_cols:
        if c not in df.columns:
            df[c] = 0

    X_ready = df[ordered_cols].copy()
    X_ready = ensure_numeric_X(X_ready)  # final safety
    return X_ready

# ---------- Main: serial read, feature extraction, predict ----------
def main():
    import serial  # imported inside to allow top-level linting without pyserial
    print("Enter user info (press Enter to skip / default).")
    name = input("Enter Name: ").strip() or "unknown"
    age = input("Enter Age: ").strip() or 0
    height = input("Enter Height (cm): ").strip() or 0
    weight = input("Enter Weight (kg): ").strip() or 0
    gender = input("Enter Gender (M/F): ").strip() or "U"
    sleep_duration = input("Enter Sleep Duration (hh:mm): ").strip() or "00:00"
    sleep_deviation = input("Deviation from circadian rhythm? (0/1): ").strip() or "0"
    meal_food = input("Enter Food Name(s): ").strip() or "none"
    meal_protein = input("Enter Protein (g): ").strip() or 0
    meal_carbs = input("Enter Carbs (g): ").strip() or 0
    meal_fibre = input("Enter Fibre (g): ").strip() or 0
    meal_fat = input("Enter Fat (g): ").strip() or 0
    meal_time = input("Enter Meal Time (hh:mm:ss): ").strip() or "00:00:00"
    diabetic = input("Diabetic? (1/0): ").strip() or "0"
    family_history = input("Family History of Diabetes? (1/0): ").strip() or "0"
    measurement_time = input("Time since last meal (minutes): ").strip() or 0

    # load or train models
    reg, clf, scaler, encoders = load_models_if_exist()
    if reg is None or clf is None:
        print("Models not found. Attempting to train from", DATA_PATH)
        try:
            reg, clf, scaler, encoders = train_models_from_data(DATA_PATH)
        except Exception as e:
            print("Could not train models automatically:", e)
            print("Please train models first or place models in the 'models/' folder.")
            return

    # prepare excel storage
    if Path(EXCEL_SAVE_PATH).exists():
        df_excel = pd.read_excel(EXCEL_SAVE_PATH, engine="openpyxl")
    else:
        df_excel = pd.DataFrame(columns=columns)

    # open serial
    try:
        ser = serial.Serial(COM_PORT, BAUDRATE, timeout=TIMEOUT)
        time.sleep(2)
        print(f"Opened serial {COM_PORT} at {BAUDRATE} baud.")
    except Exception as e:
        print("Could not open serial port:", e)
        return

    times = deque(maxlen=WINDOW_SIZE)
    values = deque(maxlen=WINDOW_SIZE)
    curr_segment = []
    curr_times = []
    segment_count = 0
    sample_count = 0

    print("Starting live collection & prediction. Press Ctrl+C to stop.")
    try:
        while segment_count < SEGMENTS_TO_COLLECT:
            line = ser.readline().decode(errors='ignore').strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 2:
                continue
            try:
                t = int(parts[0])
                v = float(parts[1])
            except:
                continue

            times.append(t)
            values.append(v)
            curr_segment.append(v)
            curr_times.append(t)
            sample_count += 1

            if len(values) >= FEATURE_WINDOW and sample_count % 50 == 0:
                window_features = extract_comprehensive_features(list(values)[-FEATURE_WINDOW:])
                print(f"[live] HR={window_features.get('HeartRate',0):.1f} AC/DC={window_features.get('ACDC_Ratio',0):.3f} SQI={window_features.get('Signal_Quality_Index',0):.2f}")

            if len(curr_segment) >= SEGMENT_LENGTH:
                print(f"Segment {segment_count+1} complete. Extracting features & predicting...")
                feats = extract_comprehensive_features(curr_segment, curr_times)
                row = {
                    "DateTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Name": name, "Age": age, "Height": height, "Weight": weight, "Gender": gender,
                    "SleepDuration": sleep_duration, "SleepDeviation": sleep_deviation,
                    "MealFood": meal_food, "Protein": meal_protein, "Carbs": meal_carbs,
                    "Fibre": meal_fibre, "Fat": meal_fat, "MealTime": meal_time,
                    "Diabetic": diabetic, "FamilyHistory": family_history, "TimeSinceMeal": measurement_time,
                }
                for k in ppg_feature_names:
                    row[k] = feats.get(k, 0)
                row["RawWaveform_Segment"] = json.dumps(curr_segment[:500])
                row["Timestamps_Segment"] = json.dumps(curr_times[:500])

                row_df = pd.DataFrame([row])
                X_ready = prepare_input_row_for_model(row_df, encoders, scaler)

                try:
                    pred_glucose = float(reg.predict(X_ready)[0])
                except Exception as e:
                    print("Regressor prediction failed:", e)
                    pred_glucose = np.nan
                try:
                    proba = clf.predict_proba(X_ready)[0]
                    target_le = encoders.get('target_encoder') if encoders else None
                    if target_le is not None:
                        classes = target_le.inverse_transform(np.arange(len(proba)))
                    else:
                        classes = GLUCOSE_LABELS
                    pred_idx = int(np.argmax(proba))
                    pred_class = classes[pred_idx] if len(classes) > pred_idx else GLUCOSE_LABELS[pred_idx]
                except Exception as e:
                    print("Classifier prediction failed:", e)
                    proba = None
                    pred_class = None

                row["Predicted_Glucose"] = pred_glucose
                row["Predicted_Category"] = pred_class if pred_class is not None else pd.NA

                # print results
                print(">>> Prediction results:")
                print(f" Predicted glucose (regression): {pred_glucose:.2f} mg/dL")
                if proba is not None:
                    for cls, p in zip(classes, proba):
                        print(f"  {cls}: {p:.3f}")
                    print(" Predicted category (model):", pred_class)
                else:
                    print(" Predicted category unavailable")
                print("-" * 40)

                df_excel = pd.concat([df_excel, pd.DataFrame([row])], ignore_index=True)
                df_excel.to_excel(EXCEL_SAVE_PATH, index=False, engine="openpyxl")
                print(f"Saved prediction to {EXCEL_SAVE_PATH} (total records: {len(df_excel)})")

                curr_segment = []
                curr_times = []
                segment_count += 1
                time.sleep(0.5)

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        try:
            ser.close()
        except:
            pass
        print("Exiting. Final save done.")

if __name__ == "__main__":
    main()

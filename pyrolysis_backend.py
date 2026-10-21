import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor

# นำเข้าอัลกอริทึม
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor

# ==========================================
# 1. ฟังก์ชันตัวช่วย (ต้องไว้นอกสุดเพื่อให้ joblib เซฟโมเดลได้)
# ==========================================
def divide_by_100(X):
    return X / 100.0

# ==========================================
# 2. การตั้งค่ากลุ่มตัวแปรและรายชื่อ Target ทั้งหมด
# ==========================================
# กลุ่มที่ 1: หาร 100 (เป็นเปอร์เซ็นต์)
pct_cols = ['LB', 'WT', 'TPO', 'PP', 'PS', 'PE', 'PVC', 'PET', 'Catalyst (%)']
# กลุ่มที่ 2: ใช้ RobustScaler (จัดการ Outlier)
robust_cols = ['C', 'H', 'N', 'S', 'O', 'Ash', 'FC', 'VM', 'M', 'HHT (C)', 'Ti (min)']

# รายชื่อผลผลิตทั้งหมด เพื่อนำไปตัดออกจาก X (ป้องกัน Data Leakage)
ALL_TARGETS = [
    'Oil (%)', 'Char (%)', 'Gas (%)', 
    'Syield Wt%', 'Sg (15.6)Yield', 'CnYield (PPM)', 'VcYield', 
    'Gcv (KJ/Kg)Yield', 'FpYield', 'Cl (ppm)Yield', 'Dt (90%)Yield'
]

# ==========================================
# 3. กำหนด Configuration ของโมเดลตามที่คุณระบุ
# ==========================================
MODELS_CONFIG = {
    # 1. โมเดลทำนาย Oil, Char, Gas (Multi-output XGBoost)
    "Yields_Multi": {
        "targets": ['Oil (%)', 'Char (%)', 'Gas (%)'],
        "test_size": 0.25, # Train 75 / Test 25
        "model": MultiOutputRegressor(
            XGBRegressor(
                learning_rate=0.05, 
                max_depth=3, 
                n_estimators=700, 
                subsample=0.8, 
                random_state=42,
                n_jobs=-1
            )
        )
    },
    # 2. โมเดลทำนาย Sulphur (XGBoost)
    "Sulphur": {
        "targets": ['Syield Wt%'],
        "test_size": 0.20, # Train 80 / Test 20
        "model": XGBRegressor(
            learning_rate=0.05, 
            max_depth=3, 
            n_estimators=200, 
            subsample=1.0, 
            random_state=42,
            n_jobs=-1
        )
    },
    # 3. โมเดลทำนาย Cetane (Neural Network / MLPRegressor)
    "Cetane": {
        "targets": ['CnYield (PPM)'],
        "test_size": 0.30, # Train 70 / Test 30
        "model": MLPRegressor(
            activation='tanh', 
            alpha=0.0001, 
            hidden_layer_sizes=(50,), 
            learning_rate_init=0.01, 
            max_iter=3000, 
            early_stopping=True,
            random_state=42
        )
    },
    # 4. โมเดลทำนาย Chlorine (XGBoost)
    "Chlorine": {
        "targets": ['Cl (ppm)Yield'],
        "test_size": 0.20, # Train 80 / Test 20
        "model": XGBRegressor(
            learning_rate=0.05,  
            max_depth=4,         
            n_estimators=300,    
            subsample=1.0,       
            random_state=42, 
            objective='reg:squarederror',
            n_jobs=-1
        )
    }
}

# ==========================================
# 4. ฟังก์ชันหลักสำหรับเทรนและสร้างไฟล์ Model
# ==========================================
def build_backend_models():
    print("🚀 เริ่มกระบวนการสร้างโมเดลสำหรับ Backend...")
    df = pd.read_csv('Prepared_Data_Pyrolysis_9_Imputed.csv')
    
    # สร้างโฟลเดอร์สำหรับเก็บโมเดล .pkl
    export_dir = 'backend_models'
    os.makedirs(export_dir, exist_ok=True)
    
    for name, config in MODELS_CONFIG.items():
        print(f"\nกำลังเทรนโมเดล: {name} | เป้าหมาย: {config['targets']}")
        
        # 1. ตัดแถวที่ค่า Target ว่างออก
        df_clean = df.dropna(subset=config['targets'])
        
        # 2. แยก X และ y (ลบ Target ออกจาก X ทั้งหมด)
        X = df_clean.drop(columns=[col for col in ALL_TARGETS if col in df_clean.columns])
        
        # ลบคอลัมน์ ID / Ref ออกถ้ามี
        cols_to_drop = ['Ref no.', 'No.']
        X = X.drop(columns=[col for col in cols_to_drop if col in X.columns])
        
        # กำหนดแกน y
        if len(config['targets']) > 1:
            y = df_clean[config['targets']]
        else:
            y = df_clean[config['targets'][0]]
            
        # 3. แบ่งชุดข้อมูล Train / Test ตามสัดส่วนของแต่ละโมเดล
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config['test_size'], random_state=42
        )
        
        # 4. ตรวจสอบคอลัมน์สำหรับการทำ Scaling
        valid_pct_cols = [col for col in pct_cols if col in X_train.columns]
        valid_robust_cols = [col for col in robust_cols if col in X_train.columns]
        
        # 5. สร้าง Preprocessor (Scaling)
        preprocessor = ColumnTransformer(
            transformers=[
                ('pct', FunctionTransformer(divide_by_100, validate=False), valid_pct_cols),
                ('robust', RobustScaler(), valid_robust_cols)
            ],
            remainder='passthrough'
        )
        
        # 6. มัดรวม Scaling กับ Model ให้เป็นท่อ (Pipeline) เดียว
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', config['model'])
        ])
        
        # 7. สั่งฝึกสอนโมเดล (Train)
        # ให้โมเดลเรียนรู้และปรับสเกล "เฉพาะชุด Train" ตามที่คุณต้องการ
        pipeline.fit(X_train, y_train)
        
        # 8. บันทึกโมเดลเป็นไฟล์ .pkl (Export)
        model_filename = os.path.join(export_dir, f"{name}_model.pkl")
        joblib.dump(pipeline, model_filename)
        print(f"✅ บันทึกสำเร็จ: {model_filename}")

    print(f"\n🎉 สร้างโมเดลเสร็จสมบูรณ์! ไฟล์ทั้งหมดอยู่ในโฟลเดอร์ '{export_dir}'")

if __name__ == "__main__":
    build_backend_models()
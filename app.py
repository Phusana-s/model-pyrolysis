from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app)

# ==========================================
# 1. ฟังก์ชันจำเป็นที่ต้องมี (เพื่อให้ Joblib โหลด Pipeline ที่มีฟังก์ชันนี้ได้)
# ==========================================
def divide_by_100(X):
    return X / 100.0

# ==========================================
# 2. โหลดโมเดล .pkl และ Database สำหรับทำ Auto-fill
# ==========================================
models = {}
# ชื่อ Key จะต้องตรงกับตอนเซฟใน MODELS_CONFIG ของ pyrolysis_backend.py
model_names = ['Yields_Multi', 'Sulphur', 'Cetane', 'Chlorine']
model_dir = 'backend_models' # โฟลเดอร์ที่คุณเซฟโมเดลไว้

print("กำลังโหลดโมเดลเข้าสู่ระบบ...")
for name in model_names:
    model_path = os.path.join(model_dir, f"{name}_model.pkl")
    if os.path.exists(model_path):
        models[name] = joblib.load(model_path)
    else:
        print(f" ❌ ไม่พบไฟล์ {model_path}")

# โหลด Database อ้างอิงเพื่อใช้หา "การทดลองที่ใกล้เคียงที่สุด" (KNN)
print("กำลังโหลดฐานข้อมูลอ้างอิงทางเคมี...")
df_ref = pd.read_csv('Prepared_Data_Pyrolysis_9_Imputed.csv')
search_features = ['LB', 'WT', 'TPO', 'PP', 'PS', 'PE', 'PVC', 'PET', 'Catalyst (%)', 'HHT (C)', 'Ti (min)']
ref_matrix = df_ref[search_features].values

EXPECTED_FEATURES = search_features + ['C', 'H', 'N', 'S', 'O', 'Ash', 'FC', 'VM', 'M']

# ==========================================
# 3. สร้าง API Endpoint สำหรับรับข้อมูลและทำนายผล
# ==========================================
@app.route('/predict', methods=['POST'])
def predict_pyrolysis():
    try:
        data = request.json
        inferred_chem = {}
        
        # เช็คว่าผู้ใช้กรอกค่าเคมีมาหรือไม่ (ดูจาก C และ Ash)
        has_chemistry = ('C' in data) and ('Ash' in data)
        
        if not has_chemistry:
            # 🌟 โหมด Auto-fill: ค้นหาการทดลองที่อิงจาก สัดส่วนวัตถุดิบ+อุณหภูมิ+เวลา ที่เหมือนกันที่สุด
            input_vec = np.array([[float(data.get(f, 0.0)) for f in search_features]])
            
            # คำนวณระยะห่าง (Euclidean Distance)
            distances = np.linalg.norm(ref_matrix - input_vec, axis=1)
            closest_idx = np.argmin(distances) # หาแถวที่ระยะห่างน้อยที่สุด (ใกล้เคียงสุด)
            closest_row = df_ref.iloc[closest_idx]
            
            # ดึงค่าเคมีจากแถวนั้นมาใช้
            inferred_chem = {
                'C': closest_row['C'], 'H': closest_row['H'], 'N': closest_row['N'],
                'S': closest_row['S'], 'O': closest_row['O'], 'Ash': closest_row['Ash'],
                'FC': closest_row['FC'], 'VM': closest_row['VM'], 'M': closest_row['M']
            }
            # อัปเดตทับลงไปในข้อมูลที่รับมา
            data.update(inferred_chem)

        # สร้าง DataFrame 1 แถว ที่พร้อมป้อนเข้าโมเดล
        input_dict = {feature: float(data.get(feature, 0.0)) for feature in EXPECTED_FEATURES}
        df_input = pd.DataFrame([input_dict])
        
        results = {}
        
        # ----------------------------------------------------
        # ทำนายผล
        # ----------------------------------------------------
        # 1. ทำนาย Oil, Char, Gas ด้วยโมเดล Yields_Multi (ได้ค่ากลับมาเป็น Array 3 ค่า)
        if 'Yields_Multi' in models:
            multi_preds = models['Yields_Multi'].predict(df_input)[0]
            results['Oil (%)'] = float(multi_preds[0])
            results['Char (%)'] = float(multi_preds[1])
            results['Gas (%)'] = float(multi_preds[2])
            
        # 2. ทำนาย Sulphur
        if 'Sulphur' in models:
            results['Syield Wt%'] = float(models['Sulphur'].predict(df_input)[0])
            
        # 3. ทำนาย Cetane
        if 'Cetane' in models:
            results['CnYield (PPM)'] = float(models['Cetane'].predict(df_input)[0])
            
        # 4. ทำนาย Chlorine
        if 'Chlorine' in models:
            results['Cl (ppm)Yield'] = float(models['Chlorine'].predict(df_input)[0])
        
        return jsonify({
            "status": "success",
            "results": results,
            "inferred_chemistry": inferred_chem  # ส่งข้อมูลที่ Auto-fill กลับไปบอกหน้าเว็บด้วย
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# ==========================================
# 4. สั่งรันเซิร์ฟเวอร์
# ==========================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
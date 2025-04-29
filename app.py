from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Kategorilere göre input alanları
CATEGORY_FIELDS = {
    "tablet": ["ram", "hafiza", "ekran_boyutu", "marka", "model", "isletim_sistemi"],
    "telefon": ["ram", "hafiza", "ekran_boyutu", "kamera_mp", "marka", "model", "isletim_sistemi"],
    "monitor": ["ekran_boyutu", "yenileme_hizi", "marka", "panel_tipi"],
    "headset": ["frekans_araligi", "empedans", "marka", "tip"],
    "akilli_saat": ["ekran_boyutu", "batarya_kapasitesi", "marka", "isletim_sistemi"],
    "laptop": ["ram", "hafiza", "ekran_boyutu", "islemci_hizi", "marka", "isletim_sistemi"]
}

@app.route('/', methods=['GET', 'POST'])
def index():
    categories = list(CATEGORY_FIELDS.keys())
    if request.method == 'POST':
        category = request.form.get('category')
        fields = CATEGORY_FIELDS[category]
        return render_template('home.html', category=category, fields=fields)
    return render_template('index.html', categories=categories)

@app.route('/predictdata', methods=['POST'])
def predict_datapoint():
    category = request.form.get('category')
    fields = CATEGORY_FIELDS[category]
    # Formdan gelen verileri al
    data_kwargs = {field: request.form.get(field) for field in fields}
    # Sayısal alanları float'a çevir
    for key in data_kwargs:
        try:
            data_kwargs[key] = float(data_kwargs[key])
        except (ValueError, TypeError):
            pass  # Kategorik alanlar string kalır

    data = CustomData(category, **data_kwargs)
    pred_df = data.get_data_as_data_frame()
    predict_pipeline = PredictPipeline(category)
    results = predict_pipeline.predict(pred_df)
    return render_template('home.html', category=category, fields=fields, results=results[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
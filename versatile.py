from flask import Flask, request, jsonify, render_template
import pandas as pd
import nltk
import re
import os
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import random

# Download NLTK data (sekali saja, tidak setiap request)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

app = Flask(__name__)

MODEL_PATH = 'mobil_model.pkl'

# ==========================
# 1. Fungsi untuk Load & Preprocess Dataset
# ==========================
def build_model():
    print("🔄 Membaca dan preprocessing dataset mobilbekas.csv...")
    df = pd.read_csv('mobilbekas.csv')
    df = df.dropna(subset=['Harga', 'Merek', 'Model', 'Tahun', 'Tipe bodi', 'Tipe bahan bakar', 'Transmisi'])
    df = df[~df['Model'].str.contains('Lain-lain', case=False, na=False)]
    df = df[~df['Varian'].str.contains('Lain-lain', case=False, na=True)]
    df['Harga'] = pd.to_numeric(df['Harga'], errors='coerce')
    df['Tahun'] = pd.to_numeric(df['Tahun'], errors='coerce', downcast='integer')

    def parse_mileage(mileage):
        if isinstance(mileage, str) and '-' in mileage:
            try:
                low, high = map(int, mileage.replace(' km', '').replace('.', '').split('-'))
                return (low + high) / 2
            except:
                return float('inf')
        return float('inf')

    df['Jarak tempuh (km)'] = df['Jarak tempuh'].apply(parse_mileage)
    df = df[df['Harga'].notnull() & df['Tahun'].notnull() & (df['Jarak tempuh (km)'] != float('inf'))]

    # Normalisasi tipe bodi untuk beberapa model populer
    df.loc[df['Model'].str.contains('Mobilio', case=False, na=False), 'Tipe bodi'] = 'MPV'
    df.loc[df['Model'].str.contains('Ertiga', case=False, na=False), 'Tipe bodi'] = 'MPV'
    df.loc[df['Model'].str.contains('Xpander', case=False, na=False), 'Tipe bodi'] = 'MPV'
    return df

# ==========================
# 2. Load Model dari Cache atau Buat Baru
# ==========================
if os.path.exists(MODEL_PATH):
    print("✅ Memuat model mobil dari cache...")
    with open(MODEL_PATH, 'rb') as f:
        df = pickle.load(f)
else:
    df = build_model()
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(df, f)
    print(f"✅ Model disimpan ke cache: {MODEL_PATH}")

# ==========================
# 3. Lexicon & Preprocessing Tools
# ==========================
lexicon = {
    'family': ['keluarga', 'family', 'mpv', 'minibus', 'minivan'],
    'suv': ['suv', 'jeep', 'tangguh', 'off road'],
    'sedan': ['sedan', 'anak muda'],
    'hatchback': ['hatchback', 'hatch back', 'city car', 'anak muda','kecil','harian'],
    'pickup': ['pick-up', 'pickup', 'truk', 'double cabin', 'niaga'],
    'coupe': ['coupe', 'sport', 'sports', 'super car', 'anak muda'],
    'youthful': ['anak muda', 'muda', 'trendy', 'stylish', 'keren', 'sporty'],
    'comfortable': ['nyaman', 'comfortable', 'mewah', 'luxury', 'prestige'],
    'budget': ['budget', 'harga', 'jtan', 'jt', 'juta', 'miliar'],
    'fuel': {
        'bensin': ['bensin', 'petrol', 'gasoline'],
        'diesel': ['diesel','solar','irit'],
        'hybrid': ['hybrid', 'listrik','irit']
    },
    'transmission': {
        'automatic': ['automatic', 'otomatis', 'auto', 'triptonic', 'matik', 'at','a/t','matic', 'metik'],
        'manual': ['manual','m/t','mt']
    },
    'year': ['tahun', 'year']
}

stop_words = set(stopwords.words('indonesian') + stopwords.words('english') + ['yang', 'untuk', 'dengan'])
lemmatizer = WordNetLemmatizer()

# ==========================
# 4. Fungsi NLP & Filter (tidak berubah)
# ==========================
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens

def extract_budget(text):
    budget_pattern = r'(\d+\.?\d*)\s*(jtan|jt|juta|miliar)'
    match = re.search(budget_pattern, text, re.IGNORECASE)
    if match:
        budget_value = float(match.group(1))
        unit = match.group(2).lower()
        if unit in ['juta', 'jt', 'jtan']:
            budget_value *= 1_000_000
        elif unit == 'miliar':
            budget_value *= 1_000_000_000
        return int(budget_value)
    return None

def extract_year(text):
    year_pattern = r'tahun\s*(>|>=|<|<=|=)\s*(\d{4})'
    match = re.search(year_pattern, text, re.IGNORECASE)
    if match:
        operator, year = match.group(1), int(match.group(2))
        return operator, year
    return None, None

def parse_query(text):
    tokens = preprocess_text(text)
    budget = extract_budget(text)
    year_operator, year_value = extract_year(text)
    car_type = None
    fuel_type = None
    transmission = None
    is_comfortable = False
    is_youthful = False

    for key, values in lexicon.items():
        if key in ['family', 'suv', 'sedan', 'hatchback', 'pickup', 'coupe']:
            for value in values:
                if value in text.lower():
                    car_type = key
                    break
        elif key == 'youthful':
            for value in values:
                if value in text.lower():
                    is_youthful = True
                    break
        elif key == 'comfortable':
            for value in values:
                if value in text.lower():
                    is_comfortable = True
                    break
        elif key == 'fuel':
            for fuel, fuel_values in values.items():
                for value in fuel_values:
                    if value in text.lower():
                        fuel_type = fuel
                        break
        elif key == 'transmission':
            for trans, trans_values in values.items():
                for value in trans_values:
                    if value in text.lower():
                        transmission = trans
                        break

    if car_type == 'family':
        car_type = 'mpv'

    if car_type == 'suv' and is_comfortable and fuel_type is None:
        fuel_type = 'bensin'

    return {
        'budget': budget,
        'car_type': car_type,
        'fuel_type': fuel_type,
        'transmission': transmission,
        'comfortable': is_comfortable,
        'youthful': is_youthful,
        'year_operator': year_operator,
        'year_value': year_value
    }

def calculate_score(row, budget, year_value, is_youthful, is_comfortable, car_type):
    price_score = 1 - abs(row['Harga'] - budget) / budget if budget else 1
    year_score = (row['Tahun'] - 2010) / (2025 - 2010) if row['Tahun'] <= 2025 else 1
    mileage_score = 1 - min(row['Jarak tempuh (km)'], 200000) / 200000
    intent_score = 0
    if is_youthful and row['Tipe bodi'] in ['Hatchback', 'Coupe', 'Sedan', 'Sports & Super Car']:
        intent_score += 0.2
    if is_comfortable and (row['Merek'] in ['Mercedes-Benz', 'BMW', 'Audi', 'Lexus', 'Land Rover', 'Porsche', 'Jaguar', 'Mini Cooper'] or row['Tahun'] >= 2018):
        intent_score += 0.2
    if car_type == 'mpv' and row['Tipe bodi'] in ['MPV', 'Minibus']:
        intent_score += 0.2
    weights = {'price': 0.4, 'year': 0.3, 'mileage': 0.2, 'intent': 0.1}
    if is_youthful:
        weights = {'price': 0.3, 'year': 0.4, 'mileage': 0.2, 'intent': 0.1}
    elif is_comfortable:
        weights = {'price': 0.3, 'year': 0.3, 'mileage': 0.2, 'intent': 0.2}
    return (weights['price'] * price_score +
            weights['year'] * year_score +
            weights['mileage'] * mileage_score +
            weights['intent'] * intent_score)

def filter_cars(parsed_query):
    filtered_df = df.copy()
    filtered_df = filtered_df[filtered_df['Tahun'] >= 2010]
    if parsed_query['budget']:
        max_budget = parsed_query['budget'] * 1.1
        filtered_df = filtered_df[filtered_df['Harga'] <= max_budget]
    if parsed_query['car_type']:
        if parsed_query['car_type'] == 'mpv':
            filtered_df = filtered_df[filtered_df['Tipe bodi'].isin(['MPV', 'Minibus']) |
                                    (filtered_df['Tipe bodi'] == 'SUV') & (filtered_df['Harga'] >= 150_000_000)]
        elif parsed_query['car_type'] == 'suv':
            filtered_df = filtered_df[filtered_df['Tipe bodi'].isin(['SUV', 'Jeep'])]
        elif parsed_query['car_type'] == 'sedan':
            filtered_df = filtered_df[filtered_df['Tipe bodi'] == 'Sedan']
        elif parsed_query['car_type'] == 'hatchback':
            filtered_df = filtered_df[filtered_df['Tipe bodi'].isin(['Hatchback', 'Compact & City Car'])]
        elif parsed_query['car_type'] == 'pickup':
            filtered_df = filtered_df[filtered_df['Tipe bodi'].isin(['Pick-up', 'Double Cabin', 'Truk'])]
        elif parsed_query['car_type'] == 'coupe':
            filtered_df = filtered_df[filtered_df['Tipe bodi'].isin(['Coupe', 'Sports & Super Car'])]
        elif parsed_query['youthful']:
            filtered_df = filtered_df[filtered_df['Tipe bodi'].isin(['Hatchback', 'Coupe', 'Sedan', 'Sports & Super Car'])]
    if parsed_query['fuel_type']:
        filtered_df = filtered_df[filtered_df['Tipe bahan bakar'].str.lower() == parsed_query['fuel_type']]
    if parsed_query['transmission']:
        filtered_df = filtered_df[filtered_df['Transmisi'].str.lower().str.contains(parsed_query['transmission'], case=False, na=False)]
    if parsed_query['year_operator'] and parsed_query['year_value']:
        op = parsed_query['year_operator']
        val = parsed_query['year_value']
        if op == '>':
            filtered_df = filtered_df[filtered_df['Tahun'] > val]
        elif op == '>=':
            filtered_df = filtered_df[filtered_df['Tahun'] >= val]
        elif op == '<':
            filtered_df = filtered_df[filtered_df['Tahun'] < val]
        elif op == '<=':
            filtered_df = filtered_df[filtered_df['Tahun'] <= val]
        elif op == '=':
            filtered_df = filtered_df[filtered_df['Tahun'] == val]
    if parsed_query['comfortable']:
        luxury_brands = ['Mercedes-Benz', 'BMW', 'Audi', 'Lexus', 'Land Rover', 'Porsche', 'Jaguar', 'Mini Cooper']
        filtered_df = filtered_df[(filtered_df['Merek'].isin(luxury_brands)) | (filtered_df['Tahun'] >= 2018)]
    return filtered_df

def generate_recommendation(parsed_query):
    filtered_cars = filter_cars(parsed_query)
    if filtered_cars.empty:
        suggestions = []
        if parsed_query['budget'] and parsed_query['budget'] < 150_000_000:
            suggestions.append("meningkatkan budget di atas 150 juta")
        if parsed_query['fuel_type']:
            suggestions.append("mengubah tipe bahan bakar (misalnya, ke bensin)")
        if parsed_query['year_value']:
            suggestions.append("mengurangi batasan tahun (misalnya, tahun >2010)")
        suggestion_text = " atau ".join(suggestions) if suggestions else "menyesuaikan kriteria lainnya"
        return (f"Maaf, tidak ada mobil yang sesuai dengan kriteria Anda. "
                f"Coba {suggestion_text}.")

    filtered_cars['Score'] = filtered_cars.apply(
        lambda row: calculate_score(row, parsed_query['budget'], parsed_query['year_value'],
                                 parsed_query['youthful'], parsed_query['comfortable'], parsed_query['car_type']), axis=1)
    filtered_cars = filtered_cars.sort_values(by='Score', ascending=False)
    filtered_cars = filtered_cars.drop_duplicates(subset=['Merek', 'Model', 'Tahun'])
    top_candidates = filtered_cars.head(10)

    if not top_candidates.empty:
        num_recommendations = min(3, len(top_candidates))
        recommendations = top_candidates.sample(n=num_recommendations, random_state=None).to_dict('records')
    else:
        return "Maaf, tidak ada mobil yang cukup relevan untuk direkomendasikan. Coba ubah kriteria Anda."

    response = "Berikut rekomendasi mobil bekas berdasarkan kriteria Anda:\n\n"
    for i, car in enumerate(recommendations, 1):
        response += (f"{i}. {car['Merek']} {car['Model']} {car['Varian'] or 'N/A'} ({int(car['Tahun'])})\n"
                     f"   Harga: Rp {car['Harga']:,}\n"
                     f"   Tipe Bodi: {car['Tipe bodi']}\n"
                     f"   Bahan Bakar: {car['Tipe bahan bakar']}\n"
                     f"   Transmisi: {car['Transmisi']}\n"
                     f"   Jarak Tempuh: {car['Jarak tempuh']}\n"
                     f"   Lokasi: {car['Lokasi']}\n\n")
    return response

def is_out_of_context(text):
    car_related_keywords = ['mobil', 'car', 'suv', 'sedan', 'mpv', 'minibus', 'hatchback', 'pickup', 
                           'coupe', 'bensin', 'diesel', 'hybrid', 'budget', 'harga', 'juta', 'miliar',
                           'tahun', 'anak muda', 'trendy', 'sporty', 'nyaman', 'mewah', 'niaga']
    return not any(keyword in text.lower() for keyword in car_related_keywords)

# ==========================
# 5. Flask Routes
# ==========================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({'response': 'Silakan masukkan pertanyaan.'})

    if user_input.lower() == 'keluar':
        return jsonify({'response': 'Terima kasih telah menggunakan chatbot!'})

    if is_out_of_context(user_input):
        return jsonify({'response': 'Maaf, pertanyaan Anda tidak relevan dengan pencarian mobil bekas. '
                                    'Silakan masukkan pertanyaan terkait mobil, seperti "mobil keluarga budget 100 juta".'})

    try:
        parsed_query = parse_query(user_input)
        if parsed_query['budget'] is None:
            return jsonify({'response': 'Maaf, silakan masukkan budget dengan format yang jelas, seperti "100 juta" atau "100jt".'})
        response = generate_recommendation(parsed_query)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'response': f'Terjadi kesalahan: {str(e)}. Silakan coba lagi dengan format yang sesuai, '
                                    'seperti "suv diesel 100 juta" atau "mobil anak muda budget 200 juta".'})

if __name__ == '__main__':
    app.run(debug=True)

import pandas as pd
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import asyncio
import platform
from uuid import uuid4

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load and clean the dataset
try:
    df = pd.read_csv('mobilbekas.csv')
    # Remove rows with missing or vague critical columns
    df = df.dropna(subset=['Harga', 'Merek', 'Model', 'Tahun', 'Tipe bodi', 'Tipe bahan bakar', 'Transmisi'])
    df = df[~df['Model'].str.contains('Lain-lain', case=False, na=False)]
    df = df[~df['Varian'].str.contains('Lain-lain', case=False, na=True)]
    # Convert Harga and Tahun to numeric
    df['Harga'] = pd.to_numeric(df['Harga'], errors='coerce')
    df['Tahun'] = pd.to_numeric(df['Tahun'], errors='coerce', downcast='integer')
    # Convert mileage to numeric (midpoint of range)
    def parse_mileage(mileage):
        if isinstance(mileage, str) and '-' in mileage:
            try:
                low, high = map(int, mileage.replace(' km', '').replace('.', '').split('-'))
                return (low + high) / 2
            except:
                return float('inf')
        return float('inf')
    df['Jarak tempuh (km)'] = df['Jarak tempuh'].apply(parse_mileage)
    # Filter valid entries
    df = df[df['Harga'].notnull() & df['Tahun'].notnull() & (df['Jarak tempuh (km)'] != float('inf'))]
    # Standardize body types (e.g., fix Honda Mobilio)
    df.loc[df['Model'].str.contains('Mobilio', case=False, na=False), 'Tipe bodi'] = 'MPV'
    df.loc[df['Model'].str.contains('Ertiga', case=False, na=False), 'Tipe bodi'] = 'MPV'
    df.loc[df['Model'].str.contains('Xpander', case=False, na=False), 'Tipe bodi'] = 'MPV'
except FileNotFoundError:
    print("Error: File 'mobilbekas.csv' not found. Please ensure the dataset is in the correct directory.")
    exit(1)

# Enhanced lexicon for intent recognition
lexicon = {
    'family': ['keluarga', 'family', 'mpv', 'minibus', 'minivan'],
    'suv': ['suv', 'jeep'],
    'sedan': ['sedan'],
    'hatchback': ['hatchback', 'hatch back', 'city car'],
    'pickup': ['pick-up', 'pickup', 'truk', 'double cabin'],
    'coupe': ['coupe', 'sport', 'sports', 'super car'],
    'youthful': ['anak muda', 'muda', 'trendy', 'stylish', 'keren', 'sporty'],
    'comfortable': ['nyaman', 'comfortable', 'mewah', 'luxury', 'prestige'],
    'budget': ['budget', 'harga', 'jtan', 'jt', 'juta'],
    'fuel': {
        'bensin': ['bensin', 'petrol', 'gasoline'],
        'diesel': ['diesel'],
        'hybrid': ['hybrid', 'listrik']
    },
    'transmission': {
        'automatic': ['automatic', 'otomatis', 'auto', 'triptonic'],
        'manual': ['manual']
    },
    'year': ['tahun', 'year']
}

# Initialize NLP tools
stop_words = set(stopwords.words('indonesian') + stopwords.words('english') + ['yang', 'untuk', 'dengan'])
lemmatizer = WordNetLemmatizer()

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens

# Function to extract budget
def extract_budget(text):
    budget_pattern = r'(\d+\.?\d*)\s*(jtan|jt|juta)'
    match = re.search(budget_pattern, text, re.IGNORECASE)
    if match:
        budget_value = float(match.group(1))
        if 'juta' in match.group(2).lower() or 'jt' in match.group(2).lower() or 'jtan' in match.group(2).lower():
            budget_value *= 1_000_000
        return int(budget_value)
    return None

# Function to extract year constraint
def extract_year(text):
    year_pattern = r'tahun\s*(>|>=|<|<=|=)\s*(\d{4})'
    match = re.search(year_pattern, text, re.IGNORECASE)
    if match:
        operator, year = match.group(1), int(match.group(2))
        return operator, year
    return None, None

# Function to parse user query
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

    # Map family to MPV (and SUVs for higher budgets)
    if car_type == 'family':
        car_type = 'mpv'

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

# Function to calculate score for ranking
def calculate_score(row, budget, year_value, is_youthful, is_comfortable, car_type):
    # Normalize price proximity (closer to budget is better)
    price_score = 1 - abs(row['Harga'] - budget) / budget if budget else 1
    # Normalize year (newer is better)
    year_score = (row['Tahun'] - 2010) / (2025 - 2010) if row['Tahun'] <= 2025 else 1
    # Normalize mileage (lower is better)
    mileage_score = 1 - min(row['Jarak tempuh (km)'], 200000) / 200000
    # Intent-specific boosts
    intent_score = 0
    if is_youthful and row['Tipe bodi'] in ['Hatchback', 'Coupe', 'Sedan', 'Sports & Super Car']:
        intent_score += 0.2
    if is_comfortable and (row['Merek'] in ['Mercedes-Benz', 'BMW', 'Audi', 'Lexus', 'Land Rover', 'Porsche', 'Jaguar', 'Mini Cooper'] or row['Tahun'] >= 2018):
        intent_score += 0.2
    if car_type == 'mpv' and row['Tipe bodi'] in ['MPV', 'Minibus']:
        intent_score += 0.2
    # Adjust weights based on intent
    if is_youthful:
        weights = {'price': 0.3, 'year': 0.4, 'mileage': 0.2, 'intent': 0.1}
    elif is_comfortable:
        weights = {'price': 0.3, 'year': 0.3, 'mileage': 0.2, 'intent': 0.2}
    else:
        weights = {'price': 0.4, 'year': 0.3, 'mileage': 0.2, 'intent': 0.1}
    return (weights['price'] * price_score +
            weights['year'] * year_score +
            weights['mileage'] * mileage_score +
            weights['intent'] * intent_score)

# Function to filter cars
def filter_cars(parsed_query):
    filtered_df = df.copy()

    # Apply minimum year constraint (2010 or newer)
    filtered_df = filtered_df[filtered_df['Tahun'] >= 2010]

    # Filter by budget (allow 10% above budget)
    if parsed_query['budget']:
        max_budget = parsed_query['budget'] * 1.1
        filtered_df = filtered_df[filtered_df['Harga'] <= max_budget]

    # Filter by car type
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

    # Filter by fuel type
    if parsed_query['fuel_type']:
        filtered_df = filtered_df[filtered_df['Tipe bahan bakar'].str.lower() == parsed_query['fuel_type']]

    # Filter by transmission
    if parsed_query['transmission']:
        filtered_df = filtered_df[filtered_df['Transmisi'].str.lower().str.contains(parsed_query['transmission'], case=False, na=False)]

    # Filter by year constraint
    if parsed_query['year_operator'] and parsed_query['year_value']:
        if parsed_query['year_operator'] == '>':
            filtered_df = filtered_df[filtered_df['Tahun'] > parsed_query['year_value']]
        elif parsed_query['year_operator'] == '>=':
            filtered_df = filtered_df[filtered_df['Tahun'] >= parsed_query['year_value']]
        elif parsed_query['year_operator'] == '<':
            filtered_df = filtered_df[filtered_df['Tahun'] < parsed_query['year_value']]
        elif parsed_query['year_operator'] == '<=':
            filtered_df = filtered_df[filtered_df['Tahun'] <= parsed_query['year_value']]
        elif parsed_query['year_operator'] == '=':
            filtered_df = filtered_df[filtered_df['Tahun'] == parsed_query['year_value']]

    # Filter for comfortable (luxury brands or newer models)
    if parsed_query['comfortable']:
        luxury_brands = ['Mercedes-Benz', 'BMW', 'Audi', 'Lexus', 'Land Rover', 'Porsche', 'Jaguar', 'Mini Cooper']
        filtered_df = filtered_df[(filtered_df['Merek'].isin(luxury_brands)) | (filtered_df['Tahun'] >= 2018)]

    return filtered_df

# Function to generate recommendation
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
    
    # Calculate scores for ranking
    filtered_cars['Score'] = filtered_cars.apply(
        lambda row: calculate_score(row, parsed_query['budget'], parsed_query['year_value'],
                                 parsed_query['youthful'], parsed_query['comfortable'], parsed_query['car_type']), axis=1)
    
    # Sort by score and remove duplicates based on Merek, Model, and Tahun
    filtered_cars = filtered_cars.sort_values(by='Score', ascending=False)
    filtered_cars = filtered_cars.drop_duplicates(subset=['Merek', 'Model', 'Tahun'])
    
    # Get top 3 recommendations
    recommendations = filtered_cars.head(3)[['Merek', 'Model', 'Varian', 'Tahun', 'Harga', 'Tipe bodi', 
                                            'Tipe bahan bakar', 'Transmisi', 'Lokasi', 'Jarak tempuh']].to_dict('records')
    
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

# Error handling for out-of-context queries
def is_out_of_context(text):
    car_related_keywords = ['mobil', 'car', 'suv', 'sedan', 'mpv', 'minibus', 'hatchback', 'pickup', 
                           'coupe', 'bensin', 'diesel', 'hybrid', 'budget', 'harga', 'juta', 'tahun', 
                           'anak muda', 'trendy', 'sporty', 'nyaman', 'mewah']
    return not any(keyword in text.lower() for keyword in car_related_keywords)

# Main chatbot function
async def chatbot():
    print("Selamat datang di Chatbot Rekomendasi Mobil Bekas!")
    print("Silakan masukkan pertanyaan seperti 'mobil keluarga budget 100 juta', 'suv diesel 300jt', atau 'mobil anak muda 200jt'.")
    print("Ketik 'keluar' untuk mengakhiri.")
    
    while True:
        user_input = input("\nPertanyaan Anda: ")
        
        if user_input.lower() == 'keluar':
            print("Terima kasih telah menggunakan chatbot!")
            break
        
        if is_out_of_context(user_input):
            print("Maaf, pertanyaan Anda tidak relevan dengan pencarian mobil bekas. "
                  "Silakan masukkan pertanyaan terkait mobil, seperti 'mobil keluarga budget 100 juta'.")
            continue
        
        try:
            parsed_query = parse_query(user_input)
            
            if parsed_query['budget'] is None:
                print("Maaf, silakan masukkan budget dengan format yang jelas, seperti '100 juta' atau '100jt'.")
                continue
            
            response = generate_recommendation(parsed_query)
            print(response)
            
        except Exception as e:
            print(f"Terjadi kesalahan: {str(e)}. Silakan coba lagi dengan format yang sesuai, "
                  "seperti 'suv diesel 100 juta' atau 'mobil anak muda budget 200 juta'.")

# Run the chatbot
if platform.system() == "Emscripten":
    asyncio.ensure_future(chatbot())
else:
    if __name__ == "__main__":
        asyncio.run(chatbot())
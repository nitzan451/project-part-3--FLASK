from flask import Flask, request, render_template
import pandas as pd
import pickle
from assets_data_prep import prepare_data, clean_address
import numpy as np

def safe_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan

def safe_int(val):
    try:
        return int(val)
    except (ValueError, TypeError):
        return np.nan

app = Flask(__name__)

with open("trained_model.pkl", "rb") as f:
    model = pickle.load(f)

df = pd.read_csv("train.csv")
addresses = sorted(df['address'].dropna().unique())
neighborhoods = sorted(df['neighborhood'].dropna().unique())

@app.route("/", methods=["GET", "POST"])
def index():
    form_data = {}
    error = None
    prediction = None

    if request.method == "POST":
        form_data = request.form.to_dict(flat=False)
        form_data = {k: v[0] if len(v) == 1 else v for k, v in form_data.items()}

        field_errors = {}

        form_data["property_type"] = form_data.get("property_type", "").strip()
        form_data["address"] = form_data.get("address", "").strip()
        form_data["neighborhood"] = form_data.get("neighborhood", "").strip()

        try:
            room_number_val = float(form_data.get("room_number", ""))
            if not (1 <= room_number_val <= 10):
                field_errors["room_number"] = "מספר החדרים חייב להיות בין 1 ל-10"
        except ValueError:
            field_errors["room_number"] = "יש להזין מספר חדרים תקין (1-10)"

        try:
            floor_val = int(form_data.get("floor", ""))
            if floor_val < 1:
                field_errors["floor"] = "קומה מינימלית חייבת להיות 1"
        except ValueError:
            field_errors["floor"] = "יש להזין מספר קומה תקין"

        try:
            total_floors_val = int(form_data.get("total_floors", ""))
            if total_floors_val < 1:
                field_errors["total_floors"] = "סה\"כ קומות חייב להיות לפחות 1"
        except ValueError:
            field_errors["total_floors"] = "יש להזין מספר תקין"
            total_floors_val = None
        else:
            try:
                floor_val = int(form_data.get("floor", ""))
                if "floor" not in field_errors and total_floors_val is not None and floor_val > total_floors_val:
                    field_errors["total_floors"] = "סה\"כ קומות חייב להיות גדול או שווה למספר קומה"
            except ValueError:
                pass


        try:
            area_val = float(form_data.get("area", ""))
            if area_val < 20:
                field_errors["area"] = "שטח מינימלי חייב להיות 20 מ\"ר"
        except ValueError:
            field_errors["area"] = "יש להזין שטח תקין"

        if not form_data["address"] and not form_data["neighborhood"]:
            field_errors["address"] = "חובה להזין שכונה או כתובת"

        if not form_data["property_type"]:
            field_errors["property_type"] = "יש לבחור סוג נכס"
            
        if field_errors:
            return render_template(
                "index.html",
                prediction=None,
                form_data=form_data,
                addresses=addresses,
                neighborhoods=neighborhoods,
                error=None,
                field_errors=field_errors
            )

        address = form_data.get("address", "").strip()
        if address == "אחר":
            address = form_data.get("custom_address", "").strip()
        # אם אין כתובת והוזנה שכונה – נשתמש בשכונה בתור כתובת
        if not address:
            neighborhood = form_data.get("neighborhood", "").strip()
            if neighborhood and neighborhood != "אחר":
                address = neighborhood
        address = clean_address(address)

        neighborhood = form_data.get("neighborhood", "").strip()
        if neighborhood == "אחר":
            neighborhood = form_data.get("custom_neighborhood", "").strip()

        raw_features = form_data.get("features", [])
        if isinstance(raw_features, str):
            features = [raw_features]
        elif isinstance(raw_features, list):
            features = raw_features
        else:
            features = []

        data = {
            "room_num": [safe_float(form_data.get("room_number"))],
            "floor": [safe_int(form_data.get("floor"))],
            "area": [safe_float(form_data.get("area"))],
            "property_type": [form_data.get("property_type", "דירה")],
            "features": [features],
            "total_floors": [safe_int(form_data.get("total_floors"))],
            "monthly_arnona": [float(form_data["monthly_arnona"] or 0)],
            "building_tax": [float(form_data["building_tax"] or 0)],
            "garden_area": [float(form_data["garden_area"] or 0)],
            "description": [""]
        }

        if address and address != "רחוב לא ידוע":
            data["address"] = [address]
            data["neighborhood"] = ["Unknown"]
        elif neighborhood:
            data["address"] = ["רחוב לא ידוע"]
            data["neighborhood"] = [neighborhood]
        else:
            data["address"] = ["רחוב לא ידוע"]
            data["neighborhood"] = ["Unknown"]

        all_features = ['מיזוג', 'משופצת', 'מרפסת', 'חניה', 'מעלית', 'סורגים',
                        'ריהוט', 'ממ\"ד', 'חיות מחמד', 'מחסן', 'גישה לנכים']
        for feature in all_features:
            data[feature] = [1 if feature in features else 0]
        data.pop("features")

        data["has_parking"] = [data["חניה"][0]]
        data["elevator"] = [data["מעלית"][0]]
        data["has_safe_room"] = [data["ממ\"ד"][0]]
        data["has_balcony"] = [data["מרפסת"][0]]
        data["is_renovated"] = [data["משופצת"][0]]
        data["is_furnished"] = [data["ריהוט"][0]]
        data["distance_from_center"] = [0]

        df_input = pd.DataFrame(data)

        try:
            df_processed = prepare_data(df_input, dataset_type="test")
            prediction = round(model.predict(df_processed)[0], 2)
        except Exception as e:
            import traceback
            traceback.print_exc()
            error = "שגיאה, אין מספיק ערכים לחיזוי"
            return render_template(
                "index.html",
                prediction=None,
                form_data=form_data,
                addresses=addresses,
                neighborhoods=neighborhoods,
                error=error,
                field_errors={}
            )

        return render_template(
            "index.html",
            prediction=prediction,
            form_data=form_data,
            addresses=addresses,
            neighborhoods=neighborhoods,
            error=None,
            field_errors={}
        )

    # החזרת דף ריק במקרה של GET
    return render_template(
        "index.html",
        prediction=None,
        form_data=form_data,
        addresses=addresses,
        neighborhoods=neighborhoods,
        error=None,
        field_errors={}
    )

if __name__ == "__main__":
    app.run(debug=True)

import pandas as pd
import numpy as np
import re
import json
import pickle
import requests
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def extract_street(address):
    """
    מחלץ את שם הרחוב מתוך כתובת, למשל:
    'לאונרדו דה וינצי 14' -> 'לאונרדו דה וינצי'
    """
    if pd.isnull(address):
        return None
    match = re.match(r'^(.+?)(?:\s+\d+.*)?$', str(address).strip())
    return match.group(1).strip() if match else address

def process_floors(df):
    """
    פונקציה זו:
    1. מפצלת ערכים כמו '8 מתוך 10' ל־floor=8, total_floors=10.
    2. מטפלת ב'קרקע', ערכים עשרוניים, ערכים חריגים (למשל 810 במקום 8 מתוך 10).
    3. מעתיקה floor ל-total_floors אם total_floors חסר ויש floor.
    4. ממלאת חסרים לפי חציון ברחוב (ולא כתובת מלאה!).
    """

    # --- שלב 1: פיצול floor ו-total_floors מתוך מחרוזות ---
    floors = []
    totals = []
    for val in df['floor'].astype(str):
        if "מתוך" in val:
            parts = val.split("מתוך")
            floor_part = parts[0].strip()
            total_part = parts[1].strip()
            floor_val = 0 if "קרקע" in floor_part else int(float(floor_part)) if floor_part.replace('.', '', 1).isdigit() else np.nan
            total_val = 0 if "קרקע" in total_part else int(float(total_part)) if total_part.replace('.', '', 1).isdigit() else np.nan
        else:
            floor_val = 0 if "קרקע" in val else int(float(val)) if val.replace('.', '', 1).isdigit() else np.nan
            total_val = np.nan
        floors.append(floor_val)
        totals.append(total_val)
    df['floor'] = floors
    if 'total_floors' in df.columns:
        missing_mask = df['total_floors'].isnull() | (df['total_floors'] == '')
        df.loc[missing_mask, 'total_floors'] = pd.Series(totals)[missing_mask].values
    else:
        df['total_floors'] = totals

    # --- שלב 2: תיקון ערכים חריגים (קומה גדולה ממספר קומות, לדוג' floor=810, total_floors=10) ---
    mask = (
        df["floor"].notna() &
        df["total_floors"].notna() &
        (df["floor"] > df["total_floors"])
    )
    for idx in df[mask].index:
        floor_val = int(df.at[idx, "floor"])
        total_val = int(df.at[idx, "total_floors"])
        # פיצול: הספרות האחרונות הן total_floors, היתר הן floor
        floor_str = str(floor_val)
        total_str = str(total_val)
        if floor_str.endswith(total_str):
            floor_only = floor_str[:len(floor_str) - len(total_str)]
            try:
                new_floor = int(floor_only)
                df.at[idx, "floor"] = new_floor
            except:
                pass  # לא הצליח? ישאיר את הערך המקורי

    # --- שלב 3: אם total_floors חסר ויש ערך ב-floor, העתק ---
    mask_missing_total = df['total_floors'].isnull() & df['floor'].notnull()
    df.loc[mask_missing_total, 'total_floors'] = df.loc[mask_missing_total, 'floor']

    # --- שלב 4: השלמת חסרים לפי שם רחוב בלבד ---
    df['street_only'] = df['address'].apply(extract_street)

    # השלמה ל-floor לפי חציון ברחוב
    mask_floor_na = df['floor'].isnull() & df['street_only'].notnull()
    for idx in df[mask_floor_na].index:
        street = df.at[idx, 'street_only']
        relevant = df[(df['street_only'] == street) & df['floor'].notnull()]
        if not relevant.empty:
            df.at[idx, 'floor'] = relevant['floor'].median()

    # השלמה ל-total_floors לפי חציון ברחוב
    mask_total_na = df['total_floors'].isnull() & df['street_only'].notnull()
    for idx in df[mask_total_na].index:
        street = df.at[idx, 'street_only']
        relevant = df[(df['street_only'] == street) & df['total_floors'].notnull()]
        if not relevant.empty:
            df.at[idx, 'total_floors'] = relevant['total_floors'].median()

    # הסרת העמודה הזמנית
    df = df.drop(columns=['street_only'])

    problem_rows = df['floor'] > df['total_floors']
    df.loc[problem_rows, 'floor'] = df.loc[problem_rows, 'total_floors']


    return df

def extract_room_num(description):
    '''
    הפונקציה מחלצת מספר חדרים מתוך טקסט חופשי.
    זה מאפשר להפוך מידע לא מובנה לפיצ’ר כמותי חשוב.
    השיטה נבחרה כי היא מדויקת למבנה הנפוץ "X חדרים" ומתמודדת גם עם ערכים עשרוניים.
    '''
    if pd.isnull(description):
        return None
    match = re.search(r'(\d+(\.\d+)?)\s*חדרים', description)
    if match:
        return float(match.group(1))
    return None

def fix_room_num(df):
    '''
    הפונקציה מתקנת ערכי חדרים חסרים או אפסיים בשלבים:
     תחילה לפי תיאור הדירה, אחר כך לפי חציון בשכונה ובשטח דומה,
     ולבסוף לפי חציון כללי.
     שיטה מדויקת להשלמת מידע חסר באופן חכם.
    '''
    # שלב 1: תיקון לפי התיאור
    mask_zero = df['room_num'] == 0
    df.loc[mask_zero, 'room_num'] = (
        df.loc[mask_zero, 'description'].apply(extract_room_num)
    )
    
    # שלב 2: אם עדיין 0, תיקון לפי חציון שכונה ושטח (±5 מ"ר)
    mask_zero = df['room_num'] == 0
    for idx in df[mask_zero].index:
        neighborhood = df.at[idx, 'neighborhood']
        area = df.at[idx, 'area']
        if pd.notnull(neighborhood) and pd.notnull(area):
            relevant = df[
                (df['neighborhood'] == neighborhood) &
                (df['area'] >= area - 5) &
                (df['area'] <= area + 5) &
                (df['room_num'] > 0) &
                df['room_num'].notnull()
            ]
            if not relevant.empty:
                df.at[idx, 'room_num'] = relevant['room_num'].median()
            else:
                # אם אין דירות כאלה, אפשר לשים חציון שכונה כללית (לא חובה)
                relevant = df[
                    (df['neighborhood'] == neighborhood) &
                    (df['room_num'] > 0) &
                    df['room_num'].notnull()
                ]
                if not relevant.empty:
                    df.at[idx, 'room_num'] = relevant['room_num'].median()
    # שלב 3: אם עדיין חסר (0 או NaN) -> חציון כללי
    mask_zero = (df['room_num'] == 0) | (df['room_num'].isnull())
    if mask_zero.any():
        general_median = df.loc[(df['room_num'] > 0) & df['room_num'].notnull(), 'room_num'].median()
        df.loc[mask_zero, 'room_num'] = general_median

    return df

def process_garden_area(df):
    '''
    הפונקציה משלימה שטח גינה חסר.
     אם הדירה בקומה גבוהה – נקבע אפס. אחרת, מחושבת לפי חציון דירות דומות בשכונה ובשטח.
     מאפשר שמירה על עקביות והיגיון בנתוני הגינה.
    '''
    mask_garden_na = df['garden_area'].isnull()
    for idx in df[mask_garden_na].index:
        floor = df.at[idx, 'floor'] if 'floor' in df.columns else np.nan
        if pd.notnull(floor) and floor > 0:
            df.at[idx, 'garden_area'] = 0
        else:
            neighborhood = df.at[idx, 'neighborhood']
            area = df.at[idx, 'area']
            relevant = df[
                (df['neighborhood'] == neighborhood) &
                (df['area'] == area) &
                df['garden_area'].notnull()
            ]
            if not relevant.empty:
                df.at[idx, 'garden_area'] = relevant['garden_area'].median()
            else:
                df.at[idx, 'garden_area'] = 0
    return df

def process_tax_col(df, col_name):
    """
    משלימה ערכים חסרים בעמודה כספית (כמו ארנונה או מס בניין).
    1. קובע 0 לוועד בית בבתי פרטי/קוטג'.
    2. מנסה להשלים לפי שטח דומה (±10%) ושאר קריטריונים.
    3. אם לא נמצא — משלים לפי חציון של אותה שכונה.
    4. אם עדיין לא נמצא — משלים לפי חציון כללי של הדאטה.
    """
    # ועד בית לבתי פרטי/קוטג' = 0
    if col_name == 'building_tax':
        df.loc[df['property_type'].astype(str).str.contains("בית פרטי|קוטג'", na=False), col_name] = 0

    mask_na_or_zero = df[col_name].isnull() | (df[col_name] == 0)

    for idx in df[mask_na_or_zero].index:
        area = df.at[idx, 'area']
        neighborhood = df.at[idx, 'neighborhood']

        relevant = pd.DataFrame()

        # שלב 1: לפי שטח דומה (±10%)
        if pd.notnull(area):
            relevant = df[
                (df['area'] >= area * 0.9) &
                (df['area'] <= area * 1.1) &
                df[col_name].notnull() &
                (df[col_name] > 0)
            ]

        # שלב 2: לפי שכונה
        if relevant.empty and pd.notnull(neighborhood):
            relevant = df[
                (df['neighborhood'] == neighborhood) &
                df[col_name].notnull() &
                (df[col_name] > 0)
            ]

        # שלב 3: אם עדיין אין — לפי כלל הדאטה
        if relevant.empty:
            relevant = df[
                df[col_name].notnull() &
                (df[col_name] > 0)
            ]
        if pd.isnull(df.at[idx, col_name]) or df.at[idx, col_name] == 0:
            df.at[idx, col_name] = df[col_name].median()

        # השלמה בפועל
        if not relevant.empty:
            df.at[idx, col_name] = relevant[col_name].median()
            
    return df
    
def fix_monthly_arnona_by_median(df, min_val=100, max_val=2000, area_tol=5):
    df = df.copy()
    mask = (df['monthly_arnona'] < min_val) | (df['monthly_arnona'] > max_val)

    for i in df[mask].index:
        n, a = df.at[i, 'neighborhood'], df.at[i, 'area']
        s = df.query(
            "@n == neighborhood and @a - @area_tol <= area <= @a + @area_tol and @min_val <= monthly_arnona <= @max_val"
        )['monthly_arnona']
        if s.empty:
            s = df.query("@n == neighborhood and @min_val <= monthly_arnona <= @max_val")['monthly_arnona']
        df.at[i, 'monthly_arnona'] = s.median() if not s.empty else df['monthly_arnona'].median()

    return df


def process_distance_from_center(df):
    '''
    הפונקציה מתקנת יחידות שגויות (מק״מ במקום מטרים),
    משלימה ערכים חסרים לפי חציון שכונה,
    ובמקרים חריגים מוסיפה ערכים שנבדקו ידנית.
    מבטיחה דיוק ואחידות במרחקים מהמרכז.
    '''
    if 'distance_from_center' in df.columns:
        # הכפלה ב-1000 לערכים <10 (לא חסרים)
        mask_under_10 = df['distance_from_center'].apply(lambda x: pd.notnull(x) and x < 10)
        df.loc[mask_under_10, 'distance_from_center'] = df.loc[mask_under_10, 'distance_from_center'] * 1000
        # השלמת חסרים: חציון לפי neighborhood
        df['distance_from_center'] = df.groupby('neighborhood')['distance_from_center']\
            .transform(lambda x: x.fillna(x.median()))
        #  השלמה ידנית לשכונות גני צהלה וכוכב הצפון אחרי בדיקה בגוגל
        df.loc[(df['neighborhood'] == 'גני צהלה') & (df['distance_from_center'].isnull()), 'distance_from_center'] = 7700
        df.loc[(df['neighborhood'] == 'כוכב הצפון') & (df['distance_from_center'].isnull()), 'distance_from_center'] = 4600
    return df
def filter_extreme_distances(df, threshold=1.5):
    """
    מחליף ערכים קיצוניים בעמודת distance_from_center לפי סטייה מהחציון לשכונה.
    פרמטרים:
    df : DataFrame
        טבלת הדירות הכוללת עמודות 'neighborhood' ו-'distance_from_center'
    threshold : float (ברירת מחדל 1.0)
        כמה סטיות תקן מעבר לחציון ייחשבו לחריגים (למשל: 1.5 = חציון + 1.5*std)
    מחזיר:
    df : DataFrame מתוקן
    """
    df = df.copy()
    # סטטיסטיקות לפי שכונה
    stats = df.groupby('neighborhood')['distance_from_center'].agg(['median', 'std']).reset_index()
    stats.columns = ['neighborhood', 'neigh_median', 'neigh_std']
    # מיזוג עם הדאטה המקורי
    df = df.merge(stats, on='neighborhood', how='left')
    # תנאי לחריגה
    mask = df['distance_from_center'] > (df['neigh_median'] + threshold * df['neigh_std'])
    # החלפת ערכים חריגים בחציון השכונה
    df.loc[mask, 'distance_from_center'] = df.loc[mask, 'neigh_median']    
    # הסרת העמודות העזר
    df.drop(columns=['neigh_median', 'neigh_std'], inplace=True)
    
    return df

def get_distance_from_center(address, neighborhood, distance):
    """
    משתמש במרחק קיים אם תקין (>=100).
    משתמש ב-API אם יש מפתח ואם distance חסר או קטן מ-100.
    לא מחזיר None אם יש ערך קיים כלשהו.
    """
    # אם המרחק תקין – נחזיר אותו מיד
    if pd.notnull(distance) and distance >= 100:
        return distance

    api_key = "Add API KEY"  # החלף במפתח אמיתי
    if not api_key or api_key == "Add API KEY":
        return distance  # לא דורס – משאיר את מה שהיה

    # כתובת מקור
    origin_raw = address if pd.notnull(address) else neighborhood
    origin_address = f"{origin_raw}, תל אביב"

    url = "https://routes.googleapis.com/directions/v2:computeRoutes"
    headers = {
        'Content-Type': 'application/json',
        'X-Goog-Api-Key': api_key,
        'X-Goog-FieldMask': 'routes.distanceMeters,routes.duration'
    }
    body = {
        "origin": {"address": origin_address},
        "destination": {"address": "כיכר דיזינגוף, תל אביב"},
        "travelMode": "DRIVE",
        "routingPreference": "TRAFFIC_AWARE"
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(body))
        if response.status_code == 200:
            data = response.json()
            return float(data['routes'][0]['distanceMeters'])
        else:
            return distance  # שמירה על ערך קודם אם הייתה שגיאה
    except:
        return distance  # שמירה על ערך קודם במקרה חריג

def clean_address(val):
        if pd.isna(val):
            return val
        # מסיר ספרות
        cleaned = re.sub(r'\d+', '', str(val)).strip()
        # אם אחרי הניקוי לא נשאר כלום - כנראה היה רק מספר
        return cleaned if cleaned else np.nan

def keep_only_text_in_address(df):
    '''
    משאיר רק את שם הרחוב ללא המספר בית
    '''

    df['address'] = df['address'].apply(clean_address)
    return df

def fill_missing_address_by_neighborhood(df):
    '''
    ערכים חסרים מחליף בערך השכיח ביותר באותה שכונה
    '''
    mode_by_neighborhood = (
        df.dropna(subset=['address'])  
        .groupby('neighborhood')['address']
        .agg(lambda x: x.mode().iloc[0]))
    def fill_address(row):
        if pd.isna(row['address']):
            return mode_by_neighborhood.get(row['neighborhood'], df['address'].mode().iloc[0])
        return row['address']
    df['address'] = df.apply(fill_address, axis=1)
    return df

df = pd.read_csv('train.csv')  

# --- פונקציה ראשית: רק מזמנת את כל הפונקציות בסדר העבודה + מטפלת בשאר עמודות ישירות ---
def prepare_data(df, dataset_type):
    df = df.copy()

    # טיפול בעמודת floor ו-total_floors
    # --- שלב 1: טיפול בעמודת floor ו-total_floors (פיצול מתוך מחרוזות) ---
    if 'floor' in df.columns:
        df = process_floors(df)
        # בעקבות שגיאה בין שתי שכונות ספציפיות אני מחליף בין העמודות לאחר בדיקה באתר והבנה שהערכים הוכנסו הפוך
       
    # המרת עמודות חשובות למספרים
    numeric_cols = [
        'floor', 'area', 'total_floors', 'monthly_arnona', 'building_tax',
        'garden_area', 'distance_from_center']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # property_type - מחיקת סוגים לא רלוונטיים ומחיקת חסרים/ריקים
    invalid_types = [
        "מרתף/פרטר", "חניה", "סאבלט", "Квартира", "מחסן",
        "באתר מופיע ערך שלא ברשימה הסגורה", "כללי", "החלפת דירות"
    ]
    if 'property_type' in df.columns:
        df = df[~df['property_type'].isin(invalid_types)]
        df = df[~df['property_type'].isnull()]
        df = df[df['property_type'].astype(str).str.strip() != '']
        df['property_type'] = df['property_type'].replace('דירה להשכרה', 'דירה')
        # ניקוי ואיחוד ערכים בעמודת property_type
        df['property_type'] = df['property_type'].str.strip()  # הסרת רווחים מיותרים    
        # איחוד פנטהאוזים
        df['property_type'] = df['property_type'].replace({
            'גג/פנטהאוז': 'פנטהאוז',
            'גג/ פנטהאוז': 'פנטהאוז',
            'גג/פנטהאוז להשכרה': 'פנטהאוז'})
        # איחוד דירות גן
        df['property_type'] = df['property_type'].replace({
            'דירת גן': 'דירת גן',
            'דירת גן להשכרה': 'דירת גן'})

    # neighborhood - השלמת חסרים ל-'Unknown'
    if 'neighborhood' in df.columns:
        df['neighborhood'] = df['neighborhood'].fillna('Unknown')

    # room num- תיקון ערכים ששווים ל-0
    if 'room_num' in df.columns:
        df = fix_room_num(df)

    # distance_from_center
    if 'distance_from_center' in df.columns:
        df.loc[df['distance_from_center'].isnull(), 'distance_from_center'] = 0
        df['distance_from_center'] = df.apply(
            lambda row: get_distance_from_center(row['address'], row['neighborhood'], row['distance_from_center'])
            if (pd.isnull(row['distance_from_center']) or row['distance_from_center'] < 100)
            else row['distance_from_center'], axis=1   )
        df = process_distance_from_center(df)
        df = filter_extreme_distances(df, threshold=1.5)        
    
    # garden_area
    if 'garden_area' in df.columns:
        df = process_garden_area(df)
        df.loc[df['garden_area'] > 100, 'garden_area'] = df['garden_area'] / 10

    if 'area' in df.columns:
        df = df[df['area'] >= 20]
        
    # days_to_enter - חסרים ל-0
    if 'days_to_enter' in df.columns:
        df = df.drop(['days_to_enter'], axis=1)


    # num_of_payments -  מחיקה ( לא רלוונטי )
    if 'num_of_payments' in df.columns:
        df = df.drop(['num_of_payments'], axis=1)

    # monthly_arnona - חציון לפי area (±10%)
    if 'monthly_arnona' in df.columns:
        df = process_tax_col(df, 'monthly_arnona')
        df['monthly_arnona'] = df['monthly_arnona'].fillna(df['monthly_arnona'].median())
        df = fix_monthly_arnona_by_median(df)



    # building_tax - חציון לפי area (±10%)
    if 'building_tax' in df.columns:
        df = process_tax_col(df, 'building_tax')
        df['building_tax'] = df['building_tax'].fillna(df['building_tax'].median())


    # num_of_images - מחיקה ( עמודה לא רלוונטית לחיזוי מחיר שכד)
    if 'num_of_images' in df.columns:
       df = df.drop(['num_of_images'], axis=1)
     

    # description -
    #מחיקה (עמודה בעייתית שקשה לעבוד איתה במודלים מכיוון שצריך ערכים מספריים, בנוסף לא הכי רלוונטית לחיזוי שכד)
    if 'description' in df.columns:
        df = df.drop(['description'], axis=1)

    # price - מחיקה של שורות עם חסר או מחיר נמוך מ-999
    # בנוסף, מחיקה של שורות עם מחיר גבוה ממיליון (כי סביר להניח שמדובר בדירות למכירה ולא להשכרה)
    if 'price' in df.columns:
        df = df[~df['price'].isnull()]
        df = df[df['price'] >= 1000] # מחיר לא הגיוני לשכר דירה בתל אביב ולכן נמחק את השורה כי זה יפגע באמינות
        df = df[df['price'] <= 25000]  # שורות עם מחיר לא הגיוני לשכירות, כנראה דירה למכירה
        df = df[(df['price'] > df['price'].quantile(0.01)) & (df['price'] < df['price'].quantile(0.99))] #מוריד קצוות אחוזונים
    
    
    df = df.sort_values(['neighborhood']).reset_index(drop=True)
    df = keep_only_text_in_address(df)
    df = fill_missing_address_by_neighborhood(df)
    
    # המרת עמודות חשובות למספרים
    numeric_cols = [
        'floor', 'area', 'total_floors', 'monthly_arnona', 'building_tax',
        'garden_area', 'distance_from_center']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # מספר חדרים למ"ר 
    df['room_density'] = df['room_num'] / (df['area'])
    # ציון 0–4 שמעיד על כמה הדירה איכותית פונקציונלית    
    df['luxury_score'] = (df['has_parking'] + df['elevator'] + df['has_safe_room'] + df['has_balcony'])
    #דירות עם יותר חדרים על אותו שטח לרוב פחות יוקרתיות – תיתן למודל יכולת להבדיל.    
    df['room_num_x_area'] = df['room_num'] * df['area']
    # משקל מרפסת יחסית לגובה – ככל שהדירה גבוהה יותר, לרוב מרפסת חשובה יותר
    df["balcony_per_floor"] = df["has_balcony"] / (df["floor"] + 1) 
    # שילוב של שיפוץ וריהוט – מצביע על מוכנות גבוהה למגורים
    df["renovated_furnished"] = ((df["is_renovated"] == 1) & (df["is_furnished"] == 1)).astype(int)  
    # נכס קטן – מתאים להשכרה ליחידים או זוגות   
    df["is_mini_property"] = ((df["area"] < 50) & (df["room_num"] <= 2)).astype(int) 
    # שילוב של שטח הדירה עם הארנונה – מייצג עלות כוללת למ"ר
    df['area_x_monthly_arnona'] = df['area'] * df['monthly_arnona']
    # שילוב של קומה עם הארנונה – עשוי לייצג בניינים יוקרתיים יותר
    df['floor_x_monthly_arnona'] = df['floor'] * df['monthly_arnona']
    # בדיקת מיקום מרכזי לנכס
    df["central_location"] = (df["distance_from_center"] < 3000).astype(int)
    df['log_area'] = np.log1p(df['area'])  # בטוח גם אם area = 0
    


    # Target Encoding לקטגוריות
    target_encoded_cols = ['neighborhood', 'address']
    onehot_col = 'property_type'
    
    if dataset_type == 'train':
        category_means = {}
    
        # Target Encoding רגיל לשכונה וכתובת
        for col in target_encoded_cols:
            means = df.groupby(col)['price'].mean()
            category_means[col] = means.to_dict()
            df[col + '_encoded'] = df[col].map(category_means[col])
    
        # One-Hot Encoding לסוג נכס
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        onehot_encoded = encoder.fit_transform(df[[onehot_col]])
        onehot_df = pd.DataFrame(onehot_encoded, columns=encoder.get_feature_names_out([onehot_col]), index=df.index)
        df = pd.concat([df.drop(columns=[onehot_col]), onehot_df], axis=1)
    
        # שמירת הקידודים וה-encoder
        with open("category_means.pkl", "wb") as f:
            pickle.dump(category_means, f)
        with open("onehot_encoder.pkl", "wb") as f:
            pickle.dump(encoder, f)
    
    elif dataset_type == 'test':
        with open("category_means.pkl", "rb") as f:
            category_means = pickle.load(f)
        # קידוד שכונה רגיל עם fallback
        df['neighborhood_encoded'] = df['neighborhood'].map(category_means['neighborhood'])
        fallback_neigh = np.mean(list(category_means['neighborhood'].values()))
        df['neighborhood_encoded'] = df['neighborhood_encoded'].fillna(fallback_neigh)

        # קידוד כתובת עם fallback לפי שכונה
        def encode_address(row):
            addr = row['address']
            neigh = row['neighborhood']
            if addr in category_means['address']:
                return category_means['address'][addr]
            elif neigh in category_means['neighborhood']:
                return category_means['neighborhood'][neigh]
            else:
                return fallback_neigh

        df['address_encoded'] = df.apply(encode_address, axis=1)
        # One-Hot Encoding ל-test
        with open("onehot_encoder.pkl", "rb") as f:
            encoder = pickle.load(f)
        onehot_encoded = encoder.transform(df[[onehot_col]])
        onehot_df = pd.DataFrame(onehot_encoded, columns=encoder.get_feature_names_out([onehot_col]), index=df.index)
        df = pd.concat([df.drop(columns=[onehot_col]), onehot_df], axis=1)

    # מחיקת העמודות המקוריות
    df.drop(columns=target_encoded_cols + [onehot_col], inplace=True, errors='ignore')

    # === פיצול ונרמול ===
    if dataset_type == 'train':
        feature_cols = df.drop(columns='price').columns
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[feature_cols])
        with open("train_columns.pkl", "wb") as f:
            pickle.dump(feature_cols, f)
        with open("scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        df_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=df.index)
        df_scaled["price"] = df["price"].values
        return df_scaled

    elif dataset_type == 'test':
        with open("train_columns.pkl", "rb") as f:
            train_columns = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        for col in train_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[train_columns]
        X_scaled = scaler.transform(df)
        df_scaled = pd.DataFrame(X_scaled, columns=train_columns, index=df.index)
        return df_scaled
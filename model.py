import pandas as pd
import itertools
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from PIL import Image
import streamlit as st
import numpy as np

df = pd.read_csv("allclothes.csv", encoding="utf-8")

label_color = LabelEncoder()
label_purpose = LabelEncoder()

df['color_enc'] = label_color.fit_transform(df['color'])
df['purpose_enc'] = label_purpose.fit_transform(df['purpose'])

tops = df[df['type']=='top'].reset_index(drop=True)
bottoms = df[df['type']=='bottom'].reset_index(drop=True)

for item_df in [tops, bottoms]:
    item_df['temp_avg'] = (item_df['temp_min'] + item_df['temp_max']) / 2

features = ['temp_avg','purpose_enc','color_enc']


dt_top = DecisionTreeClassifier()
dt_top.fit(tops[features], tops['name'])

dt_bottom = DecisionTreeClassifier()
dt_bottom.fit(bottoms[features], bottoms['name'])

color_compat = {
    ('black','white'): 1.0,
    ('white','black'): 1.0,
    ('blue','white'): 0.9,
    ('white','blue'): 0.9,
    ('black','blue'): 0.8,
    ('blue','black'): 0.8,
    ('red','black'): 0.6,
    ('black','red'): 0.8,
    ('red','white'): 0.8,
    ('white','red'): 0.8
}

def color_score(c1, c2):
    return color_compat.get((c1,c2), 0.5)

st.title("Clothes recommendation")

user_temp = st.number_input("Enter temperature:", min_value=-50, max_value=50, value=None)
user_purpose = st.selectbox("Enter purpose:", ["", "casual", "formal", "ceremonial"])
user_color = st.text_input("Enter color:").lower()

if st.button("Get result"):
    filtered_tops = tops.copy()
    filtered_bottoms = bottoms.copy()
    
    if user_temp is not None:
        filtered_tops = filtered_tops[(filtered_tops['temp_min'] <= user_temp) & (filtered_tops['temp_max'] >= user_temp)]
        filtered_bottoms = filtered_bottoms[(filtered_bottoms['temp_min'] <= user_temp) & (filtered_bottoms['temp_max'] >= user_temp)]
    
    if user_color != "":
        filtered_tops = filtered_tops[filtered_tops['color'] == user_color]
        filtered_bottoms = filtered_bottoms[filtered_bottoms['color'] == user_color]
    
    if user_purpose != "":
        purpose_enc = label_purpose.transform([user_purpose])[0]
        filtered_tops = filtered_tops[filtered_tops['purpose_enc'] == purpose_enc]
        filtered_bottoms = filtered_bottoms[filtered_bottoms['purpose_enc'] == purpose_enc]

    def get_proba(df_subset, dt_model):
        X = pd.DataFrame({
            'temp_avg': [user_temp]*len(df_subset) if user_temp is not None else df_subset['temp_avg'],
            'purpose_enc': [purpose_enc]*len(df_subset) if user_purpose != "" else df_subset['purpose_enc'],
            'color_enc': df_subset['color_enc']
        })
        proba = dt_model.predict_proba(X)
        df_subset['score'] = proba.max(axis=1)
        return df_subset

    tops_scored = get_proba(filtered_tops, dt_top)
    bottoms_scored = get_proba(filtered_bottoms, dt_bottom)

    K = 3
    top_candidates = tops_scored.nlargest(K, 'score').reset_index(drop=True)
    bottom_candidates = bottoms_scored.nlargest(K, 'score').reset_index(drop=True)
    
    pairs = list(itertools.product(top_candidates.iterrows(), bottom_candidates.iterrows()))
    ranked_pairs = []
    for (i_top, top), (i_bot, bottom) in pairs:
        pair_score = 0.6*top['score'] + 0.6*bottom['score'] + 1.0*color_score(top['color'], bottom['color'])
        ranked_pairs.append((pair_score, top, bottom))
    
    ranked_pairs.sort(reverse=True, key=lambda x: x[0])

    
    st.subheader("Top 3 match clothes")
    if len(ranked_pairs)==0:
        st.write("No match.")
    else:
        for idx, (score, top, bottom) in enumerate(ranked_pairs[:3], 1):
            st.write(f"**{idx}. Score: {score:.2f}**")
            cols = st.columns(5)
        
            top_img = Image.open(top['image']).resize((100,100))
            cols[0].image(top_img, caption=top['name'])
            
            bot_img = Image.open(bottom['image']).resize((100,100))
            cols[1].image(bot_img, caption=bottom['name'])
        
            cols[2].write(f"Top color: {top['color']}")
            cols[3].write(f"Bottom color: {bottom['color']}")
           
            st.markdown("---")
            
st.subheader("My all clothes")
for idx, row in df.iterrows():
    cols = st.columns(5)
    img = Image.open(row['image']).resize((100,100))
    cols[0].image(img, caption=row['name'])
    cols[1].write(f"Type: {row['type']}")
    cols[2].write(f"Color: {row['color']}")
    cols[3].write(f"Purpose: {row['purpose']}")
    


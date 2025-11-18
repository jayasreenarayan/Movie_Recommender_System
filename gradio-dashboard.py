import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr

import os
from PIL import Image, ImageDraw, ImageFont

load_dotenv()

movies = pd.read_csv("movies_with_emotions.csv")
movies["large_thumbnail"] = movies.get("large_thumbnail", np.nan)
movies["large_thumbnail"] = movies["large_thumbnail"].fillna("")

THUMB_DIR = "thumbnails"
os.makedirs(THUMB_DIR, exist_ok=True)

def _safe_filename(s: str) -> str:
    return "".join(c for c in s if c.isalnum() or c in (" ", "_", "-")).strip() or "untitled"

def _initials(title: str) -> str:
    parts = [p for p in str(title).replace(":", " ").split() if p and p[0].isalpha()]
    return "".join([p[0].upper() for p in parts[:3]]) or "NO"

def _bg_color_from_title(title: str) -> tuple:
    h = abs(hash(title)) % 360
    r = int(28 + 60 * ((h % 60) / 60))
    g = int(44 + 80 * (((h + 120) % 60) / 60))
    b = int(60 + 110 * (((h + 240) % 60) / 60))
    return (r, g, b)

def generate_thumbnail(title: str, size=(300, 450)) -> str:
    name = _safe_filename(title)
    out_path = os.path.join(THUMB_DIR, f"{name}.jpg")
    if os.path.exists(out_path):
        return out_path

    img = Image.new("RGB", size, color=_bg_color_from_title(title))
    d = ImageDraw.Draw(img)
    initials = _initials(title)

    try:
        font_big = ImageFont.truetype("arial.ttf", 120)
        font_small = ImageFont.truetype("arial.ttf", 22)
    except:
        font_big = ImageFont.load_default()
        font_small = ImageFont.load_default()

    w, h = d.textsize(initials, font=font_big)
    d.text(((size[0]-w)//2, (size[1]-h)//2 - 30), initials, (235, 240, 245), font=font_big)

    subtitle = " ".join(str(title).split()[:2])
    sw, sh = d.textsize(subtitle, font=font_small)
    d.text(((size[0]-sw)//2, (size[1]//2)+70), subtitle, (245, 247, 248), font=font_small)

    img.save(out_path, "JPEG", quality=92)
    return out_path

raw_documents = TextLoader("tagged_description.txt",encoding="utf-8").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=100)
documents = text_splitter.split_documents(raw_documents)
db_movies = Chroma.from_documents(documents, OpenAIEmbeddings())


def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        title_filter: str = "",
        year_filter: str = "",
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    recs = db_movies.similarity_search(query, k=initial_top_k)
    movies_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    movie_recs = movies[movies["show_id"].isin(movies_list)].head(initial_top_k)

    if category != "All":
        movie_recs = movie_recs[movie_recs["simple_category"] == category].head(final_top_k)
    else:
        movie_recs = movie_recs.head(final_top_k)
    if title_filter:
        movie_recs = movie_recs[movie_recs["title"].fillna("").str.contains(title_filter.strip(), case=False, na=False)]

    if year_filter and year_filter.strip().isdigit() and "release_year" in movie_recs.columns:  # <-- Changed from df
        yr = int(year_filter.strip())
        movie_recs = movie_recs[movie_recs["release_year"] == yr]

    if tone == "Happy":
        movie_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        movie_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        movie_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        movie_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        movie_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return movie_recs


def recommend_movies(
        query: str,
        category: str,
        tone: str,
        title_filter: str,
        year_filter: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone,title_filter, year_filter)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        director_split = row["director"].split(";")
        if len(director_split) == 2:
            director_str = f"{director_split[0]} and {director_split[1]}"
        elif len(director_split) > 2:
            director_str = f"{', '.join(director_split[:-1])}, and {director_split[-1]}"
        else:
            director_str = row["director"]

        caption = f"{row['title']} by {director_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results
print("type(movies) =", type(movies))

category = ["All"] + sorted(movies["simple_category"].unique().tolist())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic movie recommender")

    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a movies:",
                                placeholder = "e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices = category, label = "Select a category:", value = "All")
        tone_dropdown = gr.Dropdown(choices = tones, label = "Select an emotional tone:", value = "All")
        submit_button = gr.Button("Find recommendations")
    with gr.Row():
        title_filter_tb = gr.Textbox(label="Filter by movie title (optional)", placeholder="e.g., Inception")
        year_filter_tb = gr.Textbox(label="Filter by year (optional)", placeholder="e.g., 2010")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label = "Recommended movies", columns = 8, rows = 2)

    submit_button.click(fn=recommend_movies,
                        inputs=[user_query,
                                category_dropdown,
                                tone_dropdown,
                                title_filter_tb,  # <-- Add this
                                year_filter_tb],  # <-- Add this
                        outputs=output)


if __name__ == "__main__":
    dashboard.launch()
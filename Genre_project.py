import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px

# Set the page configuration at the top
st.set_page_config(page_title="Movie Genre Popularity", layout="wide")

# Caching data and model loading for efficiency
@st.cache_data  
def load_data(): #Reads the CSV file, and returns selected columns
    return pd.read_csv(
        'C:\\Users\\AL-Rehmat\\Documents\\project practice\\df_movies.csv',
        usecols=['startYear', 'genres', 'originalTitle', 'averageRating']
    )

@st.cache_data
def load_model_and_scaler(): #Loads a pre-trained machine learning model
    model = pickle.load(open("C:\\Users\\AL-Rehmat\\Documents\\project practice\\random_forest_model.pkl", "rb"))
    scaler = pickle.load(open("C:\\Users\\AL-Rehmat\\Documents\\project practice\\scaler_model.pkl", "rb"))
    return model, scaler

def predict_votes(model, scaler, average_rating, start_year, genre_encoded_value):
    data = np.array([[average_rating, start_year, genre_encoded_value]])
    data_scaled = scaler.transform(data) #It normalizes the input data before feeding it to the model.
    return model.predict(data_scaled)[0]

# Load data and models
split_genre = load_data() #load the dataset and the model as per the previous caching functions.
model, scaler = load_model_and_scaler()

# App Design
st.title("🎥 Movie Genre and Popularity Prediction")
st.markdown("Explore the most popular movie genres for any year and predict their popularity!")

# Input Section
st.sidebar.header("🔍 Input Section")
year_input = st.sidebar.number_input('Enter Year', min_value=1900, max_value=2025, value=2025)
average_rating = st.sidebar.number_input('Enter Average Rating (0-10)', min_value=0.0, max_value=10.0, value=5.0)

filtered_data = split_genre[split_genre['startYear'] == year_input] #Filters the dataset based on the selected year.

if not filtered_data.empty:
    # Handle multiple genres
    filtered_data['genres_split'] = filtered_data['genres'].str.split(', ')
    genres_flat = filtered_data.explode('genres_split') #to expand the genres column so that each movie-genre pair is a separate row.
    top_genres = genres_flat['genres_split'].value_counts().head(2).index.tolist()

    # Results Display
    col1, col2 = st.columns(2) #Displays the top genres and top movies for those genres in a two-column layout.

    with col1:
        st.subheader(f"📅 Year: {year_input}")
        st.markdown(f"The **most popular genres** for {year_input} are:")
        for idx, genre in enumerate(top_genres, start=1): # for returning index
            st.markdown(f"**{idx}. {genre}**")

    with col2:
        top_movies = genres_flat[genres_flat['genres_split'].isin(top_genres)]
        for genre in top_genres:
            movie = top_movies[top_movies['genres_split'] == genre].iloc[0]
            st.markdown(f"🎬 **Top Movie in {genre}:** {movie['originalTitle']}")

    # Genre Popularity Bar Chart
    st.subheader("📊 Genre Popularity")
    genre_counts = genres_flat['genres_split'].value_counts().head(10)  # Top 10 genres
    fig = px.bar(
        x=genre_counts.index,
        y=genre_counts.values,
        labels={'x': 'Genres', 'y': 'Number of Movies'},
        title=f"Top Genres in {year_input}",
        color=genre_counts.index,
        template='plotly_white'
    )
    st.plotly_chart(fig)

    # Enhanced Visualization for Ratings Distribution
    st.subheader("🎨 Movie Ratings Distribution by Genre")

    # Create a scatter plot using Plotly
    fig = px.scatter(
        genres_flat,
        x='genres_split',
        y='averageRating',
        color='genres_split',
        size='averageRating',
        title=f"Movie Ratings Distribution by Genre ({year_input})",
        labels={'averageRating': 'Average Rating', 'genres_split': 'Genres'},
        hover_data=['originalTitle'],
        template='plotly_white'
    )

    # Customize layout for better visuals
    fig.update_layout(
        xaxis_title="Genres",
        yaxis_title="Average Rating",
        xaxis=dict(tickangle=45, categoryorder='total descending'),
        showlegend=False,
    )

    # Add reference line for the average rating
    avg_rating = genres_flat['averageRating'].mean()
    fig.add_hline(
        y=avg_rating,
        line_dash="dot",
        annotation_text=f"Avg Rating: {avg_rating:.2f}",
        annotation_position="bottom right",
        line_color="red",
    )

    # Display the plot
    st.plotly_chart(fig)

    # Encode genres and predict
    unique_genres = split_genre['genres'].str.split(', ').explode().unique()
    genre_encoded = {genre: idx for idx, genre in enumerate(unique_genres)}

    genre_encoded_values = [genre_encoded.get(genre, -1) for genre in top_genres]
    st.write("")

    # Prediction Section
    st.subheader("🔮 Popularity Prediction")
    if st.button("Predict Number of Votes for Top Genre"):
        for genre, encoded_value in zip(top_genres, genre_encoded_values):
            if encoded_value != -1:
                predicted_votes = predict_votes(model, scaler, average_rating, year_input, encoded_value)
                st.markdown(f"📊 Predicted votes for **{genre}:** {predicted_votes:.2f}")
            else:
                st.warning(f"Could not encode genre: {genre}")

    # Votes Prediction Line Chart
    st.subheader("📈 Predicted Votes Over Years")
    years = list(range(year_input - 5, year_input + 1))  # creates a line chart showing how the predicted number of votes changes over the past 5 years.
    predicted_votes = [predict_votes(model, scaler, average_rating, year, genre_encoded_values[0]) for year in years]
    fig = px.line(
        x=years,
        y=predicted_votes,
        title="Predicted Votes Over Years",
        labels={"x": "Year", "y": "Predicted Votes"},
        template='plotly_white'
    )
    st.plotly_chart(fig)

else:
    st.error(f"No data available for the year {year_input}. Please try another year.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("📌 **Pro Tip:** Adjust the year and rating to see dynamic predictions!")
st.sidebar.markdown("Developed with ❤️ using Streamlit.")

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

train_data = pd.read_parquet("data.parquet")
mandate_embeddings = pd.read_parquet("mandate_embeddings.parquet")

party_to_colors = {
    "CDU/CSU": "black",
    "fraktionslos": "gray",
    # "Die Linke. (Gruppe) (Bundestag 2021 - 2025)": "purple",
    "DIE LINKE": "purple",
    "BSW": "orange",
    "FDP": "yellow",
    "BÜNDNIS 90/DIE GRÜNEN": "green",
    "AfD": "blue",
    "SPD": "red",
}

chart = (
    alt.Chart(mandate_embeddings)
    .mark_circle()
    .encode(
        x=alt.X("dim_0", title="", axis=None),
        y=alt.Y("dim_1", title="", axis=None),
        color=alt.Color(
            "fraction",
            title="Fraction",
            scale=alt.Scale(
                domain=list(party_to_colors.keys()),
                range=list(party_to_colors.values()),
            ),
        ),
        tooltip=[
            alt.Tooltip("mandate", title="Name"),
            alt.Tooltip("mandate_id", title="Mandate ID"),
            alt.Tooltip("fraction", title="Fraction"),
        ],
    )
    .properties(
        width=700,
        height=400,
        title="Mandates of the German Bundestag 2021 - 2025, clustered by polling behaviour",
    )
    .configure_view(
        strokeWidth=0
    )
    .interactive()
)


st.set_page_config(page_title="Politician Embeddings")
st.title("German Politician Mandates Embeddings")
st.markdown("I created embeddings of German politicians from the current legislative period of the Bundestag (2021 - 2025) based on their behavior in polls. I collected the poll data via the [API of abgeordnetenwatch.de](https://www.abgeordnetenwatch.de/api). It's just a fun project, so don't take it too seriously 😉.")
st.markdown("**Last update: 15.04.2024. Later polls are not included.**")
st.write("#")
st.altair_chart(chart, use_container_width=True)
st.markdown("""
    ## Results
    The individual points in the graphic each represent a politician. If two politicians are close to each other, 
    this means that they had similar poll behavior. It is important to note that I **did not include their parties** in the 
    algorithm, but really only **how the politicians voted** in votes taken from [here](https://www.abgeordnetenwatch.de/bundestag/abstimmungen). 
    
    I only added the colors afterwards. Hence, it is interesting to see that there are still four clear clusters that correspond to the parties:

    1. the government cluster, consisting of the SPD, BÜNDNIS 90/DIE GRÜNEN and the FDP
    2. the CDU cluster
    3. the AfD cluster
    4. the Linke/BSW cluster

    This means that usually, parties vote as one, apart from the small remainder cluster, and also some single politicians
    being in clusters of other parties but their own. Also note how the governing parties form a single cluster.
    Within this cluster, SPD and BÜNDNIS 90/DIE GRÜNEN overlap a lot according to the polling behavior of their members,
    while the FDP stands out a little. Still, all in all, the governing parties also mainly act as one when voting.
""")
st.markdown("""
    ## Methodology
    I will now walk you through how I got to this result.
    ### Data
    I used the API of abgeordnetenwatch.de to find out which politician voted for what. In the end, I had data like this:
""")
st.write(train_data.filter(["mandate", "poll", "vote"]).head(10))
st.markdown("""
    I excluded cases where politicians did not show up or did not vote for other reasons. I counted only a clear **yes**
    or **no**. 
    
    ### Model
    Then, I trained a simple matrix factorization algorithm to train a model that tries to predict whether a politician
    would vote yes or no. To be more precise, I 
    1. embedded each politician and each poll into a 8-dimensional vector
    2. took the dot product between these vectors
    3. output "yes" if the dot product is larger than 0, otherwise "no".
""")
with st.expander("See code"):
    st.code("""
        [...]
        data = pl.read_parquet("data.parquet").sample(fraction=1.0, shuffle=True)

        all_mandates = data["mandate_id"].unique().to_list()
        all_polls = data["poll_id"].unique().to_list()

        mandate_input = tf.keras.layers.Input(shape=(1,), name="mandate_id")
        mandate_as_integer = tf.keras.layers.IntegerLookup(vocabulary=all_mandates)(mandate_input)
        mandate_embedding = tf.keras.layers.Embedding(input_dim=len(all_mandates) + 1, output_dim=8)(mandate_as_integer)

        poll_input = tf.keras.layers.Input(shape=(1,), name="poll_id")
        poll_as_integer = tf.keras.layers.IntegerLookup(vocabulary=all_polls)(poll_input)
        poll_embedding = tf.keras.layers.Embedding(input_dim=len(all_polls) + 1, output_dim=8)(poll_as_integer)

        dot = tf.keras.layers.Dot(axes=2)([mandate_embedding, poll_embedding])
        flatten = tf.keras.layers.Flatten()(dot)
        squash = tf.keras.layers.Lambda(tf.nn.sigmoid)(flatten)

        model = tf.keras.Model(inputs=[mandate_input, poll_input], outputs=flatten)
        model.compile(loss="bce", metrics=[tf.keras.metrics.BinaryAccuracy()])

        model.fit(
            x={"mandate_id": data["mandate_id"], "poll_id": data["poll_id"]},
            y=data["vote_label"],
            epochs=15,
        )
    """)
st.markdown("""
    ### Dimensionality Reduction
    After the training, I could represent each politician and poll as a vector consisting of 8 numbers.
    Since this is hard to visualize, I reduced the dimension to 2 using [UMAP](https://umap-learn.readthedocs.io/en/latest/) (Uniform Manifold Approximation and Projection for Dimension Reduction). The result was what you can see in the chart now.
""")
st.markdown("""
    Created by Robert Kübler
    [LinkedIn](https://www.linkedin.com/in/robert-kuebler/)
""")

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

import librosa
import librosa.display
import numpy as np
import pickle
import tempfile
import matplotlib.pyplot as plt
from openai import OpenAI

client = OpenAI()

st.set_page_config(page_title="Analyse Audio", layout="wide")


# On charge le modele une seule fois au demarrage
@st.cache_resource
def charger_modele():
    with open("model_emotion.pkl", "rb") as f:
        donnees = pickle.load(f)
    return donnees["model"], donnees["scaler"], donnees["le"]


# On extrait les caracteristiques audio qui permettent au modele de reconnaitre les emotions
def analyser_audio(chemin_fichier):
    son, frequence = librosa.load(chemin_fichier, duration=3)

    # Les MFCC representent l empreinte de la voix
    mfcc = librosa.feature.mfcc(y=son, sr=frequence, n_mfcc=40)
    mfcc_moyenne = np.mean(mfcc.T, axis=0)
    mfcc_variation = np.std(mfcc.T, axis=0)

    # Le chroma capture les harmoniques de la voix
    chroma = librosa.feature.chroma_stft(y=son, sr=frequence)
    chroma_moyenne = np.mean(chroma.T, axis=0)

    # Le mel spectrogram represente l energie par frequence
    mel = librosa.feature.melspectrogram(y=son, sr=frequence)
    mel_moyenne = np.mean(mel.T, axis=0)

    # Le taux de passage par zero indique si la voix est douce ou agitee
    zcr = np.mean(librosa.feature.zero_crossing_rate(son))

    # Le RMS mesure le volume general de la voix
    volume = np.mean(librosa.feature.rms(y=son))

    # Le pitch correspond a la hauteur de la voix
    hauteurs, _ = librosa.piptrack(y=son, sr=frequence)
    pitch_moyen = np.mean(hauteurs[hauteurs > 0]) if np.any(hauteurs > 0) else 0
    pitch_variation = np.std(hauteurs[hauteurs > 0]) if np.any(hauteurs > 0) else 0

    # Le tempo mesure la vitesse du debit de parole
    tempo, _ = librosa.beat.beat_track(y=son, sr=frequence)
    tempo = float(np.atleast_1d(tempo)[0])

    return np.hstack([
        mfcc_moyenne, mfcc_variation, chroma_moyenne, mel_moyenne,
        [zcr, volume, pitch_moyen, pitch_variation, tempo]
    ])


# Le modele se charge automatiquement au demarrage
modele, normaliseur, encodeur = charger_modele()

# En-tete de la page
st.title("Analyse Audio par Intelligence Artificielle")
st.caption("Depose un fichier audio pour obtenir la transcription et detecter l emotion de la voix")
st.divider()

# Menu lateral avec les infos du projet
with st.sidebar:
    st.header("A propos du projet")
    st.write(
        "Cette application analyse des fichiers audio en deux etapes : "
        "elle transcrit la parole avec Whisper d OpenAI, "
        "puis detecte l emotion de la voix grace a un modele entraine sur le dataset RAVDESS."
    )
    st.divider()
    st.caption("Details du modele")
    st.write("Algorithme : Reseau de neurones MLP")
    st.write("Accuracy : 90%")
    st.write("Dataset : RAVDESS — 1440 fichiers audio")
    st.write("Emotions : neutre, calme, joie, tristesse, colere, peur, degout, surprise")


# Zone principale — upload du fichier
fichier = st.file_uploader("Selectionne un fichier audio WAV", type=["wav"])

if fichier:

    # On sauvegarde temporairement le fichier pour pouvoir l analyser
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(fichier.read())
        chemin_tmp = tmp.name

    # Lecteur audio integre
    st.audio(fichier)
    st.divider()

    colonne_gauche, colonne_droite = st.columns(2, gap="large")

    # Colonne gauche — Transcription avec Whisper
    with colonne_gauche:
        st.subheader("Ce qui a ete dit")
        with st.spinner("Whisper analyse la parole..."):
            try:
                resultat = client.audio.transcriptions.create(
                    model="gpt-4o-mini-transcribe",
                    file=open(chemin_tmp, "rb"),
                    response_format="text"
                )
                st.info(resultat)
            except Exception as e:
                st.error(f"Erreur lors de la transcription : {e}")

    # Colonne droite — Detection de l emotion
    with colonne_droite:
        st.subheader("Emotion ressentie dans la voix")

        # On extrait les caracteristiques du fichier audio
        caracteristiques = analyser_audio(chemin_tmp)
        caracteristiques_normalisees = normaliseur.transform([caracteristiques])

        # Le modele predit l emotion
        prediction_encodee = modele.predict(caracteristiques_normalisees)[0]
        emotion = encodeur.inverse_transform([prediction_encodee])[0]
        probabilites = modele.predict_proba(caracteristiques_normalisees)[0]

        # On affiche le resultat principal
        st.metric(label="Emotion detectee", value=emotion.upper())

        # On affiche les probabilites pour chaque emotion
        st.write("Niveau de confiance par emotion :")
        for nom_emotion, proba in sorted(
            zip(encodeur.classes_, probabilites),
            key=lambda x: -x[1]
        ):
            st.progress(float(proba), text=f"{nom_emotion} — {proba*100:.1f}%")

    st.divider()

    # Visualisation du signal audio
    st.subheader("Visualisation du signal audio")
    st.caption("Le spectrogramme montre les frequences dans le temps. Les MFCC representent l empreinte de la voix utilisee par le modele.")

    son, frequence = librosa.load(chemin_tmp)
    mfcc = librosa.feature.mfcc(y=son, sr=frequence, n_mfcc=13)

    fig, axes = plt.subplots(2, 1, figsize=(12, 5))

    librosa.display.specshow(
        librosa.amplitude_to_db(np.abs(librosa.stft(son)), ref=np.max),
        sr=frequence, x_axis="time", y_axis="hz",
        ax=axes[0], cmap="magma"
    )
    axes[0].set_title("Spectrogramme — frequences dans le temps", fontsize=10)

    librosa.display.specshow(mfcc, sr=frequence, x_axis="time", ax=axes[1], cmap="magma")
    axes[1].set_title("MFCC — empreinte de la voix utilisee par le modele", fontsize=10)

    plt.tight_layout()
    st.pyplot(fig)

    # On supprime le fichier temporaire
    os.unlink(chemin_tmp)
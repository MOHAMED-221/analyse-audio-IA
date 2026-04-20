from dotenv import load_dotenv
import os
load_dotenv()

from openai import OpenAI
import tkinter as tk
from tkinter import filedialog
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

client = OpenAI()

def choose_audio_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Choisis un fichier audio (.wav uniquement)",
        filetypes=[("Fichiers WAV", "*.wav")]
    )
    return file_path if file_path else None

audio_file = choose_audio_file()

if audio_file:
    print(f"Fichier sélectionné : {audio_file}")

    # Transcription
    print("Transcription en cours...")
    transcription = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=open(audio_file, "rb"),
        response_format="text"
    )
    print("\nTranscription :")
    print(transcription)

    # Chargement et analyse
    y, sr = librosa.load(audio_file)
    print(f"\nDurée : {librosa.get_duration(y=y, sr=sr):.2f} secondes")

    # MFCC et Spectrogramme
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    librosa.display.specshow(
        librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max),
        sr=sr, x_axis="time", y_axis="hz", ax=axes[0]
    )
    axes[0].set_title("Spectrogramme")

    librosa.display.specshow(mfcc, sr=sr, x_axis="time", ax=axes[1])
    axes[1].set_title("MFCC (13 coefficients)")

    plt.tight_layout()
    plt.savefig("mfcc_output.png")
    print("Graphique sauvegardé : mfcc_output.png")
    plt.show()

else:
    print("Aucun fichier sélectionné.")
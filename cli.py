import argparse
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.io as v_io

from src import models, utils


def load_models(video_weights, video_transformer_size, music_weights, music_transformer_size, device):
    video_model = models.VideoTransformer(num_layers=video_transformer_size, query_dim=512)
    music_model = models.MusicTransformer(num_layers=music_transformer_size, query_dim=512)
    
    video_model.load_state_dict(torch.load(video_weights, map_location=device))
    music_model.load_state_dict(torch.load(music_weights, map_location=device))
    
    video_model.to(device).eval()
    music_model.to(device).eval()
    
    return video_model, music_model


def get_embedding(model, preprocessed_data, device):
    """Generuje wektor osadzenia (embedding) i normalizuje go."""
    with torch.no_grad():
        inputs = preprocessed_data.to(device)
        # Zakładamy, że Twój model zwraca wektor o wymiarach [1, embedding_dim]
        embedding = model(inputs)
        # Normalizacja L2 jest kluczowa dla podobieństwa cosinusowego
        embedding = F.normalize(embedding, p=2, dim=-1)
    return embedding


def build_db_segments(db_dir, db_type, model, segment_length, device, tmpdir):
    """Przecina pliki bazy danych na segmenty i generuje dla nich wektory."""
    db_segments = []
    extension = "*.mp4" if db_type == "video" else "*.wav"
    db_files = list(db_dir.glob(extension))

    print(f"\n-> Budowanie wektorów dla bazy ({db_type})... Znaleziono {len(db_files)} plików.")

    for _, file_path in enumerate(db_files):
        try:
            if db_type == "video":
                db_segments.extend(utils.media.split_video_into_segments(
                    source_path=file_path,
                    output_dir=tmpdir,
                    segment_length=segment_length,
                ))
            else:
                db_segments.extend(utils.media.split_audio_into_segments(
                    source_path=file_path,
                    output_dir=tmpdir,
                    segment_length=segment_length,
                    sample_rate=16000,
                    channels=1
                ))
        except Exception as e:
            print(f"Błąd przetwarzania pliku z bazy {file_path}: {e}")
            sys.exit(1)

    return db_segments


def match(query_path, db_dir, query_model, db_model, query_type, segment_length, device):
    """Główny silnik dopasowywania i montażu segmentowego."""
    
    query_path = Path(query_path)
    db_dir = Path(db_dir)

    if not query_path.exists():
        print(f"[ERROR]: File '{query_path}' doesn't exist.")
        sys.exit(1)

    db_type = "music" if query_type == "video" else "video"
    output_path = f"{query_path.name}_out.mp4"

    # Używamy katalogu tymczasowego, żeby nie zaśmiecić dysku użytkownika
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. ZBUDUJ BAZĘ KANDYDATÓW
        db_segments = build_db_segments(db_dir, db_type, db_model, segment_length, device, tmpdir)
        if not db_segments:
            print("Błąd: Baza danych jest pusta lub nie udało się wyodrębnić segmentów.")
            return

        # 2. PRZETWÓRZ ZAPYTANIE
        print(f"\n-> Analiza pliku zapytania: {query_path.name}")

        if query_type == "video":
            query_segments = utils.media.split_video_into_segments(
                source_path=query_path,
                output_dir=tmpdir,
                segment_length=segment_length,
            )
        else:
            query_segments = utils.media.split_audio_into_segments(
                source_path=query_path,
                output_dir=tmpdir,
                segment_length=segment_length,
                sample_rate=16000,
                channels=1
            )
        
        for segment in query_segments:
            if query_type == "video":
                segment_tensor, _, __ = v_io.read_video(segment, pts_unit="sec")
                loaded_video.append(video_tensor)

                min_frames = min(t.shape[0] for t in loaded_video)
                formatted_tensors = [t[:min_frames, ...] for t in loaded_video]
            else:
                q_subclip.write_audiofile(temp_path, logger=None)
                data = preprocess_music(temp_path)

            query_emb = get_embedding(query_model, data, device)

            # Szukanie najlepszego dopasowania w bazie (argmax)
            best_score = -1.0
            best_clip_info = None

            for db_seg in db_segments:
                score = torch.dot(query_emb.squeeze(), db_seg["embedding"].squeeze()).item()
                if score > best_score:
                    best_score = score
                    best_clip_info = db_seg

            # Zabezpieczenie dla ostatniego, ewentualnie krótszego segmentu
            selected_clip = best_clip_info["clip"]
            if seg_duration < segment_length:
                selected_clip = selected_clip.subclip(0, seg_duration)

            print(f"   [Czas {start_t:04.1f}s - {end_t:04.1f}s] Dopasowano: {best_clip_info['name']} (Score: {best_score:.4f})")
            chosen_db_clips.append(selected_clip)

        # 3. ZŁĄCZENIE I RENDEROWANIE (ASSEMBLY)
        print("\n-> Renderowanie końcowego pliku wideo...")
        
        if query_type == "video":
            # Szukaliśmy muzyki: sklejamy muzykę i doczepiamy do oryginalnego wideo
            final_audio = concatenate_audioclips(chosen_db_clips)
            final_video = query_clip.set_audio(final_audio)
        else:
            # Szukaliśmy wideo: sklejamy klipy wideo i doczepiamy do oryginalnej muzyki
            final_video = concatenate_videoclips(chosen_db_clips, method="compose")
            final_video = final_video.set_audio(query_clip)

        # Zapis pliku wynikowego
        final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")
        
        # Zamykanie zasobów (zwalnia RAM)
        query_clip.close()
        final_video.close()
        print(f"\n✅ Zakończono pomyślnie! Zapisano plik: {output_path}\n")


def main(args: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    video_model, music_model = load_models(
        video_weights=args["video_weights"],
        video_transformer_size=args["video_transformer_size"],
        music_weights=args["music_weights"],
        music_transformer_size=args["music_transformer_size"],
        device=device
    )

    if args["command"] == "video2music":
        match(
            query_path=args["video"],
            db_dir=args["music_db"],
            query_model=video_model,
            db_model=music_model,
            query_type="video",
            segment_length=args["segment_length"],
            device=device
        )
    elif args["command"] == "music2video":
        match(
            query_path=args["music"],
            db_dir=args["video_db"],
            query_model=music_model,
            db_model=video_model,
            query_type="music",
            segment_length=args["segment_length"],
            device=device
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CLI do dopasowywania muzyki do wideo (i na odwrót) przy użyciu Transformerów kontrastowych."
    )
    
    # Tworzenie podkomend (subparsers)
    subparsers = parser.add_subparsers(dest="command", required=True, help="Wybierz tryb wyszukiwania")

    # Podkomenda: Wideo -> Muzyka
    v2m_parser = subparsers.add_parser("video2music", help="Znajdź najlepiej pasującą muzykę do podanego wideo")
    v2m_parser.add_argument("--video", "-v", required=True, type=str, help="Ścieżka do wejściowego pliku wideo (.mp4)")
    v2m_parser.add_argument("--music-db", "-m", required=True, type=str, help="Katalog z bazą utworów muzycznych")

    # Podkomenda: Muzyka -> Wideo
    m2v_parser = subparsers.add_parser("music2video", help="Znajdź najlepiej pasujące wideo do podanego utworu muzycznego")
    m2v_parser.add_argument("--music", "-m", required=True, type=str, help="Ścieżka do wejściowego pliku audio (.mp3 / .wav)")
    m2v_parser.add_argument("--video-db", "-db", required=True, type=str, help="Katalog z bazą klipów wideo")

    # Wspólne argumenty konfiguracyjne dodawane do obu podkomend
    for p in [v2m_parser, m2v_parser]:
        p.add_argument("--video-weights", default="models/video_transformer.pth", type=str, help="Ścieżka do wag video_transformer")
        p.add_argument("--video-transformer-size", default=3, type=int, help="Ilość warstw w video_transformer")
        p.add_argument("--music-weights", default="models/music_transformer.pth", type=str, help="Ścieżka do wag music_transformer")
        p.add_argument("--music-transformer-size", default=3, type=int, help="Ilość warstw w music_transformer")
        p.add_argument("--segment-length", default=5, type=int, help="Długość segmentów")

    args = vars(parser.parse_args())
    main(args)
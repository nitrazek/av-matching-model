import argparse
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

from src import models


def load_models(video_weights, music_weights, device):
    video_model = models.VideoTransformer()
    music_model = models.MusicTransformer()
    
    # Ładowanie wag (.pth)
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


def run_retrieval(query_path, db_dir, query_model, db_model, query_type, top_k, device):
    """Główna pętla przeszukująca bazę danych."""
    query_path = Path(query_path)
    db_dir = Path(db_dir)
    
    if not query_path.exists():
        print(f"Błąd: Plik zapytania {query_path} nie istnieje.")
        sys.exit(1)
    if not db_dir.is_dir():
        print(f"Błąd: Katalog bazy danych {db_dir} nie istnieje.")
        sys.exit(1)

    print(f" Przetwarzanie pliku zapytania: {query_path.name}...")
    if query_type == "video":
        query_data = preprocess_video(query_path)
    else:
        query_data = preprocess_music(query_path)
        
    query_emb = get_embedding(query_model, query_data, device)

    # Szukamy plików kandydatów w bazie danych
    # Możesz zmienić rozszerzenia (np. *.mp3, *.wav, *.mp4)
    extensions = ["*.mp4", "*.avi", "*.mkv"] if query_type == "music" else ["*.mp3", "*.wav", "*.flac"]
    candidate_paths = []
    for ext in extensions:
        candidate_paths.extend(list(db_dir.glob(ext)))

    if not candidate_paths:
        print(f"Błąd: Nie znaleziono żadnych plików kandydatów w {db_dir} z rozszerzeniami {extensions}")
        return

    print(f" Przetwarzanie i porównywanie {len(candidate_paths)} kandydatów z bazy danych...")
    results = []

    for path in candidate_paths:
        # Ekstrakcja cech i embedding dla kandydata
        if query_type == "video": # szukamy muzyki dla wideo
            cand_data = preprocess_music(path)
            cand_emb = get_embedding(db_model, cand_data, device)
        else: # szukamy wideo dla muzyki
            cand_data = preprocess_video(path)
            cand_emb = get_embedding(db_model, cand_data, device)
            
        # Obliczenie podobieństwa cosinusowego (iloczyn skalarny dla znormalizowanych wektorów)
        similarity = torch.dot(query_emb.squeeze(), cand_emb.squeeze()).item()
        results.append((path, similarity))

    # Sortowanie wyników malejąco po podobieństwie
    results.sort(key=lambda x: x[1], reverse=True)

    # Wyświetlenie wyników
    print(f"\n=== NAJLEPSZE DOPASOWANIA (Top {top_k}) ===")
    for i, (path, score) in enumerate(results[:top_k], 1):
        print(f"{i}. [{score:.4f}] -> {path.name}")
    print("=========================================\n")


def main(args: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    video_model, music_model = load_models(args["video_weights"], args["music_weights"], device)

    if args.command == "video2music":
        run_retrieval(
            query_path=args["video"],
            db_dir=args["music_db"],
            query_model=video_model,
            db_model=music_model,
            query_type="video",
            top_k=args["top_k"],
            device=device
        )
    elif args.command == "music2video":
        run_retrieval(
            query_path=args["music"],
            db_dir=args["video_db"],
            query_model=music_model,
            db_model=video_model,
            query_type="music",
            top_k=args["top_k"],
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
        p.add_argument("--music-weights", default="models/music_transformer.pth", type=str, help="Ścieżka do wag music_transformer")
        p.add_argument("--top-k", default=3, type=int, help="Liczba zwracanych najlepszych wyników")

    args = vars(parser.parse_args())
    main(args)
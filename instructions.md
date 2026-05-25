# Instrukcje korzystania z projektu

## Przygotowanie zbiorów danych

### Utwórz foldery na surowe dane:

```powershell
python scripts/prepare_data.py --init
```

### Umieść pliki źródłowe tutaj:

- `data/raw/videos`
- `data/raw/audio`

Pliki audio i wideo muszą mieć ten sam trzon nazwy, na przykład `sample_001.mp4` i `sample_001.wav`.

### Wygeneruj dane treningowe/walidacyjne:

```powershell
python scripts/prepare_data.py --overwrite
```

Przygotowane pliki i manifesty zostaną zapisane w katalogu `data/processed`.

## Trenowanie

Do udokumentowania.

## CLI

### Jak działa

- System analizuje docelowy katalog (tzw. "bazę danych") z plikami multimedialnymi, dzieli je na segmenty o stałej długości i przepuszcza przez multimodalny model Transformer w celu ekstrakcji osadzeń (embeddings) znormalizowanych w normie L2.
- Plik wejściowy zapytania (wideo lub ścieżka muzyczna) jest podobnie segmentowany i kodowany.
- Dla każdego segmentu w zapytaniu, narzędzie oblicza wynik podobieństwa względem wszystkich segmentów w bazie danych i wybiera dopasowanie o najwyższym wyniku.
- Wybrane segmenty są płynnie łączone (konkatenowane) i multipleksowane z oryginalnym medium zapytania za pomocą narzędzia ffmpeg, aby utworzyć końcowy plik wideo.

### Wymagania wstępne

- Środowisko Pythona: PyTorch, TorchAudio, TorchVision oraz tqdm.
- Narzędzia systemowe: ffmpeg musi być zainstalowany i dostępny w zmiennej środowiskowej PATH systemu, ponieważ obsługuje on niskopoziomową segmentację, konkatenację i multipleksowanie multimediów.
- Wagi Modeli: Wytrenowane wagi dla transformerów wideo oraz muzycznego muszą być wcześniej wytrenowane i dostępne lokalnie.

### Instrukcja użytkowania

Narzędzie CLI działa w oparciu o dwa główne polecenia podrzędne (subcommands) w zależności od pożądanego przepływu pracy: `video2music` i `music2video`.

1. video2music
Znajduje najlepiej dopasowane segmenty audio z muzycznej bazy danych w celu połączenia ich z wprowadzonym wideo.

Składnia:
```powershell
python cli.py video2music --video <path_to_video> --music-db <path_to_audio_folder> [OPTIONS]
```

Przykład:
```powershell
python cli.py video2music \
    --video ./data/raw/videos/video.mp4 \
    --music-db ./data/raw/audio \
    --segment-length 5 \
    --output ./outputs/vlog_scored.mp4
```

2. music2video
Znajduje najlepiej dopasowane klipy wideo z bazy danych wideo, które będą towarzyszyć wprowadzonej ścieżce dźwiękowej.

Składnia:
```powershell
python cli.py music2video --music <path_to_audio> --video-db <path_to_video_folder> [OPTIONS]
```

Przykład:
```powershell
python cli.py music2video \
    --music ./inputs/beat_track.wav \
    --video-db ./assets/b-roll_library \
    --segment-length 3 \
    --output ./outputs/music_video.mp4
```

### Konfiguracja

Poniższe argumenty mają zastosowanie do obu poleceń podrzędnych:

| Argument | Typ | Domyślnie | Opis |
| :--- | :--- | :--- | :--- |
| `--segment-length` | Integer | `5` | Długość (w sekundach), na jaką pliki źródłowe i bazodanowe mają zostać podzielone do dopasowania. |
| `--output`, `-o` | String | `<query_stem>_out.mp4` | Ścieżka i nazwa dla wygenerowanego pliku wideo. |
| `--video-weights` | String | `models/video_transformer.pth` | Ścieżka do pliku z wagami .pth dla modelu VideoTransformer. |
| `--video-transformer-size` | Integer | `3` | Liczba warstw zdefiniowana w architekturze VideoTransformera. |
| `--music-weights` | String | `models/music_transformer.pth` | Ścieżka do pliku z wagami .pth dla modelu MusicTransformer. |
| `--music-transformer-size` | Integer | `3` | Liczba warstw zdefiniowana w architekturze Music Transformera. |
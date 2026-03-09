# WIMU Design Proposal

Projekt stanowi implementację modelu zaproponowanego w artykule naukowym pt. „It’s Time for Artistic Correspondence in Music and Video”, w którym autorzy przedstawili innowacyjne podejście do automatycznego dopasowywania ścieżki dźwiękowej do materiału wideo oraz wideo do muzyki oraz przygotowanie praktycznego interfejsu CLI. Głównym celem naszej pracy jest rzetelne odtworzenie wyników bazowych na zupełnie nowym zbiorze danych oraz szczegółowe przetestowanie modelu w zróżnicowanych scenariuszach użytkowych. Aby dostosować tę technologię do wymagań rzeczywistych aplikacji, proponujemy autorskie usprawnienia zwiększające jej elastyczność i precyzję. Kluczowe modyfikacje obejmują przede wszystkim zbadanie wpływu długości fragmentów, na jakie dzielony jest film, na synchronizację z audio, a także wprowadzenie inteligentnej opcji braku przypisania audio w momentach, w których system uzna, że dany segment filmu nie wymaga podkładu muzycznego.

## Założenia projektu

### Stos technologiczny

- **Środowisko i jakość kodu:** uv, Python >= 3.12, pytest
- **Uczenie maszynowe:** PyTorch
- **Analiza danych i wizualizacja:** Scikit-learn, NumPy, Pandas, Matplotlib, Seaborn
- **Śledzenie eksperymentów:** wandb lub MLflow
- **Przetwarzanie multimediów:** FFmpeg do obsługi plików MP4 i dodawania ścieżki dźwiękowej
- **Interfejs użytkownika:** Click do przygotowania CLI

### Planowana funkcjonalność

- Dobieranie ścieżki audio dla podanego pliku MP4.
- Dobieranie ścieżki audio dla wskazanego fragmentu czasowego z podanego pliku MP4.

## Zakres eksperymentów

Punktem wyjścia będzie zbiór danych opisany w pozycji [2]. Plan obejmuje zarówno odtworzenie wyników referencyjnych, jak i rozszerzenie badań o scenariusze bardziej zbliżone do praktycznego użycia systemu.

- Odtworzenie eksperymentów z artykułu [1].
- Predykcję dla pojedynczych fragmentów filmu.
- Predykcję dla całego materiału wideo.
- Analizę wpływu długości fragmentów wideo na jakość predykcji.
- Rozszerzenie zbioru o przykłady, w których nie występuje muzyka w tle.
- Uwzględnienie dialogu w połączeniu z klipami wideo.

## Harmonogram

| Termin | Zakres prac |
| --- | --- |
| 12.03 - 19.03 | Analiza pracy naukowej i wybór architektury modelu |
| 20.03 - 26.03 | Implementacja prototypowego modelu, trening na małym zbiorze danych oraz pierwsze testy |
| 27.03 - 02.04 | Budowa modułu obsługi plików MP4: odczyt, zapis, wycinanie fragmentów oraz dodawanie ścieżki dźwiękowej |
| 03.04 - 26.04 | Trening modelu i eksperymenty na pełnym zbiorze danych |
| 27.04 - 03.05 | Dodanie doboru ścieżki dla wybranego fragmentu czasowego lub całego pliku |
| 04.05 - 10.05 | Przygotowanie prostego interfejsu CLI do wykorzystania modelu |
| 11.05 - 17.05 | Finalne testy gotowego programu |
| 18.05 - 24.05 | Finalizacja raportu |

## Bibliografia

1. Surís, Dídac, et al. *It's Time for Artistic Correspondence in Music and Video.* Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022. [arXiv:2206.07148](https://arxiv.org/pdf/2206.07148)
2. Tian, Zeyue, et al. *VidMuse: A Simple Video-to-Music Generation Framework with Long-Short-Term Modeling.* Proceedings of the Computer Vision and Pattern Recognition Conference, 2025. [arXiv:2406.04321](https://arxiv.org/abs/2406.04321)
3. Radford, Alec, et al. *Learning Transferable Visual Models from Natural Language Supervision.* International Conference on Machine Learning, PMLR, 2021. [PMLR v139](https://proceedings.mlr.press/v139/radford21a)
4. Lee, Jongpil, et al. *Metric Learning vs Classification for Disentangled Music Representation Learning.* arXiv preprint arXiv:2008.03729, 2020. [arXiv:2008.03729](https://arxiv.org/abs/2008.03729)

import os
import re
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import cv2
from skimage.metrics import structural_similarity as ssim



def psnr_numpy_per_channel(img_ref: np.ndarray, img_test: np.ndarray, data_range: float = 255.0) -> float:
    """
    PSNR między img_ref i img_test w dB, obliczany osobno dla każdego kanału RGB i uśredniony.
    - img_ref, img_test: np.ndarray (H,W,3), ten sam kształt
    - data_range: 255.0 dla 8-bit; nie zmieniać jeśli trzymamy się 0..255
    """
    if img_ref.shape != img_test.shape:
        raise ValueError("Obrazy muszą mieć ten sam kształt")
    
    # Sprawdź czy obrazy są kolorowe (3 kanały)
    if img_ref.shape[-1] != 3:
        raise ValueError("Obrazy muszą mieć 3 kanały RGB")
    
    # Oblicz PSNR dla każdego kanału osobno
    psnr_values = []
    for c in range(3):  # RGB
        x = img_ref[..., c].astype(np.float64)
        y = img_test[..., c].astype(np.float64)
        mse = np.mean((x - y) ** 2)
        if mse == 0:
            psnr_values.append(float("100")) #wartość dąży do 100 dB
        else:
            psnr = 10.0 * np.log10((data_range ** 2) / mse)
            psnr_values.append(psnr)
    
    # Zwróć średnią z trzech kanałów
    return np.mean(psnr_values)

# =============================
# Nazwy i katalogi
# =============================
script_name = os.path.splitext(os.path.basename(__file__))[0]

# Uniwersalny regex - wyciąga prefix, QP, numer projektu, numer kroku i typ
match = re.match(r'(s2a_)(x\d+_)(\d+)_(\d+)_([a-z]+)', script_name)
if not match:
    raise ValueError(f"Nazwa pliku '{script_name}' nie pasuje do wzorca 's2a_xQP_NNN_S_ttt'")

base_prefix, qp_part, num_str, step_str, script_type = match.groups()
prefix = base_prefix + qp_part  # s2a_x00_ lub s2a_x27_ itp.
print(f"Wykryto: base_prefix={base_prefix}, qp_part={qp_part}, prefix={prefix}, numer={num_str}, krok={step_str}, typ={script_type}")

# Poprzedni krok (dla pliku treningowego) - zachowaj format z zerem wiodącym
prev_step = f"{int(step_str) - 1}"  
train_type = "tgd"  # Zawsze zmieniamy na tgd dla treningu
train_script_name = f"{prefix}{num_str}_{prev_step}_{train_type}"

# Tworzymy ścieżki - katalog eksperymentu to poprzedni krok treningowy z typem tgd
exp_dir = f"exp_{prefix}{num_str}_{prev_step}_tgd"
#test_dir = os.path.join(exp_dir, f"test_{script_name}")
test_dir = os.path.join(exp_dir, f"test_{script_name}")
#train_dir = os.path.join(exp_dir, f"train_{train_script_name}")
train_dir = os.path.join(exp_dir, f"train_{prefix}{num_str}_{prev_step}_tgd")
#weights_src_dir = os.path.join(train_dir, "weights")
weights_src_dir = os.path.join(train_dir, "weights")
psnr_dir = os.path.join(test_dir, "PSNR_db")
results_dir = os.path.join(test_dir, "results")
erc_dir = os.path.join(test_dir, "EdgeRecoverCoefficient")
ssim_dir = os.path.join(test_dir, "SSIM")
f1_dir = os.path.join(test_dir, "F1_Score")

print(f"Używane ścieżki:")
print(f"- Katalog eksperymentu: {exp_dir}")
print(f"- Katalog testowy: {test_dir}")
print(f"- Katalog treningowy: {train_dir}")
print(f"- Katalog z wagami: {weights_src_dir}")

os.makedirs(test_dir, exist_ok=True)
os.makedirs(psnr_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(erc_dir, exist_ok=True)
os.makedirs(ssim_dir, exist_ok=True)
os.makedirs(f1_dir, exist_ok=True)

# =============================
# Bloki modelu (jak w treningu)
# =============================
def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                      kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                               kernel_initializer=initializer, use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result

def Generator():
    inputs = tf.keras.layers.Input(shape=[None, None, 3])
    OUTPUT_CHANNELS = 3

    down_stack = [
        downsample(32, 4, apply_batchnorm=False),
        downsample(64, 4),
        downsample(128, 4),
        downsample(256, 4),
        downsample(256, 4),
        downsample(256, 4),
        downsample(256, 4),
    ]

    up_stack = [
        upsample(256, 4, apply_dropout=True),
        upsample(256, 4, apply_dropout=True),
        upsample(256, 4, apply_dropout=True),
        upsample(256, 4),
        upsample(128, 4),
        upsample(64, 4),
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4, strides=2, padding='same',
                                           kernel_initializer=initializer, activation='tanh')

    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

# =============================
# I/O obrazów i przygotowanie
# =============================
def load_image(image_path: str) -> tf.Tensor:
    # Zwraca float32 w skali 0..255
    image = tf.io.read_file(image_path)
    image = tf.io.decode_png(image, channels=3)
    image = tf.cast(image, tf.float32)
    return image

def crop_center(image: tf.Tensor, crop_size: tuple = (128, 128)) -> tf.Tensor:
    """Wycina środkową część obrazu do zadanego rozmiaru (domyślnie 128x128)."""
    shape = tf.shape(image)
    h, w = shape[0], shape[1]
    crop_h, crop_w = crop_size
    offset_h = (h - crop_h) // 2
    offset_w = (w - crop_w) // 2
    return image[offset_h:offset_h+crop_h, offset_w:offset_w+crop_w, :]
def crop_both_same_window(input_image: tf.Tensor, 
                          org_image: tf.Tensor,
                          crop_size: tuple = (128, 128),
                          edge_threshold: int = 80,
                          grid_size: int = 3) -> tuple:
    """
    Przycinak input_image i org_image do TEGO SAMEGO fragmentu 128x128.
    Wybór okna bazuje na krawędziach w org_image (GT).
    
    Args:
        input_image: Input (H,W,3) float32 0..255
        org_image: Ground truth (H,W,3) float32 0..255
        crop_size: (128, 128)
        edge_threshold: Próg dla detekcji krawędzi
        grid_size: Podział na grid (3x3, 4x4, itp.)
    
    Returns:
        tuple: (input_cropped, org_cropped) oba (128,128,3)
    """
    org_np = org_image.numpy().astype(np.uint8)
    h, w = org_np.shape[:2]
    crop_h, crop_w = crop_size
    
    if h < crop_h or w < crop_w:
        offset_h = max(0, (h - crop_h) // 2)
        offset_w = max(0, (w - crop_w) // 2)
        return (input_image[offset_h:offset_h+crop_h, offset_w:offset_w+crop_w, :],
                org_image[offset_h:offset_h+crop_h, offset_w:offset_w+crop_w, :])
    
    # Konwertuj GT na szarość
    if len(org_np.shape) == 3 and org_np.shape[2] == 3:
        gray_img = cv2.cvtColor(org_np, cv2.COLOR_RGB2GRAY)
    else:
        gray_img = org_np
    
    # Znajdź krawędzie w GT
    edge_mask = (gray_img <= edge_threshold)
    
    best_count = 0
    best_offset_h = (h - crop_h) // 2
    best_offset_w = (w - crop_w) // 2
    
    # Podziel na grid i szukaj okna z największą liczbą krawędzi
    step_h = (h - crop_h) // (grid_size - 1) if grid_size > 1 else 0
    step_w = (w - crop_w) // (grid_size - 1) if grid_size > 1 else 0
    
    for i in range(grid_size):
        for j in range(grid_size):
            y = i * step_h if i < grid_size - 1 else h - crop_h
            x = j * step_w if j < grid_size - 1 else w - crop_w
            
            window_edges = np.sum(edge_mask[y:y+crop_h, x:x+crop_w])
            
            if window_edges > best_count:
                best_count = window_edges
                best_offset_h = y
                best_offset_w = x
    
    # Przytnij oba obrazy tym samym oknem
    input_cropped = input_image[best_offset_h:best_offset_h+crop_h, best_offset_w:best_offset_w+crop_w, :]
    org_cropped = org_image[best_offset_h:best_offset_h+crop_h, best_offset_w:best_offset_w+crop_w, :]
    
    return input_cropped, org_cropped
def normalize_for_generator(image: tf.Tensor) -> tf.Tensor:
    # 0..255 -> [-1,1]
    return (image / 127.5) - 1.0

#def przegladnij_tensor(tensor):
#    tensor_np = tensor.numpy().astype(np.uint8)
#    for y in range(tensor_np.shape[0]):      # wysokość
#       for x in range(tensor_np.shape[1]):  # szerokość
#           pixel = tensor_np[y, x]          # [R,G,B] dla danego piksela
#           print(f"Pixel ({y},{x}): {pixel}")
#           # Sprawdź czy którykolwiek kanał jest inny niż 0 lub 255
#            if not np.all(np.isin(pixel, [0, 255])):
#                print(f"Znaleziono wartość inną niż 0 lub 255 w pikselu ({y},{x}): {pixel}")
#                input("Naciśnij Enter, aby zatrzymać...")  # Zatrzymaj pętlę
                



# Przykład użycia:
# tensorwyjsciowy = edge_prog_image(input_image)
# przegladnij_tensor(tensorwyjsciowy)

# =============================
# ERC - uproszczona wersja (procent pikseli krawędziowych)
# =============================
def calculate_edge_percentage(image: tf.Tensor, edge_threshold: int = 80) -> float:
    """
    Oblicza procent pikseli krawędziowych w obrazie.
    
    Args:
        image: Obraz w formacie tensor (H,W,3) float32 0..255
        edge_threshold: Wartość progowa definiująca krawędź (0-edge_threshold)
        
    Returns:
        Procent pikseli krawędziowych w obrazie.
    """
    # Konwertuj tensor na tablicę NumPy
    img_np = image.numpy().astype(np.uint8)
    
    # Konwertuj na obraz w skali szarości, jeśli jest kolorowy
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray_img = img_np
    
    # Znajdź krawędzie (piksele o wartości 0-edge_threshold)
    edge_mask = (gray_img <= edge_threshold)
    num_edges = np.sum(edge_mask)
    
    # Liczba pikseli obrazu
    total_pixels = gray_img.shape[0] * gray_img.shape[1]
    
    # Oblicz procent krawędzi
    edge_percentage = (num_edges / total_pixels) * 100 if total_pixels > 0 else 0

    return edge_percentage


def edge_prog_image(input_image: tf.Tensor) -> tf.Tensor:
    # input_image: tf.Tensor (H,W,3) float32 0..255
    img_np = input_image.numpy().astype(np.uint8)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 200, 255,L2gradient=True)
    #print("Unikalne wartości krawędzi:", np.unique(edges))  # <-- Dodane wyświetlanie
    # Zamień na 3 kanały (RGB)
    edges = 255 - edges  # Inwersja: białe tło, czarne krawędzie
    edges_rgb = np.stack([edges]*3, axis=-1)
    tensorwyjsciowy= tf.convert_to_tensor(edges_rgb, dtype=tf.float32)
 #   przegladnij_tensor(tensorwyjsciowy)  # <-- Dodane wywołanie funkcji
    return tensorwyjsciowy
    
def create_prog_image(input_image: tf.Tensor) -> tf.Tensor:
    # Dodaj/odejmij 1 w uint8, następnie rzutuj na float32 0..255
    input_uint8 = tf.cast(input_image, tf.uint8)
    prog_uint8 = tf.where(input_uint8 < 255, input_uint8 + 1, input_uint8 - 1)
    return tf.cast(prog_uint8, tf.float32)

# Dodane: modyfikacja pojedynczego piksela w batchu
def bump_pixel_batched(img, b, y, x):
    # img: [B,H,W,3] float32 0..255
    current_values = img[b, y, x, :]
    
    # Dla każdego kanału: jeśli wartość = 0, dodaj 1; jeśli wartość = 255, odejmij 1; w przeciwnym razie dodaj 1
    # Tworzymy tensor zmian: +1 dla wartości < 255, -1 dla wartości = 255
    changes = tf.where(current_values < 255, tf.ones_like(current_values), -tf.ones_like(current_values))
    # Dodatkowo: jeśli wartość = 0, zawsze dodaj +1
    changes = tf.where(current_values <= 0, tf.ones_like(current_values), changes)
    
    # Zastosuj zmiany
    v = current_values + changes
    v = tf.clip_by_value(v, 0.0, 255.0)
    idx = [[b, y, x, 0], [b, y, x, 1], [b, y, x, 2]]
    return tf.tensor_scatter_nd_update(img, indices=idx, updates=v)


# =============================
# SSIM: funkcja dla obrazów RGB
# =============================
def ssim_numpy(img_ref: np.ndarray, img_test: np.ndarray, data_range: float = 255.0) -> float:
    """
    SSIM między img_ref i img_test, uśredniony po kanałach RGB.
    - img_ref, img_test: np.ndarray (H,W,3), ten sam kształt
    - data_range: 255.0 dla 8-bit
    """
    if img_ref.shape != img_test.shape:
        raise ValueError("Obrazy muszą mieć ten sam kształt")
    if img_ref.shape[-1] != 3:
        raise ValueError("Obrazy muszą mieć 3 kanały RGB")
    ssim_values = []
    for c in range(3):
        ssim_val = ssim(img_ref[..., c], img_test[..., c], data_range=data_range)
        ssim_values.append(ssim_val)
    return float(np.mean(ssim_values))

# =============================
# Funkcje wizualizacji oczyszczania - USUNIĘTE
# =============================

# =============================
# Formatowanie tabel do plików TXT i CSV
# =============================
def write_formatted_table(file_path, headers, data, title):
    """
    Zapisuje dane w formie ładnie sformatowanej tabeli do pliku TXT.
    
    Args:
        file_path: Ścieżka do pliku
        headers: Lista nagłówków kolumn
        data: Lista wierszy z danymi
        title: Tytuł tabeli
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(f"=== {title} ===\n\n")
        
        # Oblicz szerokości kolumn
        col_widths = [len(str(header)) for header in headers]
        for row in data:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Zapisz nagłówek
        header_row = " | ".join(str(headers[i]).ljust(col_widths[i]) for i in range(len(headers)))
        f.write(header_row + "\n")
        
        # Zapisz separator
        separator = "-|-".join("-" * col_widths[i] for i in range(len(headers)))
        f.write(separator + "\n")
        
        # Zapisz wiersze danych
        for row in data:
            data_row = " | ".join(str(row[i]).ljust(col_widths[i]) for i in range(min(len(row), len(headers))))
            f.write(data_row + "\n")
    
    # Zapisz również wersję CSV
    csv_path = file_path.replace('.txt', '.csv')
    write_csv_file(csv_path, headers, data)

def write_csv_file(file_path, headers, data):
    """
    Zapisuje dane w formacie CSV.
    
    Args:
        file_path: Ścieżka do pliku CSV
        headers: Lista nagłówków kolumn
        data: Lista wierszy z danymi
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        # Zapisz nagłówek
        f.write(','.join(str(h) for h in headers) + '\n')
        
        # Zapisz wiersze danych
        for row in data:
            f.write(','.join(str(cell) for cell in row[:len(headers)]) + '\n')

def write_or_append_summary(file_path, headers, new_data, title):
    """
    Zapisuje lub dodaje dane podsumowujące do pliku z zachowaniem formatowania.
    """
    csv_path = file_path.replace('.txt', '.csv')
    
    if not os.path.exists(file_path):
        # Utwórz nowy plik z nagłówkami
        write_formatted_table(file_path, headers, new_data, title)
    else:
        # Dodaj nowe dane do istniejącego pliku TXT
        with open(file_path, 'r', encoding='utf-8') as rf:
            lines = rf.readlines()
        
        # Oblicz szerokości kolumn na podstawie nagłówków i nowych danych
        col_widths = [len(str(header)) for header in headers]
        for row in new_data:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Dodaj nowe wiersze do TXT
        with open(file_path, 'a', encoding='utf-8') as f:
            for row in new_data:
                data_row = " | ".join(str(row[i]).ljust(col_widths[i]) for i in range(min(len(row), len(col_widths))))
                f.write(data_row + "\n")
        
        # Dodaj nowe dane do pliku CSV
        with open(csv_path, 'a', encoding='utf-8') as f:
            for row in new_data:
                f.write(','.join(str(cell) for cell in row[:len(headers)]) + '\n')
    
    # Jeśli plik CSV nie istnieje, utwórz go z nagłówkami
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write(','.join(str(h) for h in headers) + '\n')
            for row in new_data:
                f.write(','.join(str(cell) for cell in row[:len(headers)]) + '\n')

# =============================
# BOUNDARY F1 SCORE - ODS i OIS
# Kompletna implementacja zgodna z algorytmem
# =============================

def calculate_f1_all_thresholds(pred_image_rgb: np.ndarray, 
                                 gt_image_rgb: np.ndarray, 
                                 thresholds: np.ndarray,
                                 gt_edge_thresh: int = 128) -> dict:
    """
    Oblicza F1, Precision, Recall dla wszystkich progów dla jednego obrazu.
    
    Algorytm (dla każdego kanału osobno, potem suma TP/FP/FN):
    1. Predykcja sieci: ciągła mapa krawędzi P ∈ [0,1] (P = pred/255)
       P = 1 oznacza wysokie prawdopodobieństwo krawędzi (ciemny piksel = krawędź)
       UWAGA: W naszym przypadku CIEMNY piksel = krawędź, więc P = 1 - (pred/255)
    2. Ground truth: binarna mapa GT ∈ {0,1} gdzie 1 = krawędź
       (konwersja: piksel <= gt_edge_thresh to krawędź)
    3. Binaryzacja: Pred_t = (P >= t) - piksel jest krawędzią jeśli P >= próg
    4. Zliczanie TP/FP/FN piksel-po-pikselu (suma po kanałach RGB)
    5. Obliczenie Precision/Recall/F1
    
    Args:
        pred_image_rgb: Obraz predykcji (H,W,3) uint8 [0,255], ciemny = krawędź
        gt_image_rgb: Ground truth (H,W,3) uint8 [0,255], ciemny = krawędź
        thresholds: Array progów w zakresie [0,1]
        gt_edge_thresh: Próg dla GT (piksel <= próg to krawędź)
        
    Returns:
        dict: {threshold: {'f1': float, 'precision': float, 'recall': float, 
                          'tp': int, 'fp': int, 'fn': int}}
    """
    EPS = 1e-12
    
    if pred_image_rgb.shape != gt_image_rgb.shape:
        raise ValueError("Predykcja i GT muszą mieć ten sam rozmiar")
    if pred_image_rgb.shape[2] != 3:
        raise ValueError("Obrazy muszą mieć 3 kanały RGB")
    
    results = {}
    
    for t in thresholds:
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        # Przetwarzaj każdy kanał osobno i sumuj TP/FP/FN
        for c in range(3):
            pred_channel = pred_image_rgb[..., c].astype(np.float32)
            gt_channel = gt_image_rgb[..., c].astype(np.uint8)
            
            # P ∈ [0,1] - prawdopodobieństwo krawędzi
            # CIEMNY piksel = krawędź, więc P = 1 - (pred/255)
            # Im ciemniejszy piksel (mniejsza wartość), tym większe P
            p_map = 1.0 - (pred_channel / 255.0)
            
            # GT ∈ {0,1}, 1 = krawędź (ciemny piksel)
            gt_binary = (gt_channel <= gt_edge_thresh).astype(np.uint8)
            
            # Binaryzacja predykcji: Pred_t = (P >= t)
            pred_binary = (p_map >= t).astype(np.uint8)
            
            # Zliczanie TP/FP/FN
            tp = int(np.sum((pred_binary == 1) & (gt_binary == 1)))
            fp = int(np.sum((pred_binary == 1) & (gt_binary == 0)))
            fn = int(np.sum((pred_binary == 0) & (gt_binary == 1)))
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
        
        # Oblicz metryki dla tego progu (zsumowane po kanałach)
        precision = total_tp / (total_tp + total_fp + EPS)
        recall = total_tp / (total_tp + total_fn + EPS)
        f1 = 2.0 * precision * recall / (precision + recall + EPS)
        
        results[t] = {
            'f1': float(f1),
            'precision': float(precision),
            'recall': float(recall),
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn
        }
    
    return results


def calculate_ods_ois(all_images_results: list, thresholds: np.ndarray) -> dict:
    """
    Oblicza ODS i OIS na podstawie wyników ze wszystkich obrazów.
    
    ODS (Optimal Dataset Scale):
        - Dla każdego progu t oblicz F1 dla KAŻDEGO obrazu
        - Policz ŚREDNIĄ F1 po wszystkich obrazach dla tego progu
        - Wybierz próg t* który maksymalizuje tę średnią
        - ODS = ta maksymalna średnia F1
    
    OIS (Optimal Image Scale):
        - Dla każdego obrazu znajdź JEGO najlepszy próg (maksymalizujący F1)
        - Zbierz te najlepsze F1 ze wszystkich obrazów
        - OIS = średnia z tych najlepszych F1
    
    Args:
        all_images_results: Lista dict-ów z funkcji calculate_f1_all_thresholds
                           [img1_results, img2_results, ...]
                           gdzie img_results = {threshold: {'f1', 'precision', 'recall', 'tp', 'fp', 'fn'}}
        thresholds: Array progów użytych do obliczeń
        
    Returns:
        dict: {
            'ods': {'f1': float, 'threshold': float, 'precision': float, 'recall': float},
            'ois': {'f1': float, 'precision': float, 'recall': float}
        }
    """
    EPS = 1e-12
    num_images = len(all_images_results)
    
    if num_images == 0:
        raise ValueError("Lista wyników jest pusta")
    
    # ============ ODS: Znajdź najlepszy GLOBALNY próg ============
    # Dla każdego progu oblicz ŚREDNIĄ F1 po wszystkich obrazach
    ods_best_f1 = -1.0
    ods_best_threshold = 0.0
    ods_best_precision = 0.0
    ods_best_recall = 0.0
    
    for t in thresholds:
        # Zbierz F1 dla tego progu ze wszystkich obrazów
        f1_values = [img_results[t]['f1'] for img_results in all_images_results]
        mean_f1 = np.mean(f1_values)
        
        if mean_f1 > ods_best_f1:
            ods_best_f1 = mean_f1
            ods_best_threshold = float(t)
            # Oblicz też średnią precision i recall dla tego progu
            precision_values = [img_results[t]['precision'] for img_results in all_images_results]
            recall_values = [img_results[t]['recall'] for img_results in all_images_results]
            ods_best_precision = float(np.mean(precision_values))
            ods_best_recall = float(np.mean(recall_values))
    
    # ============ OIS: Dla każdego obrazu znajdź jego najlepszy próg ============
    best_f1_per_image = []
    best_precision_per_image = []
    best_recall_per_image = []
    
    for img_results in all_images_results:
        # Znajdź próg z najwyższym F1 dla tego obrazu
        best_f1 = -1.0
        best_precision = 0.0
        best_recall = 0.0
        
        for t in thresholds:
            if img_results[t]['f1'] > best_f1:
                best_f1 = img_results[t]['f1']
                best_precision = img_results[t]['precision']
                best_recall = img_results[t]['recall']
        
        best_f1_per_image.append(best_f1)
        best_precision_per_image.append(best_precision)
        best_recall_per_image.append(best_recall)
    
    # OIS = średnia z najlepszych F1 dla każdego obrazu
    ois_f1 = float(np.mean(best_f1_per_image))
    ois_precision = float(np.mean(best_precision_per_image))
    ois_recall = float(np.mean(best_recall_per_image))
    
    return {
        'ods': {
            'f1': float(ods_best_f1),
            'threshold': ods_best_threshold,
            'precision': ods_best_precision,
            'recall': ods_best_recall
        },
        'ois': {
            'f1': ois_f1,
            'precision': ois_precision,
            'recall': ois_recall
        }
    }


def get_best_f1_for_image(image_results: dict, thresholds: np.ndarray) -> dict:
    """
    Znajduje najlepszy próg i metryki dla pojedynczego obrazu.
    
    Args:
        image_results: Dict z funkcji calculate_f1_all_thresholds
        thresholds: Array progów
        
    Returns:
        dict: {'f1': float, 'threshold': float, 'precision': float, 'recall': float}
    """
    best_f1 = -1.0
    best_threshold = 0.0
    best_precision = 0.0
    best_recall = 0.0
    
    for t in thresholds:
        if image_results[t]['f1'] > best_f1:
            best_f1 = image_results[t]['f1']
            best_threshold = float(t)
            best_precision = image_results[t]['precision']
            best_recall = image_results[t]['recall']
    
    return {
        'f1': float(best_f1),
        'threshold': best_threshold,
        'precision': float(best_precision),
        'recall': float(best_recall)
    }

# apply_optimal_threshold - USUNIĘTE

# =============================
# Wczytanie wag
# =============================
if not os.path.exists(weights_src_dir):
    raise ValueError(f"Katalog z wagami nie istnieje: {weights_src_dir}")

def extract_step_num(filename):
    match = re.search(r'step_(\d+)', filename)
    return int(match.group(1)) if match else -1

weights_files = sorted(
    [f for f in os.listdir(weights_src_dir) if f.endswith('.h5')],
    key=extract_step_num
)

if not weights_files:
    raise ValueError(f"Nie znaleziono plików wag .h5 w katalogu {weights_src_dir}")

print(f"Znaleziono {len(weights_files)} plików wag:")
for wf in weights_files:
    print(f" - {wf}")

# =============================
# Zbiory testowe
# =============================
possible_input_dirs = [
    pathlib.Path("/mnt/home/test_3000_qp/test_3000_gt_out_pngs_qp57"),
]

possible_output_dirs = [
    pathlib.Path("/mnt/home/test_3000_gt_out_200_255"),
]

input_test_dir = None
for dir_path in possible_input_dirs:
    if dir_path.exists():
        input_test_dir = dir_path
        print(f"Znaleziono katalog wejściowy: {input_test_dir}")
        break

output_test_dir = None
for dir_path in possible_output_dirs:
    if dir_path.exists():
        output_test_dir = dir_path
        print(f"Znaleziono katalog wyjściowy: {output_test_dir}")
        break

if input_test_dir is None:
    input_test_dir = possible_input_dirs[0]
    print(f"Uwaga: Nie znaleziono katalogu wejściowego. Używam: {input_test_dir}")

if output_test_dir is None:
    output_test_dir = possible_output_dirs[0]
    print(f"Uwaga: Nie znaleziono katalogu wyjściowego. Używam: {output_test_dir}")

input_images = sorted(list(input_test_dir.glob('*.png')))
output_images = sorted(list(output_test_dir.glob('*.png')))

if not input_images:
    raise ValueError(f"Nie znaleziono obrazów wejściowych w katalogu {input_test_dir}")
if len(input_images) != len(output_images):
    raise ValueError(f"Liczba obrazów wejściowych ({len(input_images)}) i wyjściowych ({len(output_images)}) się nie zgadza")

print(f"Znaleziono {len(input_images)} par obrazów testowych")

# =============================
# Model
# =============================
generator = Generator()

# Inicjalizacja modelu
example_input = tf.random.uniform(shape=(1, 128, 128, 3), minval=-1.0, maxval=1.0, dtype=tf.float32)
_ = generator(example_input, training=False)

# =============================
# ZMODYFIKOWANA PĘTLA TESTOWA Z ODS/OIS
# =============================

# Zakres progów (rozszerzony do 0.5 zgodnie z wymaganiami)
THRESHOLDS = np.linspace(0.001, 0.95, 25)

for i, weight_file in enumerate(weights_files):
    weight_path = os.path.join(weights_src_dir, weight_file)
    print(f"Wczytywanie wag z pliku: {weight_file}")
    generator.load_weights(weight_path)

    iter_match = re.search(r'step_(\d+)', weight_file)
    iter_num = iter_match.group(1) if iter_match else f"{i+1}"

    # Używamy już zdefiniowanych ścieżek
    iter_results_dir = os.path.join(results_dir, f"iter_{iter_num}")
    os.makedirs(iter_results_dir, exist_ok=True)

    results_file = os.path.join(psnr_dir, f"iter_{iter_num}.txt")
    erc_results_file = os.path.join(erc_dir, f"iter_{iter_num}.txt")
    ssim_results_file = os.path.join(ssim_dir, f"iter_{iter_num}.txt")
    f1_results_file = os.path.join(f1_dir, f"iter_{iter_num}.txt")
    
    # Nagłówki dla sformatowanych tabel
    psnr_headers = ["n_img", "prog_org_psnr", "gen_org_psnr", "psnr_improv"]
    erc_headers = ["n_img", "erc_prog", "erc_gen", "erc_improv"]
    ssim_headers = ["n_img", "prog_org_ssim", "gen_org_ssim", "ssim_improv"]
    f1_headers = ["n_img", "f1_prog(ODS)", "f1_gen(ODS)", "f1_improv(ODS)", "thresh_prog", "thresh_gen"]
    
    summary_file_path = os.path.join(psnr_dir, f"mean_PSNR_{script_name}.txt")
    erc_summary_file_path = os.path.join(erc_dir, f"mean_ERC_{script_name}.txt")
    ssim_summary_file_path = os.path.join(ssim_dir, f"mean_SSIM_{script_name}.txt")
    f1_summary_file_path = os.path.join(f1_dir, f"mean_F1_{script_name}.txt")
    
    # Listy do zbierania metryk
    prog_org_psnrs = []
    gen_org_psnrs = []
    psnr_improvs = []
    ercs_prog = []
    ercs_gen = []
    ercs_improv = []
    prog_org_ssims = []
    gen_org_ssims = []
    ssim_improvs = []

    # Kontenery danych dla sformatowanych plików
    psnr_data = []
    erc_data = []
    ssim_data = []
    f1_data = []
    
    # Listy do zbierania wyników F1 dla wszystkich obrazów (do ODS/OIS)
    all_prog_f1_results = []  # Lista dict-ów z calculate_f1_all_thresholds dla prog
    all_gen_f1_results = []   # Lista dict-ów z calculate_f1_all_thresholds dla gen
    
    # Lista do przechowywania danych obrazów do wizualizacji
    images_data = []
    input_images_data = []  # Tymczasowy kontener na dane obrazów

 
    print(f"  Przetwarzanie {len(input_images)} obrazów...")
    
    for j, (input_path, output_path) in enumerate(zip(input_images, output_images)):
        # Wczytaj obrazy (float32, 0..255)
        input_image = load_image(str(input_path))
        org_image = load_image(str(output_path))
        
        # === CROP DO 128x128 (model był trenowany na tym rozmiarze) ===
        input_image, org_image = crop_both_same_window(input_image, org_image, crop_size=(128, 128),grid_size=3)

        # Obraz prog (Canny)
        prog_image = edge_prog_image(input_image)[tf.newaxis, ...]

        # Generacja
        normalized_input = normalize_for_generator(input_image)
        normalized_input_batch = normalized_input[tf.newaxis, ...]
        gen_normalized = generator(normalized_input_batch, training=False)

        # Denormalizacja do 0..255 + clip
        gen_image = (gen_normalized + 1.0) * 127.5
        gen_image = tf.clip_by_value(gen_image, 0.0, 255.0)

        # Konwersja do NumPy uint8
        prog_image_np = tf.squeeze(prog_image, axis=0).numpy().astype(np.uint8)
        gen_image_np = tf.squeeze(gen_image, axis=0).numpy().astype(np.uint8)
        org_image_np = org_image.numpy().astype(np.uint8)

        # PSNR
        prog_org_psnr = psnr_numpy_per_channel(prog_image_np, org_image_np)
        gen_org_psnr = psnr_numpy_per_channel(gen_image_np, org_image_np)
        psnr_improv = gen_org_psnr - prog_org_psnr
        prog_org_psnrs.append(prog_org_psnr)
        gen_org_psnrs.append(gen_org_psnr)
        psnr_improvs.append(psnr_improv)

        # ERC (procent krawędzi)
        erc_prog = calculate_edge_percentage(tf.squeeze(prog_image, axis=0))
        erc_gen = calculate_edge_percentage(tf.squeeze(gen_image, axis=0))
        erc_improv = erc_gen - erc_prog
        ercs_prog.append(erc_prog)
        ercs_gen.append(erc_gen)
        ercs_improv.append(erc_improv)

        # SSIM
        prog_org_ssim = ssim_numpy(prog_image_np, org_image_np)
        gen_org_ssim = ssim_numpy(gen_image_np, org_image_np)
        ssim_improv = gen_org_ssim - prog_org_ssim
        prog_org_ssims.append(prog_org_ssim)
        gen_org_ssims.append(gen_org_ssim)
        ssim_improvs.append(ssim_improv)

        # ============ F1 Score: Oblicz dla WSZYSTKICH progów ============
        # Prog vs GT
        prog_f1_results = calculate_f1_all_thresholds(prog_image_np, org_image_np, THRESHOLDS)
        all_prog_f1_results.append(prog_f1_results)
        
        # Gen vs GT
        gen_f1_results = calculate_f1_all_thresholds(gen_image_np, org_image_np, THRESHOLDS)
        all_gen_f1_results.append(gen_f1_results)
        
        # Najlepszy F1 dla tego obrazu (OIS - per image)
        prog_best = get_best_f1_for_image(prog_f1_results, THRESHOLDS)
        gen_best = get_best_f1_for_image(gen_f1_results, THRESHOLDS)
        
        f1_improv = gen_best['f1'] - prog_best['f1']

        # Zapisz dane do kontenerów
        psnr_data.append([j+1, f"{prog_org_psnr:.4f}", f"{gen_org_psnr:.4f}", f"{psnr_improv:.4f}"])
        erc_data.append([j+1, f"{erc_prog:.6f}", f"{erc_gen:.6f}", f"{erc_improv:.6f}"])
        ssim_data.append([j+1, f"{prog_org_ssim:.6f}", f"{gen_org_ssim:.6f}", f"{ssim_improv:.6f}"])
        # f1_data będzie wypełnione później po obliczeniu ODS
        
        # Zbierz dane obrazów do tymczasowego kontenera (pierwsze 5)
        if j < 5:
            input_images_data.append({
                'input_image': input_image.numpy().astype(np.uint8),
                'org_image': org_image_np,
                'prog_image': prog_image_np,
                'gen_image': gen_image_np,
            })

        if (j+1) % 100 == 0:
            print(f"    Przetworzono {j+1}/{len(input_images)} obrazów...")

    # ============ Oblicz ODS i OIS ============
    print("  Obliczanie ODS i OIS...")
    
    # Dla Prog
    prog_ods_ois = calculate_ods_ois(all_prog_f1_results, THRESHOLDS)
    
    # Dla Gen
    gen_ods_ois = calculate_ods_ois(all_gen_f1_results, THRESHOLDS)
    
    # ============ Przelicz F1 data używając progów ODS ============
    f1_data = []  # Wyczyść i przelicz z progami ODS
    prog_ods_threshold = prog_ods_ois['ods']['threshold']
    gen_ods_threshold = gen_ods_ois['ods']['threshold']
    
    for j in range(len(all_prog_f1_results)):
        # Użyj progów ODS do obliczenia F1 dla tego obrazu
        prog_f1_ods = all_prog_f1_results[j][prog_ods_threshold]['f1']
        gen_f1_ods = all_gen_f1_results[j][gen_ods_threshold]['f1'] 
        f1_improv_ods = gen_f1_ods - prog_f1_ods
        
        f1_data.append([j+1, f"{prog_f1_ods:.6f}", f"{gen_f1_ods:.6f}", f"{f1_improv_ods:.6f}", 
                       f"{prog_ods_threshold:.4f}", f"{gen_ods_threshold:.4f}"])
                       
        # Przechowaj dane do wizualizacji (pierwsze 5 obrazów) - używaj ODS
        if j < 5:
            images_data.append({
                'input_image': input_images_data[j]['input_image'],
                'org_image': input_images_data[j]['org_image'],
                'prog_image': input_images_data[j]['prog_image'],
                'gen_image': input_images_data[j]['gen_image'],
                'f1_prog': prog_f1_ods,
                'f1_gen': gen_f1_ods,
                'thresh_prog': prog_ods_threshold,
                'thresh_gen': gen_ods_threshold,
            })
    
    # Wyświetl wyniki
    print(f"\n  === WYNIKI F1 SCORE ===")
    print(f"  PROG (Canny):")
    print(f"    OIS: F1={prog_ods_ois['ois']['f1']:.6f}, P={prog_ods_ois['ois']['precision']:.6f}, R={prog_ods_ois['ois']['recall']:.6f}")
    print(f"    ODS: F1={prog_ods_ois['ods']['f1']:.6f}, P={prog_ods_ois['ods']['precision']:.6f}, R={prog_ods_ois['ods']['recall']:.6f}, t={prog_ods_ois['ods']['threshold']:.4f}")
    print(f"  GENERATOR:")
    print(f"    OIS: F1={gen_ods_ois['ois']['f1']:.6f}, P={gen_ods_ois['ois']['precision']:.6f}, R={gen_ods_ois['ois']['recall']:.6f}")
    print(f"    ODS: F1={gen_ods_ois['ods']['f1']:.6f}, P={gen_ods_ois['ods']['precision']:.6f}, R={gen_ods_ois['ods']['recall']:.6f}, t={gen_ods_ois['ods']['threshold']:.4f}")
    
    # Poprawa
    ois_improv = gen_ods_ois['ois']['f1'] - prog_ods_ois['ois']['f1']
    ods_improv = gen_ods_ois['ods']['f1'] - prog_ods_ois['ods']['f1']
    print(f"  POPRAWA:")
    print(f"    OIS: {ois_improv:+.6f}")
    print(f"    ODS: {ods_improv:+.6f}")

    # ============ Prosta wizualizacja 4 obrazów ============
    print("  Generowanie wizualizacji...")
    
    for j, img_data in enumerate(images_data[:5]):
        # Wizualizacja 4 obrazów: Input, GT, Prog, Generator
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(img_data['input_image'])
        axes[0].set_title('Input (skompresowany)')
        axes[0].axis('off')
        
        axes[1].imshow(img_data['org_image'])
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')

        axes[2].imshow(img_data['prog_image'])
        axes[2].set_title(f'Prog/Canny (F1: {img_data["f1_prog"]:.3f})')
        axes[2].axis('off')

        axes[3].imshow(img_data['gen_image'])
        axes[3].set_title(f'Generator (F1: {img_data["f1_gen"]:.3f})')
        axes[3].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(iter_results_dir, f"comparison_{j+1}.png"))
        plt.close()

    # ============ Średnie PSNR, ERC, SSIM ============
    avg_prog_org_psnr = float(np.mean(prog_org_psnrs))
    avg_gen_org_psnr = float(np.mean(gen_org_psnrs))
    avg_psnr_improv = float(np.mean(psnr_improvs))
    
    avg_erc_prog = float(np.mean(ercs_prog))
    avg_erc_gen = float(np.mean(ercs_gen))
    avg_erc_improv = float(np.mean(ercs_improv))

    avg_prog_org_ssim = float(np.mean(prog_org_ssims))
    avg_gen_org_ssim = float(np.mean(gen_org_ssims))
    avg_ssim_improv = float(np.mean(ssim_improvs))

    # Dodaj wiersze podsumowujące
    psnr_data.append(["ŚREDNIA", f"{avg_prog_org_psnr:.4f}", f"{avg_gen_org_psnr:.4f}", f"{avg_psnr_improv:.4f}"])
    erc_data.append(["ŚREDNIA", f"{avg_erc_prog:.6f}", f"{avg_erc_gen:.6f}", f"{avg_erc_improv:.6f}"])
    ssim_data.append(["ŚREDNIA", f"{avg_prog_org_ssim:.6f}", f"{avg_gen_org_ssim:.6f}", f"{avg_ssim_improv:.6f}"])
    
    # Dodaj podsumowanie F1 (OIS i ODS)
    f1_data.append(["---", "---", "---", "---", "---", "---"])
    f1_data.append(["OIS_PROG", f"{prog_ods_ois['ois']['f1']:.6f}", "-", "-", 
                   f"P={prog_ods_ois['ois']['precision']:.4f}", f"R={prog_ods_ois['ois']['recall']:.4f}"])
    f1_data.append(["OIS_GEN", "-", f"{gen_ods_ois['ois']['f1']:.6f}", f"{ois_improv:+.6f}", 
                   f"P={gen_ods_ois['ois']['precision']:.4f}", f"R={gen_ods_ois['ois']['recall']:.4f}"])
    f1_data.append(["ODS_PROG", f"{prog_ods_ois['ods']['f1']:.6f}", "-", "-", 
                   f"t={prog_ods_ois['ods']['threshold']:.4f}", f"P={prog_ods_ois['ods']['precision']:.4f}"])
    f1_data.append(["ODS_GEN", "-", f"{gen_ods_ois['ods']['f1']:.6f}", f"{ods_improv:+.6f}", 
                   f"t={gen_ods_ois['ods']['threshold']:.4f}", f"P={gen_ods_ois['ods']['precision']:.4f}"])

    # Zapisz sformatowane pliki z wynikami
    write_formatted_table(results_file, psnr_headers, psnr_data, f"WYNIKI PSNR - ITERACJA {iter_num}")
    write_formatted_table(erc_results_file, erc_headers, erc_data, f"WYNIKI ERC - ITERACJA {iter_num}")
    write_formatted_table(ssim_results_file, ssim_headers, ssim_data, f"WYNIKI SSIM - ITERACJA {iter_num}")
    write_formatted_table(f1_results_file, f1_headers, f1_data, f"WYNIKI F1 SCORE - ITERACJA {iter_num}")

    # ============ Pliki podsumowujące ============
    # Nagłówki dla plików podsumowujących (rozszerzone o ODS precision/recall)
    summary_headers = ["iter_num", "avg_prog_org_psnr", "avg_gen_org_psnr", "avg_psnr_improv"]
    erc_summary_headers = ["iter_num", "avg_erc_prog", "avg_erc_gen", "avg_erc_improv"]
    ssim_summary_headers = ["iter_num", "avg_prog_org_ssim", "avg_gen_org_ssim", "avg_ssim_improv"]
    f1_summary_headers = ["iter_num", "f1_prog(ODS)", "f1_gen(ODS)", "f1_improv(ODS)", 
                          "f1_gen(OIS)", "threshold(ODS_gen)", "precision(ODS)", "recall(ODS)"]

    # Dane podsumowujące
    summary_psnr_data = [[iter_num, f"{avg_prog_org_psnr:.4f}", f"{avg_gen_org_psnr:.4f}", f"{avg_psnr_improv:.4f}"]]
    summary_erc_data = [[iter_num, f"{avg_erc_prog:.6f}", f"{avg_erc_gen:.6f}", f"{avg_erc_improv:.6f}"]]
    summary_ssim_data = [[iter_num, f"{avg_prog_org_ssim:.6f}", f"{avg_gen_org_ssim:.6f}", f"{avg_ssim_improv:.6f}"]]
    summary_f1_data = [[iter_num, 
                        f"{prog_ods_ois['ods']['f1']:.6f}", 
                        f"{gen_ods_ois['ods']['f1']:.6f}", 
                        f"{ods_improv:+.6f}",
                        f"{gen_ods_ois['ois']['f1']:.6f}", 
                        f"{gen_ods_ois['ods']['threshold']:.4f}",
                        f"{gen_ods_ois['ods']['precision']:.6f}",
                        f"{gen_ods_ois['ods']['recall']:.6f}"]]

    # Zapisz lub dodaj do plików podsumowujących
    write_or_append_summary(summary_file_path, summary_headers, summary_psnr_data, f"PODSUMOWANIE PSNR - {script_name}")
    write_or_append_summary(erc_summary_file_path, erc_summary_headers, summary_erc_data, f"PODSUMOWANIE ERC - {script_name}")
    write_or_append_summary(ssim_summary_file_path, ssim_summary_headers, summary_ssim_data, f"PODSUMOWANIE SSIM - {script_name}")
    write_or_append_summary(f1_summary_file_path, f1_summary_headers, summary_f1_data, f"PODSUMOWANIE F1 SCORE - {script_name}")

    # Wyświetl podsumowanie
    print(f"\n  Wyniki zapisane do: {test_dir}")
    print(f"  === PODSUMOWANIE ITERACJI {iter_num} ===")
    print(f"  PSNR: prog={avg_prog_org_psnr:.4f}dB, gen={avg_gen_org_psnr:.4f}dB, improv={avg_psnr_improv:+.4f}dB")
    print(f"  ERC:  prog={avg_erc_prog:.4f}%, gen={avg_erc_gen:.4f}%, improv={avg_erc_improv:+.4f}%")
    print(f"  SSIM: prog={avg_prog_org_ssim:.6f}, gen={avg_gen_org_ssim:.6f}, improv={avg_ssim_improv:+.6f}")
    print(f"  F1 ODS: prog={prog_ods_ois['ods']['f1']:.6f}, gen={gen_ods_ois['ods']['f1']:.6f}, improv={ods_improv:+.6f}")
    print(f"  F1 OIS: prog={prog_ods_ois['ois']['f1']:.6f}, gen={gen_ods_ois['ois']['f1']:.6f}, improv={ois_improv:+.6f}")

print("\nTestowanie zakończone!")
print(f"Wyniki PSNR zostały zapisane w katalogu: {psnr_dir}")
print(f"Wyniki ERC zostały zapisane w katalogu: {erc_dir}")
print(f"Wyniki SSIM zostały zapisane w katalogu: {ssim_dir}")
print(f"Wyniki F1 Score zostały zapisane w katalogu: {f1_dir}")
print(f"Wizualizacje porównawcze zostały zapisane w katalogu: {results_dir}")

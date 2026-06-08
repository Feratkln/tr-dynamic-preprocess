import math
import numpy as np
from numpy.linalg import norm
from collections import defaultdict

def hesapla_tfidf(metin_listesi: list) -> dict:
    """
    Külliyat genelinde kelimelerin TF-IDF skorlarını hesaplar.
    """
    N = len(metin_listesi)
    kelime_tf = defaultdict(int)
    kelime_df = defaultdict(int)

    for metin in metin_listesi:
        kelimeler = str(metin).split()
        for k in kelimeler:
            kelime_tf[k] += 1
        for k in set(kelimeler):
            kelime_df[k] += 1

    max_f = max(kelime_tf.values()) if kelime_tf else 1

    return {
        k: math.log10(1 + f / max_f) * math.log10(N / kelime_df[k])
        for k, f in kelime_tf.items()
        if kelime_df[k] > 0
    }

def dirsek_esigi_bul(skorlar: list) -> tuple:
    """
    Kosinüs benzerliği ve mesafe projeksiyonu ile TF-IDF eğrisinin 
    kırılma (dirsek) noktasını bulur.
    """
    if len(skorlar) < 3:
        return 0.0, 0

    skorlar_sirali = sorted(skorlar, reverse=True)
    n = len(skorlar_sirali)
    coords = np.vstack((range(n), skorlar_sirali)).T

    first, last = coords[0], coords[-1]
    line = last - first
    line_norm = line / np.linalg.norm(line)
    vecs = coords - first
    proj = np.outer(np.dot(vecs, line_norm), line_norm)
    dist = np.linalg.norm(vecs - proj, axis=1)

    dirsek_idx = np.argmax(dist)
    dirsek_skor = skorlar_sirali[dirsek_idx]

    return dirsek_skor, dirsek_idx

def dinamik_esik(uzunluk: int) -> float:
    """
    Cümlenin uzunluğuna bağlı olarak dinamik eşik değerini hesaplar.
    """
    return 1.0 - (0.6 / (uzunluk + 2))

def cumle_vektoru(kelimeler: list, w2v_model) -> np.ndarray:
    """
    Cümledeki kelimelerin Word2Vec vektörlerinin ortalamasını alır.
    """
    vek = [w2v_model.wv[k] for k in kelimeler if k in w2v_model.wv]
    return np.mean(vek, axis=0) if vek else np.zeros(w2v_model.vector_size)

def kosinus(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    İki cümle vektörü arasındaki Kosinüs Benzerliğini hesaplar.
    """
    n1, n2 = norm(v1), norm(v2)
    return float(np.dot(v1, v2) / (n1 * n2)) if n1 and n2 else 0.0

def baglamsal_filtre(metin: str, supheliler: set, ozel_korunacaklar: set, w2v_model) -> str:
    """
    Word2Vec anlamsal sapma ve dinamik eşik değerleri kullanarak 
    cümledeki gereksiz kelimeleri temizler.
    """
    metin = str(metin)
    kelimeler = metin.split()
    if not kelimeler:
        return metin

    guncel = kelimeler.copy()

    for aday in [k for k in kelimeler if k in supheliler]:
        # Cümle çok kısaldıysa durdur
        if len(guncel) <= 2:
            break

        # Özel korunacaklar listesindeyse geç
        if aday in ozel_korunacaklar:
            continue

        vec_tam = cumle_vektoru(guncel, w2v_model)
        vec_eksik = cumle_vektoru([k for k in guncel if k != aday], w2v_model)
        benzerlik = kosinus(vec_tam, vec_eksik)
        esik = dinamik_esik(len(guncel))

        if benzerlik >= esik:
            # Anlam değişmedi, gerçek stopword'dür, sil
            guncel = [k for k in guncel if k != aday]

    return ' '.join(guncel)

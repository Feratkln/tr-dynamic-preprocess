import re
import jpype

def temel_temizlik(metin: str) -> str:
    """
    HTML etiketlerini, noktalama işaretlerini ve sayıları temizler.
    Kökleri ve ekleri korur.
    """
    metin = re.sub(r'<.*?>', ' ', str(metin))  # HTML etiketleri
    metin = metin.lower()                       # Küçük harfe çevir
    metin = re.sub(r'[^\w\s]', ' ', metin)     # Noktalama işaretlerini kaldır
    metin = re.sub(r'\d+', ' ', metin)         # Sayıları kaldır
    metin = re.sub(r'\s+', ' ', metin)         # Fazla boşlukları temizle
    return metin.strip()

def levenshtein(s1: str, s2: str) -> int:
    """
    İki kelime arasındaki Levenshtein (düzenleme) mesafesini hesaplar.
    """
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    prev = range(len(s2) + 1)
    for c1 in s1:
        curr = [0] * (len(s2) + 1)
        curr[0] = prev[0] + 1
        for i, c2 in enumerate(s2):
            curr[i+1] = min(prev[i] + (c1 != c2), curr[i] + 1, prev[i+1] + 1)
        prev = curr
    return prev[-1]

def zemberek_tanir_mi(kelime: str, morphology) -> bool:
    """
    Zemberek morfolojik analiz motorunun kelimeyi tanıyıp tanımadığını kontrol eder.
    """
    try:
        sonuc = morphology.analyzeAndDisambiguate(kelime)
        analizler = sonuc.bestAnalysis()
        if analizler.size() == 0:
            return False
        return 'Unknown' not in str(analizler.get(0))
    except:
        return False

def oov_duzelt(kelime: str, morphology, ft_model) -> str:
    """
    Zemberek tarafından bilinmeyen kelimeleri FastText komşuları
    ve Levenshtein mesafesi kullanarak düzeltir.
    """
    if len(kelime) < 3:
        return kelime
    if zemberek_tanir_mi(kelime, morphology):
        return kelime
    
    # FastText ile en yakın komşuları bul
    try:
        adaylar = [k for _, k in ft_model.get_nearest_neighbors(kelime, k=5)]
    except:
        return kelime

    for aday in adaylar:
        if zemberek_tanir_mi(aday, morphology) and levenshtein(kelime, aday) <= 2:
            return aday
            
    return kelime

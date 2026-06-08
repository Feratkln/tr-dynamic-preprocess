import jpype

def zemberek_pos(kelime: str, morphology) -> str:
    """
    Zemberek morfolojik analiz motorunu kullanarak kelimenin POS (Sözcük Türü) etiketini çıkarır.
    """
    try:
        sonuc = morphology.analyzeAndDisambiguate(kelime)
        analizler = sonuc.bestAnalysis()
        if analizler.size() == 0:
            return 'UNKNOWN'
        
        analiz_str = str(analizler.get(0))
        # Özel isim kontrolü
        if 'Prop' in analiz_str:
            return 'PROPN'
            
        return analiz_str.split(':')[0].strip('[]') if ':' in analiz_str else 'UNKNOWN'
    except:
        return 'UNKNOWN'

def korunacak_mi(kelime: str, pos: str, korunacaklar_listesi: set) -> bool:
    """
    Kelimenin özel isim veya görev bazlı koruma listesinde olup olmadığını kontrol eder.
    """
    if kelime in korunacaklar_listesi:
        return True
    if pos == 'PROPN':
        return True
    return False

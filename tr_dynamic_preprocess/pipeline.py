import os
import json
import pandas as pd
from tqdm import tqdm
import jpype
import fasttext
from gensim.models import Word2Vec

from .normalizer import temel_temizlik, oov_duzelt
from .pos_tagger import zemberek_pos, korunacak_mi
from .stopword_detector import (hesapla_tfidf, dirsek_esigi_bul, 
                                baglamsal_filtre)

class TurkishDynamicPreprocessor:
    def __init__(self, ft_model_path: str, zemberek_jar_path: str):
        """
        TurkishDynamicPreprocessor sınıfını ilklendirir, JVM ve FastText modelini yükler.
        """
        self.ft_model_path = ft_model_path
        self.zemberek_jar_path = zemberek_jar_path
        
        # JVM Başlat
        if not jpype.isJVMStarted():
            jpype.startJVM(classpath=[self.zemberek_jar_path])
            
        TurkishMorphology = jpype.JClass('zemberek.morphology.TurkishMorphology')
        self.morphology = TurkishMorphology.createWithDefaults()
        
        # FastText yükle
        print("FastText model yükleniyor (Bu işlem model boyutuna göre zaman alabilir)...")
        self.ft_model = fasttext.load_model(self.ft_model_path)
        print("✅ Başlatma başarılı: Zemberek ve FastText yüklendi.")
        
        # Caches
        self.duzeltme_sozlugu = {}
        self.pos_cache = {}
        self.ozel_korunacaklar = set()
        self.supheliler = set()
        self.w2v_model = None
        
        # Korunacaklar listeleri
        self.koruma_listeleri = {
            'duygu_analizi': {
                'değil', 'ama', 'fakat', 'lakin', 'hiç', 'çok', 'az',
                'fazla', 'yetersiz', 'berbat', 'harika', 'maalesef',
                'kesinlikle', 'asla', 'hiçbir', 'bile', 'ne kadar'
            },
            'haber': set(),
            'genel': set()
        }

    def load_caches(self, sozluk_yolu=None, pos_cache_yolu=None, korunacaklar_yolu=None):
        """
        Daha önceki çalışmalardan kalan önbellek dosyalarını yükler.
        """
        if sozluk_yolu and os.path.exists(sozluk_yolu):
            with open(sozluk_yolu, encoding='utf-8') as f:
                self.duzeltme_sozlugu = json.load(f)
            print(f"✅ Düzeltme sözlüğü yüklendi: {len(self.duzeltme_sozlugu):,} kelime")
            
        if pos_cache_yolu and os.path.exists(pos_cache_yolu):
            with open(pos_cache_yolu, encoding='utf-8') as f:
                self.pos_cache = json.load(f)
            print(f"✅ POS önbelleği yüklendi: {len(self.pos_cache):,} kelime")
            
        if korunacaklar_yolu and os.path.exists(korunacaklar_yolu):
            with open(korunacaklar_yolu, encoding='utf-8') as f:
                self.ozel_korunacaklar = set(json.load(f))
            print(f"✅ Korunacak kelime listesi yüklendi: {len(self.ozel_korunacaklar):,} kelime")

    def save_caches(self, sozluk_yolu, pos_cache_yolu, korunacaklar_yolu):
        """
        Analiz sonuçlarını önbellek dosyalarına kaydeder.
        """
        with open(sozluk_yolu, 'w', encoding='utf-8') as f:
            json.dump(self.duzeltme_sozlugu, f, ensure_ascii=False)
        with open(pos_cache_yolu, 'w', encoding='utf-8') as f:
            json.dump(self.pos_cache, f, ensure_ascii=False)
        with open(korunacaklar_yolu, 'w', encoding='utf-8') as f:
            json.dump(list(self.ozel_korunacaklar), f, ensure_ascii=False)
        print("✅ Tüm önbellekler başarıyla kaydedildi.")

    def zemberek_lemma(self, kelime: str) -> str:
        """
        Zemberek ile tek bir kelimenin kökünü bulur.
        """
        try:
            sonuc = self.morphology.analyzeAndDisambiguate(kelime)
            anal = sonuc.bestAnalysis()
            if anal.size() > 0:
                lemmas = anal.get(0).getLemmas()
                if lemmas and lemmas.size() > 0:
                    kok = str(lemmas.get(0)).lower()
                    if kok != 'unk':
                        return kok
        except:
            pass
        return kelime

    def metin_lemma(self, metin: str) -> str:
        """
        Yorumdaki tüm kelimelerin köklerini bulur.
        """
        if pd.isna(metin):
            return ''
        return ' '.join(self.zemberek_lemma(k) for k in str(metin).split())

    def fit(self, df: pd.DataFrame, text_column: str, task='genel', w2v_vector_size=100, w2v_epochs=10):
        """
        Veri seti üzerinde analizleri çalıştırır:
        1. Temel temizlik ve OOV düzeltmeleri yapar.
        2. POS analizi ve korunacak kelimeleri belirler.
        3. TF-IDF ve Dirsek noktası analiziyle stopword adaylarını belirler.
        4. Word2Vec anlamsal uzayını eğitir.
        """
        print("\n--- Adım 1: Temel Temizlik ve OOV Düzeltme başlatılıyor ---")
        df_clean = df[text_column].apply(temel_temizlik)
        
        # OOV kelimeleri tespit et ve düzeltme sözlüğü oluştur
        tum_kelimeler = set()
        for metin in df_clean:
            tum_kelimeler.update(str(metin).split())
            
        islenmemis = [k for k in tum_kelimeler if k not in self.duzeltme_sozlugu]
        print(f"Toplam benzersiz: {len(tum_kelimeler):,}")
        print(f"İşlenecek OOV: {len(islenmemis):,}")
        
        if islenmemis:
            for kelime in tqdm(islenmemis, desc="OOV Düzeltme"):
                self.duzeltme_sozlugu[kelime] = oov_duzelt(kelime, self.morphology, self.ft_model)
                
        # Düzeltilmiş metinleri oluştur
        df_duzeltilmis = df_clean.apply(
            lambda m: ' '.join(self.duzeltme_sozlugu.get(k, k) for k in str(m).split())
        )
        
        print("\n--- Adım 2: POS Analizi ve Kelime Koruma başlatılıyor ---")
        # Benzersiz kelimelerin POS türlerini çıkart
        tum_kelimeler_duzeltilmis = set()
        for metin in df_duzeltilmis:
            tum_kelimeler_duzeltilmis.update(str(metin).split())
            
        korunacaklar_listesi = self.koruma_listeleri.get(task, set())
        
        islenmemis_pos = [k for k in tum_kelimeler_duzeltilmis if k not in self.pos_cache]
        if islenmemis_pos:
            for kelime in tqdm(islenmemis_pos, desc="POS Analizi"):
                pos = zemberek_pos(kelime, self.morphology)
                self.pos_cache[kelime] = pos
                if korunacak_mi(kelime, pos, korunacaklar_listesi):
                    self.ozel_korunacaklar.add(kelime)
        else:
            # Önbellekten yüklenen korunacakları güncelle
            for kelime in tum_kelimeler_duzeltilmis:
                pos = self.pos_cache.get(kelime, 'UNKNOWN')
                if korunacak_mi(kelime, pos, korunacaklar_listesi):
                    self.ozel_korunacaklar.add(kelime)

        print("\n--- Adım 3: TF-IDF ve Dirsek Noktası Analizi başlatılıyor ---")
        # Kök bulma işlemi (Sadece TF-IDF için)
        print("Lemmatizasyon uygulanıyor...")
        df_lemma = df_duzeltilmis.apply(self.metin_lemma)
        
        tfidf_skorlari = hesapla_tfidf(df_lemma.tolist())
        dirsek_skor, dirsek_idx = dirsek_esigi_bul(list(tfidf_skorlari.values()))
        
        self.supheliler = set(
            k for k, s in tfidf_skorlari.items()
            if s < dirsek_skor and k not in self.ozel_korunacaklar
        )
        print(f"Dirsek noktası   : {dirsek_idx}. kelime, skor={dirsek_skor:.4f}")
        print(f"Şüpheli kelime   : {len(self.supheliler):,}")

        print("\n--- Adım 4: Word2Vec Eğitimi başlatılıyor ---")
        tokenized = [str(m).split() for m in df_duzeltilmis.tolist() if str(m).strip()]
        self.w2v_model = Word2Vec(
            sentences=tokenized,
            vector_size=w2v_vector_size,
            window=5,
            min_count=2,
            sg=0,
            workers=4,
            epochs=w2v_epochs
        )
        print(f"Word2Vec eğitimi tamamlandı. Kelime sayısı: {len(self.w2v_model.wv):,}")

    def transform(self, df: pd.DataFrame, text_column: str) -> pd.Series:
        """
        Verilen veri setine eğitilmiş parametrelerle bağlamsal filtre uygular.
        """
        if self.w2v_model is None:
            raise ValueError("Hata: Önce fit() fonksiyonunu çalıştırmalı veya modeli yüklemelisiniz.")
            
        print("\nMetinlere bağlamsal filtre uygulanıyor...")
        
        # Temizlik ve OOV Düzeltmeleri
        df_clean = df[text_column].apply(temel_temizlik)
        df_duzeltilmis = df_clean.apply(
            lambda m: ' '.join(self.duzeltme_sozlugu.get(k, k) for k in str(m).split())
        )
        
        # Bağlamsal stopword filtresi
        filtrelenmis = [
            baglamsal_filtre(m, self.supheliler, self.ozel_korunacaklar, self.w2v_model)
            for m in tqdm(df_duzeltilmis.tolist(), desc="Filtreleme")
        ]
        
        # Lemmatize edip geri döndür
        series_filtrelenmis = pd.Series(filtrelenmis).apply(self.metin_lemma)
        return series_filtrelenmis

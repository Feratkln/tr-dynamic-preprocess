# Turkish Dynamic Preprocess (tr_dynamic_preprocess)

Türkçe metinler için Zemberek (Morfoloji ve POS Tagging) ve FastText/Word2Vec anlamsal uzay modellerini birleştirerek çalışan, göreve özel (task-aware) **Dinamik ve Bağlamsal Stopword Temizleme** kütüphanesidir.

## Özellikler

- **OOV (Out-Of-Vocabulary) Yazım Düzeltmesi:** Zemberek'in tanımadığı kelimeleri FastText anlamsal yakınlığı ve Levenshtein mesafesi ile otomatik düzeltir.
- **Göreve Duyarlı Kelime Koruma (Task-Aware):** Duygu analizi, konu sınıflandırma gibi farklı NLP görevlerine göre kritik anlam taşıyan kelimeleri otomatik korur.
- **Dinamik Stopword Algılaması (TF-IDF Elbow):** Veri setinin TF-IDF dağılımını çıkartıp dirsek noktası tespiti ile stopword adaylarını belirler.
- **Bağlamsal Filtreleme (Word2Vec Cosine):** Cümledeki bir kelime silindiğinde cümle anlamındaki sapmayı Word2Vec vektör benzerliğiyle ölçer. Dinamik cümle uzunluğu eşiği ile gereksiz kelimeleri cümleden eler.

---

## Kurulum (Installation)

Kütüphaneyi doğrudan GitHub deponuz üzerinden kurmak için:

```bash
pip install git+https://github.com/KULLANICI_ADI/DEPO_ADI.git
```

### Bağımlılıklar (Dependencies)
Kütüphane kurulurken şu bağımlılıklar otomatik olarak yüklenir:
- `jpype1` (Zemberek JVM köprüsü için)
- `fasttext-wheel` (Hızlı FastText vektör sorguları için)
- `gensim` (Word2Vec eğitimi için)
- `pandas`, `numpy`, `tqdm`, `scikit-learn`

---

## Hızlı Başlangıç (Quick Start)

### 1. Kütüphaneyi İçe Aktarma ve İlklendirme

```python
from tr_dynamic_preprocess import TurkishDynamicPreprocessor

# Model ve Zemberek yollarını belirterek sınıfı başlatın
preprocessor = TurkishDynamicPreprocessor(
    ft_model_path='/content/drive/MyDrive/cc.tr.300.bin',
    zemberek_jar_path='/content/zemberek-nlp/all/target/zemberek-full.jar'
)
```

### 2. Önbellekleri Yükleme (Opsiyonel)
İşlemleri hızlandırmak için önceden eğitilmiş sözlük ve POS cache dosyalarınızı yükleyebilirsiniz:

```python
preprocessor.load_caches(
    sozluk_yolu='/content/drive/MyDrive/duzeltme_sozlugu.json',
    pos_cache_yolu='/content/drive/MyDrive/pos_cache.json',
    korunacaklar_yolu='/content/drive/MyDrive/ozel_korunacaklar.json'
)
```

### 3. Model Eğitimi (Fitting)
Algoritmayı veri setinize göre eğitmek, TF-IDF sınırını belirlemek ve Word2Vec modelini kurmak için:

```python
# df: Pandas DataFrame
# text_column: Metinlerin olduğu sütun adı
# task: 'duygu_analizi', 'haber' veya 'genel'
preprocessor.fit(df, text_column='review', task='duygu_analizi')
```

### 4. Metinleri Dönüştürme (Transforming)
Filtreleme işlemini uygulayıp temizlenmiş metinleri almak için:

```python
df['filtrelenmis'] = preprocessor.transform(df, text_column='review')
```

### 5. Önbellekleri Kaydetme
Gelecekteki çalıştırmalarda zaman kazanmak için yeni analiz sonuçlarını önbelleğe kaydedin:

```python
preprocessor.save_caches(
    sozluk_yolu='/content/drive/MyDrive/duzeltme_sozlugu.json',
    pos_cache_yolu='/content/drive/MyDrive/pos_cache.json',
    korunacaklar_yolu='/content/drive/MyDrive/ozel_korunacaklar.json'
)
```

# Armut ARL — Birliktelik Kuralı Tabanlı Öneri Sistemi

Türkiye'nin en büyük online hizmet platformu **Armut**'un kullanıcı-hizmet verisi üzerinde **Association Rule Learning (Apriori)** kullanılarak geliştirilen hizmet öneri sistemi. *Miuul Data Scientist Bootcamp* kapsamında hazırlanmıştır.

## İş Problemi

Hizmet alan kullanıcıların satın aldığı servis ve kategorilere ait veriler kullanılarak, birliktelik kuralı öğrenmesi ile bir hizmet tavsiye sistemi oluşturmak.

## Veri Seti

4 değişken, 162.523 gözlem

| Değişken | Açıklama |
|---|---|
| `UserId` | Müşteri numarası |
| `ServiceId` | Kategoriye özel anonimleştirilmiş servis (aynı ServiceId farklı kategorilerde farklı hizmeti ifade edebilir) |
| `CategoryId` | Anonimleştirilmiş kategori (Temizlik, nakliyat, tadilat vb.) |
| `CreateDate` | Hizmetin satın alındığı tarih |

## Yöntem

1. **Veri hazırlama** — `ServiceId` ve `CategoryId`, `Hizmet` değişkeninde birleştirildi (örn. `4_5`). Veri setinde sepet/fatura tanımı olmadığından, sepet tanımı *"bir kullanıcının bir ay içinde aldığı hizmetler"* olarak kuruldu → `SepetID` (örn. `25446_2017-08`).
2. **Sepet–Hizmet pivot table** — Sepetler satır, hizmetler sütun olacak şekilde binary (0/1) matris oluşturuldu.
3. **Apriori & Association Rules** — `mlxtend` ile sık hizmet kombinasyonları ve bunlara ait support/confidence/lift değerlerini içeren kurallar üretildi.
4. **`arl_recommender`** — Verilen bir hizmeti öncül (antecedent) olarak içeren kuralları lift değerine göre sıralayıp en ilgili hizmetleri öneren fonksiyon.

## Örnek Sonuç

En son `2_0` hizmetini alan bir kullanıcı için üretilen öneriler:

```
['22_0', '25_0', '15_1']
```

## Kurulum ve Çalıştırma

```bash
pip install -r requirements.txt
python armut_arl.py
```

## Proje Yapısı

```
ArmutARL/
├── armut_arl.py        # veri hazırlama, apriori, arl_recommender
├── data/
│   └── armut_data.csv  # ham veri seti
├── requirements.txt
└── README.md
```

## Kullanılan Kütüphaneler

- pandas
- mlxtend (apriori, association_rules)

## Kaynak

Miuul Data Scientist Bootcamp — Association Rule Learning modülü, Armut projesi.

"""
Armut - Birliktelik Kuralı Tabanlı Öneri Sistemi (Association Rule Learning)
=============================================================================
İş Problemi
-----------
Türkiye'nin en büyük online hizmet platformu olan Armut, hizmet verenler ile
hizmet almak isteyenleri buluşturmaktadır. Hizmet alan kullanıcıların aldığı
servis ve kategorilere ait veri seti kullanılarak, Association Rule Learning
ile bir hizmet tavsiye sistemi oluşturulması amaçlanmaktadır.

Veri Seti
---------
UserId      : Müşteri numarası
ServiceId   : Her kategoriye ait anonimleştirilmiş servis (örn. Temizlik
              kategorisi altında koltuk yıkama servisi). Aynı ServiceId,
              farklı kategoriler altında farklı hizmetleri ifade edebilir.
CategoryId  : Anonimleştirilmiş kategori (örn. Temizlik, nakliyat, tadilat)
CreateDate  : Hizmetin satın alındığı tarih
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)


# ---------------------------------------------------------------------------
# GÖREV 1: Veriyi Hazırlama
# ---------------------------------------------------------------------------
def load_data(csv_path: str = "data/armut_data.csv") -> pd.DataFrame:
    """Adım 1: armut_data.csv dosyasını okutur."""
    return pd.read_csv(csv_path)


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adım 2: ServiceId ve CategoryId'yi "_" ile birleştirerek her hizmeti
    temsil eden 'Hizmet' değişkenini oluşturur (örn. 4_5).

    Adım 3: Veri setinde sepet (fatura) tanımı bulunmadığından, sepet
    tanımı "bir kullanıcının bir aydaki tüm hizmet alımları" olarak
    kurulur. Yıl-ay bilgisini içeren 'New_Date' ve kullanıcı+ay bazlı
    benzersiz 'SepetID' değişkenleri üretilir (örn. 25446_2017-08).
    """
    df = df.copy()

    # Adım 2
    df["Hizmet"] = df["ServiceId"].astype(str) + "_" + df["CategoryId"].astype(str)

    # Adım 3
    df["CreateDate"] = pd.to_datetime(df["CreateDate"])
    df["New_Date"] = df["CreateDate"].dt.strftime("%Y-%m")
    df["SepetID"] = df["UserId"].astype(str) + "_" + df["New_Date"]

    return df


# ---------------------------------------------------------------------------
# GÖREV 2: Birliktelik Kuralları Üretiniz ve Öneride Bulununuz
# ---------------------------------------------------------------------------
def create_invoice_product_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adım 1: Sepet (SepetID) x Hizmet şeklinde, bir hizmetin o sepette
    alınıp alınmadığını (1/0) gösteren pivot table'ı oluşturur.
    """
    invoice_product_df = (
        df.groupby(["SepetID", "Hizmet"])["Hizmet"]
        .count()
        .unstack()
        .fillna(0)
    )
    return invoice_product_df.astype(bool)


def create_rules(invoice_product_df: pd.DataFrame, min_support: float = 0.01) -> pd.DataFrame:
    """Adım 2: Apriori algoritması ile sık hizmet kombinasyonlarını ve
    bunlardan doğan birliktelik kurallarını (support, confidence, lift...)
    üretir."""
    frequent_itemsets = apriori(invoice_product_df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=min_support)
    return rules


def arl_recommender(rules: pd.DataFrame, product_id: str, rec_count: int = 1) -> list:
    """
    Adım 3: Verilen bir hizmeti (product_id) antecedent (öncül) olarak
    içeren kuralları lift değerine göre büyükten küçüğe sıralar ve
    consequent (ardıl) hizmetlerden benzersiz bir öneri listesi döndürür.

    Parametreler
    ------------
    rules       : association_rules() çıktısı
    product_id  : Öneri temel alınacak hizmet, örn. "2_0"
    rec_count   : Döndürülecek öneri sayısı
    """
    sorted_rules = rules.sort_values("lift", ascending=False)
    recommendations = []

    for _, row in sorted_rules.iterrows():
        if product_id in row["antecedents"]:
            for service in row["consequents"]:
                if service not in recommendations:
                    recommendations.append(service)
        if len(recommendations) >= rec_count:
            break

    return recommendations[:rec_count]


def main():
    df = load_data()
    df = prepare_data(df)

    invoice_product_df = create_invoice_product_df(df)
    rules = create_rules(invoice_product_df)

    print(f"Sepet sayısı: {invoice_product_df.shape[0]}, Hizmet sayısı: {invoice_product_df.shape[1]}")
    print(f"Üretilen kural sayısı: {len(rules)}\n")

    # En son "2_0" hizmetini alan bir kullanıcıya hizmet önerisi
    product_id = "2_0"
    recommendation = arl_recommender(rules, product_id, rec_count=3)
    print(f"'{product_id}' hizmetini alan kullanıcı için önerilen hizmetler: {recommendation}")


if __name__ == "__main__":
    main()

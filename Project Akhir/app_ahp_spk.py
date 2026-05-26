import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SPK AHP – Optimasi Restocking Produk Retail",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# AHP HELPERS
# ─────────────────────────────────────────────
SAATY_RI = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90,
            5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}

SAATY_SCALE = {
    "1 – Sama penting": 1,
    "3 – Sedikit lebih penting": 3,
    "5 – Lebih penting": 5,
    "7 – Sangat lebih penting": 7,
    "9 – Mutlak lebih penting": 9,
    "1/3": 1/3,
    "1/5": 1/5,
    "1/7": 1/7,
    "1/9": 1/9,
}

def ahp_weights(matrix: np.ndarray):
    """Return (weights, lambda_max, CI, CR)."""
    n = matrix.shape[0]
    col_sum = matrix.sum(axis=0)
    norm = matrix / col_sum
    weights = norm.mean(axis=1)
    weighted_sum = matrix @ weights
    lambda_vals = weighted_sum / weights
    lambda_max = lambda_vals.mean()
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0
    ri = SAATY_RI.get(n, 1.49)
    cr = ci / ri if ri != 0 else 0
    return weights, lambda_max, ci, cr

def normalize_minmax(df: pd.DataFrame, cols_benefit: list, cols_cost: list) -> pd.DataFrame:
    out = df.copy()
    for c in cols_benefit:
        mn, mx = df[c].min(), df[c].max()
        out[c] = (df[c] - mn) / (mx - mn) if mx != mn else 0
    for c in cols_cost:
        mn, mx = df[c].min(), df[c].max()
        out[c] = (mx - df[c]) / (mx - mn) if mx != mn else 0
    return out

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("retail_indonesia_55k.csv")
    return df

@st.cache_data
def aggregate_by_product(df):
    grp = df.groupby("nama_produk").agg(
        total_penjualan   = ("total_penjualan",  "sum"),
        rata_profit       = ("profit",            "mean"),
        rata_margin       = ("margin_persen",     "mean"),
        total_qty         = ("qty",               "sum"),
        rata_diskon       = ("diskon_persen",     "mean"),
        kategori          = ("kategori",          "first"),
    ).reset_index()
    grp.columns.name = None
    return grp

df_raw = load_data()
df_prod = aggregate_by_product(df_raw)

CRITERIA = {
    "C1 – Total Penjualan": "total_penjualan",
    "C2 – Rata-rata Profit": "rata_profit",
    "C3 – Rata-rata Margin (%)": "rata_margin",
    "C4 – Total Qty Terjual": "total_qty",
    "C5 – Rata-rata Diskon (%)": "rata_diskon",
}
CRITERIA_COLS = list(CRITERIA.values())
CRITERIA_LABELS = list(CRITERIA.keys())
N = len(CRITERIA)

# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
st.sidebar.image(
    "Logo.png",
    width=250,
)
st.sidebar.title("SPK Restocking Retail")
st.sidebar.caption("Sistem Pendukung Keputusan Berbasis AHP\nOptimasi Keputusan Restocking Produk\nbagi Manajer Retail Indonesia")

page = st.sidebar.radio(
    "Navigasi",
    ["Dataset", "Hitung SPK", "Hasil & Grafik"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Jumlah Data**")
st.sidebar.metric("Total Transaksi", f"{len(df_raw):,}")
st.sidebar.metric("Jumlah Produk", df_prod.shape[0])
st.sidebar.metric("Jumlah Kriteria", N)


# ═══════════════════════════════════════════════
# PAGE 1 – DATASET
# ═══════════════════════════════════════════════
if page == "Dataset":
    st.title("Dataset Transaksi Retail Indonesia")
    st.markdown(
        "Dataset berisi **55.000 transaksi** retail dari berbagai kota di Indonesia. "
        "Untuk keperluan SPK restocking, data diagregasi per produk menjadi **26 alternatif** "
        "yang akan dievaluasi kelayakannya untuk diprioritaskan dalam pengadaan stok."
    )

    tab1, tab2 = st.tabs(["Data Mentah (Raw)", "Data Agregat per Produk"])

    with tab1:
        st.subheader("Data Transaksi Mentah")

        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
            kat_opts = ["Semua"] + sorted(df_raw["kategori"].dropna().unique().tolist())
            kat_sel = st.selectbox("Filter Kategori", kat_opts)
        with col_filter2:
            kota_opts = ["Semua"] + sorted(df_raw["kota"].dropna().unique().tolist())
            kota_sel = st.selectbox("Filter Kota", kota_opts)

        df_view = df_raw.copy()
        if kat_sel != "Semua":
            df_view = df_view[df_view["kategori"] == kat_sel]
        if kota_sel != "Semua":
            df_view = df_view[df_view["kota"] == kota_sel]

        st.caption(f"Menampilkan {len(df_view):,} baris")
        st.dataframe(df_view.head(500), use_container_width=True, height=420)
        st.info("Ditampilkan maks. 500 baris agar performa tetap optimal.")

    with tab2:
        st.subheader("Data Agregat per Produk (Alternatif Restocking)")
        st.markdown(
            "Setiap baris = 1 alternatif produk. Nilai merupakan **agregat** dari seluruh transaksi produk tersebut. "
            "Produk dengan skor AHP tertinggi akan direkomendasikan untuk di-restock terlebih dahulu."
        )
        st.dataframe(df_prod.style.format({
            "total_penjualan": "{:,.0f}",
            "rata_profit": "{:,.0f}",
            "rata_margin": "{:.2f}",
            "total_qty": "{:,.0f}",
            "rata_diskon": "{:.2f}",
        }), use_container_width=True, height=450)


# ═══════════════════════════════════════════════
# PAGE 2 – HITUNG SPK
# ═══════════════════════════════════════════════
elif page == "Hitung SPK":
    st.title("Perhitungan SPK – Metode AHP")
    st.markdown("""
    **Tujuan:** Menentukan prioritas produk yang perlu di-restock terlebih dahulu
    berdasarkan performa penjualan, profitabilitas, dan efisiensi diskon.

    **Langkah AHP:**
    1. Tentukan bobot kriteria lewat **matriks perbandingan berpasangan** (Pairwise Comparison).
    2. Hitung vektor prioritas (bobot) dan uji konsistensi (**CR < 0.10**).
    3. Normalisasi nilai alternatif, lalu hitung **skor akhir** = Σ(bobot × nilai ternormalisasi).
    """)

    st.markdown("---")

    # ── Input Jenis Kriteria ──────────────────────
    st.subheader("① Tentukan Jenis Kriteria")
    st.caption("Benefit = semakin besar semakin baik | Cost = semakin kecil semakin baik")

    col_types = st.columns(N)
    crit_types = {}
    default_types = ["benefit", "benefit", "benefit", "benefit", "cost"]
    for i, (lbl, col_key) in enumerate(CRITERIA.items()):
        with col_types[i]:
            crit_types[col_key] = st.selectbox(
                lbl.split("–")[0].strip(),
                ["benefit", "cost"],
                index=0 if default_types[i] == "benefit" else 1,
                key=f"type_{col_key}",
            )

    st.markdown("---")

    # ── Matriks Perbandingan Berpasangan ──────────
    st.subheader("② Matriks Perbandingan Berpasangan (Pairwise Comparison)")
    st.caption(
        "Isi sel **baris vs kolom** menggunakan skala Saaty (1–9). "
        "Diagonal otomatis = 1. Nilai reciprocal dihitung otomatis."
    )

    SCALE_OPTIONS = list(SAATY_SCALE.keys())
    matrix_vals = np.ones((N, N))

    rows = st.columns([1] * N)
    # Header labels
    header_cols = st.columns([1.5] + [1] * N)
    header_cols[0].markdown("**Kriteria**")
    for j in range(N):
        header_cols[j + 1].markdown(f"**C{j+1}**")

    for i in range(N):
        row_cols = st.columns([1.5] + [1] * N)
        row_cols[0].markdown(f"**C{i+1}** {CRITERIA_LABELS[i].split('–')[1].strip()}")
        for j in range(N):
            if i == j:
                row_cols[j + 1].markdown("**1**")
                matrix_vals[i][j] = 1
            elif i < j:
                sel = row_cols[j + 1].selectbox(
                    "", SCALE_OPTIONS,
                    index=0,
                    key=f"pair_{i}_{j}",
                    label_visibility="collapsed",
                )
                matrix_vals[i][j] = SAATY_SCALE[sel]
                matrix_vals[j][i] = 1 / SAATY_SCALE[sel]
            else:
                val = matrix_vals[i][j]
                if val < 1:
                    disp = f"1/{int(round(1/val))}"
                else:
                    disp = str(int(val))
                row_cols[j + 1].markdown(f"*{disp}*")

    st.markdown("---")

    # ── Top-N Filter ─────────────────────────────
    st.subheader("③ Filter Alternatif")
    col_a, col_b = st.columns(2)
    with col_a:
        top_n = st.slider("Tampilkan Top-N Produk", min_value=5, max_value=26, value=10, step=1)
    with col_b:
        kat_filter = st.multiselect(
            "Filter Kategori Produk",
            options=sorted(df_prod["kategori"].unique().tolist()),
            default=sorted(df_prod["kategori"].unique().tolist()),
        )

    st.markdown("---")

    # ── TOMBOL HITUNG ────────────────────────────
    st.subheader("④ Jalankan Perhitungan")
    hitung_btn = st.button("Hitung AHP Sekarang!", type="primary", use_container_width=True)

    if hitung_btn:
        st.session_state["matrix_vals"] = matrix_vals.copy()
        st.session_state["crit_types"] = crit_types.copy()
        st.session_state["top_n"] = top_n
        st.session_state["kat_filter"] = kat_filter

        # ── Hitung bobot AHP
        weights, lambda_max, ci, cr = ahp_weights(matrix_vals)
        st.session_state["weights"] = weights
        st.session_state["lambda_max"] = lambda_max
        st.session_state["ci"] = ci
        st.session_state["cr"] = cr

        # ── Filter alternatif
        df_alt = df_prod[df_prod["kategori"].isin(kat_filter)].copy()

        # ── Normalisasi
        benefit_cols = [c for c, t in crit_types.items() if t == "benefit"]
        cost_cols    = [c for c, t in crit_types.items() if t == "cost"]
        df_norm = normalize_minmax(df_alt[CRITERIA_COLS], benefit_cols, cost_cols)

        # ── Skor akhir
        skor = df_norm.values @ weights
        df_alt = df_alt.copy()
        df_alt["Skor AHP"] = skor
        df_alt["Peringkat"] = df_alt["Skor AHP"].rank(ascending=False, method="min").astype(int)
        df_alt = df_alt.sort_values("Peringkat")

        st.session_state["df_alt"] = df_alt
        st.session_state["df_norm"] = df_norm
        st.session_state["computed"] = True

        st.success("Perhitungan selesai! Lihat hasil di halaman **Hasil & Grafik**.")

    # ── Tampilkan ringkasan bobot & CR — selalu muncul jika sudah pernah dihitung ──
    if st.session_state.get("computed"):
        saved_matrix = st.session_state["matrix_vals"]
        saved_weights = st.session_state["weights"]
        saved_lambda  = st.session_state["lambda_max"]
        saved_ci      = st.session_state["ci"]
        saved_cr      = st.session_state["cr"]
        saved_crit_types = st.session_state["crit_types"]

        st.markdown("---")
        st.subheader("Hasil Matriks & Bobot Kriteria")

        # Matriks ternormalisasi
        col_sum = saved_matrix.sum(axis=0)
        norm_matrix = saved_matrix / col_sum
        df_matrix_raw = pd.DataFrame(
            saved_matrix,
            index=[f"C{i+1}" for i in range(N)],
            columns=[f"C{j+1}" for j in range(N)],
        )
        df_matrix_norm = pd.DataFrame(
            norm_matrix,
            index=[f"C{i+1}" for i in range(N)],
            columns=[f"C{j+1}" for j in range(N)],
        )

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("**Matriks Perbandingan Berpasangan**")
            st.dataframe(df_matrix_raw.style.format("{:.4f}"), use_container_width=True)
        with col_m2:
            st.markdown("**Matriks Ternormalisasi**")
            st.dataframe(df_matrix_norm.style.format("{:.4f}"), use_container_width=True)

        df_weights = pd.DataFrame({
            "Kode": [f"C{i+1}" for i in range(N)],
            "Kriteria": CRITERIA_LABELS,
            "Tipe": [saved_crit_types[c] for c in CRITERIA_COLS],
            "Bobot (Prioritas)": saved_weights,
            "Bobot (%)": saved_weights * 100,
        })
        st.markdown("**Vektor Prioritas (Bobot Kriteria)**")
        st.dataframe(df_weights.style.format({
            "Bobot (Prioritas)": "{:.4f}",
            "Bobot (%)": "{:.2f}%",
        }), use_container_width=True)

        # Uji Konsistensi
        cr_status = "Konsisten (CR < 0.10)" if saved_cr < 0.10 else "Tidak Konsisten (CR ≥ 0.10)"

        st.markdown("**Uji Konsistensi (Consistency Ratio)**")
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        col_c1.metric("λ max", f"{saved_lambda:.4f}")
        col_c2.metric("CI", f"{saved_ci:.4f}")
        col_c3.metric("RI (n=5)", f"{SAATY_RI[N]:.2f}")
        col_c4.metric("CR", f"{saved_cr:.4f}", delta=cr_status, delta_color="normal")

        if saved_cr >= 0.10:
            st.warning("CR ≥ 0.10! Perbandingan berpasangan kurang konsisten. Silakan revisi matriks.")

    else:
        st.info("Isi matriks perbandingan, lalu klik tombol **Hitung AHP Sekarang!**")


# ═══════════════════════════════════════════════
# PAGE 3 – HASIL & GRAFIK
# ═══════════════════════════════════════════════
elif page == "Hasil & Grafik":
    st.title("Hasil Perangkingan Prioritas Restocking")

    if not st.session_state.get("computed"):
        st.warning("Belum ada hasil. Silakan jalankan perhitungan di halaman **Hitung SPK** terlebih dahulu.")
        st.stop()

    df_alt: pd.DataFrame = st.session_state["df_alt"]
    weights: np.ndarray  = st.session_state["weights"]
    cr: float            = st.session_state["cr"]
    top_n: int           = st.session_state["top_n"]

    # ── Tabel hasil ──────────────────────────────
    st.subheader("Tabel Prioritas Restocking Produk")

    display_cols = ["Peringkat", "nama_produk", "kategori",
                    "total_penjualan", "rata_profit", "rata_margin",
                    "total_qty", "rata_diskon", "Skor AHP"]

    df_show = df_alt[display_cols].head(top_n).copy()
    df_show.columns = ["Peringkat", "Produk", "Kategori",
                       "Total Penjualan", "Avg Profit", "Avg Margin (%)",
                       "Total Qty", "Avg Diskon (%)", "Skor AHP"]

    def color_rank(val):
        if val == 1:   return "background-color:#FFD700; font-weight:bold"
        if val == 2:   return "background-color:#C0C0C0; font-weight:bold"
        if val == 3:   return "background-color:#CD7F32; font-weight:bold"
        return ""

    st.dataframe(
        df_show.style
            .format({
                "Total Penjualan": "{:,.0f}",
                "Avg Profit":      "{:,.0f}",
                "Avg Margin (%)":  "{:.2f}",
                "Total Qty":       "{:,.0f}",
                "Avg Diskon (%)":  "{:.2f}",
                "Skor AHP":        "{:.4f}",
            })
            .applymap(color_rank, subset=["Peringkat"]),
        use_container_width=True,
        height=420,
    )

    st.caption(
        f"CR = **{cr:.4f}** {'Konsisten' if cr < 0.10 else 'Tidak Konsisten'} | "
        f"Menampilkan Top-{top_n} dari {len(df_alt)} produk yang difilter."
    )

    st.markdown("---")

    # ── Grafik ───────────────────────────────────
    st.subheader("Grafik Skor AHP per Produk")

    df_chart = df_alt.head(top_n).sort_values("Skor AHP", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.5)))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    colors_map = {
        "Aksesoris": "#4C9BE8",
        "Footwear":  "#F4A261",
        "Apparel":   "#2A9D8F",
        "Home":      "#E76F51",
        "Stationery":"#E9C46A",
    }
    bar_colors = [colors_map.get(k, "#888") for k in df_chart["kategori"]]

    bars = ax.barh(df_chart["nama_produk"], df_chart["Skor AHP"],
                   color=bar_colors, edgecolor="none", height=0.6)

    # Tambah nilai di ujung bar
    for bar, val in zip(bars, df_chart["Skor AHP"]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=8,
                color="white")

    ax.set_xlabel("Skor AHP", color="white", fontsize=10)
    ax.set_title(f"Top-{top_n} Produk Prioritas Restocking berdasarkan AHP", color="white", fontsize=13, pad=12)
    ax.tick_params(colors="white")
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.xaxis.grid(True, color="#333", linewidth=0.5)
    ax.set_axisbelow(True)

    legend_handles = [mpatches.Patch(color=c, label=k) for k, c in colors_map.items()]
    ax.legend(handles=legend_handles, loc="lower right",
              facecolor="#1a1a2e", edgecolor="none", labelcolor="white", fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")

    # ── Bobot kriteria pie ────────────────────────
    st.subheader("Distribusi Bobot Kriteria")
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    fig2.patch.set_facecolor("#0e1117")
    ax2.set_facecolor("#0e1117")

    pie_colors = ["#4C9BE8","#F4A261","#2A9D8F","#E76F51","#E9C46A"]
    wedges, texts, autotexts = ax2.pie(
        weights,
        labels=[f"C{i+1}" for i in range(N)],
        autopct="%1.1f%%",
        colors=pie_colors,
        startangle=140,
        textprops={"color": "white"},
        wedgeprops={"edgecolor": "#0e1117", "linewidth": 2},
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax2.set_title("Bobot Kriteria AHP", color="white", fontsize=12)
    leg = ax2.legend(
        [f"C{i+1} – {CRITERIA_LABELS[i].split('–')[1].strip()}" for i in range(N)],
        loc="best", fontsize=8, facecolor="#1a1a2e",
        edgecolor="none", labelcolor="white",
    )
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=False)
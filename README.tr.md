# VeroKernel 1.0

**RWKV-7** ve **Transformer (flash) attention** için birleştirilmiş (fused) **Pallas TPU kernel'leri**; ayrıca veri hazırlığı ve CPU referans kernel'leri için **Rust** eklentisi.

*English: [README.md](README.md)*

```python
import verokernel as vk

y = vk.rwkv7(r, lw, k, v, kk, a, doc=doc)                 # RWKV-7 token mixing, (B,T,H,N), türevlenebilir
o = vk.attention(q, k, v, doc, valid, causal=True)         # flash attention, (B,H,T,hd), türevlenebilir
```

Saf JAX fonksiyonlarıdır: `jit`, `grad`, `remat`, Flax / Equinox / Optax ile çalışır. TPU'da Pallas kernel'leri çalışır, diğer her yerde otomatik olarak birebir aynı sonucu veren XLA gerçeklemesine düşer (`backend='auto'`).

## Hız

> Yazarın kendi model eğitiminde bu kernel'lere geçmek yaklaşık **3x daha hızlı eğitim** sağladı (uçtan uca). Bu, tek bir modelde ve tek bir düzende (TPU v5e) ölçülmüş sonuçtur, genel bir garanti değildir: kazanç dizi uzunluğuna, head boyutuna, batch'e ve adım süresinin ne kadarını token mixer'ın aldığına bağlıdır. Kendi durumunuzu ölçmek için kendi donanımınızda `benchmarks/bench.py` çalıştırın.

## Özellikler

**RWKV-7 kernel'i**
- Birleştirilmiş ileri **ve** elle yazılmış geri (backward) kernel'i; yön başına tek `pallas_call`.
- Tasarımdan güvenli sayısal davranış: 16 token'lık üçgensel çözüm blokları (daha uzun bloklar, komşu key'ler birbirine benzediğinde, örn. tekrarlanan metinde, tüm basamakları kaybeder); 8 blok 128 token'lık programlarda paketlenir, böylece her yoğun işlem 128x128 MXU şeklinde bir matmul olur.
- **Doküman izolasyonu**: aynı satıra paketlenmiş birden çok doküman birbirine state sızdırmaz (doküman id'si değişince state sıfırlanır).
- **Padding atlama**: tamamen padding olan programlar hesaplanmaz; padding satırları her zaman tam sıfır döner.
- Transpose'suz yerleşim: `(B,T,H,N)`, `(B,T,H*N)` olarak okunur; 128 lane'lik blok tam head'ler içerir.
- Geri geçiş, ileri geçişte kaydedilen çözüm matrisi tersini kullanır (kapalı formda gradyan); sonuçlar `checkpoint_name('wkv_res')` ile işaretlenir, böylece remat politikası ileri kernel'i yeniden çalıştırmayabilir.

**Attention kernel'i**
- Flash attention ileri + `dq` + `dk/dv` kernel'leri, bf16 veya float32, `(T,T)` skor matrisi yok.
- Doküman maskesi + key geçerlilik maskesi, **causal veya çift yönlü**.
- **Blok atlama**: ilgisiz dokümanların (q-blok, k-blok) çiftleri, tamamen padding olan bloklar ve (causal'da) gelecek bloklar hiç maliyet getirmez.
- İsteğe bağlı maksimum logit çıktısı (`return_max_logit=True`).

**Rust eklentisi** (`verokernel._native`, isteğe bağlı ama wheel'lerin içinde)
- `doc_ids_from_lengths`, `doc_ids_from_starts`, `valid_length`, `block_doc_ranges`: kernel'lerin kullandığı tabloların veri yükleyici tarafında hazırlanması.
- `rwkv7_cpu`, `attention_cpu`: çok iş parçacıklı float32 CPU referans kernel'leri (sadece ileri). Doğrulama, dizüstünde hata ayıklama ve çok küçük CPU çalıştırmaları için.
- Eklenti kurulu değilse NumPy'a otomatik düşer; `verokernel.NATIVE` hangisinin aktif olduğunu söyler.

**Diğer**
- `import verokernel` anında biter ve jax'i import etmez; jax ve kernel'ler ilk kullanımda yüklenir.
- `verokernel.autotune(...)`: Pallas ve XLA'yı cihazda zamanlar, korele key'li doğruluk kapısı ve çökmeye dayanıklı önbellek içerir.
- `verokernel.reference`: kernel'lerin test edildiği birebir XLA gerçeklemeleri.

## Kurulum

```bash
# 1) GitHub Releases sayfasından wheel (Linux / macOS / Windows, Rust eklentisi dahil, Rust kurmak gerekmez)
pip install verokernel-1.0.0-<platform>.whl

# 2) doğrudan depodan (Rust toolchain gerekir: https://rustup.rs)
pip install git+https://github.com/<kullanici>/<depo>.git

# 3) geliştirme kurulumu
pip install maturin && maturin develop --release && pytest tests -q

# TPU VM'de önce TPU destekli JAX'i kurun
pip install "jax[tpu]>=0.11.2" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

Rust eklentisi olmadan (`python/verokernel` klasörünün düz kopyası) her şey NumPy yedeğiyle yine çalışır.

Sürüm yayınlamak için `1.0.0` gibi bir tag push'layın; `build` iş akışı wheel'leri ve kaynak paketini GitHub Release olarak yayınlar.

## API

```python
vk.rwkv7(r, lw, k, v, kk, a, doc=None, *, backend='auto', precision='mixed')
```
- `r, lw, k, v, kk, a`: `(B,T,H,N)`, float32'ye çevrilir. `lw` log decay'dir (`<= 0`), `kk` L2-normalize edilmiş silme key'i, `a` bağlam içi öğrenme oranı.
- `doc`: `(B,T)` int32 doküman id'leri; id değişince state sıfırlanır; `>= vk.PAD_DOC` olanlar padding satırıdır (çıktı sıfır).
- `backend`: `'auto'` | `'pallas'` (desteklenmiyorsa hata verir; TPU dışında yavaş interpret modunda çalışır) | `'xla'`.
- `precision`: `'mixed'` (varsayılan) veya `'high'` (tüm dot'lar 3-geçişli bf16, daha yavaş).

```python
vk.attention(q, k, v, doc=None, valid=None, *, causal=True, block=128, backend='auto', return_max_logit=False)
```
`q, k, v`: `(B,H,T,hd)`; logit `q.k / sqrt(hd)`'dir (qk-norm / pozisyon kodlamasını siz uygulayın). Sorgu, kendi dokümanındaki `valid=True` key'lere bakar (causal ise `key_pos <= query_pos`). Bakacak yeri olmayan sorgular sıfır döner.

TPU mesh'inde veri paralelliği: `'data'` eksenli bir mesh ile `vk.set_mesh(mesh)`.

## Limitler

**RWKV-7**
- Hızlı yol **sadece TPU**. CPU/GPU'da birebir XLA yedeği (daha yavaş). GPU/Triton kernel'i yok.
- Head boyutu `N`, 128'i bölmeli veya 128'in katı olmalı; head sayısı `128/N`'nin katı olmalı (N=64 -> çift `H`; N=32 -> `H % 4 == 0`). `vk.rwkv7_supported((B,T,H,N))` ile kontrol edin; `auto` desteklenmeyen şekilde XLA'ya düşer.
- Float32 kernel (girdiler çevrilir, çıktı float32).
- Başlangıç state'i girişi ve son state çıkışı **yok**: her çağrı sıfır state ile başlar. Bu bir **eğitim / tam dizi** kernel'idir, durum taşıyan artımlı (incremental) decoder değildir.
- `T` içeride 16'nın katına tamamlanır; 128'in katları en hızlısıdır.
- Decay aralığı: `lw <= 0` olmalı. Herhangi bir 16 token'lık blokta toplam `|lw|` yaklaşık 88'in altında kalmalı (float32 `exp` taşması). Float32'de token başına sabit `|lw|` 5'e kadar sonlu ve ~5e-7 doğrulukta çıktı; 6'da NaN oldu. Referans eğitim kurulumu `lw`'yi `[-0.6065, 0)` aralığında kullandı (decay `(0.545, 1)`), yani güvenli aralığın çok içinde.
- Padding'i satırın sonuna koymak en iyisidir (atlanan o kısımdır); ortadaki padding hesaplanır ama sıfırlanır.
- Backward için artık bellek yaklaşık token başına head başına 512 byte (kaydedilen 128x128 tersler) + 128 token başına head başına bir `N x N` state.
- Çözüm blokları bilerek 16 token'da sabit; "daha büyük chunk = daha hızlı" ayarı yok, sayısal olarak güvensiz.

**Attention**
- Head boyutu 8'in katı (64, 128, 256 için lowering ve test edildi); `block` 128'in katı (128 veya 256 önerilir).
- Sadece self-attention (`Tq == Tk`), q/k/v için aynı head sayısı (GQA/MQA yok), softmax ölçeği `1/sqrt(hd)` sabit, dropout yok, ek bias yok, sliding window yok (sadece doküman maskesi), KV-cache decode yolu yok.
- Bu, Transformer attention **kernel'idir**; Hugging Face `transformers` kütüphanesi entegrasyonu değildir.

**Genel**
- Sadece JAX (PyTorch / PyTorch-XLA yok).
- TPU dışında `'pallas'` backend'i Pallas interpret modunda çalışır: doğru ama çok yavaş, sadece testler için.

## Uyumluluk

| | durum |
|---|---|
| Python | 3.10+ |
| JAX | 0.12.0.1 ile doğrulandı (CPU testleri, TPU lowering); yazar tarafından gerçek TPU eğitimlerinde `jax 0.11.2` + `libtpu 0.0.48` çifti. Daha eski JAX sürümleri test edilmedi. JAX'in Mosaic derleyicisinden eski `libtpu`, yüklemede *"Unsupported version"* hatası verir: ikisini birlikte güncelleyin. |
| TPU donanımı | **TPU v5e** üzerinde eğitildi ve ölçüldü (Colab / Kaggle). Diğer TPU nesillerinin çalışması beklenir ama test edilmedi. |
| CPU / GPU | XLA yedeği (JAX'in desteklediği her backend) |
| Framework | her JAX kodu: Flax, Equinox, Haiku, Optax, `jit`, `grad`, `jax.checkpoint` (`vmap` test edilmedi) |
| OS (Rust wheel) | Linux x86_64 derlendi ve test edildi; macOS / Windows / aarch64 wheel'leri dahil CI iş akışıyla üretilir |

## Neler test edildi

- `cargo test`: Rust birim testleri (blok tabloları, doküman id'leri, elle hesaplanmış RWKV-7 adımları, attention maskeleri).
- `pytest tests`: Rust ve NumPy yedeği; Pallas kernel'leri (interpret modu) ile token token recurrence ve yoğun float32 attention karşılaştırması, ileri **ve** gradyan (aynı key'li / tekrarlı metin girdileri, katı olmayan uzunluklar, padding, bf16 dahil); Rust CPU kernel'leri ile JAX; ve her kernel'in (ileri ve geri) `jax.export` ile **TPU için Mosaic lowering**'i.
- Bu deponun CI'ında fiziksel TPU üzerinde çalıştırma **yoktur**. TPU sayısal davranışı ve hız rakamı yazarın eğitim çalıştırmalarından gelir; güvenmeden önce kendi TPU'nuzda `benchmarks/bench.py` ve `autotune` çalıştırın.

## Lisans

MIT (`LICENSE`). RWKV-7, Peng ve ark.'nın mimarisidir; flash attention Dao ve ark.'nı izler; bu depo bağımsız bir gerçeklemedir.

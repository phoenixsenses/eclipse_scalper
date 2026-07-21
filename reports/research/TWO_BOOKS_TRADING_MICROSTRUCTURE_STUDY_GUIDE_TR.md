# İki Kitaplık Trading ve Microstructure Çalışma Kitabı

Kaynaklar:
- Larry Harris, *Trading and Exchanges: Market Microstructure for Practitioners*
- Ernest P. Chan, *Algorithmic Trading: Winning Strategies and Their Rationale*

Amaç: Bu not, iki kitabın birebir kopyası veya telifli metnin yeniden üretimi değildir. Kitapların ana fikirlerini, trader ve sistem geliştirici gözüyle yeniden anlatır. Özellikle Eclipse/S34 için gereken kararları öne çıkarır: veri doğruluğu, sinyal tanımı, execution, maliyet, risk, backtest hataları ve forward validation.

---

## 0. Büyük Resim

İki kitap birlikte şunu söyler:

Bir strateji, sadece “fiyat buradan gider” fikri değildir. Gerçek strateji şu zincirin tamamıdır:

1. Piyasada tekrar eden bir davranış var mı?
2. Bu davranışı veriyle doğru ölçüyor muyuz?
3. Sinyal no-lookahead mı?
4. Entry fiyatı gerçek hayatta alınabilir mi?
5. Spread, fee, slippage, latency ve missed-fill sonrası hâlâ edge var mı?
6. Risk yönetimi edge’i öldürüyor mu, yoksa koruyor mu?
7. Backtest sonucu forward’da tekrarlanıyor mu?
8. Örnek sayısı karar vermeye yeter mi?

Harris kitabı daha çok “piyasa nasıl çalışır?” sorusunu cevaplıyor.
Chan kitabı daha çok “stratejiyi nasıl test eder ve çalıştırırsın?” sorusunu cevaplıyor.

S34 için ana ders:

Liquidation verisini yakalamak sadece ilk adım. Asıl mesele, forced-flow anında gerçekten executable edge olup olmadığını kanıtlamak.

---

# Kitap 1: Trading and Exchanges

## 1. Piyasa Nedir?

Piyasa, alıcı ve satıcıların karşılaştığı mekanik bir yer değildir sadece. Piyasa aynı zamanda bilgi, likidite, risk transferi, maliyet ve rekabet sistemidir.

Bir trader için piyasanın görevi:
- Almak isteyenle satmak isteyeni eşleştirmek.
- Fiyatı bilgiye göre güncellemek.
- Likidite sağlayana risk karşılığı ödeme yaptırmak.
- Acele eden trader’dan sabırlı trader’a değer transfer etmek.

Buradaki en önemli ayrım:

Sabırsız trader genellikle spread ve slippage öder.
Sabırlı trader likidite sağlar ama adverse selection riski taşır.

S34 bağlantısı:

Liquidation anlarında piyasa çok aceleci traderlarla dolar. Bu edge yaratabilir. Ama aynı anda spread açılır, book incelir, fiyat hızlı kayar. Yani sinyal kuvvetlenirken execution zorlaşır.

## 2. Emir Tipleri

Temel emirler:

- Market order: Hemen girer, fiyat garantisi yok.
- Limit order: Fiyat garantisi var, fill garantisi yok.
- Stop order: Belirli seviyede tetiklenir, çoğu zaman market order gibi davranır.
- Market-if-touched: Fiyat bir seviyeye değince market order üretir.

Trader’ın gerçek seçimi şudur:

Hemen gireyim ama kötü fiyat alayım mı?
Bekleyeyim ama fırsatı kaçırayım mı?

S34 bağlantısı:

Mevcut paper runner entry’yi mark price’dan varsayıyor. Bu ne market order ne limit order gerçekliği. Gerçek sistemde üç ayrı model lazım:

- Taker model: spread crossing + taker fee + slippage.
- Maker model: limit order, fill olursa düşük maliyet, ama missed fill riski.
- Passive-then-taker model: önce limit bekle, dolmazsa market gir.

## 3. Market Structure

Piyasalar farklı yapılarda olabilir:

- Dealer market: Dealer alım-satım kotasyonu verir.
- Order-driven market: Emir defteri eşleşir.
- Auction market: Belirli anda tek fiyat veya sürekli açık artırma.
- Crossing network: Emirler belirli kurallarla eşleşir.

Crypto futures çoğunlukla elektronik, order-driven, continuous limit order book yapısındadır.

Bu ne demek?

Fiyat sadece “son işlem” değildir. Fiyat; book depth, spread, queue position, aggressive flow ve passive liquidity’nin etkileşimidir.

S34 bağlantısı:

Liquidation feed bize forced-flow’u gösterir. Ama entry kalitesi için bookTicker/spread/depth gerekir. S34 sadece liquidation+mark ile çalışıyorsa, execution kalitesinin en kritik kısmı eksiktir.

## 4. Trader Tipleri

Harris traderları motivasyona göre ayırır:

- Utilitarian traders: Hedge, yatırım, nakit ihtiyacı gibi sebeplerle trade eder.
- Informed traders: Bilgi avantajıyla trade eder.
- Dealers/liquidity providers: Spread kazanır, inventory ve adverse selection riski taşır.
- Arbitrageurs: Fiyat uyumsuzluklarını kapatır.
- Order anticipators: Büyük emirleri veya forced-flow’u önceden tahmin edip pozisyon alır.
- Bluffers/manipulators: Diğerlerinin algısını değiştirmeye çalışır.
- Futile traders: Sistematik avantajı olmadan trade eder.

S34 açısından en önemli grup:

Order anticipators.

Çünkü S34 de bir çeşit forced-flow/order-anticipation stratejisidir. Liquidation cluster sonrası fiyatın nasıl hareket edeceğini tahmin etmeye çalışır.

Bu iyi haber ve kötü haber:

İyi haber: Bu davranış gerçek piyasa mekanizmasına dayanır.
Kötü haber: Bunu herkes görür; edge hızlı arbitrage edilebilir.

## 5. Informed Trading ve Market Efficiency

Fiyatlar bilgiyle hareket eder. Bilgili traderlar fiyatı doğru değere yaklaştırır. Ama bilgi fiyatlara anında ve kusursuz geçmez.

Edge buradan doğar:

Bilginin fiyatlanması geciktiğinde.
Likidite yetersiz olduğunda.
Traderlar panik veya zorunlu işlem yaptığında.

S34 bağlantısı:

Liquidation bir bilgi mi?

Evet, ama klasik bilgi değil. Bu, “zorunlu işlem oldu” bilgisidir. Yani fiyatın neden hareket ettiğine dair microstructure sinyali verir.

Soru:

Bu bilgi fiyatlanmadan önce yakalanabiliyor mu?
Yoksa liquidation geldiğinde fiyat zaten hareketi bitirmiş mi?

Bu yüzden S34 için latency ve fill modeli kritik.

## 6. Order Anticipators

Order anticipator, başkasının trade edeceğini tahmin edip önden pozisyon alır.

Örnekler:
- Büyük kurumsal emirleri tahmin etmek.
- Stop bölgelerini tahmin etmek.
- Zorunlu liquidation akışını takip etmek.
- Sıkışmış pozisyonların unwind edeceğini görmek.

S34 bağlantısı:

S34 liquidation cluster gördüğünde aslında şunu soruyor:

“Bu forced-flow devam edecek mi, yoksa piyasa bunu absorbe edip dönecek mi?”

Bu iki farklı strateji doğurur:

- Momentum/cascade continuation: Flow aynı yönde devam eder.
- Reversion/absorption: Liquidation sonrası satış/alım tükenir, fiyat geri döner.

Mevcut ETH BUY liq LONG kuralı bu iki yorumdan hangisine dayanıyor açıkça yazılmalı. Eğer BUY forceOrder “short liquidation buyback” ise LONG mantığı momentum olabilir. Eğer yorum tersse strateji yönü ters kurulmuş olabilir.

## 7. Dealers, Spread ve Adverse Selection

Spread sadece komisyon değildir. Spread’in içinde birkaç maliyet vardır:

- Order processing cost
- Inventory risk
- Adverse selection

Adverse selection şudur:

Likidite sağlayıcı, kendinden daha bilgili veya daha agresif bir akışa karşı işlem yapar. Bu yüzden spread ister.

S34 bağlantısı:

Liquidation anında passive entry almak risklidir. Çünkü sen “ucuz fill aldım” sanarken, aslında piyasa senden geçip gidiyor olabilir.

Bu yüzden paper runner şu metrikleri kaydetmeli:

- entry anındaki spread
- entry sonrası ilk 5/10/30 saniyede adverse move
- MFE/MAE
- passive fill olsaydı doldu mu?
- taker girseydik net ne olurdu?

## 8. Liquidity

Likidite dört parçadan oluşur:

- Tightness: Spread ne kadar dar?
- Depth: Yakın fiyatlarda ne kadar miktar var?
- Immediacy: Hemen trade edebilir misin?
- Resiliency: Book şok sonrası ne kadar hızlı toparlanıyor?

S34 için bu doğrudan stop/TP kalitesine bağlanır.

Eğer liquidation anında:
- spread açılıyorsa,
- depth boşalıyorsa,
- price impact artıyorsa,
- book geç toparlanıyorsa,

paper PnL canlıda ciddi bozulabilir.

## 9. Transaction Cost Measurement

Transaction cost sadece fee değildir.

Gerçek maliyet:

`explicit fee + spread crossing + slippage + market impact + missed opportunity + adverse selection`

S34 mevcut net PnL:

`gross_bps - 8 bps`

Bu eksik. 8 bps sadece round-trip fee varsayımı. Spread/slippage yok.

Kitaptan çıkan sistem kararı:

Her paper trade dört ayrı PnL ile raporlanmalı:

- theoretical mark PnL
- taker executable PnL
- passive executable PnL
- latency-adjusted PnL

## 10. Performance Evaluation

Kötü performans ölçümü iyi görünen ama sahte alpha üretir.

En büyük riskler:

- sadece kazanan dönemi seçmek,
- küçük örnekle karar vermek,
- outlier trade’lere dayanmak,
- maliyetleri eksik saymak,
- execution gerçekliğini atlamak.

S34 mevcut durum:

41 closed trade pozitif ortalama veriyor ama:
- median negatif,
- top 3 trade toplam edge’in çoğunu üretiyor,
- 06-07 günü ağır basıyor,
- yeni rejim filtresi henüz forward’da trade açmadı.

Bu “öldü” demek değil. Ama “kanıtlandı” hiç değil.

---

# Kitap 2: Algorithmic Trading

## 1. Backtesting ve Automated Execution

Chan’ın en önemli mesajı:

Backtest çoğu zaman live performanstan daha iyi görünür. Bu yüzden backtest’e şüpheyle yaklaşmalısın.

Backtest’in amacı “beni zengin edecek mi?” değil.

Backtest’in amacı:

- strateji fikri mekanik olarak tutarlı mı?
- veri hatası var mı?
- lookahead var mı?
- maliyet sonrası edge var mı?
- parametreler stabil mi?
- forward test’e değer mi?

S34 bağlantısı:

06-07 completed-day regime filter lookahead idi. Bu yüzden canlıda kullanılamaz.

No-lookahead day-so-far filter doğru yaklaşım. Ama bu filtre de 3 günlük veriyle seçildiği için henüz sadece hipotez.

## 2. Lookahead Bias

Lookahead bias, sistemin karar anında bilmediği bilgiyi kullanmasıdır.

Örnek:

Günün sonunda “bugün range %7 oldu” deyip sabah trade açmak.

S34’de birebir yaşandı:

Completed-day filtre 06-07’yi seçti ama o bilgi gün bitmeden bilinemezdi.

Doğru çözüm:

Rejim metrikleri sadece sinyal anına kadar olan veriden hesaplanmalı:

- day-so-far trend
- day-so-far range
- day-so-far BUY liq notional
- day-so-far agg trade count

Bu artık doğru yönde.

## 3. Data-Snooping Bias

Data-snooping, çok fazla parametre deneyip kazananı seçmek ve bunu gerçek edge sanmaktır.

Örnek:

50 eşik denedin, biri çalıştı. Bu edge olmayabilir; sadece şansa iyi görünmüş olabilir.

S34 bağlantısı:

Regime thresholds:
- trend >= 1%
- range >= 2.5%
- BUY liq >= 5M
- agg count >= 250k

Bu eşikler forward test edilmeden kesin edge değildir.

Kural:

Bu eşikleri sabitle.
Her gün değiştirme.
Forward journal biriktir.

## 4. Statistical Significance

Bir stratejinin kazanması yetmez. Kaç örnekle kazandığı önemlidir.

41 trade ile:

- bir fikir gözlenebilir,
- ama istatistiksel güven oluşmaz.

Yaklaşık sample ihtiyacı:

- 100 trade: erken koklama
- 300 trade: daha ciddi ama hâlâ sınırlı
- 600+ trade: anlamlı istatistik konuşulabilir
- 1000+ trade: canlı risk tartışması daha sağlam olur

S34 için karar:

Bugünkü paper run “proof” değil, “evidence collection”.

## 5. Mean Reversion

Mean reversion fikri:

Fiyat veya spread aşırı uzaklaştığında ortalamaya dönme eğilimindedir.

Ama çalışması için:

- seri stationarity göstermeli,
- half-life ölçülmeli,
- transaction cost düşük olmalı,
- stop mantığı doğru kurulmalı.

S34 bağlantısı:

ETH BUY liq LONG kuralı mean reversion mı momentum mu net değil.

Eğer liquidation sonrası fiyat aşırı satış/alım sonrası dönüyorsa mean reversion.
Eğer liquidation cascade devam ediyorsa momentum.

Bunu ayırmak için her signal sonrası path analizi gerekir:

- ilk 10 saniye ne oluyor?
- ilk 1 dakika ne oluyor?
- 5 dakika sonra ne oluyor?
- MFE/MAE sırası nasıl?

## 6. Cointegration ve Pairs

Chan pairs/cointegration anlatırken şunu öğretir:

İki varlık birlikte hareket ediyor diye trade edilebilir spread oluşmaz. İlişkinin stationarity ve ekonomik mantığı olmalı.

S34/BTC-ETH bağlantısı:

BTC ETH’i etkiliyor olabilir. Ama “BTC önce hareket etti, ETH sonra geldi” gözlemi tek başına edge değil.

Gerekli test:

- BTC move threshold
- ETH response latency
- reverse direction kontrolü
- execution delay sonrası PnL
- out-of-sample stability

Bu PR #9 lead-lag test mantığıyla uyumlu.

## 7. Intraday Momentum

Intraday momentum stratejileri, haber, açılış gap’i veya hızlı flow sonrası devam hareketini yakalamaya çalışır.

Chan’ın mesajı:

Bu stratejilerde zamanlama çok önemlidir. Edge genellikle kısa sürelidir.

S34 bağlantısı:

Liquidation momentum edge’i varsa, latency floor çok önemlidir.

Eğer edge ilk 1-5 saniyede bitiyorsa bizim için kullanılamaz.
Eğer 1-60 dakika arası sürüyorsa, current paper runner daha anlamlı olur.

Bu yüzden S34 için “time-to-edge” raporu şart.

## 8. Risk Management

Chan risk yönetimini stratejinin parçası olarak ele alır. Risk sadece “kaç dolar kaybederim?” değildir.

Risk şu soruları kapsar:

- leverage ne kadar?
- max drawdown ne?
- stop loss gerçekten performansı iyileştiriyor mu?
- position sizing edge’e uygun mu?
- farklı stratejiler aynı anda zarar eder mi?
- volatility rejimi değişince ne olur?

S34 current risk:

- equity 100 USDT
- risk per trade 0.25%
- leverage 10x
- max open 1
- daily max SL 3
- cooldown after 2 SL

Bu güvenli ama şu soru açık:

Risk layer edge’i koruyor mu, yoksa iyi sinyalleri de çok fazla blokluyor mu?

Bu yüzden skipped trades’in “would-have-been PnL” analizi gerekir.

## 9. Stop Loss

Chan’ın stop loss yaklaşımından çıkan önemli ders:

Stop koymak otomatik olarak stratejiyi iyileştirmez. Stop:

- tail risk’i azaltabilir,
- ama noise içinde iyi trade’i erken öldürebilir,
- özellikle mean-reversion sistemlerde zararlı olabilir.

S34 için:

40 bps SL ve 30 bps BE mekanik değerler.

Bunların iyi olup olmadığı şu analizlerle anlaşılır:

- SL olan trade sonra TP’ye gidiyor muydu?
- BE olan trade sonra büyük winner oluyor muydu?
- TP olan trade daha fazla gidebilir miydi?
- TIME exit sonrası fiyat ne yaptı?

Bu “best stop-loss management” için gerçek veri yoludur.

---

# İki Kitabın Ortak Dersleri

## Ders 1: Edge ve Execution Ayrıdır

Sinyal doğru olabilir ama execution kötü olabilir.

S34 örneği:

Liquidation cluster doğru yönü gösterebilir. Ama o anda spread/slippage yüzünden net zarar edebilirsin.

## Ders 2: Paper Fill Gerçek Fill Değildir

Mark price’dan entry, temiz bir araştırma varsayımıdır.

Canlıya yaklaşmak için:

- bid/ask spread gerekir,
- order book depth gerekir,
- taker/maker ayrımı gerekir,
- latency gerekir,
- missed fills gerekir.

## Ders 3: Küçük Sample Tehlikelidir

41 trade ile heyecanlanmak normal.
Ama karar vermek erken.

Özellikle:

- median negatifse,
- outlier kazananlar büyükse,
- tek güne yığılmışsa,
- tek semboldeyse,

edge kırılgan olabilir.

## Ders 4: Rejim Filtresi Gerekli Ama Tehlikeli

Rejim filtresi noise günlerini keser.
Ama geçmişe bakarak seçilirse overfit olur.

Çözüm:

Bir kez seç.
Sabitle.
Forward’da ölç.

## Ders 5: Risk Yönetimi Edge’i Saklayabilir

Risk gate çok sıkıysa kötü trade’leri keser ama iyi trade’leri de kaçırabilir.

S34 için her skip reason ayrı test edilmeli:

- REGIME_FILTER doğru mu blokladı?
- DAILY_MAX_SL iyi mi korudu?
- COOLDOWN kazananları kaçırdı mı?
- MAX_OPEN_TRADES ikinci iyi setup’ı engelledi mi?

## Ders 6: Piyasa Rakiplerle Doludur

S34’ün gördüğü liquidation bilgisini sadece biz görmüyoruz.

Bu yüzden edge şuradan gelmeli:

- daha doğru yorum,
- daha iyi rejim filtresi,
- daha iyi entry timing,
- daha iyi execution,
- daha iyi stop/BE yönetimi.

Sadece “liq geldi, long aç” yeterli değil.

---

# S34 İçin Kitaplardan Çıkan Uygulama Planı

## A. Veri Doğruluğu

1. Binance forceOrder side semantiğini doğrula.
2. BUY liquidation ekonomik olarak ne demek kesin yaz.
3. DB’de side alanı ham mı, çevrilmiş mi ayır.
4. Her signal’a raw/liquidation interpretation etiketi ekle.

## B. Signal Path

1. Signal yalnızca liquidations üzerinden mi, agg ile mi?
2. Mevcut runner’da agg sadece regime count.
3. Sinyal için ayrı, rejim için ayrı, execution için ayrı feature set yaz.

## C. Execution Model

Her paper trade şu modlarda raporlansın:

- mark model
- taker model
- maker model
- passive-then-taker model
- latency 0s / 2s / 5s / 8s / 15s

## D. Trade Path Analytics

Her trade için:

- MFE
- MAE
- time-to-MFE
- time-to-MAE
- time-to-TP
- time-to-SL
- BE helped/hurt
- post-exit drift

## E. Risk Gate Audit

Her skipped trade için:

- neden skip oldu?
- eğer girseydi ne olurdu?
- gate para mı kurtardı, para mı kaçırdı?

## F. Sample Gate

Karar seviyeleri:

- 0-100 trade: sadece gözlem
- 100-300 trade: erken hipotez
- 300-600 trade: preliminary edge
- 600-1000 trade: istatistik konuşulabilir
- 1000+ trade: canlı risk tartışması yapılabilir

---

# Benim Net Sonucum

Bu iki kitabı S34’e uygularsak karar şu:

S34 fikri çöpe atılacak bir fikir değil. Forced-flow ve liquidation microstructure gerçek bir piyasa davranışına dayanıyor. Ama şu anki evidence “alpha kanıtlandı” seviyesinde değil.

Şu an doğru strateji:

- liquidation data akmaya devam etsin,
- no-lookahead regime filter sabit kalsın,
- paper runner trade üretirse journal tutsun,
- execution realism eklenmeden live risk artırılmasın,
- stop/BE/TP ancak MFE/MAE analiziyle optimize edilsin,
- 41 trade değil, yüzlerce forward trade beklensin.

Kısaca:

Harris bize “piyasada asıl para execution ve liquidity gerçekliğinde kaybolur” diyor.
Chan bize “backtest/paper sonucu ancak bias ve maliyetlerden sağ çıkarsa anlamlıdır” diyor.

S34 için en iyi yol:

Yeni entry aramak değil; mevcut entry’nin gerçek hayatta executable olup olmadığını kanıtlamak.

